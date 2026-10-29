"""Local real Qdrant plus synthetic vectors/HTML; never live proof."""
import json
import tempfile
import unittest
from unittest.mock import patch,MagicMock
from types import SimpleNamespace as NS
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import curated_evidence as ce
from rag import hosted_embeddings as he


class CuratedEvidenceTests(unittest.TestCase):
    def test_hybrid_rerank_promotes_lexical_support_and_excludes_instruction_only_matches(self):
        def hit(identifier,body):return NS(payload={'chunk_id':identifier,'text':body},score=.7)
        hits=[hit('first','Antibiotics storage fixture.'),hit('second','Antibiotics do not work against viruses. Viruses fixture.'),hit('third','General education requires clinician review. Source date fixture.')]
        result=ce.hybrid_rerank('Do antibiotics work against viruses? Cite source date and state general education requiring clinician review.',hits,2)
        self.assertEqual(result[0]['hit'].payload['chunk_id'],'second')
        self.assertEqual(result[0]['semantic_rank'],2);self.assertEqual(result[0]['lexical_rank'],1)
        self.assertNotIn('third',[item['hit'].payload['chunk_id'] for item in result])
        self.assertEqual(ce.hybrid_rerank('Cite source date general education',hits,6),[])
        self.assertEqual(ce.hybrid_rerank('xylophone gyroscope',hits,6),[])

    def source(self,text=None):
        text=text or ('Antibiotic synthetic fixture guidance,not real medical evidence. '*7)
        return ce.parse_source(ce.URLS[0],'<h1>Synthetic fixture</h1><time datetime="2024-01-01"></time><div class="cdc-dfe-body"><p>'+text+'</p></div>','fixture-policy-hash')

    def test_parse_limits_and_preserves_source_body_date_and_reuse_metadata(self):
        source=self.source();self.assertIn('not real medical evidence',source['text']);self.assertEqual(source['source_date'],'2024-01-01');self.assertEqual(source['license_url'],ce.POLICY)
        with self.assertRaises(ValueError):ce.parse_source(ce.URLS[0],'<h1>No body or date</h1>','test')
        self.assertEqual(list(ce.chunks('a'*2200))[0],'a'*1100)

    def test_actual_schema_and_embedding_space_mismatch_never_recreates_collection(self):
        for size,tag in ((8,ce.MODEL),(384,'another-model')):
            c=QdrantClient(location=':memory:')
            try:
                c.create_collection(ce.collection(),vectors_config=qm.VectorParams(size=size,distance=qm.Distance.COSINE))
                c.upsert(ce.collection(),[qm.PointStruct(id=1,vector=[1.0]*size,payload={'embedding_model':tag,'embedding_dimensions':size})])
                with self.assertRaises(ValueError):ce.ensure_compatible(c,np.ones((1,384)))
                with patch.object(c,'create_payload_index') as add_index:
                    with self.assertRaises(ValueError):ce.ensure_compatible(c,np.ones((1,384)),allow_create=True)
                    add_index.assert_not_called()
                self.assertEqual(c.count(ce.collection()).count,1)
            finally:c.close()

    def test_ingestion_repeat_search_missing_evidence_and_archive_preserve_versions(self):
        with tempfile.TemporaryDirectory(prefix='raphael-local-corpus-') as path:
            with patch.object(ce,'client',side_effect=lambda:QdrantClient(path=path)),patch.object(ce,'encode',side_effect=lambda texts:np.ones((len(texts),384),dtype=np.float32)):
                source=self.source();first=ce.ingest([source]);repeat=ce.ingest([source])
                self.assertEqual(first['active_documents'],1);self.assertEqual(repeat['vectors_written'],0);self.assertEqual(repeat['vectors_after'],first['vectors_after'])
                self.assertTrue(ce.search('antibiotic fixture'));self.assertEqual(ce.search('unfindable xylophone gyro'),[])
                hit=ce.search('antibiotic fixture')[0];self.assertEqual(ce.citation(hit['chunk_id'])['content_sha256'],source['content_sha256'])
                changed=self.source('Antibiotic revised synthetic fixture content,not real clinical guidance. '*8);after=ce.ingest([changed])
                self.assertGreater(after['vectors_after'],first['vectors_after']);self.assertEqual(after['active_documents'],1)
                self.assertIsNone(ce.citation(hit['chunk_id']))

    def test_withdrawal_preserves_vectors_excludes_search_and_requires_readmission_review(self):
        with tempfile.TemporaryDirectory() as path:
            with patch.object(ce,'client',side_effect=lambda:QdrantClient(path=path)),patch.object(ce,'encode',side_effect=lambda texts:np.ones((len(texts),384),dtype=np.float32)):
                source=self.source();first=ce.ingest([source]);hit=ce.search('antibiotic fixture')[0]
                result=ce.withdraw_source(source['url'],'Synthetic explicit safety review withdrawal')
                self.assertEqual(result['vectors_preserved'],first['vectors_after'])
                self.assertEqual(ce.search('antibiotic fixture'),[]);self.assertIsNone(ce.citation(hit['chunk_id']))
                self.assertEqual(ce.withdraw_source(source['url'],'Synthetic repeat withdrawal')['writes'],0)
                self.assertEqual(ce.refresh_status([source])['sources'][0]['status'],'withdrawn_requires_review')
                with self.assertRaisesRegex(ValueError,'Withdrawn'):ce.ingest([source])

    def test_failed_new_version_batch_leaves_old_active_and_retry_publishes_complete_version(self):
        with tempfile.TemporaryDirectory() as path:
            factory=lambda:QdrantClient(path=path)
            with patch.object(ce,'client',side_effect=factory),patch.object(ce,'encode',side_effect=lambda texts:np.ones((len(texts),384),dtype=np.float32)):
                old=self.source();ce.ingest([old]);old_hit=ce.search('antibiotic fixture')[0]
                revised=self.source('Antibiotic revised synthetic fixture text. '*1800)
                def failing_client():
                    client=factory();original=client.upsert;calls=[]
                    def upsert(*args,**kwargs):
                        calls.append(1)
                        if len(calls)==2:raise RuntimeError('Synthetic bounded provider failure')
                        return original(*args,**kwargs)
                    client.upsert=upsert;return client
                with patch.object(ce,'client',side_effect=failing_client):
                    with self.assertRaises(RuntimeError):ce.ingest([revised])
                self.assertIsNotNone(ce.citation(old_hit['chunk_id']))
                self.assertTrue(all(hit['content_sha256']==old['content_sha256'] for hit in ce.search('antibiotic fixture')))
                recovered=ce.ingest([revised]);self.assertGreater(recovered['vectors_deduplicated'],0)
                self.assertIsNone(ce.citation(old_hit['chunk_id']))
                self.assertTrue(all(hit['content_sha256']==revised['content_sha256'] for hit in ce.search('antibiotic fixture')))

    def test_refresh_is_read_only_and_older_source_date_cannot_replace_current(self):
        with tempfile.TemporaryDirectory() as path:
            with patch.object(ce,'client',side_effect=lambda:QdrantClient(path=path)),patch.object(ce,'encode',side_effect=lambda texts:np.ones((len(texts),384),dtype=np.float32)):
                source=self.source();ce.ingest([source]);self.assertEqual(ce.refresh_status([source])['sources'][0]['status'],'unchanged')
                changed=self.source('Antibiotic changed synthetic fixture. '*10)
                self.assertEqual(ce.refresh_status([changed])['sources'][0]['status'],'update_available')
                older=dict(changed,source_date='2023-01-01')
                with self.assertRaisesRegex(ValueError,'older source date'):ce.ingest([older])
                self.assertEqual(ce.search('antibiotic fixture')[0]['content_sha256'],source['content_sha256'])

    def test_hosted_adapter_rejects_wrong_model_dimensions_indices_and_zero_vectors(self):
        for model,data in [('another-model',[NS(index=0,embedding=[1.0]*384)]),(he.MODEL,[NS(index=0,embedding=[1.0]*8)]),(he.MODEL,[NS(index=1,embedding=[1.0]*384)]),(he.MODEL,[NS(index=0,embedding=[0.0]*384)])]:
            with patch.dict('os.environ',{'LLM_BASE_URL':'https://provider.invalid/v1','LLM_API_KEY':'synthetic-only'}),patch.object(he,'OpenAI') as sdk:
                sdk.return_value.__enter__.return_value.embeddings.create.return_value=NS(model=model,data=data)
                with self.assertRaises(ValueError):he.encode(['synthetic probe'])
                self.assertEqual(sdk.call_args.kwargs['max_retries'],0)
