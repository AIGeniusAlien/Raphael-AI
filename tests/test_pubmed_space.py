"""Local synthetic Qdrant safety;not live PubMed or medical validation."""
import asyncio
import unittest
from unittest.mock import patch,AsyncMock
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from pubmed import ingest
from pubmed import client as discovery


class PubMedSpaceTests(unittest.TestCase):
    def test_cached_encoder_cannot_silently_reuse_another_models_vectors(self):
        first,second=object(),object()
        constructor=Mock(side_effect=[first,second])
        ingest.get_model.cache_clear();self.addCleanup(ingest.get_model.cache_clear)
        with patch.dict('sys.modules',{'sentence_transformers':SimpleNamespace(SentenceTransformer=constructor)}):
            self.assertIs(ingest.get_model('model-a'),first)
            self.assertIs(ingest.get_model('model-a'),first)
            self.assertIs(ingest.get_model('model-b'),second)
        self.assertEqual([call.args[0] for call in constructor.call_args_list],['model-a','model-b'])

    def test_same_dimensions_different_or_untagged_model_never_mutates(self):
        for model in ('another-model',None):
            client=QdrantClient(location=':memory:');self.addCleanup(client.close)
            client.create_collection('fixture',vectors_config=qm.VectorParams(size=3,distance=qm.Distance.COSINE))
            client.upsert('fixture',[qm.PointStruct(id=1,vector=[1.,1.,1.],payload={'embedding_model':model,'embedding_dimensions':3})])
            with self.assertRaises(ValueError):ingest.ensure_collection(client,'fixture',3,'expected-model')
            self.assertEqual(client.count('fixture').count,1)
            self.assertEqual(client.scroll('fixture')[0][0].payload['embedding_model'],model)

    def test_actual_encoder_dimension_mismatch_does_not_create_target(self):
        client=QdrantClient(location=':memory:');self.addCleanup(client.close)
        with patch.object(client,'close'),patch.object(ingest,'QdrantClient',return_value=client),patch.object(ingest,'esearch_pubmed',new=AsyncMock(return_value=['123'])),patch.object(ingest,'esummary_pubmed',new=AsyncMock(return_value={'result':{'123':{'title':'Synthetic fixture'}}})),patch.object(ingest,'embed_texts',return_value=np.ones((1,8))):
            with self.assertRaises(ValueError):asyncio.run(ingest.ingest_pubmed_query(qdrant_url='https://unused.invalid',qdrant_api_key='',collection='fixture',embed_model='synthetic',embed_dim=3,query='synthetic',retmax=1))
        self.assertEqual(client.get_collections().collections,[])

    def test_unbounded_discovery_rejected_before_client_or_network(self):
        with patch.object(ingest,'QdrantClient') as client:
            for limit in (0,21,True):
                with self.assertRaises(ValueError):asyncio.run(ingest.ingest_pubmed_query(qdrant_url='https://unused.invalid',qdrant_api_key='',collection='fixture',embed_model='synthetic',embed_dim=3,query='synthetic',retmax=limit))
            client.assert_not_called()

    def test_explicit_retraction_metadata_is_excluded_without_creating_collection(self):
        client=QdrantClient(location=':memory:');self.addCleanup(client.close)
        with patch.object(client,'close'),patch.object(ingest,'QdrantClient',return_value=client),patch.object(ingest,'esearch_pubmed',new=AsyncMock(return_value=['123'])),patch.object(ingest,'esummary_pubmed',new=AsyncMock(return_value={'result':{'123':{'title':'Synthetic retracted fixture','pubtype':['Retracted Publication']}}})),patch.object(ingest,'embed_texts') as encoder:
            result=asyncio.run(ingest.ingest_pubmed_query(qdrant_url='https://unused.invalid',qdrant_api_key='',collection='fixture',embed_model='synthetic',embed_dim=3,query='synthetic',retmax=1))
            self.assertEqual(result['chunks_indexed'],0);self.assertEqual(result['excluded_records'][0]['pmid'],'123')
            self.assertFalse(result['full_text_reuse_verified']);encoder.assert_not_called()
        self.assertEqual(client.get_collections().collections,[])

    def test_retracted_existing_record_is_retained_but_marked_inactive(self):
        client=QdrantClient(location=':memory:');self.addCleanup(client.close)
        client.create_collection('fixture',vectors_config=qm.VectorParams(size=3,distance=qm.Distance.COSINE))
        identifier=ingest.point_id_for(ingest.sha256('pubmed|123'))
        client.upsert('fixture',[qm.PointStruct(id=identifier,vector=[1.,1.,1.],payload={'embedding_model':'synthetic','embedding_dimensions':3,'text':'Historical synthetic discovery record'})])
        with patch.object(client,'close'),patch.object(ingest,'QdrantClient',return_value=client),patch.object(ingest,'esearch_pubmed',new=AsyncMock(return_value=['123'])),patch.object(ingest,'esummary_pubmed',new=AsyncMock(return_value={'result':{'123':{'pubtype':['Retracted Publication']}}})),patch.object(ingest,'embed_texts') as encoder:
            asyncio.run(ingest.ingest_pubmed_query(qdrant_url='https://unused.invalid',qdrant_api_key='',collection='fixture',embed_model='synthetic',embed_dim=3,query='synthetic',retmax=1))
            encoder.assert_not_called()
        self.assertEqual(client.count('fixture').count,1)
        record=client.scroll('fixture')[0][0].payload
        self.assertTrue(record['retracted']);self.assertFalse(record['is_current'])
        self.assertEqual(record['text'],'Historical synthetic discovery record')

    def test_discovery_rejects_invalid_or_excessive_returned_identifiers(self):
        for identifiers in (['123','456'],['123','123'],[{}],['１２３'],['']):
            with patch.object(discovery,'_get_json',new=AsyncMock(return_value={'esearchresult':{'idlist':identifiers}})):
                with self.assertRaises(ValueError):asyncio.run(discovery.esearch_pubmed('synthetic',retmax=1))

    def test_metadata_response_body_is_bounded(self):
        transport=httpx.MockTransport(lambda request:httpx.Response(200,content=b' ' * 2_000_001))
        client=httpx.AsyncClient(transport=transport)
        with patch.object(discovery.httpx,'AsyncClient',return_value=client):
            with self.assertRaises(ValueError):asyncio.run(discovery._get_json('esearch.fcgi',{}))
