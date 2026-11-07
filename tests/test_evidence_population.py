"""Real local Qdrant and synthetic corpus only;no remote mutation or embeddings."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from qdrant_client import QdrantClient
import curated_evidence as ce


class EvidencePopulationTests(unittest.TestCase):
    def setup_corpus(self,path):
        def source(url):
            return ce.parse_source(url,'<h1>Synthetic test source</h1><time datetime="2024-01-01"></time><div class="cdc-dfe-body"><p>'+('Antibiotic synthetic fixture population test,not medical guidance. '*6)+'</p></div>','test-policy')
        factory=lambda:QdrantClient(path=path)
        with patch.object(ce,'client',side_effect=factory),patch.object(ce,'encode',side_effect=lambda texts:np.ones((len(texts),384),dtype=np.float32)):
            ce.ingest([source(url) for url in ce.URLS[:3]])
        return factory

    def test_reviewed_catalog_has_bounded_verified_specialty_scopes(self):
        self.assertEqual(len(ce.URLS),9)
        self.assertEqual(ce.source_population(ce.URLS[1]),'adult')
        self.assertEqual(ce.source_population(ce.URLS[2]),'pediatric')
        self.assertEqual(ce.source_population(ce.URLS[6]),'adult')
        self.assertEqual(ce.source_population(ce.URLS[7]),'general')
        self.assertEqual(ce.source_population(ce.URLS[8]),'general')
        self.assertEqual(ce.source_population('https://example.org/pediatric.html'),'unknown')
        with self.assertRaises(ValueError):ce.source_scope('https://example.org/unreviewed')

    def test_population_filter_excludes_opposite_scope_and_marks_unknown_age(self):
        with tempfile.TemporaryDirectory(prefix='raphael-population-') as path:
            factory=self.setup_corpus(path)
            with patch.object(ce,'client',side_effect=factory),patch.object(ce,'encode',side_effect=lambda texts:np.ones((len(texts),384),dtype=np.float32)):
                adult=ce.search('antibiotic fixture',population='adult');pediatric=ce.search('antibiotic fixture',population='pediatric');unknown=ce.search('antibiotic fixture')
                self.assertEqual({item['population'] for item in adult},{'adult','general'})
                self.assertEqual({item['population'] for item in pediatric},{'pediatric','general'})
                self.assertEqual(len(unknown),3)
                self.assertTrue(all(item['population_applicability']=='patient_population_unknown_requires_review' for item in unknown if item['population']!='general'))
                with self.assertRaises(ValueError):ce.search('antibiotic fixture',population='any-claimed-age')

    def test_explicit_migration_preserves_vectors_ids_content_and_repeat_is_noop(self):
        with tempfile.TemporaryDirectory(prefix='raphael-population-migration-') as path:
            factory=self.setup_corpus(path);client=factory()
            try:
                before,_=client.scroll(ce.collection(),limit=100,with_payload=True,with_vectors=True)
                original={str(p.id):(p.vector,p.payload['text'],p.payload['content_sha256']) for p in before}
                client.delete_payload(ce.collection(),keys=['population','specialty','population_scope_note','population_tag_version'],points=[p.id for p in before])
            finally:client.close()
            with patch.object(ce,'client',side_effect=factory),patch.object(ce,'encode',side_effect=lambda texts:np.ones((len(texts),384),dtype=np.float32)):
                with self.assertRaises(ValueError):ce.search('antibiotic fixture',population='adult')
                migrated=ce.migrate_population_metadata();repeat=ce.migrate_population_metadata()
                self.assertEqual(migrated['points_metadata_updated'],3);self.assertEqual(migrated['vectors_reembedded'],0)
                self.assertEqual(repeat['points_metadata_updated'],0)
                self.assertEqual(migrated['points_before'],migrated['points_after'])
                self.assertEqual({p['population'] for p in ce.search('antibiotic fixture',population='adult')},{'adult','general'})
            client=factory()
            try:
                after,_=client.scroll(ce.collection(),limit=100,with_payload=True,with_vectors=True)
                self.assertEqual(original,{str(p.id):(p.vector,p.payload['text'],p.payload['content_sha256']) for p in after})
            finally:client.close()

    def test_conflicting_tag_refuses_all_changes_before_first_write(self):
        with tempfile.TemporaryDirectory(prefix='raphael-population-conflict-') as path:
            factory=self.setup_corpus(path);client=factory()
            try:
                points,_=client.scroll(ce.collection(),limit=100,with_payload=True)
                general=next(p for p in points if p.payload['population']=='general');adult=next(p for p in points if p.payload['population']=='adult')
                client.delete_payload(ce.collection(),keys=['population'],points=[general.id])
                client.set_payload(ce.collection(),payload={'population':'pediatric'},points=[adult.id])
                with patch.object(ce,'client',return_value=client),patch.object(client,'close'),patch.object(client,'set_payload') as write,patch.object(client,'create_payload_index') as index:
                    with self.assertRaises(ValueError):ce.migrate_population_metadata()
                    write.assert_not_called();index.assert_not_called()
                self.assertEqual(client.count(ce.collection()).count,3)
            finally:client.close()
