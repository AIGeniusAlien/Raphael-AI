"""Port-routing regression uses real SDK construction without any requests."""
import unittest
from qdrant_client import QdrantClient
from qdrant_config import connection_options


class QdrantRouteTests(unittest.TestCase):
    def test_https_default_and_explicit_ports_reach_the_requested_route(self):
        for url, expected in [('https://cluster.invalid',443),('https://cluster.invalid:443',443),('https://cluster.invalid:6333',6333),('https://cluster.invalid:8443',8443),('http://localhost:6333',6333)]:
            with self.subTest(url=url):
                client=QdrantClient(url=url,**connection_options(url))
                try:self.assertEqual(client._client._port,expected)
                finally:client.close()

    def test_known_different_encoder_is_rejected_even_when_dimensions_match(self):
        from qdrant_config import verify_legacy_encoder
        from qdrant_client.http import models as qm
        client=QdrantClient(location=':memory:')
        try:
            client.create_collection('synthetic',vectors_config=qm.VectorParams(size=384,distance=qm.Distance.COSINE))
            client.upsert('synthetic',[qm.PointStruct(id=1,vector=[1.0]*384,payload={'embedding_model':'other-provider-model'})])
            with self.assertRaises(ValueError):verify_legacy_encoder(client,'synthetic','expected-model')
            self.assertEqual(client.count('synthetic').count,1)
        finally:client.close()
