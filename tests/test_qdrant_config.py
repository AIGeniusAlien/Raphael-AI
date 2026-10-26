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
