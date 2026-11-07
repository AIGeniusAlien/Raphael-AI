import unittest
from scripts.audit_dependency_versions import audit_distribution


class DependencyAuditTests(unittest.TestCase):
    def test_local_cpu_version_is_checked_against_public_version(self):
        urls = []
        result = audit_distribution('torch', '2.6.0+cpu', lambda url: urls.append(url) or {
            'vulnerabilities': [{'id': 'active', 'withdrawn': None}, {'id': 'withdrawn', 'withdrawn': '2026-01-01'}]})
        self.assertEqual(urls, ['https://pypi.org/pypi/torch/2.6.0/json'])
        self.assertEqual(result['status'], 'vulnerable')
        self.assertEqual([v['id'] for v in result['vulnerabilities']], ['active'])

    def test_failed_or_missing_metadata_never_passes(self):
        for fetch in (lambda url: {}, lambda url: {'vulnerabilities': None}):
            self.assertEqual(audit_distribution('torch', '2.15.0+cpu', fetch)['status'], 'error')
        def offline(url):
            raise OSError('offline')
        self.assertEqual(audit_distribution('torch', '2.15.0+cpu', offline)['status'], 'error')

    def test_empty_advisories_preserve_exact_installed_version(self):
        result = audit_distribution('torch', '2.15.0+cpu', lambda url: {'vulnerabilities': []})
        self.assertEqual(result['installed_version'], '2.15.0+cpu')
        self.assertEqual(result['status'], 'no_public_advisories')


class LocalMLCompatibilityTests(unittest.TestCase):
    def test_installed_local_stack_runs_without_model_downloads(self):
        import importlib.util
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers', 'sentence_transformers')):
            self.skipTest('Optional local ML runtime is absent in the lightweight profile')
        import numpy as np
        import torch
        from transformers import BertConfig, BertModel
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import BoW
        from rag import embedder
        from unittest.mock import patch
        self.assertTrue(torch.equal(torch.tensor([1, 2]) + torch.tensor([3, 4]), torch.tensor([4, 6])))
        # Random tiny synthetic model checks API compatibility, not medical quality.
        bert = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                                    num_attention_heads=2, intermediate_size=32))
        with torch.no_grad():
            result = bert(input_ids=torch.tensor([[1, 2, 3]]))
        self.assertEqual(tuple(result.last_hidden_state.shape), (1, 3, 16))
        local = SentenceTransformer(modules=[BoW(['synthetic', 'local', 'tensor'])], device='cpu')
        with patch.object(embedder, '_MODEL', local):
            vectors = embedder.embed_texts(['synthetic local', 'tensor'], 'unused-no-download')
        self.assertEqual(vectors.shape, (2, 3))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1], atol=1e-6)
