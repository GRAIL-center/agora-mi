from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.cluster_stats import cluster_bootstrap_selection_frequency, cluster_permutation_pvalues
from data.io import load_agora_records
from data.matching import MatchConfig, greedy_match_rows, pairwise_text_similarity
from data.public_values import assign_public_value_families, load_public_value_spec, proxy_slug


class PublicValuePipelineTests(unittest.TestCase):
    def test_proxy_slug(self):
        self.assertEqual(proxy_slug("Risk factors: Safety"), "risk_factors_safety")

    def test_family_assignment_uses_yaml(self):
        spec = load_public_value_spec(ROOT / "configs" / "public_value_families.yaml")
        hits = assign_public_value_families(
            ["Risk factors: Safety", "Harms: Ecological harm"],
            spec,
            include_secondary=True,
        )
        self.assertIn("sustainability", hits)
        self.assertEqual(hits["sustainability"]["tier"], "mixed")

    def test_agora_records_include_matching_metadata(self):
        records = load_agora_records(ROOT / "data" / "raw" / "agora")
        row = records[0]
        self.assertIn("jurisdiction", row)
        self.assertIn("document_form", row)
        self.assertIn("application_tags", row)
        self.assertIn("collection_values", row)
        self.assertIn("tags", row)
        self.assertIn("all_tags", row)
        self.assertNotIn("binary_tags", row)
        self.assertNotIn("segment_binary_tags", row)
        self.assertNotIn("document_binary_tags", row)

    def test_greedy_matching_prefers_stronger_candidate(self):
        positives = [
            {
                "segment_id": "p1",
                "text": "privacy rights and accountability",
                "metadata": {
                    "authority": "A",
                    "jurisdiction": "us_federal",
                    "document_form": "law",
                    "year": 2024,
                    "application_tags": ["Applications: Security"],
                },
            }
        ]
        candidates = [
            {
                "segment_id": "n1",
                "text": "privacy and accountability",
                "metadata": {
                    "authority": "A",
                    "jurisdiction": "us_federal",
                    "document_form": "law",
                    "year": 2024,
                    "application_tags": ["Applications: Security"],
                },
            },
            {
                "segment_id": "n2",
                "text": "subsidies for innovation pilots",
                "metadata": {
                    "authority": "B",
                    "jurisdiction": "other",
                    "document_form": "policy_document",
                    "year": 2018,
                    "application_tags": ["Applications: Transportation"],
                },
            },
        ]
        sim = pairwise_text_similarity(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        matched, diag = greedy_match_rows(positives, candidates, similarity_matrix=sim, config=MatchConfig(method="tfidf"))
        self.assertEqual(matched[0]["segment_id"], "n1")
        self.assertAlmostEqual(diag["authority_match_rate"], 1.0)

    def test_cluster_stats_return_feature_length(self):
        pos = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0]], dtype=np.float32)
        neg = np.asarray([[0.0, 1.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
        pvals = cluster_permutation_pvalues(
            pos,
            neg,
            pos_cluster_ids=np.asarray(["a", "a", "b"], dtype=object),
            neg_cluster_ids=np.asarray(["c", "d", "d"], dtype=object),
            n_perm=10,
            seed=0,
        )
        freqs = cluster_bootstrap_selection_frequency(
            pos,
            neg,
            pos_cluster_ids=np.asarray(["a", "a", "b"], dtype=object),
            neg_cluster_ids=np.asarray(["c", "d", "d"], dtype=object),
            topk=1,
            n_boot=10,
            seed=0,
        )
        self.assertEqual(pvals.shape[0], 2)
        self.assertEqual(freqs.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
