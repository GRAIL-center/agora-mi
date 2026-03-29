from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assistant.documents import normalize_document_input, segment_document
from assistant.experiments import _build_family_cards, _summary_table_rows
from assistant.render import build_document_summary_note, build_segment_note


class PolicyAnalysisAssistantTests(unittest.TestCase):
    def test_normalize_document_input_with_text(self) -> None:
        payload = "Paragraph one.\n\nParagraph two."
        document = normalize_document_input(payload, document_id="doc_x", title="Example", source_type="plain_text")
        self.assertEqual(document["document_id"], "doc_x")
        self.assertEqual(document["title"], "Example")
        self.assertEqual(document["source_type"], "plain_text")
        self.assertFalse(document["segments_provided"])
        self.assertIn("Paragraph one", document["raw_text"])

    def test_segment_document_prefers_paragraphs(self) -> None:
        document = normalize_document_input("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
        segments = segment_document(document, chunk_chars=50, overlap_chars=10)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["segment_id"], "seg_1")
        self.assertIn("First paragraph", segments[0]["segment_text"])

    def test_build_segment_note_contains_numeric_fields(self) -> None:
        card = {
            "family": "individual_rights",
            "family_display_name": "Individual rights",
            "proxy_anchor": "risk_factors_privacy",
            "proxy_anchor_display_name": "Privacy",
            "concern_score": 0.82,
            "related_support_score": 0.61,
            "reliability_score": 0.74,
            "priority_score": 0.76,
            "document_rank": 2,
            "document_segment_count": 9,
            "supporting_feature_count": 11,
            "top_feature_ids": [12, 98, 105],
            "mean_feature_stability": 0.68,
            "causal_badge": "passed",
        }
        note = build_segment_note(card)
        self.assertIn("priority 0.76", note)
        self.assertIn("concern score 0.82", note)
        self.assertIn("11 sparse features", note)
        self.assertIn("Top sparse features: 12, 98, 105", note)
        self.assertIn("causal status passed", note)

    def test_build_document_summary_note_uses_family_scores(self) -> None:
        brief = {
            "dominant_families": [
                {
                    "family_display_name": "Transparency and accountability",
                    "document_family_score": 0.81,
                },
                {
                    "family_display_name": "Individual rights",
                    "document_family_score": 0.66,
                },
            ],
            "review_priority_order": [
                {
                    "family_display_name": "Transparency and accountability",
                    "top_segment_id": "seg_3",
                    "top_priority_score": 0.84,
                }
            ],
        }
        note = build_document_summary_note(brief)
        self.assertIn("Transparency and accountability (0.81)", note)
        self.assertIn("seg_3", note)

    def test_summary_table_rows_orders_methods_by_core_score(self) -> None:
        summaries = [
            {
                "method_name": "method_a",
                "highlighting": [
                    {
                        "segment_auroc": 0.80,
                        "segment_auprc": 0.70,
                        "precision_at_3": 0.60,
                        "recall_at_3": 0.50,
                        "mean_first_relevant_rank": 2.0,
                    }
                ],
                "retrieval": [
                    {
                        "hit_at_5": 0.60,
                        "mrr_at_5": 0.50,
                        "ndcg_at_5": 0.55,
                        "within_minus_cross_rate": 0.20,
                    }
                ],
                "triage": [
                    {
                        "recall_at_3": 0.55,
                        "first_relevant_rank": 2.0,
                        "average_review_depth": 2.0,
                        "ndcg_at_5": 0.58,
                    }
                ],
            },
            {
                "method_name": "method_b",
                "highlighting": [
                    {
                        "segment_auroc": 0.70,
                        "segment_auprc": 0.60,
                        "precision_at_3": 0.50,
                        "recall_at_3": 0.40,
                        "mean_first_relevant_rank": 3.0,
                    }
                ],
                "retrieval": [
                    {
                        "hit_at_5": 0.45,
                        "mrr_at_5": 0.40,
                        "ndcg_at_5": 0.42,
                        "within_minus_cross_rate": 0.05,
                    }
                ],
                "triage": [
                    {
                        "recall_at_3": 0.42,
                        "first_relevant_rank": 3.0,
                        "average_review_depth": 3.0,
                        "ndcg_at_5": 0.44,
                    }
                ],
            },
        ]
        rows = _summary_table_rows(summaries)
        self.assertEqual(rows[0]["method_name"], "method_a")
        self.assertGreater(rows[0]["AssistantCoreScore"], rows[1]["AssistantCoreScore"])

    def test_sparse_family_cards_include_feature_evidence(self) -> None:
        rows = [
            {
                "document_id": "doc_1",
                "segment_id": "seg_1",
                "text": "Privacy and rights safeguards are required.",
                "char_start": 0,
                "char_end": 40,
                "section_hint": None,
            }
        ]
        family_def = {
            "family": "individual_rights",
            "family_display_name": "Individual rights",
            "left_task_id": "risk_factors_privacy",
            "right_task_id": "harms_violation_of_civil_or_human_rights_including_privacy",
            "left_display_name": "Privacy",
            "right_display_name": "Rights violation",
        }
        artifact = {
            "method_name": "sparse_sae_feature_bank",
            "task_scores_norm": {
                "risk_factors_privacy": [0.9],
                "harms_violation_of_civil_or_human_rights_including_privacy": [0.4],
            },
            "task_reliability": {"risk_factors_privacy": 0.8},
            "task_bootstrap_means": {"risk_factors_privacy": 0.7},
            "task_dense_sparse_agreement": {"risk_factors_privacy": 0.6},
            "task_layers": {"risk_factors_privacy": 20},
            "task_feature_ids": {"risk_factors_privacy": [11, 22, 33]},
            "task_feature_weights": {"risk_factors_privacy": [0.9, 0.5, 0.1]},
            "task_feature_stability": {"risk_factors_privacy": {11: 0.72, 22: 0.61, 33: 0.40}},
            "task_selected_feature_activations": {"risk_factors_privacy": np.asarray([[1.0, 0.4, 0.2]], dtype=np.float32)},
            "family_sparse_vectors": {"individual_rights": np.asarray([[1.0, 0.4, 0.2]], dtype=np.float32)},
        }
        cards = _build_family_cards(
            rows,
            family_def,
            artifact,
            {"weights": {"concern": 0.6, "related_support": 0.25, "reliability": 0.15}},
            {"proxy_causal_badges": {"risk_factors_privacy": "passed"}},
        )
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card["top_feature_ids"], [11, 22, 33])
        self.assertEqual(card["causal_badge"], "passed")
        self.assertAlmostEqual(card["mean_feature_stability"], (0.72 + 0.61 + 0.40) / 3.0, places=6)
        self.assertEqual(card["selected_layer"], 20)


if __name__ == "__main__":
    unittest.main()
