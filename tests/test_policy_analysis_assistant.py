from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assistant.documents import normalize_document_input, segment_document
from assistant.experiments import _summary_table_rows
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
        }
        note = build_segment_note(card)
        self.assertIn("priority 0.76", note)
        self.assertIn("concern score 0.82", note)
        self.assertIn("11 sparse features", note)

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


if __name__ == "__main__":
    unittest.main()
