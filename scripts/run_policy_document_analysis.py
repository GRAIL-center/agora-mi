from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from assistant.documents import load_document_payload
from assistant.experiments import analyze_document_with_sparse_assistant
from runtime import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a document with the sparse policy analysis assistant.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input text or JSON payload.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("configs") / "policy_analysis_assistant.yaml"),
        help="Path to the policy analysis assistant config.",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--document_id", type=str, default="document_1", help="Document identifier.")
    parser.add_argument("--title", type=str, default="Untitled document", help="Document title.")
    parser.add_argument("--source_type", type=str, default="user_text", help="Document source type.")
    args = parser.parse_args()

    setup_logging("run_policy_document_analysis")
    payload = load_document_payload(args.input_path)
    analyze_document_with_sparse_assistant(
        payload,
        config_path=args.config,
        output_path=args.output_path,
        document_id=args.document_id,
        title=args.title,
        source_type=args.source_type,
    )


if __name__ == "__main__":
    main()
