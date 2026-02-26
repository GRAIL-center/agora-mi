from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _common import read_config, setup_logging
from data.io import read_jsonl, write_jsonl
from model.prompt import build_prompts, load_prompt_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run.yaml")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--prompt_config", default=None)
    parser.add_argument("--template_key", default="template_v1")
    parser.add_argument("--out_jsonl", required=True)
    args = parser.parse_args()

    setup_logging("make_prompts")
    run_cfg = read_config(args.config)
    rows = read_jsonl(args.input_jsonl)
    prompt_path = args.prompt_config or run_cfg.get("prompt_config", "configs/prompt_templates.yaml")
    p_cfg = load_prompt_config(prompt_path)
    template = str(p_cfg[args.template_key])

    prompts = build_prompts(rows, template)
    out = []
    for r, p in zip(rows, prompts):
        row = dict(r)
        row["prompt"] = p
        out.append(row)

    write_jsonl(args.out_jsonl, out)
    logging.info("Wrote prompts: %s (%d rows)", args.out_jsonl, len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
