from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_prompt_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def render_prompt(template: str, text: str) -> str:
    return template.replace("{TEXT}", text)


def build_prompts(
    records: list[dict[str, Any]],
    template: str,
    tokenizer=None,
    use_chat_template: bool = False
) -> list[str]:
    prompts = [render_prompt(template, str(r["text"])) for r in records]
    if use_chat_template and tokenizer is not None:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]
    return prompts
