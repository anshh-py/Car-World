"""Prompt length benchmark for CarWorld AI.

Measures the automotive system prompt's estimated token count using the
deterministic estimator in ``app.modules.chat.token_count``. The prompt is
extracted via AST (no app import, so no API keys or network are required).

Usage:
    python -m scripts.benchmark_prompt [--src app.py] [--label Label]

Output is a short JSON record: chars, words, estimated_tokens, and a
note that the estimate is heuristic (not provider-reported). It is intended for
before/after comparison as the prompt is optimized; it never reports a number
as if it were a provider billing figure.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
BACKEND = ROOT / "backend" / "app" / "modules" / "chat"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from token_count import token_report  # noqa: E402


def extract_system_prompt(src_path: Path, name: str = "SYSTEM_PROMPT") -> str:
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
    raise SystemExit(f"Could not find {name} in {src_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=ROOT / "app.py")
    parser.add_argument("--label", type=str, default="system_prompt")
    args = parser.parse_args(argv)

    prompt = extract_system_prompt(args.src)
    report = token_report(prompt, label=args.label)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
