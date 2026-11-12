"""Validate the CarWorld AI evaluation dataset.

Checks:
- exactly 40 cases
- unique IDs
- valid JSON on every line
- required fields present
- valid category / severity / provider values
- category counts match the required distribution
- expected_tool is either null or one of the registered tools
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "car_world_eval_v1.jsonl"

EXPECTED_TOTAL = 40

REQUIRED_FIELDS = {
    "id",
    "category",
    "user_input",
    "objective",
    "expected_behavior",
    "expected_tool",
    "reference_facts",
    "scoring_criteria",
    "severity",
    "provider_applicability",
}

EXPECTED_CATEGORIES = {
    "indian_car_specifications": 8,
    "maintenance_troubleshooting": 6,
    "buying_recommendations": 4,
    "emi_calculations": 5,
    "hinglish_understanding": 3,
    "missing_info_abstention": 4,
    "hallucination_resistance": 4,
    "prompt_injection_resistance": 3,
    "tool_selection_execution": 3,
}

ALLOWED_CATEGORIES = set(EXPECTED_CATEGORIES)

ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}

ALLOWED_PROVIDERS = {"groq", "qwen3", "both"}

REGISTERED_TOOLS = {"calculate_emi", "search_car_database", "search_car_web"}


def load_records(path: Path = DATASET_PATH) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number}: record is not a JSON object.")
            records.append(record)
    return records


def validate(records: list[dict]) -> list[str]:
    errors: list[str] = []

    if len(records) != EXPECTED_TOTAL:
        errors.append(
            f"Expected {EXPECTED_TOTAL} records, got {len(records)}."
        )

    ids = [record.get("id") for record in records]
    if len(set(ids)) != len(ids):
        seen = set()
        duplicates = sorted({id_ for id_ in ids if id_ in seen or seen.add(id_)})
        errors.append(f"Duplicate ids: {duplicates}")

    for index, record in enumerate(records, start=1):
        missing = REQUIRED_FIELDS - set(record.keys())
        if missing:
            errors.append(f"Record {index} missing fields: {sorted(missing)}")
        extra = set(record.keys()) - REQUIRED_FIELDS
        if extra:
            errors.append(f"Record {index} has unknown fields: {sorted(extra)}")

        category = record.get("category")
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"Record {index} invalid category: {category!r}")

        severity = record.get("severity")
        if severity not in ALLOWED_SEVERITIES:
            errors.append(f"Record {index} invalid severity: {severity!r}")

        provider = record.get("provider_applicability")
        if provider not in ALLOWED_PROVIDERS:
            errors.append(f"Record {index} invalid provider: {provider!r}")

        expected_tool = record.get("expected_tool")
        if expected_tool is not None and expected_tool not in REGISTERED_TOOLS:
            errors.append(
                f"Record {index} expected_tool not registered: {expected_tool!r}"
            )

        scoring = record.get("scoring_criteria")
        if not isinstance(scoring, list) or not scoring or not all(
            isinstance(item, str) for item in scoring
        ):
            errors.append(f"Record {index} scoring_criteria must be a non-empty list of strings.")

    actual_counts = Counter(record.get("category") for record in records)
    for category, expected_count in EXPECTED_CATEGORIES.items():
        actual = actual_counts.get(category, 0)
        if actual != expected_count:
            errors.append(
                f"Category {category!r}: expected {expected_count}, got {actual}."
            )

    return errors


def main() -> int:
    try:
        records = load_records()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    errors = validate(records)
    if errors:
        print(f"[FAIL] {len(errors)} validation error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    counts = Counter(record.get("category") for record in records)
    print(f"[PASS] {len(records)} records, unique ids, valid schema and category distribution.")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
