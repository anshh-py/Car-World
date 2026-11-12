"""Independent ground-truth verification for the CarWorld evaluation dataset.

Verifies two things without invoking any model and without modifying the dataset:

1. Every CSV-grounded factual expectation (``reference_facts`` that name a
   ``source_field`` row) against ``Data/car_dataset.csv``.
2. Every EMI expected value, recomputed with the *same* financial formula used
   by the production ``calculate_emi`` in ``app.py`` (including zero-interest
   handling and total-interest/circle-trip rounding).

Discrepancies are reported separately from verified expectations. Nothing here
ever edits ``car_world_eval_v1.jsonl``; it only reads it.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = (
    PROJECT_ROOT / "backend" / "evaluation" / "datasets" / "car_world_eval_v1.jsonl"
)
CSV_PATH = PROJECT_ROOT / "Data" / "car_dataset.csv"

# Map a dataset reference_facts key to the real CSV column name.
CSV_FACT_FIELDS = {
    "ex_showroom_price": "Ex-Showroom_Price",
    "fuel_type": "Fuel_Type",
    "body_type": "Body_Type",
    "seating_capacity": "Seating_Capacity",
}

# Keys that are identifiers/annotations, not ground-truth values to compare.
_NON_COMPARABLE_FACT_KEYS = {
    "model",
    "variant",
    "source_field",
    "note",
    "formula_source",
    "expected_arg",
    "routing",
}


def clean_int(value: Any) -> int:
    """Normalize any currency/quantity string to a plain int.

    Handles Indian digit grouping (9,99,994), a leading rupee symbol, and the
    trailing ``.000000000000000`` decimals present in the CSV: only the digits
    before the decimal point are kept.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d[\d,]*", str(value))
    if not match:
        return 0
    return int(match.group(0).replace(",", ""))


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def emi_expected(principal: float, annual_interest_rate: float, tenure_months: int) -> dict:
    """Recompute the expected EMI using the production formula verbatim."""
    monthly_rate = float(annual_interest_rate) / 1200.0
    tenure = float(tenure_months)
    if monthly_rate == 0:
        emi = float(principal) / tenure
    else:
        factor = (1.0 + monthly_rate) ** tenure
        emi = float(principal) * monthly_rate * factor / (factor - 1.0)
    total_payment = emi * tenure
    return {
        "monthly_emi": round(emi, 2),
        "total_interest": round(total_payment - float(principal), 2),
        "total_payment": round(total_payment, 2),
    }


def load_csv_rows(path: Path = CSV_PATH) -> list[dict]:
    """Return CSV records as dicts keyed by the header row. Raises on missing file."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def find_csv_row(rows: list[dict], model: str, variant: str) -> tuple[dict | None, int | None]:
    """Find the first data row matching Make+Model and Variant exactly.

    Returns ``(row, data_row_number)`` where ``data_row_number`` is 1-based over
    data rows (the header is file line 1). Matching uses the joined
    ``Make Model`` text against the reference ``model`` and the ``Variant``
    column against the reference ``variant``, so it never matches on Make alone.
    """
    target_variant = _norm_text(variant)
    target_model = _norm_text(model)
    for idx, row in enumerate(rows, start=1):
        joined = _norm_text(f"{row.get('Make', '')} {row.get('Model', '')}")
        if joined != target_model:
            continue
        if target_variant and _norm_text(row.get("Variant", "")) != target_variant:
            continue
        return row, idx
    return None, None


def verify_csv_case(record: dict, rows: list[dict]) -> dict:
    """Check one record's CSV-grounded facts against the dataset file."""
    facts = record.get("reference_facts") or {}
    source_field = str(facts.get("source_field") or "")
    model = str(facts.get("model") or "")
    variant = str(facts.get("variant") or "")

    if not source_field or not model or not variant:
        return {"id": record.get("id"), "status": "skipped",
                "reason": "no source_field/model/variant in reference_facts"}

    row, row_number = find_csv_row(rows, model, variant)
    if row is None:
        return {"id": record.get("id"), "status": "unverifiable",
                "reason": f"no CSV row for model={model!r} variant={variant!r}"}

    # The dataset's source_field cites the CSV *file line* (header is line 1),
    # while row_number is 1-based over data rows, so file line = row_number + 1.
    file_line = row_number + 1
    stated_row_match = source_field and (str(row_number) in source_field or str(file_line) in source_field)
    checks: list[dict] = []
    all_ok = True

    for key, column in CSV_FACT_FIELDS.items():
        if key not in facts:
            continue
        expected = facts[key]
        actual = row.get(column)
        if column == "Seating_Capacity":
            expected_cmp = clean_int(expected)
            actual_cmp = clean_int(actual)
        elif column == "Ex-Showroom_Price":
            expected_cmp = clean_int(expected)
            actual_cmp = clean_int(actual)
        else:
            expected_cmp = _norm_text(expected)
            actual_cmp = _norm_text(actual)
        ok = expected_cmp == actual_cmp
        all_ok = all_ok and ok
        checks.append({"field": key, "expected": expected, "actual": actual,
                       "ok": ok})

    status = "verified" if all_ok else "mismatch"
    return {"id": record.get("id"), "status": status, "model": model,
            "variant": variant, "csv_data_row": row_number,
            "stated_row_matches": stated_row_match, "checks": checks}


def verify_emi_case(record: dict) -> dict:
    """Check one record's EMI expectation against an independent recomputation."""
    facts = record.get("reference_facts") or {}
    required = {"principal", "annual_interest_rate", "tenure_months",
                "monthly_emi", "total_interest", "tolerance_inr"}
    if not required.issubset(facts.keys()):
        return {"id": record.get("id"), "status": "skipped",
                "reason": "reference_facts missing EMI expectation fields"}

    expected = emi_expected(facts["principal"], facts["annual_interest_rate"],
                            facts["tenure_months"])
    tol = float(facts["tolerance_inr"])
    monthly_ok = abs(expected["monthly_emi"] - float(facts["monthly_emi"])) <= tol
    interest_ok = abs(expected["total_interest"] - float(facts["total_interest"])) <= tol
    # An explicitly zero-interest case must produce zero interest exactly.
    zero_ok = (clean_int(facts["annual_interest_rate"]) == 0) == (
        expected["total_interest"] == 0.0
    )
    ok = monthly_ok and interest_ok and zero_ok
    return {"id": record.get("id"),
            "status": "verified" if ok else "mismatch",
            "recomputed": expected,
            "recorded": {"monthly_emi": facts["monthly_emi"],
                         "total_interest": facts["total_interest"],
                         "total_payment": facts.get("total_payment")},
            "tolerance_inr": tol,
            "monthly_within_tolerance": monthly_ok,
            "total_interest_within_tolerance": interest_ok,
            "zero_interest_handled": zero_ok}


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            import json
            records.append(json.loads(line))
    return records


def run_full_verification(records: list[dict] | None = None) -> list[dict]:
    """Verify every CSV and EMI expectation across the whole dataset."""
    records = records if records is not None else load_dataset()
    rows = load_csv_rows()
    results: list[dict] = []
    for record in records:
        category = record.get("category")
        if category == "emi_calculations":
            results.append(verify_emi_case(record))
        elif category == "indian_car_specifications":
            results.append(verify_csv_case(record, rows))
        else:
            results.append({"id": record.get("id"), "status": "skipped",
                            "reason": f"category {category} has no deterministic CSV/EMI ground truth"})
    return results


def _print_report(results: list[dict]) -> int:
    """Print a report; return exit code (1 if any mismatch/unverifiable)."""
    status_counts: dict[str, int] = {}
    mismatches = [r for r in results if r["status"] == "mismatch"]
    unverifiable = [r for r in results if r["status"] == "unverifiable"]
    verified = [r for r in results if r["status"] == "verified"]
    skipped = [r for r in results if r["status"] == "skipped"]
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    print(f"Ground-truth verification report ({len(results)} cases)")
    print(f"  verified: {len(verified)}  mismatch: {len(mismatches)}  "
          f"unverifiable: {len(unverifiable)}  skipped: {len(skipped)}")

    for group, title in ((mismatches, "MISMATCH (incorrect expectation)"),
                         (unverifiable, "UNVERIFIABLE (no matching CSV row)"),
                         (verified, "Verified")):
        if not group:
            continue
        print(f"\n-- {title} --")
        for r in group:
            print(f"  {r['id']} [{r.get('model','')} {r.get('variant','').strip()}] "
                  f"csv_row={r.get('csv_data_row')} stated_row_matches={r.get('stated_row_matches')}")
            for check in r.get("checks", []):
                mark = "ok" if check["ok"] else "FAIL"
                print(f"      {check['field']}: expected={check['expected']!r} "
                      f"actual={check['actual']!r} -> {mark}")
            if r.get("reason"):
                print(f"      reason: {r['reason']}")
            if "recomputed" in r:
                print(f"      recomputed: {r['recomputed']}")
                print(f"      recorded:   {r['recorded']}")
                print(f"      tolerance_inr: {r['tolerance_inr']}")

    return 0 if not mismatches and not unverifiable else 1


def main(argv: list[str] | None = None) -> int:
    results = run_full_verification()
    return _print_report(results)


if __name__ == "__main__":
    sys.exit(main())
