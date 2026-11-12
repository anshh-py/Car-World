"""Reconcile live Groq evaluation reports against the 40-case dataset.

Builds a single canonical, machine-readable report with exactly one result per
dataset ID (40 unique IDs), each produced by a real Groq run
(``provider_used == "groq"``), with a usable recorded reply, a corrected score
computed from the *recorded* tool evidence, and a full audit trail of every
historical result for that ID (so duplicates and prior API errors are never
lost, e.g. cw-eval-019's smoke-run RecursionError).

Read-only with respect to the dataset and the historical reports: nothing here
modifies the dataset, the expected answers, or any existing report file. The
consolidation is written to a new file only (``--output``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent))

from validate_eval_dataset import load_records  # noqa: E402
from scoring import score_case  # noqa: E402
from runner import extract_reported_facts  # noqa: E402

DATASET_PATH = EVAL_DIR / "datasets" / "car_world_eval_v1.jsonl"
DEFAULT_OUTPUT = RESULTS_DIR / "car_world_eval_canonical_groq.json"

# Live-per provider reports to reconcile. `eval_results.json` is mock mode and is
# deliberately excluded; the smoke file is included as *historical* evidence.
LIVE_REPORTS = sorted(p for p in RESULTS_DIR.glob("eval_live_groq_batch_*.json"))
EMI_REPORT = RESULTS_DIR / "eval_live_groq_emi.json"
SMOKE_REPORT = RESULTS_DIR / "eval_live_smoke.json"


def _load(report: Path) -> list[dict]:
    if not report.is_file():
        return []
    data = json.loads(report.read_text(encoding="utf-8"))
    if data.get("mode") != "live":
        return []
    results = data.get("results") or []
    return [r for r in results if r.get("provider_used") == "groq"]


def _is_api_error(result: dict) -> bool:
    evidence = result.get("evidence") or {}
    return bool(evidence.get("model_error"))


def _canonical_rank(result: dict) -> tuple:
    """Lower is better. Preferred: used Groq, no API error, usable reply."""
    evidence = result.get("evidence") or {}
    has_reply = bool(evidence.get("reply"))
    return (
        1 if _is_api_error(result) else 0,          # API error sorts last
        1 if not has_reply else 0,                   # usable reply preferred
        1 if result.get("outcome") == "error" else 0,
    )


def _select_canonical(history: list[dict]) -> dict:
    """Pick the best single result for an ID from its full history."""
    if not history:
        return {}
    return min(history, key=_canonical_rank)


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def build_canonical(records: list[dict] | None = None,
                    report_paths: list[Path] | None = None) -> dict:
    records = records if records is not None else load_records(DATASET_PATH)
    paths = report_paths if report_paths is not None else (LIVE_REPORTS + [EMI_REPORT, SMOKE_REPORT])

    # Collect every historical live-groq result per ID (audit trail).
    history: dict[str, list[dict]] = {}
    for path in paths:
        for entry in _load(path):
            history.setdefault(entry["id"], []).append({**entry, "source": path.name})

    canonical_results: list[dict] = []
    for record in records:
        rid = record["id"]
        entries = history.get(rid, [])

        if not entries:
            canonical_results.append({
                "id": rid, "category": record.get("category"),
                "status": "missing_live_result",
                "provider_used": None, "reply": None,
                "audit_trail": [],
            })
            continue

        selected = _select_canonical(entries)
        evidence = {k: v for k, v in selected.get("evidence", {}).items()
                    if k != "provider_requested"}

        # Correct evidence extraction from the recorded DB tool output.
        reference = record.get("reference_facts") or {}
        if reference.get("variant"):
            evidence["reported_facts"] = extract_reported_facts(
                evidence.get("calls", []), reference)

        scored = score_case(record, evidence)

        # Derive the corrected outcome from the fixed scorer.
        if _is_api_error(selected):
            outcome = "error"
        elif scored.get("review", {}).get("human_review_required"):
            outcome = "human_review_required"
        else:
            outcome = "pass" if (scored.get("scoring") or {}).get("pass") else "fail"

        canonical_results.append({
            "id": rid,
            "category": record.get("category"),
            "provider_requested": "groq",
            "provider_used": "groq" if selected.get("provider_used") == "groq" else None,
            "outcome": outcome,
            "score": scored,
            "evidence": evidence,
            "canonical_source": selected.get("source"),
            "audit_trail": [
                {"source": e["source"], "outcome": e.get("outcome"),
                 "provider_used": e.get("provider_used"),
                 "model_error": (e.get("evidence") or {}).get("model_error")}
                for e in entries
            ],
        })

    return {
        "mode": "live",
        "providers_requested": ["groq"],
        "total_cases": len(canonical_results),
        "canonical": True,
        "results": canonical_results,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_canonical(report: dict, records: list[dict] | None = None) -> list[str]:
    """Return a list of hard violations; empty means the report is clean.

    Fails if any dataset ID is missing, any ID is duplicated, any result is not
    from Groq, any result has an API error, or any result lacks a usable reply.
    Human-review and genuine "fail" outcomes are allowed.
    """
    records = records if records is not None else load_records(DATASET_PATH)
    dataset_ids = [r["id"] for r in records]
    violations: list[str] = []

    results = report.get("results") or []
    seen: set[str] = set()

    for entry in results:
        rid = entry.get("id")
        if rid is None:
            violations.append("result missing id")
            continue
        if rid in seen:
            violations.append(f"duplicate canonical id: {rid}")
        seen.add(rid)
        if entry.get("status") == "missing_live_result":
            violations.append(f"missing live result for {rid}")
            continue
        if entry.get("provider_used") != "groq":
            violations.append(f"{rid}: provider_used = {entry.get('provider_used')!r} (expected groq)")
        if entry.get("outcome") == "error":
            violations.append(f"{rid}: API error outcome in canonical report")
        evidence = entry.get("evidence") or {}
        if "model_error" in evidence:
            violations.append(f"{rid}: evidence contains model_error")
        if not evidence.get("reply"):
            violations.append(f"{rid}: no usable reply")

    missing = set(dataset_ids) - set(seen)
    for rid in sorted(missing):
        violations.append(f"missing dataset id in canonical report: {rid}")
    extra = set(seen) - set(dataset_ids)
    for rid in sorted(extra):
        violations.append(f"unknown id in canonical report: {rid}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = build_canonical()
    violations = verify_canonical(report)
    if violations:
        print("CANONICAL REPORT VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote canonical report: {args.output}")
    print(f"total_cases={report['total_cases']} canonical=40 unique")
    return 0


if __name__ == "__main__":
    sys.exit(main())
