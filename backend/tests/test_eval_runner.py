"""Tests for the M2.3 evaluation framework.

Covers dataset integrity, independent ground-truth verification (CSV facts and
EMI recomputation), and the mock evaluation runner (no API quota) including
provider-applicability handling and qualitative human-review marking.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = BACKEND_DIR / "evaluation"
for _p in (EVAL_DIR, BACKEND_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from validate_eval_dataset import load_records  # noqa: E402
from ground_truth import run_full_verification, emi_expected  # noqa: E402
from scoring import score_emi_output, qualitative_review_required, score_case  # noqa: E402
import runner as eval_runner  # noqa: E402

DATASET_PATH = EVAL_DIR / "datasets" / "car_world_eval_v1.jsonl"


def _records() -> list[dict]:
    return load_records(DATASET_PATH)


def _record(case_id: str) -> dict:
    for record in _records():
        if record["id"] == case_id:
            return record
    raise KeyError(case_id)


@pytest.fixture(scope="module")
def mock_target():
    """Load the real agent module once; used only for deterministic helpers."""
    from app.core.config import settings

    os.environ.setdefault("GROQ_API_KEY", "unused-mock-key")
    settings.groq_api_key = "unused-mock-key"
    return eval_runner.load_agent_module()


# ---------------------------------------------------------------------------
# 1. Dataset integrity + independent ground-truth verification
# ---------------------------------------------------------------------------

def test_dataset_has_exactly_40_records():
    assert len(_records()) == 40


def test_case_ids_are_unique():
    ids = [r["id"] for r in _records()]
    assert len(set(ids)) == len(ids)


def test_all_csv_and_emi_ground_truth_verified():
    """Every deterministic expectation must verify; none may be incorrect or
    unverifiable. This is the contract the runner's deterministic scoring relies on."""
    results = run_full_verification(_records())
    statuses = [r["status"] for r in results]
    assert "mismatch" not in statuses, [r for r in results if r["status"] == "mismatch"]
    assert "unverifiable" not in statuses, [r for r in results if r["status"] == "unverifiable"]


def test_emi_expectations_match_independent_formula():
    from ground_truth import load_dataset

    for record in load_dataset(DATASET_PATH):
        if record["category"] != "emi_calculations":
            continue
        facts = record["reference_facts"]
        expected = emi_expected(facts["principal"], facts["annual_interest_rate"],
                                facts["tenure_months"])
        tolerance = float(facts["tolerance_inr"])
        assert abs(expected["monthly_emi"] - float(facts["monthly_emi"])) <= tolerance
        assert abs(expected["total_interest"] - float(facts["total_interest"])) <= tolerance


@pytest.mark.parametrize(
    "case_id,expected_monthly",
    [("cw-eval-019", 10258.27), ("cw-eval-020", 19306.89),
     ("cw-eval-021", 19343.12), ("cw-eval-022", 21247.04),
     ("cw-eval-023", 16666.67)],
)
def test_emi_case_expected_values(case_id, expected_monthly):
    record = _record(case_id)
    facts = record["reference_facts"]
    computed = emi_expected(facts["principal"], facts["annual_interest_rate"],
                            facts["tenure_months"])
    assert computed["monthly_emi"] == pytest.approx(expected_monthly, abs=0.05)


def test_zero_interest_case_has_zero_total_interest():
    record = _record("cw-eval-023")
    facts = record["reference_facts"]
    computed = emi_expected(facts["principal"], facts["annual_interest_rate"],
                            facts["tenure_months"])
    assert computed["total_interest"] == 0.0
    assert computed["monthly_emi"] == pytest.approx(600000 / 36, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Scoring helpers
# ---------------------------------------------------------------------------

def test_emi_scoring_accepts_expected_output():
    record = _record("cw-eval-019")
    tool_output = {"loan_amount": 500000, "annual_interest_rate_percent": 8.5,
                   "tenure_months": 60, "monthly_emi": 10258.27,
                   "total_interest": 115495.94, "total_payment": 615495.94,
                   "currency": "INR"}
    scored = score_emi_output(tool_output, record["reference_facts"])
    assert scored["pass"] is True


def test_emit_scoring_rejects_wrong_value():
    record = _record("cw-eval-019")
    tool_output = {"monthly_emi": 9999.0, "total_interest": 100000.0,
                   "total_payment": 600000.0}
    scored = score_emi_output(tool_output, record["reference_facts"])
    assert scored["pass"] is False


def test_qualitative_cases_are_marked_for_human_review():
    for category in ("maintenance_troubleshooting", "buying_recommendations",
                     "hinglish_understanding", "prompt_injection_resistance"):
        review = qualitative_review_required(category)
        assert review["human_review_required"] is True
        assert review["reason"]


# ---------------------------------------------------------------------------
# 3. Mock runner behavior (no API quota)
# ---------------------------------------------------------------------------

def test_mock_runner_emi_case_passes(mock_target):
    results = eval_runner.run_case(mock_target, mock_target, _record("cw-eval-019"),
                                   ["groq"], "mock", qwen3_tools=False)
    assert len(results) == 1
    assert results[0]["outcome"] == "pass"
    assert results[0]["provider_used"] == "mock"


def test_mock_runner_tool_emi_case_passes(mock_target):
    """The tool_selection_execution EMI case is also scored deterministically."""
    results = eval_runner.run_case(mock_target, mock_target, _record("cw-eval-039"),
                                   ["groq"], "mock", qwen3_tools=False)
    assert results[0]["outcome"] == "pass"


def test_mock_runner_csv_case_requires_live_data(mock_target):
    results = eval_runner.run_case(mock_target, mock_target, _record("cw-eval-001"),
                                   ["groq"], "mock", qwen3_tools=False)
    assert results[0]["outcome"] == "requires_live_data"


def test_mock_runner_qualitative_case_human_review(mock_target):
    results = eval_runner.run_case(mock_target, mock_target, _record("cw-eval-009"),
                                   ["groq"], "mock", qwen3_tools=False)
    assert results[0]["outcome"] == "human_review_required"


def test_qwen3_tool_case_not_applicable_without_tools(mock_target):
    """A tool-dependent case on Qwen3 (tool calling disabled) is N/A, not a failure."""
    # cw-eval-031 (Nissan Magnite hallucination case) is tool-dependent and
    # marked provider_applicability='both', so it must run on both providers.
    results = eval_runner.run_case(
        mock_target, mock_target, _record("cw-eval-031"),
        ["groq", "qwen3_fallback"], "mock", qwen3_tools=False,
    )
    outcomes = {r["provider_requested"]: r["outcome"] for r in results}
    assert outcomes["groq"] == "requires_live_data"
    assert outcomes["qwen3_fallback"] == "not_applicable"
    na = next(r for r in results if r["provider_requested"] == "qwen3_fallback")
    assert "tool calling disabled" in na["reason"]


def test_groq_only_case_excludes_qwen3(mock_target):
    """cci case marked provider_applicability='groq' never runs on Qwen3."""
    results = eval_runner.run_case(
        mock_target, mock_target, _record("cw-eval-038"),
        ["groq", "qwen3_fallback"], "mock", qwen3_tools=True,
    )
    assert {r["provider_requested"] for r in results} == {"groq"}


# ---------------------------------------------------------------------------
# 4. Output-file overwrite protection (M2.4)
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_existing_output_preserved_byte_for_byte_without_overwrite(tmp_path):
    """A pre-existing report is left untouched when --overwrite is absent."""
    out = tmp_path / "report.json"
    original = '{"original": true, "sentinel": 42}'
    out.write_text(original, encoding="utf-8")

    rc = eval_runner.main(["--mode", "mock", "--providers", "groq",
                           "--output", str(out)])

    assert rc == 1
    assert out.read_text(encoding="utf-8") == original  # byte-for-byte


def test_new_output_persists_after_every_completed_case(tmp_path):
    """A fresh output file is written (and complete) for the chosen cases."""
    out = tmp_path / "report.json"

    rc = eval_runner.main(["--mode", "mock", "--providers", "groq",
                           "--ids", "cw-eval-019,cw-eval-020",
                           "--output", str(out)])

    assert rc == 0
    data = _read_json(out)
    assert data["total_cases"] == 2
    assert [r["id"] for r in data["results"]] == ["cw-eval-019", "cw-eval-020"]


def test_explicit_overwrite_replaces_existing_report(tmp_path):
    """--overwrite deliberately replaces an existing report in full."""
    out = tmp_path / "report.json"
    _write_json(out, {"old": True})

    rc = eval_runner.main(["--mode", "mock", "--providers", "groq",
                           "--ids", "cw-eval-019",
                           "--output", str(out), "--overwrite"])

    assert rc == 0
    data = _read_json(out)
    assert data["total_cases"] == 1
    assert data["results"][0]["id"] == "cw-eval-019"
    assert "old" not in data


def test_interrupted_run_retains_completed_cases(tmp_path, monkeypatch):
    """Partial results survive an interruption; no run mixing occurs.

    Deterministic source-level test: patches the exact ``run_case`` reference
    ``main`` uses, wraps the actual module-level ``_persist`` (recording every
    invocation and delegating to the original), interrupts on the 4th case,
    and asserts exactly three persistence calls happened before interruption.
    """
    out = tmp_path / "report.json"
    calls = {"n": 0}
    real_run_case = eval_runner.run_case  # capture before patching

    def counting_run_case(target, module, record, requested, mode, qwen3_tools):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt
        return real_run_case(target, module, record, requested, mode,
                             qwen3_tools)

    monkeypatch.setattr(eval_runner, "run_case", counting_run_case)

    persist_calls = []
    real_persist = eval_runner._persist

    def wrapping_persist(output, mode, requested, total_cases, results):
        persist_calls.append((output, total_cases, len(results)))
        return real_persist(output, mode, requested, total_cases, results)

    monkeypatch.setattr(eval_runner, "_persist", wrapping_persist)

    with pytest.raises(KeyboardInterrupt):
        eval_runner.main(["--mode", "mock", "--providers", "groq",
                          "--output", str(out)])

    # Exactly three persistence calls happened before the interruption.
    assert persist_calls == [
        (out, 40, 1),
        (out, 40, 2),
        (out, 40, 3),
    ]

    # The last persist wrote exactly the three completed records.
    data = _read_json(out)
    assert data["total_cases"] == 40
    assert len(data["results"]) == 3


def test_default_output_path_is_protected(tmp_path, monkeypatch):
    """Protection also applies to the built-in default output path."""
    fake_default = tmp_path / "eval_results.json"
    fake_default.write_text('{"default": true}', encoding="utf-8")
    monkeypatch.setattr(eval_runner, "DEFAULT_OUTPUT", fake_default)

    rc = eval_runner.main(["--mode", "mock", "--providers", "groq"])

    assert rc == 1
    assert fake_default.read_text(encoding="utf-8") == '{"default": true}'


# ---------------------------------------------------------------------------
# 5. tool_selection_execution scoring correctness (M2.5 fix)
# ---------------------------------------------------------------------------

def _toolsel_record() -> dict:
    return _record("cw-eval-038")  # search_car_database, no EMI reference


def test_tool_selection_scoring_passes_when_routed_and_executed():
    """A correctly routed-and-executed tool case must score pass (not fail)."""
    evidence = {"calls": [{"name": "search_car_database", "status": "ok"}],
                "tool_output": None,
                "reported_facts": {},
                "reply": "Found the Creta in the database."}
    scored = score_case(_toolsel_record(), evidence)
    assert scored["tool_selection"]["pass"] is True
    assert scored["tool_execution"]["pass"] is True
    assert scored["scoring"]["pass"] is True


def test_tool_selection_scoring_fails_when_wrong_tool():
    """Picking the wrong tool fails the case."""
    evidence = {"calls": [{"name": "search_car_web", "status": "results_found"}],
                "tool_output": None,
                "reported_facts": {},
                "reply": "Web result."}
    scored = score_case(_toolsel_record(), evidence)
    assert scored["tool_selection"]["pass"] is False
    assert scored["scoring"]["pass"] is False


def test_tool_selection_scoring_fails_on_missing_evidence():
    """No recorded calls => no tool selected/executed => fail."""
    evidence = {"calls": [], "tool_output": None,
                "reported_facts": {}, "reply": "I couldn't find it."}
    scored = score_case(_toolsel_record(), evidence)
    assert scored["scoring"]["pass"] is False
    assert scored["tool_selection"]["pass"] is False
    assert scored["tool_execution"]["pass"] is False


def test_tool_selection_scoring_keeps_emi_numeric_check():
    """The EMI tool-selection case must still use the deterministic numeric check."""
    record = _record("cw-eval-039")
    facts = record["reference_facts"]
    tool_output = {"loan_amount": 1000000.0,
                   "annual_interest_rate_percent": 8.0,
                   "tenure_months": 60,
                   "monthly_emi": 20276.39,
                   "total_interest": 216583.66,
                   "total_payment": 1216583.66,
                   "currency": "INR"}
    evidence = {"calls": [{"name": "calculate_emi", "status": "ok"}],
                "tool_output": tool_output,
                "reported_facts": {},
                "reply": "EMI is ₹20,276."}
    scored = score_case(record, evidence)
    assert scored["scoring"]["pass"] is True


# ---------------------------------------------------------------------------
# 6. Canonical live Groq reconciliation (M2.5)
# ---------------------------------------------------------------------------

import reconcile_live  # noqa: E402


def test_canonical_report_has_40_unique_groq_ids():
    report = reconcile_live.build_canonical()
    data = report["results"]
    assert report["total_cases"] == 40
    ids = [r["id"] for r in data]
    assert len(ids) == 40
    assert len(set(ids)) == 40
    assert set(ids) == {r["id"] for r in _records()}
    for entry in data:
        assert entry["provider_used"] == "groq"
        assert entry.get("status") != "missing_live_result"
        assert bool((entry["evidence"] or {}).get("reply"))


def test_canonical_report_deduplicates_case_019():
    """Case 019 resolves to the valid EMI result, keeping the smoke error in audit."""
    report = reconcile_live.build_canonical()
    entry = next(r for r in report["results"] if r["id"] == "cw-eval-019")
    assert entry["canonical_source"] == "eval_live_groq_emi.json"
    assert entry["outcome"] == "pass"
    sources = {t["source"] for t in entry["audit_trail"]}
    assert "eval_live_smoke.json" in sources
    smoke = next(t for t in entry["audit_trail"] if t["source"] == "eval_live_smoke.json")
    assert "RecursionError" in (smoke.get("model_error") or "")


def test_canonical_report_passes_verification():
    """The canonical report must have zero hard violations."""
    report = reconcile_live.build_canonical()
    violations = reconcile_live.verify_canonical(report)
    assert violations == [], violations


# ---------------------------------------------------------------------------
# 7. Numeric normalization (M2.6 quality gate)
# ---------------------------------------------------------------------------

def test_normalize_number_currency_formatting_equals_raw_decimal():
    """'₹9,99,994' and '999994.00' must normalize to the same number."""
    from scoring import normalize_number

    assert normalize_number("₹9,99,994") == normalize_number("999994.00")
    assert normalize_number("999994.00") == 999994.0
    assert normalize_number("₹9,99,994") == 999994.0


def test_normalize_number_preserves_meaningful_decimals():
    """12.5 must NOT equal 12; decimals are not discarded."""
    from scoring import normalize_number

    assert normalize_number("12.5") == 12.5
    assert normalize_number("12") == 12.0
    assert normalize_number("12.5") != normalize_number("12")
    assert normalize_number("20,276.39") == 20276.39
    assert normalize_number("₹16,66,666.67") == 1666666.67


def test_normalize_number_emi_values():
    """EMI figures with decimals normalize exactly for comparison."""
    from scoring import normalize_number

    assert normalize_number("10,258.27") == 10258.27
    assert normalize_number("₹1,15,495.94") == 115495.94
    assert normalize_number("615,495.94") == 615495.94
    assert normalize_number("16666.67") == 16666.67


def test_normalize_number_non_numeric_returns_zero():
    from scoring import normalize_number

    assert normalize_number(None) == 0.0
    assert normalize_number("") == 0.0


# ---------------------------------------------------------------------------
# 8. Evidence extraction (M2.6 quality gate)
# ---------------------------------------------------------------------------

def test_extract_reported_facts_missing_tool_output_is_empty():
    """No DB calls -> no facts; nothing fabricated."""
    assert eval_runner.extract_reported_facts([], {"variant": "S3 2WD"}) == {}


def test_extract_reported_facts_only_from_structured_db_output():
    """Facts come only from search_car_database structured rows."""
    calls = [
        {
            "name": "search_car_database",
            "status": "ok",
            "result": {
                "status": "ok",
                "count": 1,
                "results": [
                    {"variant": "S3 2WD", "ex_showroom_price": "999994.00",
                     "fuel_type": "Diesel", "body_type": "SUV"}
                ],
            },
        }
    ]
    facts = eval_runner.extract_reported_facts(calls, {"variant": "S3 2WD"})
    assert facts["ex_showroom_price"] == "999994.00"
    assert facts["fuel_type"] == "Diesel"
    assert facts["body_type"] == "SUV"


def test_extract_reported_facts_ignores_non_database_tools():
    """Web-search results are not DB facts."""
    calls = [
        {"name": "search_car_web", "status": "results_found",
         "result": {"results": [{"variant": "S3 2WD",
                                  "ex_showroom_price": "999994.00"}]}}
    ]
    assert eval_runner.extract_reported_facts(calls, {"variant": "S3 2WD"}) == {}


def test_extract_reported_facts_missing_fields_not_fabricated():
    """Missing fields in a DB row are omitted, never invented."""
    calls = [
        {
            "name": "search_car_database",
            "status": "ok",
            "result": {
                "status": "ok",
                "count": 1,
                "results": [
                    {"variant": "S3 2WD", "ex_showroom_price": "999994.00"}
                ],
            },
        }
    ]
    facts = eval_runner.extract_reported_facts(calls, {"variant": "S3 2WD"})
    assert facts == {"ex_showroom_price": "999994.00"}
    assert "fuel_type" not in facts
    assert "body_type" not in facts
