"""CarWorld evaluation runner.

Executes the 40-case dataset against the real CarWorld agent without changing
any normal application behavior. It reuses the actual ``ProviderRouter`` and
tool implementations; it does not duplicate provider logic and never exposes
chain-of-thought or modifies the public ``ChatResponse`` schema.

Modes
-----
* ``--mode mock`` — consumes no API quota. Uses the real agent objects for the
  deterministic parts: tool *routing* (``select_tools_for_request``) and the
  pure EMI calculator. Qualitative/CSV-retrieval parts are marked for human
  review (never fabricated). Skipped because it does not hit live web/DB data.
* ``--mode live`` — runs the actual agent end-to-end against the configured
  providers (Groq primary, Qwen3 fallback) and records which provider served
  each request.

Provider applicability is respected: a case marked ``groq`` only runs on Groq;
a tool-dependent case is skipped (not failed) on a Qwen3 config that has tool
calling disabled.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from validate_eval_dataset import load_records  # noqa: E402
from ground_truth import run_full_verification  # noqa: E402
from scoring import score_case  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.modules.chat import agent as agent_loader  # noqa: E402
from app.modules.chat.providers import ProviderResponse  # noqa: E402

DEFAULT_OUTPUT = EVAL_DIR / "results" / "eval_results.json"
TOOL_DEPENDENT = {"search_car_database", "search_car_web", "calculate_emi"}

# Substrings that identify a Groq rate-limit / exhaustion condition.
RATE_LIMIT_MARKERS = ("rate limit", "RateLimit", "429", "exhausted",
                      "too many requests")


def _is_rate_limited_outcome(entry: dict, evidence: dict) -> bool:
    """True when a live groq request was throttled.

    Two signals:
      * the provider surfaced a rate-limit error (model_error text), or
      * a groq-requested case was in fact served by the Qwen3 fallback,
        which only happens when Groq is rate-limited/exhausted.
    """
    model_error = (evidence.get("model_error") or "")
    if any(m.lower() in model_error.lower() for m in RATE_LIMIT_MARKERS):
        return True
    if (entry.get("provider_requested") == "groq"
            and entry.get("provider_used") == "qwen3_fallback"):
        return True
    return False


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def _ensure_agent_key() -> None:
    """Allow mock mode to load the agent without a real Groq key.

    Only the deterministic helpers are used in mock mode; nothing in mock mode
    makes a network call. If a real key is already configured it is untouched.
    """
    if settings.groq_api_key is None:
        os.environ.setdefault("GROQ_API_KEY", "unused-mock-key")
        settings.groq_api_key = "unused-mock-key"


def load_agent_module():
    """Load app.py as a module (the same module the tests use).

    The deterministic helpers ``select_tools_for_request`` and
    ``calculate_emi`` live at module scope in app.py, while ``agent.run`` and
    ``agent.execute_tool`` are instance methods on the CarWorldAgent.
    """
    agent_loader._load_gradio_agent_module.cache_clear()
    return agent_loader._load_gradio_agent_module()


def load_agent():
    return load_agent_module().agent


# ---------------------------------------------------------------------------
# Provider applicability helpers
# ---------------------------------------------------------------------------

def resolve_requested_providers(arg: str) -> list[str]:
    if arg == "groq":
        return ["groq"]
    if arg == "qwen3":
        return ["qwen3_fallback"]
    return ["groq", "qwen3_fallback"]


def applicable_providers(record: dict, requested: list[str]) -> list[str]:
    """Which requested providers this case may run on, given its applicability."""
    applicability = record.get("provider_applicability", "both")
    if applicability == "groq":
        return ["groq"] if "groq" in requested else []
    if applicability == "qwen3":
        return ["qwen3_fallback"] if "qwen3_fallback" in requested else []
    # "both" → any requested provider.
    return list(requested)


def qwen3_can_use_tools() -> bool:
    return bool(settings.qwen3_supports_tools)


# ---------------------------------------------------------------------------
# Mock executor (no API quota)
# ---------------------------------------------------------------------------

def mock_execute(module, record: dict) -> dict:
    """Deterministic, offline execution using the real agent objects."""
    user_input = record.get("user_input", "")
    expected_tool = record.get("expected_tool")
    evidence: dict = {
        "provider_requested": "mock",
        "provider_used": "mock",
        "calls": [],
        "tool_output": None,
        "reported_facts": {},
        "reply": None,
        "requires_live_data": False,
    }

    # 1) Verify deterministic tool routing (no network).
    try:
        selected = module.select_tools_for_request(
            [{"role": "user", "content": user_input}], module.TOOLS
        )
        selected_names = [t["function"]["name"] for t in selected]
    except Exception as exc:  # pragma: no cover - defensive
        selected_names = []
        evidence["routing_error"] = f"{type(exc).__name__}: {exc}"
    evidence["routing_selected"] = selected_names
    evidence["routing_matches_expected"] = bool(
        expected_tool and expected_tool in selected_names
    )

    # 2) Execute the deterministic EMI calculator whenever calculate_emi is the
    #     expected tool and the numeric inputs are present (covers both the
    #     emi_calculations category and the tool_selection_execution EMI case).
    facts = record.get("reference_facts") or {}
    if expected_tool == "calculate_emi" and all(
        k in facts for k in ("principal", "annual_interest_rate", "tenure_months")
    ):
        try:
            evidence["tool_output"] = module.calculate_emi(
                facts["principal"], facts["annual_interest_rate"],
                facts["tenure_months"],
            )
            evidence["calls"].append({"name": "calculate_emi", "status": "ok"})
        except Exception as exc:
            evidence["calls"].append({"name": "calculate_emi", "status": "error"})
            evidence["tool_error"] = f"{type(exc).__name__}: {exc}"
        return evidence

    # 3) CSV / web retrieval needs live data sources; never fabricate in mock.
    if expected_tool in ("search_car_database", "search_car_web"):
        evidence["requires_live_data"] = True
        evidence["calls"].append({"name": expected_tool, "status": "mock_not_executed"})
        return evidence

    return evidence


# ---------------------------------------------------------------------------
# Live executor (real providers)
# ---------------------------------------------------------------------------

def extract_reported_facts(calls: list[dict], reference: dict) -> dict:
    """Extract the CSV-grounded facts a live database lookup actually returned.

    Populates ``reported_facts`` from the recorded ``search_car_database``
    tool output (the real stored record), matching the dataset's expected
    variant. Only fields that are actually present in the tool output are
    populated; nothing is fabricated. This fixes the false-negative spec cases
    where ``reported_facts`` was left empty and every criterion scored False.
    """
    import re as _re
    variant = str(reference.get("variant") or "").strip()
    if not variant:
        return {}
    norm = lambda s: _re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())
    target = norm(variant)
    for call in calls:
        if call.get("name") != "search_car_database":
            continue
        result = call.get("result") or {}
        for row in result.get("results") or []:
            if not isinstance(row, dict):
                continue
            if norm(row.get("variant", "")) != target:
                continue
            facts: dict = {}
            if "ex_showroom_price" in row and row.get("ex_showroom_price") is not None:
                facts["ex_showroom_price"] = row.get("ex_showroom_price")
            if "fuel_type" in row and row.get("fuel_type") is not None:
                facts["fuel_type"] = row.get("fuel_type")
            if "body_type" in row and row.get("body_type") is not None:
                facts["body_type"] = row.get("body_type")
            if "seating_capacity" in row and row.get("seating_capacity") is not None:
                facts["seating_capacity"] = row.get("seating_capacity")
            return facts
    return {}


class _InstrumentedRecorder:
    """Thin, behavior-preserving wrappers that record provider and tool use."""

    def __init__(self, agent, module):
        self.agent = agent
        self.module = module
        self.provider_used = None
        self.calls: list[dict] = []
        self._orig_router = getattr(agent, "provider_router", None)
        # The agent's run() calls the module-level execute_tool (app.py:527),
        # so we wrap that function, not an instance attribute.
        self._orig_execute = getattr(module, "execute_tool", None)
        self._installed = False

    def _attach(self):
        if not self._orig_router:
            return
        router = self._orig_router
        original_chat = router.chat_completion  # bound method captured before overriding

        def _recorded_chat(*args, **kwargs):
            response = original_chat(*args, **kwargs)
            if isinstance(response, ProviderResponse) and response.provider_name:
                self.provider_used = response.provider_name
            return response

        self.agent.provider_router.chat_completion = _recorded_chat
        self._installed = True

        if self._orig_execute:
            def _recorded_execute(name, arguments):
                try:
                    result = self._orig_execute(name, arguments)
                    status = result.get("status", "ok") if isinstance(result, dict) else "ok"
                except Exception:  # pragma: no cover - defensive
                    result = {"status": "error"}
                    status = "error"
                self.calls.append({"name": name, "status": status,
                                  "result": result})
                return result

            self.module.execute_tool = _recorded_execute

    def detach(self):
        if not self._installed:
            return
        if self._orig_router is not None:
            self.agent.provider_router.chat_completion = self._orig_router.chat_completion
        if self._orig_execute is not None:
            self.module.execute_tool = self._orig_execute
        self._installed = False


def live_execute(module, agent, record: dict) -> dict:
    evidence: dict = {
        "provider_requested": record.get("provider_applicability", "both"),
        "provider_used": None,
        "calls": [],
        "tool_output": None,
        "reported_facts": {},
        "reply": None,
    }
    recorder = _InstrumentedRecorder(agent, module)
    recorder._attach()
    try:
        conversation = [{"role": "user", "content": record.get("user_input", "")}]
        replies = list(agent.run(conversation))
        evidence["reply"] = replies[-1] if replies else None
    except Exception as exc:
        evidence["model_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        evidence["provider_used"] = recorder.provider_used
        evidence["calls"] = copy.deepcopy(recorder.calls)
        recorder.detach()
    # Expose the deterministic tool output for scoring (e.g. calculate_emi
    # returns a result dict that score_emi_output compares against the dataset).
    if record.get("expected_tool") == "calculate_emi":
        for call in evidence["calls"]:
            if call.get("name") == "calculate_emi" and call.get("status") == "ok":
                evidence["tool_output"] = call.get("result")
                break
    # Fill reported_facts from the actual recorded database tool output, so
    # spec cases are scored against real retrieved records, not an empty dict.
    reference = record.get("reference_facts") or {}
    if reference.get("variant"):
        evidence["reported_facts"] = extract_reported_facts(evidence["calls"], reference)
    return evidence


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_evidence(target, module, mode: str, record: dict, provider: str) -> dict:
    if mode == "mock":
        evidence = mock_execute(target, record)
    else:
        evidence = live_execute(module, target, record)
    evidence["provider_requested"] = provider
    evidence["provider_used"] = evidence.get("provider_used") or provider
    return evidence


def run_case(target, module, record: dict, requested: list[str], mode: str,
             qwen3_tools: bool) -> list[dict]:
    results: list[dict] = []
    for provider in applicable_providers(record, requested):
        entry = {"id": record.get("id"), "category": record.get("category"),
                 "provider_requested": provider, "mode": mode}

        # Tool-dependent case on a Qwen3 config without tool calling → N/A.
        if record.get("expected_tool") in TOOL_DEPENDENT and provider == "qwen3_fallback" and not qwen3_tools:
            entry["outcome"] = "not_applicable"
            entry["reason"] = "qwen3_fallback has tool calling disabled"
            results.append(entry)
            continue

        evidence = _build_evidence(target, module, mode, record, provider)
        entry["provider_used"] = evidence.get("provider_used")
        entry["evidence"] = {k: v for k, v in evidence.items()
                             if k != "provider_requested"}

        scored = score_case(record, evidence)
        entry["score"] = scored
        if evidence.get("requires_live_data"):
            entry["outcome"] = "requires_live_data"
        elif _is_rate_limited_outcome(entry, evidence):
            entry["outcome"] = "rate_limited"
            entry.setdefault("reason", "groq rate limit / fallback")
        elif evidence.get("model_error"):
            entry["outcome"] = "error"
        elif scored.get("review", {}).get("human_review_required"):
            entry["outcome"] = "human_review_required"
        else:
            scoring = scored.get("scoring") or {}
            entry["outcome"] = "pass" if scoring.get("pass") else "fail"
        results.append(entry)
    return results


def _persist(output: Path, mode: str, requested: list[str], total_cases: int,
             results: list[dict]) -> None:
    """Write the current accumulated results immediately.

    Module-level so it is a stable, patchable reference and can be wrapped in
    tests to record every invocation (incremental per-case persistence).
    """
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"mode": mode, "providers_requested": requested,
                   "total_cases": total_cases, "results": results},
                  handle, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--ids", type=str, default="",
                        help="comma-separated case ids; default: all")
    parser.add_argument("--providers", choices=["groq", "qwen3", "both"],
                        default="both")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing --output report")
    parser.add_argument("--ground-truth-only", action="store_true",
                        help="verify CSV/EMI ground truth and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Refuse to clobber an existing report unless the user explicitly asked to
    # overwrite it. Checked before any work is done so the existing file is
    # never truncated. The --ground-truth-only path writes no output file.
    if not args.ground_truth_only and args.output.exists() and not args.overwrite:
        print(
            f"[E] Output file already exists: {args.output}\n"
            f"    Refusing to overwrite an existing evaluation report.\n"
            f"    Choose a different filename with --output, or pass "
            f"--overwrite to deliberately replace it.",
            file=sys.stderr,
        )
        return 1

    dataset_path = args.dataset or Path(EVAL_DIR) / "datasets" / "car_world_eval_v1.jsonl"
    records = load_records(dataset_path)

    if args.ground_truth_only:
        report = run_full_verification(records)
        # Reuse the same report rendering as ground_truth.main but keep it internal.
        from ground_truth import _print_report  # noqa: PLC0415
        return _print_report(report)

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    if ids:
        records = [r for r in records if r.get("id") in ids]
        missing = set(ids) - {r.get("id") for r in records}
        if missing:
            print(f"[WARN] unknown ids: {sorted(missing)}")

    requested = resolve_requested_providers(args.providers)
    qwen3_tools = qwen3_can_use_tools()

    _ensure_agent_key() if args.mode == "mock" else None
    module = load_agent_module()
    agent = module.agent
    target = agent if args.mode == "live" else module
    mode = "live" if args.mode == "live" else "mock"

    all_results: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for record in records:
        all_results.extend(run_case(target, module, record, requested, mode, qwen3_tools))
        _persist(args.output, mode, requested, len(records), all_results)  # after every case

    from collections import Counter
    outcomes = Counter(r.get("outcome") for r in all_results)
    print(f"Ran {len(records)} case(s) in {mode} mode -> {args.output}")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome}: {count}")

    if args.verbose:
        for r in all_results:
            print(f"  {r['id']} [{r['provider_requested']}] -> {r.get('outcome')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
