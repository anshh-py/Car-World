# CarWorld AI — Evaluation Framework (M2.3)

A reproducible evaluation harness for the 40-case dataset
`datasets/car_world_eval_v1.jsonl`. It reuses the application's real
`ProviderRouter` (Groq primary → Qwen3 fallback) and tool implementations; it
does not duplicate provider logic, does not change the public `ChatResponse`
schema, and never exposes chain-of-thought.

## Layout

| Path | Purpose |
|---|---|
| `datasets/car_world_eval_v1.jsonl` | The 40-case dataset (see `datasets/README.md`). |
| `validate_eval_dataset.py` | Existing schema/count/ID validator. |
| `ground_truth.py` | Independent ground-truth verification (CSV facts + EMI recomputation). |
| `scoring.py` | Deterministic scorers + qualitative human-review marking. |
| `runner.py` | CLI runner (mock or live). |
| `results/` | Emitted run reports (JSON). |

## Ground-truth verification

`python evaluation/ground_truth.py` (from `backend/`) checks, without any model:

- Every `indian_car_specifications` expectation against `Data/car_dataset.csv`
  (price, fuel type, body type, seating capacity) using the exact Make+Model+Variant
  row. 8/8 verified.
- Every `emi_calculations` expectation against an independent recomputation using
  the production formula (monthly rate = annual/1200; zero-interest short-circuits
  to `principal/tenure`; values rounded to 2dp). 5/5 verified, all within the
  dataset's stated tolerances.

Exit code is non-zero if any expectation is incorrect or unverifiable. The
script never edits the dataset.

## Running the runner

```
# Mock mode (no API quota; deterministic routing + EMI; the rest marked for review)
python evaluation/runner.py --mode mock --providers both --verbose

# Only selected cases
python evaluation/runner.py --mode mock --ids cw-eval-019,cw-eval-039

# Live mode (real providers; consumes API quota)
python evaluation/runner.py --mode live --providers both

# Ground-truth only
python evaluation/runner.py --ground-truth-only
```

Arguments: `--dataset <path>`, `--ids a,b,c`, `--providers groq|qwen3|both`,
`--mode mock|live`, `--output <path>`, `--ground-truth-only`, `--verbose`.

### Outcome taxonomy

| Outcome | Meaning |
|---|---|
| `pass` | Deterministic scoring passed (EMI math, tool selection/execution). |
| `fail` | Deterministic scoring failed. |
| `human_review_required` | Qualitative case (maintenance, buying, Hinglish, injection, hallucination, abstention nuance); marked with a reason. No LLM judge is used. |
| `requires_live_data` | CSV/web retrieval needs live data sources; not graded in mock mode. |
| `not_applicable` | Tool-dependent case on Qwen3 with tool calling disabled — not a failure. |
| `error` | The run raised an exception. |

### Provider handling

- Each case's `provider_applicability` is respected: `groq`-only cases never run
  on Qwen3; `both` cases run on the requested providers.
- The provider actually used is recorded per result (`provider_used`).
- A tool-dependent case on Qwen3 (when `QWEN3_SUPPORTS_TOOLS=false`) is recorded
  as `not_applicable`, never as a model failure.
- Live mode instruments `agent.provider_router.chat_completion` and
  `agent.execute_tool` with thin wrappers to record the provider and tool calls;
  the wrappers are restored after each case. No application behavior changes.

## Scoring model

- **Deterministic** (no keyword-as-semantics): EMI output within tolerance,
  CSV-grounded figure match, expected tool selected, tool executed without error,
  abstention-marker presence for abstention cases.
- **Qualitative**: explicitly flagged `human_review_required` with a reason.
  There is intentionally no LLM-as-judge sole scorer.

## Known limitations

- Mock mode does not fabricate CSV/web retrieval; those cases are reported as
  `requires_live_data` and must be evaluated in live mode or via the dataset's
  ground-truth verification.
- Live mode requires a configured `GROQ_API_KEY` (and `QWEN3_SPACE_ID`/endpoint
  for Qwen3) and consumes provider quota.
- `human_review_required` cases need a human to grade against each case's
  `scoring_criteria`; the runner records the reply for that review.
