# CarWorld AI Evaluation Dataset v1

A reproducible, 40-case automotive evaluation dataset for CarWorld AI (M2: Answer Reliability & Evaluation).

## File

- `car_world_eval_v1.jsonl` — 40 JSON objects, one per line.

## Schema

Each record is a JSON object with these fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Stable unique identifier (`cw-eval-NNN`). |
| `category` | string | yes | One of the 9 distribution categories below. |
| `user_input` | string | yes | The exact user message to send to the agent. |
| `objective` | string | yes | What this case is designed to measure. |
| `expected_behavior` | string | yes | What a passing model should do. |
| `expected_tool` | string | null | Expected tool name, or `null` if no tool is expected. |
| `reference_facts` | object | null | Ground-truth reference (values, `source_field`, or a clarifying note). `null` when a rubric-based answer applies. |
| `scoring_criteria` | array[string] | yes | Explicit, deterministic checks that make a pass/fail judgment. |
| `severity` | string | yes | Impact if the model fails: `critical` / `high` / `medium` / `low`. |
| `provider_applicability` | string | yes | Which provider the case applies to: `groq`, `qwen3`, or `both`. |

## Category distribution (exactly 40)

| Category | Count | IDs |
|---|---|---|
| Indian car specifications | 8 | 001–008 |
| Maintenance and troubleshooting | 6 | 009–014 |
| Buying recommendations | 4 | 015–018 |
| EMI and calculations | 5 | 019–023 |
| Hinglish understanding | 3 | 024–026 |
| Missing information and abstention | 4 | 027–030 |
| Hallucination resistance | 4 | 031–034 |
| Prompt-injection resistance | 3 | 035–037 |
| Tool selection and execution | 3 | 038–040 |

## Ground-truth sources

- **Factual cases (001–008):** values come from `Data/car_dataset.csv` rows (ex-showroom price, fuel type, body type, seating capacity). Each record names its `source_field` (row number) so answers can be checked deterministically against the stored catalog.
- **EMI cases (019–023):** expected values are computed with the exact `calculate_emi` formula in `app.py` (`monthly_rate = annual_interest_rate / 1200`; `emi = P·r·(1+r)^n / ((1+r)^n − 1)`; zero-interest handled as `P/n`). Each case records `tolerance_inr` (±₹50, or ±₹1 for the zero-interest case) to allow display/rounding variance.
- **Rubric cases (009–018, 024–034):** no single exact answer; the `scoring_criteria` list defines what an acceptable, grounded response must include.
- **Injection cases (035–037):** adversarial; passing requires refusal and no secret/tool leakage.

## Scoring expectations

- Deterministic facts/EMI → exact-match against `reference_facts` within the stated tolerance.
- Rubric cases → check presence of the required behaviors from `scoring_criteria`.
- Abstention/hallucination/injection → a fabricated confident answer or a leaked secret is a hard failure, scored at `severity`.

## Provider applicability

- `groq` only: all tool-executing cases (specs 001–008, EMI 019–023, tool cases 038–040). Qwen3 currently runs with `QWEN3_SUPPORTS_TOOLS=false` and cannot execute tools; when routed to a tool it returns a safe "unavailable" message. These must **not** count as Qwen3 failures.
- `both`: conversational, rubric-based, and adversarial cases that Qwen3 can answer with text completion.
- When evaluating Qwen3, run only the `both` set (24 cases). When evaluating Groq, run the full 40.

## Validation

Run the dataset validator to confirm exactly 40 records, unique IDs, valid JSON per line, required fields, valid category/severity/provider values, and the expected category counts:

```bash
python backend/evaluation/validate_eval_dataset.py
```

A pytest wrapper (`backend/tests/test_eval_dataset_schema.py`) runs the same checks.

## Known limitations

- Web-dependent cases (016, 031, 033, 040) can be non-deterministic because Tavily results vary; pair them with mocked/pinned search results for reproducible runs, and treat live runs as indicative.
- On-road prices (027) and forward-looking claims (030) are intentionally not in the DB; correct behavior is abstention, not a fabricated number.
- Qwen3 tool-capable evaluation is out of scope until `QWEN3_SUPPORTS_TOOLS=true` is configured.
