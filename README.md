# Task Router / Task Value Lab

Task Router is a local Flask application for two related decisions:

1. Estimate the economic value and safe model budget for GDPVal-style occupational tasks.
2. Measure which configured model actually performs best on reproducible tool-calling, spreadsheet, and optional research evaluations.

The economic-value model and routing benchmark stay deliberately separate. GDPVal supplies the economic-work anchor; BFCL, SpreadsheetBench, and optional GAIA runs supply measured quality, cost, latency, and reliability evidence. Non-GDPVal tasks never receive invented dollar-value labels.

## Current status

This repository is a working decision-support prototype. Its GDPVal target is a transparent proxy derived from configured wages, estimated human effort, employer load, and realizability assumptions. It is not observed business value.

The active recommendation path still uses the dated model catalog's heuristic capability scores. Eval-derived routing is marked experimental until it passes every locked-holdout promotion check. A passing report makes the measured router eligible for promotion; it does not silently rewrite existing forecast artifacts.

## Features

- Deterministic 70/10/20 GDPVal train, calibration, and test splits.
- Gradient-boosted proxy-value forecasts with split-conformal P10/P50/P90 intervals.
- A maximum model-cost policy equal to 10% of predicted P50 value, with an operational budget equal to 10% of P10.
- A reproducible 88-task core routing benchmark and an optional 100-task GAIA profile.
- Native benchmark scoring, bounded model/tool execution, token-based cost accounting, and reliability reruns.
- Calibration-only success estimates and locked-holdout comparisons against five routing baselines.
- Flask APIs and a dashboard for forecasts, model diagnostics, eval runs, provenance, and promotion status.

## Quick start

Python 3.11–3.13 is recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

export GDPVAL_DATA_PATH=/path/to/gdpval/output_data.csv
.venv/bin/python train.py
.venv/bin/python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The GDPVal CSV must contain these columns:

| Column | Purpose |
| --- | --- |
| `task_id` | Stable task identity |
| `sector` | Economic sector |
| `occupation` | Occupation used for wage assumptions |
| `prompt` | Task text used for feature extraction |

If `GDPVAL_DATA_PATH` is unset, the current development default is `/Users/manyam2025/src/hf-evals/gdpval/output_data.csv`.

## Dashboard

The frontend has four tabs:

| Tab | Contents |
| --- | --- |
| **Task forecasts** | Forecast intervals, value assumptions, budgets, and current recommendations |
| **Model & splits** | GDPVal split counts, held-out metrics, residuals, predictions, and feature importance |
| **Router evals** | Benchmark coverage, measured model quality/cost/latency, failures, calibration, baselines, provenance, and promotion checks |
| **Model catalog** | Provider-neutral catalog snapshot and normalized token pricing |

The Router evals tab discovers complete, partial, failed, and absent runs. Complete runs show measured outcomes and promotion checks; incomplete runs retain their recorded status without changing the active router.

## Evaluation profiles

Task selection is committed in [`evals/manifest.json`](./evals/manifest.json).

| Profile | BFCL v4 | SpreadsheetBench Verified | GAIA validation | Calibration | Holdout | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `local-core` | 64 | 24 | 0 | 66 | 22 | 88 |
| `local-plus-gaia` | 64 | 24 | 12 | 75 | 25 | 100 |

The benchmark sources serve different purposes:

| Source | Coverage | Native scoring | License/access |
| --- | --- | --- | --- |
| [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html) | 16 each from simple, multiple-call, parallel-call, and relevance-detection categories | Function name, call count, argument, and relevance evaluation | Apache-2.0, public |
| [SpreadsheetBench Verified](https://github.com/Reality2byte/spreadsheetbench) | Four tasks in each official task-type × workbook-complexity tertile | Official answer-range workbook comparison | CC-BY-SA-4.0, public |
| [GAIA](https://ai.meta.com/research/publications/gaia-a-benchmark-for-general-ai-assistants/) | Four supported validation tasks from each difficulty level | Normalized exact answer | Gated upstream terms; redistribution prohibited |

GDPVal remains the sole source for dollar-value and budget-policy evaluation.

### Reproducibility and integrity

The manifest records source revisions, SHA-256 file checksums, task IDs, selection keys, grader versions, licenses, access policies, and split assignments. Builds enforce:

- Seed `42` and SHA-256 ordering within every stratum.
- Exact source and split counts.
- A 75/25 calibration/holdout split in every stratum.
- Full SpreadsheetBench population validation across all 400 verified records.
- Spreadsheet complexity tertiles based on populated cells, formula cells, and worksheet count.
- GAIA filtering to file types supported by the local scaffold.
- Exact and five-gram near-duplicate prompt detection across included sources.
- Rejection of missing assets, changed upstream files, changed records, duplicate IDs, and stale built profiles.

Pinned revisions:

- BFCL/Gorilla: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- SpreadsheetBench: `ab0b742b0fc95b946f212d80ac7771b5531272e4`
- GAIA: `682dd723ee1e1697e00360edccf2366dc8418dd9`

Downloaded questions, answers, workbooks, and gated files live under `.eval-cache/`, which is ignored by Git.

## Run the core benchmark

### 1. Build the normalized task cache

```bash
.venv/bin/python -m task_value_mvp.evals build --profile local-core
```

The build prints its task counts, split counts, source revisions, manifest hash, and normalized `tasks.jsonl` path. It downloads data but does not call any model APIs.

### 2. Configure exactly three models

```bash
cp eval_models.example.yaml eval_models.local.yaml
```

Edit the ignored local file so it contains one distinct catalog model for each role: `low`, `middle`, and `premium`. Each model needs:

- A unique alias and catalog model ID.
- An OpenAI-compatible base URL and provider model name.
- The name of an environment variable containing its API credential.
- Input, output, and optional cached-input prices per million tokens.

The runner rejects unknown or duplicate catalog IDs, malformed endpoint URLs, missing roles, and missing credential variables. Credential values are never written to manifests, traces, results, or run metadata.

Using the names from the example configuration:

```bash
export TASK_ROUTER_LOW_API_KEY=...
export TASK_ROUTER_MIDDLE_API_KEY=...
export TASK_ROUTER_PREMIUM_API_KEY=...
```

### 3. Run and report

```bash
.venv/bin/python -m task_value_mvp.evals run \
  --profile local-core \
  --models-config eval_models.local.yaml
```

The `run` command creates a unique run directory and generates `report.json` after execution. Regenerate a report later with:

```bash
.venv/bin/python -m task_value_mvp.evals report --run-id <run-id>
```

Use `--cache` or `--artifacts` on the CLI to override their default locations.

The core profile begins with 264 task-model executions. Each task selected for reliability reruns adds six more executions: two trials across all three models.

## Optional GAIA profile

GAIA is gated. Request and receive access upstream, authenticate the Hugging Face CLI, and then build the optional profile:

```bash
.venv/bin/hf auth login
.venv/bin/python -m task_value_mvp.evals build --profile local-plus-gaia
.venv/bin/python -m task_value_mvp.evals run \
  --profile local-plus-gaia \
  --models-config eval_models.local.yaml
```

GAIA questions, answers, and attachments remain only in the ignored cache. They must not be committed or redistributed.

## Execution behavior

Every initial task-model pair runs once with the same scaffold, temperature `0`, seed `42`, timeout, tool budget, output-token limit, and tool-output-size limit. A task receives two additional trials for all three models when:

- The initial models disagree on pass/fail.
- Any initial execution returns an error.
- The task is in the deterministic 10% reliability audit sample.

The runner provides source-specific execution:

- **BFCL:** OpenAI-compatible function schemas with native call and argument grading.
- **SpreadsheetBench:** A constrained workbook tool for inspecting, writing, copying, inserting, deleting, and sorting cells. Every task uses a temporary workspace. If LibreOffice/`soffice` is installed, outputs are recalculated before grading so formula values are cached.
- **GAIA:** Packaged file/image input plus bounded public-web search and fetch tools. URL validation blocks private and non-routable targets, nonstandard ports, and credential-bearing URLs.

HTTP 429, 5xx, network, and timeout failures are bounded and recorded. Missing provider usage fields default to zero and remain visible in trial results rather than being estimated silently.

## Reports, baselines, and promotion

Success estimates are fitted only on calibration tasks. The local evaluator considers only the three models in the run and evaluates routing only on locked holdout tasks.

Reports include overall, per-model, and per-source measurements for:

- Pass rate, normalized score, native grader results, coverage, severe failures, and execution errors.
- Total cost, cost per task, cost per successful task, and cached-token-aware API cost.
- P50/P95 latency and token usage.
- Brier score and expected calibration error.
- Quality regret and cost regret.
- Budget-violation rate, abstention coverage, and abstention precision.
- Stability across repeated trials.

The holdout comparison includes:

- Cheapest-only
- Premium-only
- Seeded-random
- Current catalog heuristic
- Evidence-backed router
- Oracle

The evidence-backed router is eligible for promotion only when all checks pass:

| Check | Required result |
| --- | --- |
| Cost per successful task | At least 20% lower than premium-only |
| Overall pass rate | Within 3 percentage points of premium-only |
| Per-source pass rate | Within 5 percentage points of premium-only |
| Severe failures | No increase over premium-only |
| Budget violations | At most 1% |
| Paired bootstrap | 95% interval supports positive savings and quality non-inferiority |

Until then, reports remain experimental and the existing recommendation path stays active.

## Generated artifacts

GDPVal training writes:

```text
artifacts/
├── model.pkl
├── metrics.json
├── forecasts.json
└── forecasts.csv
```

Evaluation builds and runs write:

```text
.eval-cache/
├── sources/                    # downloaded public or gated data
└── built/<profile>/
    ├── build.json              # manifest hash, revisions, and counts
    └── tasks.jsonl             # normalized tasks, including private grader data

artifacts/evals/<run-id>/
├── run.json                    # models, limits, source build, and run status
├── results.jsonl               # trial-level scores, usage, cost, errors, and provenance
├── report.json                 # aggregates, baselines, intervals, and promotion gate
└── artifacts/<task>/<model>/<trial>/
    ├── trace.json
    └── case-*.xlsx             # SpreadsheetBench outputs when applicable
```

Run directories are uniquely named and never reused by the runner. `.eval-cache/`, `eval_models.local.yaml`, and `artifacts/evals/` are ignored.

Only load `model.pkl` when it was generated locally; pickle files from untrusted sources can execute code.

## HTTP API

| Method and path | Response |
| --- | --- |
| `GET /health` | Application status and GDPVal artifact metadata |
| `GET /api/tasks` | Compact forecast list |
| `GET /api/tasks/<task_id>` | Complete forecast and recommendation record |
| `GET /api/model` | Training metrics, chart data, and model catalog |
| `GET /api/export.json` | Download complete forecasts as JSON |
| `GET /api/export.csv` | Download flat forecasts as CSV |
| `GET /api/evals/runs` | Eval runs ordered newest first |
| `GET /api/evals/runs/<run_id>` | Run metadata and report when available |
| `GET /api/evals/runs/<run_id>/results` | Trial-level run results |

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GDPVAL_DATA_PATH` | Development path shown above | Input GDPVal CSV |
| `TASK_VALUE_ARTIFACT_DIR` | `./artifacts` | GDPVal and eval artifact root |
| `TASK_ROUTER_EVAL_CACHE` | `./.eval-cache` | Downloaded and normalized eval cache |
| `FLASK_HOST` | `127.0.0.1` | Flask bind address |
| `FLASK_PORT` | `5000` | Flask port |
| `FLASK_DEBUG` | `0` | Set to `1` for Flask debug mode |

Provider credential variable names are configured per model in `eval_models.local.yaml`.

The example model file also defines these run controls:

| Setting | Example | Behavior |
| --- | ---: | --- |
| `max_cost_per_task_usd` | `1.00` | Budget used to calculate the holdout violation rate; it is not a pre-call spend cutoff |
| `max_output_tokens` | `2048` | Maximum requested completion length per model response |
| `max_tool_calls` | `12` | Tool-call ceiling per task trial |
| `max_tool_output_chars` | `20000` | Maximum tool result returned to the model |
| `timeout_seconds` | `90` | Task/request time bound used by the execution scaffold |

## Tests

Run the complete suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover:

- Existing GDPVal splits, forecasts, budgets, metadata, exports, and Flask routes.
- Manifest counts, deterministic selection, locked split assignments, and duplicate detection.
- Correct, incorrect, malformed, and corrupt grader fixtures.
- OpenAI-compatible rate-limit retries, tool-call parsing, and missing usage fields.
- Hand-computed synthetic report metrics, all baselines, and promotion checks.
- Empty and partial eval API states, plus complete synthetic report generation.

Useful additional checks:

```bash
node --check static/app.js
.venv/bin/python -m compileall -q app.py task_value_mvp tests
git diff --check
```

## Project layout

```text
app.py                         # Flask application and APIs
train.py                       # GDPVal training entry point
task_value_mvp/
├── catalog.py                 # provider-neutral model catalog and active heuristic
├── data.py                    # GDPVal loading and deterministic splits
├── modeling.py                # proxy target, model training, intervals, and forecasts
├── service.py                 # forecast artifact store
└── evals/
    ├── manifest.py            # pinned source adapters and integrity validation
    ├── schema.py              # EvalTask, EvalResult, and model contracts
    ├── runner.py              # bounded three-model execution
    ├── graders.py             # native benchmark scoring
    ├── reporting.py           # calibration, baselines, metrics, and promotion gate
    └── store.py               # eval API artifact store
evals/manifest.json            # committed source/task/split manifest
eval_models.example.yaml       # safe three-tier configuration template
templates/ and static/         # dashboard UI
tests/                         # GDPVal and eval coverage
```

## Model catalog and economic-value discipline

The candidate catalog is a curated [PricePerToken](https://pricepertoken.com/) snapshot dated August 5, 2026. It covers several providers and price/capability tiers. Retrain after catalog changes to regenerate GDPVal recommendations and versioned catalog metadata.

GDPVal modeling uses:

- 154 rows (70%) to fit the feature encoder and `GradientBoostingRegressor`.
- 22 rows (10%) to calibrate an 80% split-conformal interval.
- 44 rows (20%) for final held-out metrics and visualization.

The maximum allowed cost is exactly 10% of the P50 forecast. The operational recommendation budget is conservatively 10% of P10.

Environment-heavy suites such as SWE-bench Verified, τ-bench, WorkArena, and OSWorld remain future extensions after the lightweight runner is proven stable.

See [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) for the original MVP architecture and economic-target construction.
