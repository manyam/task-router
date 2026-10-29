# Task Value Lab MVP

A local Flask application that forecasts economic value for GDPVal-style tasks, applies a flat 10% maximum-cost policy, and measures routing decisions with reproducible tool-calling and spreadsheet evaluations.

## Important limitation

GDPVal provides prompts and evaluation rubrics, but it does not provide observed human time, compensation, realized business value, or repeated model execution outcomes. This MVP therefore trains on a transparent **proxy economic-value target** derived from configured occupation wages and prompt-effort signals. Model-selection quality is also heuristic. The UI labels both limitations explicitly.

This is a working pipeline and decision-support prototype, not a validated economic valuation system.

The frontend has four top-level tabs:

- **Task forecasts** for browsing forecasts, proxy assumptions, budgets, and recommendations.
- **Model & splits** for the 70/10/20 allocation, held-out metrics, residuals, predictions, and feature importance.
- **Router evals** for benchmark coverage, measured model quality/cost/latency, holdout baselines, calibration, and the promotion gate.
- **Model catalog** for the provider-neutral PricePerToken snapshot and normalized pricing metadata.

## Setup

Python 3.11–3.13 is recommended because scientific-package support for the newest Python releases may lag.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The default data path is `/Users/manyam2025/src/hf-evals/gdpval/output_data.csv`. Override it with:

```bash
GDPVAL_DATA_PATH=/path/to/tasks.csv .venv/bin/python train.py
```

The CSV must contain `task_id`, `sector`, `occupation`, and `prompt`.

## Test

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Routing evaluations

The default `local-core` profile contains 88 deterministic tasks: 64 BFCL v4 tasks across simple, multiple-call, parallel-call, and relevance detection, plus 24 SpreadsheetBench Verified tasks balanced by task type and workbook-complexity tertile. The split is 66 calibration tasks and 22 locked holdout tasks. GDPVal remains the only source used for economic-value and budget-policy labels.

Build the normalized task cache, create an untracked three-model configuration, and run the suite:

```bash
.venv/bin/python -m task_value_mvp.evals build --profile local-core
cp eval_models.example.yaml eval_models.local.yaml
# Edit the three catalog mappings, prices, endpoints, and credential variable names.
export TASK_ROUTER_LOW_API_KEY=...
export TASK_ROUTER_MIDDLE_API_KEY=...
export TASK_ROUTER_PREMIUM_API_KEY=...
.venv/bin/python -m task_value_mvp.evals run --models-config eval_models.local.yaml
.venv/bin/python -m task_value_mvp.evals report --run-id <run-id>
```

`eval_models.local.yaml`, downloaded data, answers, workbooks, and gated assets are ignored. Credentials are read only from the named environment variables and are excluded from manifests, traces, and run metadata. Immutable run outputs are written to `artifacts/evals/<run-id>/`.

The optional profile adds 12 GAIA validation tasks, four per difficulty level:

```bash
.venv/bin/hf auth login
.venv/bin/python -m task_value_mvp.evals build --profile local-plus-gaia
.venv/bin/python -m task_value_mvp.evals run --profile local-plus-gaia --models-config eval_models.local.yaml
```

GAIA access must already be approved on Hugging Face. Its questions, answers, and files remain in `.eval-cache/` and must not be redistributed.

The committed [evaluation manifest](./evals/manifest.json) pins source revisions, task IDs, graders, licenses, access rules, and the calibration/holdout assignment. Builds reject missing records/assets and duplicate or near-duplicate prompts. BFCL uses Apache-2.0 data at revision `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`; SpreadsheetBench uses CC-BY-SA-4.0 data at revision `ab0b742b0fc95b946f212d80ac7771b5531272e4`; the gated GAIA revision is `682dd723ee1e1697e00360edccf2366dc8418dd9`.

Each task-model pair receives one fixed-seed trial. Disagreements, execution errors, and a deterministic 10% reliability sample receive two additional trials. The report compares the fitted router against cheapest-only, premium-only, seeded-random, current-heuristic, and oracle baselines on the locked holdout. Eval-derived routing remains experimental unless all cost, quality, severe-failure, budget, and paired-bootstrap promotion checks pass.

Spreadsheet outputs are recalculated with a local LibreOffice/`soffice` binary when one is available before the official answer-range comparison. The runner records whether recalculation occurred in each trace.

## Outputs

Training writes these generated files under `artifacts/`:

- `model.pkl` — locally generated feature encoder, gradient-boosted regressor, and calibration radius;
- `metrics.json` — split metadata, held-out metrics, chart points, feature importance, and model catalog;
- `forecasts.json` — complete per-task records; and
- `forecasts.csv` — flat export.

Only load `model.pkl` when it was generated locally; pickle is not safe for untrusted artifacts.

## Model pricing catalog

The current candidate set is a curated snapshot of [PricePerToken](https://pricepertoken.com/) dated August 5, 2026. It includes models from OpenAI, Anthropic, Google, DeepSeek, and Moonshot AI across multiple price and capability tiers. Legacy and near-duplicate table entries are excluded. Prices are versioned in generated metrics; retrain after changing the catalog to regenerate all recommendations.

The catalog's capability scores and task-family strengths remain the active MVP recommendation heuristic. Evaluation runs fit experimental success estimates on the calibration partition and evaluate them only on the locked holdout; they replace the active path only after the promotion gate passes.

## Split and modeling discipline

- 154 rows (70%) train the feature encoder and `GradientBoostingRegressor`.
- 22 rows (10%) calibrate an 80% split-conformal interval.
- 44 rows (20%) are held out for final metrics and visualization.
- The maximum allowed cost is exactly 10% of the P50 forecast.
- The operational recommendation budget is conservatively 10% of P10.

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for architecture, target construction, API contracts, acceptance criteria, and the path to measured labels.
