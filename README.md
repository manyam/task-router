# Task Value Lab MVP

A local Flask application that forecasts economic value for GDPVal-style tasks, applies a flat 10% maximum-cost policy, and recommends a candidate model from a dated, provider-neutral pricing catalog.

## Important limitation

GDPVal provides prompts and evaluation rubrics, but it does not provide observed human time, compensation, realized business value, or repeated model execution outcomes. This MVP therefore trains on a transparent **proxy economic-value target** derived from configured occupation wages and prompt-effort signals. Model-selection quality is also heuristic. The UI labels both limitations explicitly.

This is a working pipeline and decision-support prototype, not a validated economic valuation system.

The frontend has three top-level tabs:

- **Task forecasts** for browsing forecasts, proxy assumptions, budgets, and recommendations.
- **Model & splits** for the 70/10/20 allocation, held-out metrics, residuals, predictions, and feature importance.
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

## Outputs

Training writes these generated files under `artifacts/`:

- `model.pkl` — locally generated feature encoder, gradient-boosted regressor, and calibration radius;
- `metrics.json` — split metadata, held-out metrics, chart points, feature importance, and model catalog;
- `forecasts.json` — complete per-task records; and
- `forecasts.csv` — flat export.

Only load `model.pkl` when it was generated locally; pickle is not safe for untrusted artifacts.

## Model pricing catalog

The current candidate set is a curated snapshot of [PricePerToken](https://pricepertoken.com/) dated August 5, 2026. It includes models from OpenAI, Anthropic, Google, DeepSeek, and Moonshot AI across multiple price and capability tiers. Legacy and near-duplicate table entries are excluded. Prices are versioned in generated metrics; retrain after changing the catalog to regenerate all recommendations.

The catalog's capability scores and task-family strengths remain MVP heuristics. They must ultimately be replaced by repeated GDPVal-style task executions and calibrated quality estimates.

## Split and modeling discipline

- 154 rows (70%) train the feature encoder and `GradientBoostingRegressor`.
- 22 rows (10%) calibrate an 80% split-conformal interval.
- 44 rows (20%) are held out for final metrics and visualization.
- The maximum allowed cost is exactly 10% of the P50 forecast.
- The operational recommendation budget is conservatively 10% of P10.

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for architecture, target construction, API contracts, acceptance criteria, and the path to measured labels.
