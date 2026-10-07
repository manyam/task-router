# MVP Implementation Plan

## Product boundary

The MVP is a local Flask application that reads GDPVal-style tasks from a CSV, forecasts task economic value with a gradient-boosted regressor, applies a maximum AI cost of 10% of that forecast, and recommends a model from a versioned local Fireworks-style catalog.

The supplied GDPVal data has no observed human time, compensation, realized business value, or model-run outcomes. Therefore, this MVP will generate a transparent deterministic **proxy target** for economic value from occupation wage assumptions and prompt-derived task effort. The UI and exported data will label it `proxy_economic_value`, not ground truth. The learned model demonstrates training, calibration, testing, uncertainty, and serving. Replacing proxy labels with measured values later will not require changing the application contract.

## Acceptance criteria

1. Load all 220 rows from `/Users/manyam2025/src/hf-evals/gdpval/output_data.csv`.
2. Use only `sector`, `occupation`, and `prompt` as predictive task inputs.
3. Produce an exact deterministic 70%/10%/20% train/calibration/test allocation: 154/22/44 rows.
4. Train a scikit-learn `GradientBoostingRegressor` on dense, interpretable task features.
5. Fit an uncertainty interval using absolute residuals on the untouched calibration split.
6. Report test MAE, RMSE, R², MAPE, and interval coverage without training on test data.
7. Forecast P10/P50/P90 task value and compute `max_cost = 0.10 × P50` plus a conservative `operational_budget = 0.10 × P10`.
8. Estimate full execution cost for each catalog model and recommend the highest expected-quality eligible model that fits the operational budget; abstain if none qualifies.
9. Serve two frontend tabs:
   - **Task forecasts:** filterable task list and selected-task forecast/model details.
   - **Model & splits:** split counts, metrics, actual-vs-predicted test visualization, residual distribution, and feature importance.
10. Export all forecasts as JSON and CSV through Flask endpoints.
11. Include automated tests for split sizes, budget arithmetic, interval ordering, recommendation validity, and HTTP endpoints.

## Data pipeline

### Input

The loader validates the required fields and retains `task_id` for identity. Multiline prompts are supported through Python's standard CSV parser.

### Proxy target construction

Create the label independently of model training:

```text
proxy_value = estimated_human_hours
              × occupation_hourly_wage
              × 1.35 employer-load factor
              × 0.65 realizability factor
              × risk/complexity adjustment
```

Occupation wages come from an explicit local mapping with a documented fallback. Estimated hours use a deterministic formula based on prompt word count, enumerated requirements, attachment mentions, quantitative/spreadsheet indicators, artifact complexity, research requirements, and regulatory language. Values are rounded only for display.

This construction is intentionally inspectable. It must never be presented as observed value.

### Predictive features

Use a compact dense feature matrix so ordinary gradient boosting remains appropriate:

- one-hot sector;
- one-hot occupation;
- prompt word, character, sentence, and line counts;
- numeric/list requirement counts;
- keyword-family counts for analysis, spreadsheets, writing, coding, research, media, compliance, and files;
- task-risk flags;
- expected input/output token estimates; and
- required-modality flags.

Categorical vocabularies are learned on the train split only. Unknown categories get all-zero one-hot representations. Feature names are retained for visualization.

## Split and leakage controls

Use a fixed seed and deterministic stratified allocation where feasible. The exact row counts are 154 train, 22 calibration, and 44 test. Because every occupation has only five examples, exact three-way occupation stratification cannot simultaneously guarantee the requested global ratio. The splitter will stratify by sector for stable coverage, record indices, and assert disjointness.

- **Train:** fit feature vocabulary and gradient-boosted model.
- **Calibration:** compute a split-conformal absolute residual quantile for prediction intervals. It does not tune model parameters.
- **Test:** final metrics and visualizations only.

The saved split assignment is attached to every task so the UI can display it.

## Model training and uncertainty

Start with conservative fixed hyperparameters rather than searching across the small dataset:

- 160 estimators;
- learning rate 0.04;
- maximum depth 3;
- minimum samples per leaf 5;
- Huber loss;
- fixed random seed.

For a prediction `ŷ`, let `q` be the finite-sample corrected 80th percentile of calibration absolute residuals. Emit:

```text
P10 = max(0, ŷ - q)
P50 = max(0, ŷ)
P90 = ŷ + q
```

These are empirical prediction bounds, not a claim that the economic target itself is objectively measured. Report test interval coverage.

## Model recommendation

Seed a local, dated model catalog from publicly displayed Fireworks model entries and prices. Each entry includes modalities, context, price per million input/output tokens, and a coarse capability score. The catalog is configuration data, not training data, and can later be replaced by API ingestion.

For task `i` and model `m`:

```text
execution_cost = expected_input_tokens × input_rate
               + expected_output_tokens × output_rate
               + retry/validation multiplier

expected_success = sigmoid(model_capability
                           - task_complexity
                           + modality_match
                           + domain bonuses)
```

Filter out modality/context mismatches and candidates whose conservative execution cost exceeds `10% × P10`. Enforce a risk-tier quality floor. Among survivors, maximize:

```text
expected_net_value = expected_success × P50 - expected_execution_cost
```

This selection score is explicitly heuristic until real task-model execution labels exist. Return the top recommendation, alternatives, and rejection reasons.

## Application structure

```text
task-router/
├── app.py                         Flask application factory and routes
├── train.py                       CLI training/artifact generation
├── requirements.txt
├── README.md
├── task_value_mvp/
│   ├── __init__.py
│   ├── config.py                  paths and model constants
│   ├── data.py                    CSV validation and splits
│   ├── features.py                proxy labels and feature extraction
│   ├── modeling.py                train/calibrate/evaluate/save
│   ├── catalog.py                 model catalog and recommendation
│   └── service.py                 artifact loading and forecast assembly
├── templates/
│   └── index.html                 two-tab application shell
├── static/
│   ├── app.js                     filtering, detail panels, canvas charts
│   └── styles.css
├── artifacts/                     generated model, metrics, forecasts
└── tests/
    └── test_mvp.py
```

Persist the trained model with `pickle` because the artifact is locally generated and locally consumed. Persist portable forecast and metric data as JSON/CSV. Do not load untrusted pickle files.

## Flask routes

- `GET /` — application UI.
- `GET /api/tasks` — compact searchable task/forecast records.
- `GET /api/tasks/<task_id>` — full task, interval, cost ceiling, recommendation, alternatives, and assumptions.
- `GET /api/model` — model metadata, split counts, metrics, calibration, feature importance, and chart points.
- `GET /api/export.json` — all full forecast records.
- `GET /api/export.csv` — flattened forecast export.
- `GET /health` — artifact and row-count health status.

## Frontend behavior

### Task forecasts tab

- summary cards for task count, median value, median budget, and recommendation coverage;
- text, sector, occupation, and split filters;
- paginated task table;
- task detail drawer/panel showing prompt, proxy assumptions, P10/P50/P90, exact 10% calculations, selected model, expected cost/success, and alternatives;
- labels that distinguish proxy labels, ML forecasts, and heuristic model-selection scores.

### Model & splits tab

- 70/10/20 split cards and methodology note;
- test metrics and calibration coverage;
- canvas scatter plot of test proxy target versus prediction, colored by split where relevant;
- residual histogram;
- horizontal feature-importance chart;
- model catalog table and current pricing snapshot date.

Use native browser canvas and CSS so the UI has no CDN runtime dependency.

## Verification

Automated tests will assert:

- exact split counts and no overlap;
- train-only category vocabulary;
- finite nonnegative forecasts;
- `P10 ≤ P50 ≤ P90`;
- max cost equals exactly 10% of P50;
- operational budget equals exactly 10% of P10;
- recommended models meet modality/context/budget constraints;
- model/test metrics and exports are present; and
- Flask routes return successful responses and valid payloads.

A final smoke test will train artifacts, run the tests, start Flask on localhost, query `/health`, and stop the server.

## Known limitations and next upgrade

The largest limitation is label validity, not algorithm choice. The first upgrade should collect observed practitioner time, review time, loaded cost, realized benefit, and repeated task-model outcomes. Those data can replace the proxy target and heuristic recommender while preserving the UI, APIs, split discipline, and optimizer interface built by this MVP.
