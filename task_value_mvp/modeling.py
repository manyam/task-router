from __future__ import annotations

import csv
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .catalog import CATALOG_SNAPSHOT_DATE, CATALOG_SOURCE_URL, public_catalog, recommend_model
from .config import INTERVAL_COVERAGE, RANDOM_SEED
from .data import assign_splits, exact_split_counts
from .features import DenseTaskFeatureEncoder, derive_task_signals, proxy_targets, summarize_categories


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def conformal_quantile(residuals: np.ndarray, coverage: float = INTERVAL_COVERAGE) -> float:
    n = len(residuals)
    if n == 0:
        raise ValueError("Calibration residuals cannot be empty")
    corrected = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(residuals, corrected, method="higher"))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    nonzero = np.maximum(np.abs(y_true), 1e-9)
    return {
        "mae_usd": _safe_float(mean_absolute_error(y_true, y_pred)),
        "rmse_usd": _safe_float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": _safe_float(r2_score(y_true, y_pred)),
        "mape_percent": _safe_float(np.mean(np.abs(y_true - y_pred) / nonzero) * 100),
    }


@dataclass
class TrainedBundle:
    encoder: DenseTaskFeatureEncoder
    model: GradientBoostingRegressor
    conformal_radius: float
    metadata: dict[str, Any]

    def predict_interval(self, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions = np.maximum(0.0, self.model.predict(self.encoder.transform(rows)))
        lower = np.maximum(0.0, predictions - self.conformal_radius)
        upper = predictions + self.conformal_radius
        return lower, predictions, upper


def train_bundle(rows: list[dict[str, Any]]) -> tuple[TrainedBundle, list[dict[str, Any]], dict[str, Any]]:
    split_names = assign_splits(rows)
    split_indices = {
        name: [index for index, split in enumerate(split_names) if split == name]
        for name in ("train", "calibration", "test")
    }
    subsets = {name: [rows[index] for index in indices] for name, indices in split_indices.items()}
    encoder = DenseTaskFeatureEncoder()
    x_train = encoder.fit_transform(subsets["train"])
    y_train = proxy_targets(subsets["train"])
    model = GradientBoostingRegressor(
        n_estimators=160,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=5,
        loss="huber",
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)

    x_cal = encoder.transform(subsets["calibration"])
    y_cal = proxy_targets(subsets["calibration"])
    cal_pred = np.maximum(0.0, model.predict(x_cal))
    radius = conformal_quantile(np.abs(y_cal - cal_pred))

    x_test = encoder.transform(subsets["test"])
    y_test = proxy_targets(subsets["test"])
    test_pred = np.maximum(0.0, model.predict(x_test))
    test_metrics = _regression_metrics(y_test, test_pred)
    coverage = float(np.mean((y_test >= np.maximum(0.0, test_pred - radius)) & (y_test <= test_pred + radius)))
    test_metrics["interval_coverage_percent"] = coverage * 100

    metadata = {
        "trained_at": _utc_now(),
        "target": "proxy_economic_value_usd",
        "target_warning": "Proxy target derived from configured wage and prompt-effort assumptions; not observed business value.",
        "algorithm": "GradientBoostingRegressor",
        "hyperparameters": model.get_params(),
        "split_seed": RANDOM_SEED,
        "split_counts": {name: len(indices) for name, indices in split_indices.items()},
        "requested_split_fractions": {"train": 0.70, "calibration": 0.10, "test": 0.20},
        "conformal_coverage_target_percent": INTERVAL_COVERAGE * 100,
        "conformal_radius_usd": radius,
        "test_metrics": test_metrics,
        "feature_count": len(encoder.feature_names),
    }
    bundle = TrainedBundle(encoder=encoder, model=model, conformal_radius=radius, metadata=metadata)
    lower, median, upper = bundle.predict_interval(rows)

    forecasts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        signals = derive_task_signals(row)
        p10, p50, p90 = map(float, (lower[index], median[index], upper[index]))
        max_cost = p50 * 0.10
        operational_budget = p10 * 0.10
        recommendation = recommend_model(signals, p50, operational_budget)
        selected = recommendation["selected"]
        forecasts.append(
            {
                "task_id": row["task_id"],
                "sector": row["sector"],
                "occupation": row["occupation"],
                "prompt": row["prompt"],
                "split": split_names[index],
                "task_profile": {
                    "task_families": signals.task_families,
                    "required_modalities": signals.required_modalities,
                    "risk_tier": signals.risk_tier,
                    "estimated_input_tokens": int(signals.values["estimated_input_tokens"]),
                    "estimated_output_tokens": int(signals.values["estimated_output_tokens"]),
                },
                "proxy_label": {
                    "value_usd": signals.proxy_value,
                    "assumptions": signals.proxy_assumptions,
                    "warning": metadata["target_warning"],
                },
                "economic_value_forecast": {"p10_usd": p10, "p50_usd": p50, "p90_usd": p90},
                "cost_policy": {
                    "budget_fraction": 0.10,
                    "max_cost_usd": max_cost,
                    "operational_budget_usd": operational_budget,
                },
                "recommendation": recommendation,
                "summary": {
                    "selected_model": selected["display_name"] if selected else "Abstain",
                    "expected_model_cost_usd": selected["expected_execution_cost_usd"] if selected else None,
                    "expected_success_probability": selected["expected_success_probability"] if selected else None,
                },
            }
        )

    importance = sorted(
        (
            {"feature": name, "importance": float(value)}
            for name, value in zip(encoder.feature_names, model.feature_importances_)
            if value > 0
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )[:20]
    test_points = []
    for local_index, global_index in enumerate(split_indices["test"]):
        test_points.append(
            {
                "task_id": rows[global_index]["task_id"],
                "sector": rows[global_index]["sector"],
                "actual": float(y_test[local_index]),
                "predicted": float(test_pred[local_index]),
                "residual": float(y_test[local_index] - test_pred[local_index]),
            }
        )
    metrics = {
        "metadata": metadata,
        "feature_importance": importance,
        "test_points": test_points,
        "sectors": summarize_categories(rows, "sector"),
        "occupations": summarize_categories(rows, "occupation"),
        "catalog_snapshot_date": CATALOG_SNAPSHOT_DATE,
        "catalog_source_url": CATALOG_SOURCE_URL,
        "catalog_method": "Curated current decision set from PricePerToken; legacy and near-duplicate entries excluded.",
        "model_catalog": public_catalog(),
    }
    return bundle, forecasts, metrics


def save_artifacts(
    bundle: TrainedBundle,
    forecasts: list[dict[str, Any]],
    metrics: dict[str, Any],
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "model.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)
    (artifact_dir / "forecasts.json").write_text(json.dumps(forecasts, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fieldnames = [
        "task_id", "sector", "occupation", "split", "proxy_value_usd", "forecast_p10_usd",
        "forecast_p50_usd", "forecast_p90_usd", "max_cost_usd", "operational_budget_usd",
        "recommendation_status", "selected_model", "expected_model_cost_usd", "expected_success_probability",
    ]
    with (artifact_dir / "forecasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in forecasts:
            writer.writerow(
                {
                    "task_id": item["task_id"],
                    "sector": item["sector"],
                    "occupation": item["occupation"],
                    "split": item["split"],
                    "proxy_value_usd": item["proxy_label"]["value_usd"],
                    "forecast_p10_usd": item["economic_value_forecast"]["p10_usd"],
                    "forecast_p50_usd": item["economic_value_forecast"]["p50_usd"],
                    "forecast_p90_usd": item["economic_value_forecast"]["p90_usd"],
                    "max_cost_usd": item["cost_policy"]["max_cost_usd"],
                    "operational_budget_usd": item["cost_policy"]["operational_budget_usd"],
                    "recommendation_status": item["recommendation"]["status"],
                    "selected_model": item["summary"]["selected_model"],
                    "expected_model_cost_usd": item["summary"]["expected_model_cost_usd"],
                    "expected_success_probability": item["summary"]["expected_success_probability"],
                }
            )


def load_bundle(path: Path) -> TrainedBundle:
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, TrainedBundle):
        raise TypeError("Artifact does not contain a TrainedBundle")
    return bundle
