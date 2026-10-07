from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ForecastStore:
    def __init__(self, artifact_dir: Path) -> None:
        forecasts_path = artifact_dir / "forecasts.json"
        metrics_path = artifact_dir / "metrics.json"
        if not forecasts_path.exists() or not metrics_path.exists():
            raise FileNotFoundError("Training artifacts are missing; run `python train.py` first.")
        self.forecasts: list[dict[str, Any]] = json.loads(forecasts_path.read_text(encoding="utf-8"))
        self.metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.by_id = {item["task_id"]: item for item in self.forecasts}

    def compact_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": item["task_id"],
                "sector": item["sector"],
                "occupation": item["occupation"],
                "split": item["split"],
                "prompt_preview": " ".join(item["prompt"].split())[:180],
                "p10_usd": item["economic_value_forecast"]["p10_usd"],
                "p50_usd": item["economic_value_forecast"]["p50_usd"],
                "p90_usd": item["economic_value_forecast"]["p90_usd"],
                "max_cost_usd": item["cost_policy"]["max_cost_usd"],
                "operational_budget_usd": item["cost_policy"]["operational_budget_usd"],
                "selected_model": item["summary"]["selected_model"],
                "recommendation_status": item["recommendation"]["status"],
            }
            for item in self.forecasts
        ]

