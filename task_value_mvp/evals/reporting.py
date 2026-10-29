from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from task_value_mvp.catalog import MODEL_CATALOG

from .schema import EvalResult


def _read_results(path: Path) -> list[EvalResult]:
    return [
        EvalResult.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _aggregate_cells(results: list[EvalResult]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[EvalResult]] = defaultdict(list)
    for result in results:
        grouped[(result.eval_id, result.model_alias)].append(result)
    cells = {}
    for key, values in grouped.items():
        first = values[0]
        cells[key] = {
            "eval_id": first.eval_id,
            "model_alias": first.model_alias,
            "catalog_model_id": first.catalog_model_id,
            "source": first.source,
            "category": first.category,
            "split": first.split,
            "score": statistics.fmean(value.normalized_score for value in values),
            "passed": statistics.fmean(float(value.passed) for value in values) >= 0.5,
            "pass_rate": statistics.fmean(float(value.passed) for value in values),
            "severe_failure": any(value.severe_failure for value in values),
            "error": any(value.error for value in values),
            "cost_usd": statistics.fmean(value.cost_usd for value in values),
            "latency_ms": statistics.fmean(value.latency_ms for value in values),
            "token_usage": statistics.fmean(value.input_tokens + value.output_tokens for value in values),
            "trials": len(values),
            "trial_stability": max(
                statistics.fmean(float(value.passed) for value in values),
                1.0 - statistics.fmean(float(value.passed) for value in values),
            ),
            "capability_tags": first.provenance.get("capability_tags", []),
        }
    return cells


def _summary(cells: list[dict[str, Any]], expected: int | None = None) -> dict[str, Any]:
    if not cells:
        return {"tasks": 0, "coverage": 0.0, "pass_rate": 0.0, "mean_score": 0.0, "severe_failures": 0, "errors": 0, "cost_usd": 0.0, "cost_per_task_usd": None, "cost_per_pass_usd": None, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p50_token_usage": 0.0, "p95_token_usage": 0.0, "repeated_trial_stability": None}
    passes = sum(cell["passed"] for cell in cells)
    total_cost = sum(cell["cost_usd"] for cell in cells)
    latencies = [cell["latency_ms"] for cell in cells]
    tokens = [cell["token_usage"] for cell in cells]
    repeated = [cell["trial_stability"] for cell in cells if cell["trials"] > 1]
    return {
        "tasks": len(cells),
        "coverage": min(1.0, len(cells) / expected) if expected else 1.0,
        "pass_rate": passes / len(cells),
        "mean_score": statistics.fmean(cell["score"] for cell in cells),
        "severe_failures": sum(cell["severe_failure"] for cell in cells),
        "errors": sum(cell["error"] for cell in cells),
        "cost_usd": total_cost,
        "cost_per_task_usd": total_cost / len(cells),
        "cost_per_pass_usd": total_cost / passes if passes else None,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p50_token_usage": _percentile(tokens, 0.50),
        "p95_token_usage": _percentile(tokens, 0.95),
        "repeated_trial_stability": statistics.fmean(repeated) if repeated else None,
    }


def _fit_predictions(cells: dict[tuple[str, str], dict[str, Any]], models: list[str]) -> dict[tuple[str, str], float]:
    calibration = [cell for cell in cells.values() if cell["split"] == "calibration"]
    model_rates = {}
    for model in models:
        values = [float(cell["passed"]) for cell in calibration if cell["model_alias"] == model]
        model_rates[model] = statistics.fmean(values) if values else 0.5
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for cell in calibration:
        groups[(cell["model_alias"], cell["source"], cell["category"])].append(float(cell["passed"]))
    predictions = {}
    for cell in cells.values():
        key = (cell["model_alias"], cell["source"], cell["category"])
        values = groups.get(key, [])
        predictions[(cell["eval_id"], cell["model_alias"])] = (
            sum(values) + 2.0 * model_rates[cell["model_alias"]]
        ) / (len(values) + 2.0)
    return predictions


def _ece(points: list[tuple[float, float]], bins: int = 10) -> float:
    if not points:
        return 0.0
    total = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [(pred, actual) for pred, actual in points if low <= pred < high or (index == bins - 1 and pred == 1)]
        if selected:
            confidence = statistics.fmean(pred for pred, _ in selected)
            accuracy = statistics.fmean(actual for _, actual in selected)
            total += len(selected) / len(points) * abs(confidence - accuracy)
    return total


def _heuristic_alias(model_metadata: list[dict[str, Any]], source: str) -> str:
    catalog = {model.model_id: model for model in MODEL_CATALOG}
    family = {"bfcl": "analysis", "spreadsheetbench": "spreadsheet", "gaia": "research"}.get(source, "analysis")
    ranked = []
    for item in model_metadata:
        model = catalog.get(item["catalog_model_id"])
        capability = model.capability if model else 0.5
        overlap = 0.07 if model and family in model.strengths else 0.0
        ranked.append((capability + overlap, -float(item["input_per_million_usd"]), item["alias"]))
    return max(ranked)[2]


def _choose_baselines(
    task_id: str,
    task_cells: dict[str, dict[str, Any]],
    model_metadata: list[dict[str, Any]],
    predictions: dict[tuple[str, str], float],
) -> dict[str, str | None]:
    aliases = [item["alias"] for item in model_metadata]
    roles = {item["role"]: item["alias"] for item in model_metadata}
    max_prediction = max(predictions[(task_id, alias)] for alias in aliases)
    eligible = [alias for alias in aliases if predictions[(task_id, alias)] >= max_prediction - 0.03]
    predicted_cost = {alias: task_cells[alias]["cost_usd"] for alias in aliases}
    evidence = min(eligible, key=lambda alias: (predicted_cost[alias], alias)) if max_prediction >= 0.5 else None
    oracle = max(aliases, key=lambda alias: (task_cells[alias]["score"], -task_cells[alias]["cost_usd"]))
    random_alias = aliases[int(hashlib.sha256(f"42:{task_id}".encode()).hexdigest(), 16) % len(aliases)]
    source = next(iter(task_cells.values()))["source"]
    return {
        "cheapest_only": roles["low"],
        "premium_only": roles["premium"],
        "seeded_random": random_alias,
        "current_heuristic": _heuristic_alias(model_metadata, source),
        "evidence_router": evidence,
        "oracle": oracle,
    }


def _baseline_summary(selections: list[tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]]) -> dict[str, Any]:
    chosen = [cell for cell, _ in selections if cell is not None]
    passes = sum(cell["passed"] for cell in chosen)
    total_cost = sum(cell["cost_usd"] for cell in chosen)
    all_count = len(selections)
    severe = sum(cell["severe_failure"] for cell in chosen)
    regrets = []
    cost_regrets = []
    for cell, task_cells in selections:
        best_score = max(value["score"] for value in task_cells.values())
        regrets.append(best_score - (cell["score"] if cell else 0.0))
        passing_costs = [value["cost_usd"] for value in task_cells.values() if value["passed"]]
        if cell and cell["passed"] and passing_costs:
            cost_regrets.append(cell["cost_usd"] - min(passing_costs))
    return {
        "tasks": all_count,
        "coverage": len(chosen) / all_count if all_count else 0.0,
        "pass_rate": passes / all_count if all_count else 0.0,
        "severe_failures": severe,
        "cost_usd": total_cost,
        "cost_per_pass_usd": total_cost / passes if passes else None,
        "mean_quality_regret": statistics.fmean(regrets) if regrets else 0.0,
        "mean_cost_regret_usd": statistics.fmean(cost_regrets) if cost_regrets else None,
    }


def _bootstrap_gate(
    evidence: list[tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]],
    premium_alias: str,
    samples: int = 1000,
) -> dict[str, list[float] | None]:
    if not evidence:
        return {"cost_savings_95ci": None, "quality_gap_95ci": None}
    rng = random.Random(42)
    savings, gaps = [], []
    for _ in range(samples):
        sample = [evidence[rng.randrange(len(evidence))] for _ in evidence]
        chosen = [cell for cell, _ in sample if cell]
        premium = [task_cells[premium_alias] for _, task_cells in sample]
        chosen_passes = sum(cell["passed"] for cell in chosen)
        premium_passes = sum(cell["passed"] for cell in premium)
        chosen_cpp = sum(cell["cost_usd"] for cell in chosen) / chosen_passes if chosen_passes else math.inf
        premium_cpp = sum(cell["cost_usd"] for cell in premium) / premium_passes if premium_passes else math.inf
        if math.isfinite(chosen_cpp) and math.isfinite(premium_cpp) and premium_cpp > 0:
            savings.append((premium_cpp - chosen_cpp) / premium_cpp)
        gaps.append(premium_passes / len(sample) - chosen_passes / len(sample))
    return {
        "cost_savings_95ci": [_percentile(savings, 0.025), _percentile(savings, 0.975)] if savings else None,
        "quality_gap_95ci": [_percentile(gaps, 0.025), _percentile(gaps, 0.975)],
    }


def generate_report(run_dir: Path) -> dict[str, Any]:
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    results = _read_results(run_dir / "results.jsonl")
    cells = _aggregate_cells(results)
    model_metadata = metadata["models"]
    aliases = [item["alias"] for item in model_metadata]
    predictions = _fit_predictions(cells, aliases)

    expected_tasks = int(metadata["task_count"])
    per_model = {alias: _summary([cell for cell in cells.values() if cell["model_alias"] == alias], expected_tasks) for alias in aliases}
    sources = sorted({cell["source"] for cell in cells.values()})
    source_counts = metadata.get("source_counts") or {
        source: len({cell["eval_id"] for cell in cells.values() if cell["source"] == source}) for source in sources
    }
    per_source = {source: _summary([cell for cell in cells.values() if cell["source"] == source], int(source_counts[source]) * len(aliases)) for source in sources}
    holdout_ids = sorted({cell["eval_id"] for cell in cells.values() if cell["split"] == "holdout"})
    baseline_rows: dict[str, list[tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]]] = defaultdict(list)
    abstentions = []
    calibration_points = []
    budget = float(metadata["run_config"]["max_cost_per_task_usd"])
    budget_violations = 0
    source_router: dict[str, list[tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]]] = defaultdict(list)
    for task_id in holdout_ids:
        task_cells = {alias: cells[(task_id, alias)] for alias in aliases}
        choices = _choose_baselines(task_id, task_cells, model_metadata, predictions)
        for name, alias in choices.items():
            baseline_rows[name].append((task_cells[alias] if alias else None, task_cells))
        evidence_alias = choices["evidence_router"]
        evidence_cell = task_cells[evidence_alias] if evidence_alias else None
        source_router[next(iter(task_cells.values()))["source"]].append((evidence_cell, task_cells))
        if evidence_cell and evidence_cell["cost_usd"] > budget:
            budget_violations += 1
        if evidence_alias is None:
            abstentions.append(all(not value["passed"] for value in task_cells.values()))
        for alias in aliases:
            calibration_points.append((predictions[(task_id, alias)], float(task_cells[alias]["passed"])))

    baselines = {name: _baseline_summary(rows) for name, rows in baseline_rows.items()}
    evidence_rows = baseline_rows["evidence_router"]
    evidence = baselines["evidence_router"]
    premium = baselines["premium_only"]
    premium_alias = next(item["alias"] for item in model_metadata if item["role"] == "premium")
    intervals = _bootstrap_gate(evidence_rows, premium_alias)
    cost_savings = None
    if evidence["cost_per_pass_usd"] is not None and premium["cost_per_pass_usd"] not in {None, 0}:
        cost_savings = (premium["cost_per_pass_usd"] - evidence["cost_per_pass_usd"]) / premium["cost_per_pass_usd"]
    source_gaps = {}
    router_per_source = {}
    for source, rows in source_router.items():
        routed = _baseline_summary(rows)
        router_per_source[source] = routed
        premium_source = _baseline_summary([(task_cells[premium_alias], task_cells) for _, task_cells in rows])
        source_gaps[source] = premium_source["pass_rate"] - routed["pass_rate"]
    budget_rate = budget_violations / len(holdout_ids) if holdout_ids else 0.0
    promotion_checks = {
        "cost_savings_at_least_20_percent": cost_savings is not None and cost_savings >= 0.20,
        "overall_quality_noninferiority": premium["pass_rate"] - evidence["pass_rate"] <= 0.03,
        "per_source_quality_noninferiority": all(gap <= 0.05 for gap in source_gaps.values()),
        "no_severe_failure_increase": evidence["severe_failures"] <= premium["severe_failures"],
        "budget_violations_at_most_1_percent": budget_rate <= 0.01,
        "confidence_interval_supports_savings": bool(intervals["cost_savings_95ci"] and intervals["cost_savings_95ci"][0] > 0),
        "confidence_interval_supports_quality": bool(intervals["quality_gap_95ci"] and intervals["quality_gap_95ci"][1] <= 0.03),
    }
    promotion_passed = all(promotion_checks.values())
    report = {
        "run_id": metadata["run_id"],
        "profile": metadata["profile"],
        "status": metadata["status"],
        "experimental": not promotion_passed,
        "task_count": metadata["task_count"],
        "result_count": len(results),
        "overall": _summary(list(cells.values()), expected_tasks * len(aliases)),
        "per_model": per_model,
        "per_source": per_source,
        "baselines": baselines,
        "router": {
            "brier_score": statistics.fmean((prediction - actual) ** 2 for prediction, actual in calibration_points) if calibration_points else 0.0,
            "expected_calibration_error": _ece(calibration_points),
            "abstention_coverage": evidence["coverage"],
            "abstention_precision": statistics.fmean(abstentions) if abstentions else None,
            "abstention_count": len(abstentions),
            "budget_violation_rate": budget_rate,
            "repeated_trial_stability": statistics.fmean(
                cell["trial_stability"] for cell in cells.values() if cell["trials"] > 1
            ) if any(cell["trials"] > 1 for cell in cells.values()) else None,
            "source_quality_gaps": source_gaps,
            "per_source": router_per_source,
            "cost_savings_vs_premium": cost_savings,
            **intervals,
        },
        "promotion_gate": {"passed": promotion_passed, "checks": promotion_checks},
        "provenance": {
            "started_at": metadata["started_at"],
            "completed_at": metadata.get("completed_at"),
            "models": model_metadata,
            "run_config": metadata["run_config"],
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
