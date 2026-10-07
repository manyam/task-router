from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


CATALOG_SNAPSHOT_DATE = "2026-08-05"
CATALOG_SOURCE_URL = "https://pricepertoken.com/"


@dataclass(frozen=True)
class CatalogModel:
    model_id: str
    display_name: str
    provider: str
    input_per_million_usd: float
    output_per_million_usd: float
    cached_input_per_million_usd: float | None
    context_tokens: int
    capability: float
    modalities: tuple[str, ...] = ("text",)
    strengths: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["modalities"] = list(self.modalities)
        value["strengths"] = list(self.strengths)
        return value


# Curated decision set from PricePerToken's provider-neutral table. The source
# lists 300+ models; this subset removes legacy and near-duplicate entries while
# retaining a spread of providers, prices, and quality tiers suitable for the
# GDPVal MVP. Capability remains an MVP heuristic informed by the source's
# benchmark fields; it is not a substitute for GDPVal execution results.
MODEL_CATALOG = (
    CatalogModel(
        "deepseek/deepseek-v4-flash",
        "DeepSeek V4 Flash",
        "DeepSeek",
        0.088,
        0.176,
        0.0028,
        1_000_000,
        0.76,
        strengths=("analysis", "research"),
    ),
    CatalogModel(
        "openai/gpt-5-mini",
        "GPT-5 Mini",
        "OpenAI",
        0.125,
        1.00,
        0.0125,
        400_000,
        0.79,
        modalities=("text", "vision"),
        strengths=("analysis", "writing", "files"),
    ),
    CatalogModel(
        "openai/gpt-5.1-codex-mini",
        "GPT-5.1 Codex Mini",
        "OpenAI",
        0.125,
        1.00,
        0.03,
        400_000,
        0.84,
        strengths=("coding", "analysis", "spreadsheet"),
    ),
    CatalogModel(
        "google/gemini-3.5-flash-lite",
        "Gemini 3.5 Flash Lite",
        "Google",
        0.15,
        1.25,
        0.015,
        1_000_000,
        0.83,
        modalities=("text", "vision"),
        strengths=("writing", "media", "files", "analysis"),
    ),
    CatalogModel(
        "moonshot/kimi-k2.7-code",
        "Kimi K2.7 Code",
        "Moonshot AI",
        0.73,
        3.50,
        0.15,
        262_000,
        0.90,
        modalities=("text", "vision"),
        strengths=("analysis", "spreadsheet", "coding"),
    ),
    CatalogModel(
        "google/gemini-3.6-flash",
        "Gemini 3.6 Flash",
        "Google",
        0.75,
        3.75,
        0.075,
        1_000_000,
        0.93,
        modalities=("text", "vision"),
        strengths=("analysis", "research", "writing", "media", "files"),
    ),
    CatalogModel(
        "anthropic/claude-3.5-haiku",
        "Claude 3.5 Haiku",
        "Anthropic",
        0.80,
        4.00,
        None,
        200_000,
        0.78,
        modalities=("text", "vision"),
        strengths=("writing", "analysis", "files"),
    ),
    CatalogModel(
        "openai/gpt-5.3-codex",
        "GPT-5.3 Codex",
        "OpenAI",
        1.75,
        14.00,
        0.175,
        400_000,
        0.94,
        strengths=("coding", "analysis", "spreadsheet", "files"),
    ),
    CatalogModel(
        "anthropic/claude-sonnet-5",
        "Claude Sonnet 5",
        "Anthropic",
        2.00,
        10.00,
        0.20,
        1_000_000,
        0.91,
        modalities=("text", "vision"),
        strengths=("analysis", "writing", "coding", "files", "compliance"),
    ),
    CatalogModel(
        "google/gemini-3.1-pro-preview",
        "Gemini 3.1 Pro Preview",
        "Google",
        2.00,
        12.00,
        0.20,
        1_000_000,
        0.96,
        modalities=("text", "vision"),
        strengths=("analysis", "research", "writing", "media", "files", "compliance"),
    ),
    CatalogModel(
        "openai/gpt-5.6-sol",
        "GPT-5.6 Sol",
        "OpenAI",
        2.50,
        15.00,
        0.25,
        1_100_000,
        0.92,
        modalities=("text", "vision"),
        strengths=("analysis", "research", "writing", "coding", "files", "compliance"),
    ),
    CatalogModel(
        "anthropic/claude-sonnet-4.6",
        "Claude Sonnet 4.6",
        "Anthropic",
        3.00,
        15.00,
        0.30,
        1_000_000,
        0.90,
        modalities=("text", "vision"),
        strengths=("analysis", "writing", "coding", "files", "compliance"),
    ),
    CatalogModel(
        "anthropic/claude-opus-5",
        "Claude Opus 5",
        "Anthropic",
        5.00,
        25.00,
        0.50,
        1_000_000,
        0.97,
        modalities=("text", "vision"),
        strengths=("analysis", "research", "compliance"),
    ),
)


QUALITY_FLOORS = {"ordinary": 0.55, "consequential": 0.64, "regulated": 0.72}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def evaluate_model(
    model: CatalogModel,
    task_signals: Any,
    value_p50: float,
    operational_budget: float,
) -> dict[str, Any]:
    required = set(task_signals.required_modalities) - {"spreadsheet", "audio"}
    supported = set(model.modalities)
    rejection_reasons: list[str] = []
    if not required.issubset(supported):
        rejection_reasons.append("required modality is unsupported")

    input_tokens = int(task_signals.values["estimated_input_tokens"])
    output_tokens = int(task_signals.values["estimated_output_tokens"])
    if input_tokens + output_tokens > model.context_tokens:
        rejection_reasons.append("estimated tokens exceed context window")

    base_cost = (
        input_tokens * model.input_per_million_usd / 1_000_000
        + output_tokens * model.output_per_million_usd / 1_000_000
    )
    complexity = min(
        0.95,
        0.28
        + task_signals.values["word_count"] / 5000
        + task_signals.values["requirement_count"] * 0.025
        + task_signals.values["family_diversity"] * 0.025,
    )
    strength_overlap = len(set(model.strengths) & set(task_signals.task_families))
    risk_penalty = {"ordinary": 0.0, "consequential": 0.10, "regulated": 0.18}[task_signals.risk_tier]
    raw_score = (model.capability - complexity - risk_penalty + strength_overlap * 0.07) * 5.0
    success = _sigmoid(raw_score)

    retry_multiplier = 1.0 + max(0.0, 0.86 - success) * 1.8
    validation_cost = base_cost * (0.15 if task_signals.risk_tier != "ordinary" else 0.05)
    expected_cost = base_cost * retry_multiplier + validation_cost
    p90_cost = expected_cost * 1.65
    quality_floor = QUALITY_FLOORS[task_signals.risk_tier]
    if p90_cost > operational_budget:
        rejection_reasons.append("P90 execution cost exceeds operational budget")
    if success < quality_floor:
        rejection_reasons.append("expected success is below task quality floor")

    return {
        "model_id": model.model_id,
        "display_name": model.display_name,
        "provider": model.provider,
        "expected_success_probability": success,
        "expected_execution_cost_usd": expected_cost,
        "p90_execution_cost_usd": p90_cost,
        "expected_net_value_usd": success * value_p50 - expected_cost,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "quality_floor": quality_floor,
        "eligible": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
    }


def recommend_model(task_signals: Any, value_p50: float, operational_budget: float) -> dict[str, Any]:
    candidates = [evaluate_model(model, task_signals, value_p50, operational_budget) for model in MODEL_CATALOG]
    eligible = sorted(
        (candidate for candidate in candidates if candidate["eligible"]),
        key=lambda item: (item["expected_net_value_usd"], item["expected_success_probability"]),
        reverse=True,
    )
    if not eligible:
        return {
            "status": "ABSTAIN",
            "reason": "No catalog model meets modality, context, quality, and conservative budget constraints.",
            "selected": None,
            "alternatives": [],
            "all_candidates": candidates,
        }
    return {
        "status": "RECOMMEND",
        "reason": "Highest heuristic risk-adjusted net value among eligible candidates.",
        "selected": eligible[0],
        "alternatives": eligible[1:3],
        "all_candidates": candidates,
    }


def public_catalog() -> list[dict[str, Any]]:
    return [model.public_dict() for model in MODEL_CATALOG]
