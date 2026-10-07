from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import (
    EMPLOYER_LOAD_FACTOR,
    FALLBACK_HOURLY_WAGE,
    OCCUPATION_WAGES,
    REALIZABILITY_FACTOR,
)


KEYWORD_FAMILIES = {
    "spreadsheet": ("spreadsheet", "excel", "workbook", "column", "tab", "csv", "formula"),
    "analysis": ("analy", "calculate", "compare", "evaluate", "forecast", "metric", "variance"),
    "writing": ("write", "draft", "memo", "report", "document", "summary", "presentation"),
    "coding": ("code", "python", "software", "script", "api", "database", "sql"),
    "research": ("research", "source", "citation", "market", "investigate", "benchmark"),
    "media": ("image", "video", "audio", "graphic", "design", "edit", "visual"),
    "compliance": ("audit", "compliance", "regulat", "legal", "risk", "policy", "financial"),
    "files": ("attached", "file", "pdf", "document", "workbook", "deliverable"),
}


@dataclass
class TaskSignals:
    values: dict[str, float]
    task_families: list[str]
    required_modalities: list[str]
    risk_tier: str
    estimated_human_hours: float
    hourly_wage: float
    proxy_value: float
    proxy_assumptions: dict[str, float]


def _count_terms(text: str, terms: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term) for term in terms)


def derive_task_signals(row: dict[str, Any]) -> TaskSignals:
    prompt = row["prompt"]
    lowered = prompt.lower()
    words = re.findall(r"\b\w+\b", prompt)
    sentences = [part for part in re.split(r"[.!?]+", prompt) if part.strip()]
    numbered = re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+", prompt)
    family_counts = {name: _count_terms(prompt, terms) for name, terms in KEYWORD_FAMILIES.items()}
    task_families = [name for name, count in family_counts.items() if count > 0]

    modalities = ["text"]
    if family_counts["spreadsheet"]:
        modalities.append("spreadsheet")
    if any(term in lowered for term in ("image", "video", "graphic", "screenshot")):
        modalities.append("vision")
    if "audio" in lowered:
        modalities.append("audio")

    compliance_score = family_counts["compliance"]
    if compliance_score >= 5 or any(term in lowered for term in ("regulatory filing", "legal opinion", "patient")):
        risk_tier = "regulated"
        risk_multiplier = 1.25
    elif compliance_score >= 2:
        risk_tier = "consequential"
        risk_multiplier = 1.12
    else:
        risk_tier = "ordinary"
        risk_multiplier = 1.0

    word_count = len(words)
    requirement_count = len(numbered)
    file_mentions = family_counts["files"]
    complexity_points = (
        1.0
        + min(word_count / 900.0, 2.5)
        + min(requirement_count * 0.22, 2.0)
        + min(family_counts["analysis"] * 0.09, 1.5)
        + min(family_counts["spreadsheet"] * 0.11, 1.5)
        + min(file_mentions * 0.08, 1.0)
        + (0.45 if len(set(task_families)) >= 4 else 0.0)
    )
    estimated_hours = max(0.5, min(24.0, 0.55 + complexity_points * 0.82))
    hourly_wage = OCCUPATION_WAGES.get(row["occupation"], FALLBACK_HOURLY_WAGE)
    proxy_value = (
        estimated_hours
        * hourly_wage
        * EMPLOYER_LOAD_FACTOR
        * REALIZABILITY_FACTOR
        * risk_multiplier
    )

    estimated_input_tokens = max(128, int(len(prompt) / 4.0) + file_mentions * 550)
    estimated_output_tokens = max(
        256,
        int(180 + requirement_count * 115 + family_counts["writing"] * 80 + family_counts["analysis"] * 35),
    )
    values = {
        "word_count": float(word_count),
        "character_count": float(len(prompt)),
        "sentence_count": float(len(sentences)),
        "line_count": float(prompt.count("\n") + 1),
        "requirement_count": float(requirement_count),
        "family_diversity": float(len(set(task_families))),
        "estimated_input_tokens": float(estimated_input_tokens),
        "estimated_output_tokens": float(estimated_output_tokens),
        "is_consequential": float(risk_tier in {"consequential", "regulated"}),
        "is_regulated": float(risk_tier == "regulated"),
    }
    values.update({f"kw_{name}": float(count) for name, count in family_counts.items()})
    return TaskSignals(
        values=values,
        task_families=task_families or ["general"],
        required_modalities=modalities,
        risk_tier=risk_tier,
        estimated_human_hours=estimated_hours,
        hourly_wage=hourly_wage,
        proxy_value=proxy_value,
        proxy_assumptions={
            "estimated_human_hours": estimated_hours,
            "hourly_wage_usd": hourly_wage,
            "employer_load_factor": EMPLOYER_LOAD_FACTOR,
            "realizability_factor": REALIZABILITY_FACTOR,
            "risk_multiplier": risk_multiplier,
        },
    )


class DenseTaskFeatureEncoder:
    def __init__(self) -> None:
        self.sectors: list[str] = []
        self.occupations: list[str] = []
        self.numeric_names: list[str] = []
        self.feature_names: list[str] = []

    def fit(self, rows: list[dict[str, Any]]) -> "DenseTaskFeatureEncoder":
        self.sectors = sorted({row["sector"] for row in rows})
        self.occupations = sorted({row["occupation"] for row in rows})
        first = derive_task_signals(rows[0])
        self.numeric_names = sorted(first.values)
        self.feature_names = (
            self.numeric_names
            + [f"sector={name}" for name in self.sectors]
            + [f"occupation={name}" for name in self.occupations]
        )
        return self

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not self.feature_names:
            raise RuntimeError("Encoder must be fit before transform")
        matrix = np.zeros((len(rows), len(self.feature_names)), dtype=float)
        sector_offset = len(self.numeric_names)
        occupation_offset = sector_offset + len(self.sectors)
        sector_index = {name: i for i, name in enumerate(self.sectors)}
        occupation_index = {name: i for i, name in enumerate(self.occupations)}
        for row_index, row in enumerate(rows):
            signals = derive_task_signals(row)
            for col_index, name in enumerate(self.numeric_names):
                matrix[row_index, col_index] = signals.values[name]
            if row["sector"] in sector_index:
                matrix[row_index, sector_offset + sector_index[row["sector"]]] = 1.0
            if row["occupation"] in occupation_index:
                matrix[row_index, occupation_offset + occupation_index[row["occupation"]]] = 1.0
        return matrix

    def fit_transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return self.fit(rows).transform(rows)


def proxy_targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([derive_task_signals(row).proxy_value for row in rows], dtype=float)


def summarize_categories(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in Counter(row[field] for row in rows).most_common()]
