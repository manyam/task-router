from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class EvalTask:
    eval_id: str
    source: str
    source_revision: str
    source_task_id: str
    prompt: str
    category: str
    capability_tags: list[str]
    grader: str
    split: str
    access_policy: str
    modalities: list[str] = field(default_factory=lambda: ["text"])
    assets: list[str] = field(default_factory=list)
    grader_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in {"calibration", "holdout"}:
            raise ValueError(f"Invalid eval split: {self.split}")
        if not self.eval_id or not self.source_task_id or not self.prompt:
            raise ValueError("Eval task identity and prompt must be non-empty")

    def to_dict(self, include_private: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_private:
            value.pop("grader_payload", None)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalTask":
        return cls(**value)


@dataclass(frozen=True)
class GradeResult:
    native_score: float
    normalized_score: float
    passed: bool
    severe_failure: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.normalized_score) <= 1.0:
            raise ValueError("Normalized score must be in [0, 1]")


@dataclass(frozen=True)
class ModelConfig:
    alias: str
    role: str
    catalog_model_id: str
    api_model: str
    base_url: str
    api_key_env: str
    input_per_million_usd: float
    output_per_million_usd: float
    cached_input_per_million_usd: float | None = None

    def __post_init__(self) -> None:
        if self.role not in {"low", "middle", "premium"}:
            raise ValueError(f"Invalid model role: {self.role}")
        if self.input_per_million_usd < 0 or self.output_per_million_usd < 0 or (
            self.cached_input_per_million_usd is not None and self.cached_input_per_million_usd < 0
        ):
            raise ValueError("Model prices must be non-negative")
        endpoint = urlparse(self.base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
            raise ValueError("Model base URL must be HTTP or HTTPS")
        if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError("Model base URL cannot contain credentials, query parameters, or fragments")


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    eval_id: str
    source: str
    category: str
    split: str
    model_alias: str
    catalog_model_id: str
    trial: int
    native_score: float
    normalized_score: float
    passed: bool
    severe_failure: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    artifact_paths: list[str]
    error: str | None
    grader_details: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalResult":
        return cls(**value)
