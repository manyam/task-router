from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class EvalRunStore:
    def __init__(self, artifact_dir: Path) -> None:
        self.root = artifact_dir / "evals"

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        runs = []
        for path in self.root.iterdir():
            metadata = path / "run.json"
            if not path.is_dir() or not metadata.exists():
                continue
            value = json.loads(metadata.read_text(encoding="utf-8"))
            report_path = path / "report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                value["promotion_gate"] = report.get("promotion_gate")
            runs.append(value)
        return sorted(runs, key=lambda item: item.get("started_at", ""), reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) or run_id in {".", ".."}:
            return None
        path = self.root / run_id
        if not path.is_dir() or path.parent != self.root:
            return None
        metadata_path = path / "run.json"
        if not metadata_path.exists():
            return None
        value = {"metadata": json.loads(metadata_path.read_text(encoding="utf-8"))}
        report_path = path / "report.json"
        value["report"] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        return value

    def get_results(self, run_id: str) -> list[dict[str, Any]] | None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) or run_id in {".", ".."}:
            return None
        path = self.root / run_id / "results.jsonl"
        if not path.exists() or path.parent.parent != self.root:
            return None
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
