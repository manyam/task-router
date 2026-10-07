from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import CALIBRATION_FRACTION, RANDOM_SEED, TEST_FRACTION, TRAIN_FRACTION


REQUIRED_COLUMNS = {"task_id", "sector", "occupation", "prompt"}


def load_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"GDPVal CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        rows = []
        for index, row in enumerate(reader):
            if not all((row.get(field) or "").strip() for field in REQUIRED_COLUMNS):
                raise ValueError(f"Row {index + 2} has blank required fields")
            rows.append(
                {
                    "task_id": row["task_id"].strip(),
                    "sector": row["sector"].strip(),
                    "occupation": row["occupation"].strip(),
                    "prompt": row["prompt"].strip(),
                }
            )
    if not rows:
        raise ValueError("CSV contains no task rows")
    return rows


def exact_split_counts(n_rows: int) -> dict[str, int]:
    train = int(math.floor(n_rows * TRAIN_FRACTION))
    calibration = int(math.floor(n_rows * CALIBRATION_FRACTION))
    test = n_rows - train - calibration
    return {"train": train, "calibration": calibration, "test": test}


def assign_splits(rows: list[dict[str, Any]], seed: int = RANDOM_SEED) -> list[str]:
    """Create exact global 70/10/20 splits with approximate sector stratification."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["sector"]].append(index)
    for indices in groups.values():
        rng.shuffle(indices)

    targets = exact_split_counts(len(rows))
    assignments: list[str | None] = [None] * len(rows)
    allocated = {name: 0 for name in targets}

    # Allocate every sector proportionally, then repair to exact global counts.
    for sector in sorted(groups):
        indices = groups[sector]
        n = len(indices)
        local_train = int(math.floor(n * TRAIN_FRACTION))
        local_cal = int(math.floor(n * CALIBRATION_FRACTION))
        local = ["train"] * local_train + ["calibration"] * local_cal
        local += ["test"] * (n - len(local))
        for index, split in zip(indices, local):
            assignments[index] = split
            allocated[split] += 1

    # Deterministic repairs move rows from surplus splits to deficits while
    # spreading moves across sectors.
    while allocated != targets:
        deficits = [name for name in targets if allocated[name] < targets[name]]
        surpluses = [name for name in targets if allocated[name] > targets[name]]
        if not deficits or not surpluses:
            break
        destination = max(deficits, key=lambda name: targets[name] - allocated[name])
        source = max(surpluses, key=lambda name: allocated[name] - targets[name])
        candidates = [i for i, split in enumerate(assignments) if split == source]
        rng.shuffle(candidates)
        chosen = candidates[0]
        assignments[chosen] = destination
        allocated[source] -= 1
        allocated[destination] += 1

    result = [str(split) for split in assignments]
    if {name: result.count(name) for name in targets} != targets:
        raise AssertionError("Failed to produce exact split sizes")
    return result

