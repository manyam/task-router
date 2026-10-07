from __future__ import annotations

import argparse
import json

from task_value_mvp.config import ARTIFACT_DIR, DEFAULT_DATA_PATH
from task_value_mvp.data import load_tasks
from task_value_mvp.modeling import save_artifacts, train_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GDPVal economic-value MVP")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--artifacts", type=str, default=str(ARTIFACT_DIR))
    args = parser.parse_args()

    from pathlib import Path

    rows = load_tasks(Path(args.data))
    bundle, forecasts, metrics = train_bundle(rows)
    save_artifacts(bundle, forecasts, metrics, Path(args.artifacts))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "artifacts": str(Path(args.artifacts).resolve()),
                "split_counts": metrics["metadata"]["split_counts"],
                "test_metrics": metrics["metadata"]["test_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

