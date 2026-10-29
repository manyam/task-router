from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_value_mvp.config import ARTIFACT_DIR

from .manifest import DEFAULT_CACHE_DIR, build_profile
from .reporting import generate_report
from .runner import run_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, run, and report Task Router evaluations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Download and normalize a pinned eval profile")
    build_parser.add_argument("--profile", choices=("local-core", "local-plus-gaia"), default="local-core")
    build_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)

    run_parser = subparsers.add_parser("run", help="Run three configured models over a built profile")
    run_parser.add_argument("--profile", choices=("local-core", "local-plus-gaia"), default="local-core")
    run_parser.add_argument("--models-config", type=Path, required=True)
    run_parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    run_parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR / "evals")

    report_parser = subparsers.add_parser("report", help="Generate metrics for a completed eval run")
    report_parser.add_argument("--run-id", required=True)
    report_parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR / "evals")

    args = parser.parse_args()
    if args.command == "build":
        path, metadata = build_profile(args.profile, args.cache)
        print(json.dumps({**metadata, "tasks_path": str(path.resolve())}, indent=2))
    elif args.command == "run":
        run_dir = run_profile(args.profile, args.models_config, args.cache, args.artifacts)
        report = generate_report(run_dir)
        print(json.dumps({"run_id": report["run_id"], "run_dir": str(run_dir.resolve()), "promotion_gate": report["promotion_gate"]}, indent=2))
    else:
        report = generate_report(args.artifacts / args.run_id)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
