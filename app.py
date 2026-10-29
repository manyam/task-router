from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_file

from task_value_mvp.config import ARTIFACT_DIR
from task_value_mvp.evals.store import EvalRunStore
from task_value_mvp.service import ForecastStore


def create_app(artifact_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    resolved_artifacts = Path(artifact_dir or ARTIFACT_DIR)
    store = ForecastStore(resolved_artifacts)
    eval_store = EvalRunStore(resolved_artifacts)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "task_count": len(store.forecasts),
                "trained_at": store.metrics["metadata"]["trained_at"],
                "target": store.metrics["metadata"]["target"],
            }
        )

    @app.get("/api/tasks")
    def tasks():
        return jsonify(store.compact_tasks())

    @app.get("/api/tasks/<task_id>")
    def task_detail(task_id: str):
        item = store.by_id.get(task_id)
        if item is None:
            abort(404)
        return jsonify(item)

    @app.get("/api/model")
    def model_data():
        return jsonify(store.metrics)

    @app.get("/api/export.json")
    def export_json():
        return send_file(resolved_artifacts / "forecasts.json", as_attachment=True)

    @app.get("/api/export.csv")
    def export_csv():
        return send_file(resolved_artifacts / "forecasts.csv", as_attachment=True)

    @app.get("/api/evals/runs")
    def eval_runs():
        return jsonify(eval_store.list_runs())

    @app.get("/api/evals/runs/<run_id>")
    def eval_run(run_id: str):
        value = eval_store.get_run(run_id)
        if value is None:
            abort(404)
        return jsonify(value)

    @app.get("/api/evals/runs/<run_id>/results")
    def eval_results(run_id: str):
        value = eval_store.get_results(run_id)
        if value is None:
            abort(404)
        return jsonify(value)

    return app


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
