from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from app import create_app
from task_value_mvp.config import DEFAULT_DATA_PATH
from task_value_mvp.data import assign_splits, exact_split_counts, load_tasks
from task_value_mvp.modeling import save_artifacts, train_bundle


class MvpTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_tasks(DEFAULT_DATA_PATH)
        cls.bundle, cls.forecasts, cls.metrics = train_bundle(cls.rows)
        cls.temp = tempfile.TemporaryDirectory()
        cls.artifact_dir = Path(cls.temp.name)
        save_artifacts(cls.bundle, cls.forecasts, cls.metrics, cls.artifact_dir)
        cls.app = create_app(cls.artifact_dir)
        cls.app.config.update(TESTING=True)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_exact_split_sizes_and_disjoint_assignment(self) -> None:
        splits = assign_splits(self.rows)
        expected = exact_split_counts(len(self.rows))
        self.assertEqual({name: splits.count(name) for name in expected}, expected)
        self.assertEqual(sum(expected.values()), len(self.rows))
        self.assertEqual(expected, {"train": 154, "calibration": 22, "test": 44})

    def test_forecast_and_budget_invariants(self) -> None:
        self.assertEqual(len(self.forecasts), 220)
        catalog_ids = {item["model_id"] for item in self.metrics["model_catalog"]}
        for item in self.forecasts:
            forecast = item["economic_value_forecast"]
            policy = item["cost_policy"]
            self.assertTrue(0 <= forecast["p10_usd"] <= forecast["p50_usd"] <= forecast["p90_usd"])
            self.assertTrue(math.isclose(policy["max_cost_usd"], forecast["p50_usd"] * 0.10))
            self.assertTrue(math.isclose(policy["operational_budget_usd"], forecast["p10_usd"] * 0.10))
            selected = item["recommendation"]["selected"]
            if selected:
                self.assertIn(selected["model_id"], catalog_ids)
                self.assertTrue(selected["eligible"])
                self.assertLessEqual(selected["p90_execution_cost_usd"], policy["operational_budget_usd"])

    def test_model_metadata(self) -> None:
        metadata = self.metrics["metadata"]
        self.assertEqual(metadata["algorithm"], "GradientBoostingRegressor")
        self.assertEqual(metadata["split_counts"], {"train": 154, "calibration": 22, "test": 44})
        self.assertEqual(len(self.metrics["test_points"]), 44)
        self.assertGreater(len(self.metrics["feature_importance"]), 0)

    def test_http_routes(self) -> None:
        for path in ("/", "/health", "/api/tasks", "/api/model", "/api/export.json", "/api/export.csv", "/api/evals/runs"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            response.close()
        tasks = self.client.get("/api/tasks").get_json()
        self.assertEqual(len(tasks), 220)
        detail = self.client.get(f"/api/tasks/{tasks[0]['task_id']}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(self.client.get("/api/tasks/not-real").status_code, 404)
        self.assertEqual(self.client.get("/api/evals/runs").get_json(), [])
        self.assertEqual(self.client.get("/api/evals/runs/not-real").status_code, 404)
        run_dir = self.artifact_dir / "evals" / "fixture-run"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({"run_id": "fixture-run", "status": "running", "started_at": "2026-01-01T00:00:00Z"}), encoding="utf-8")
        (run_dir / "results.jsonl").write_text(json.dumps({"eval_id": "bfcl:1", "passed": True}) + "\n", encoding="utf-8")
        runs = self.client.get("/api/evals/runs").get_json()
        self.assertEqual(runs[0]["run_id"], "fixture-run")
        self.assertEqual(self.client.get("/api/evals/runs/fixture-run").status_code, 200)
        results = self.client.get("/api/evals/runs/fixture-run/results").get_json()
        self.assertEqual(results, [{"eval_id": "bfcl:1", "passed": True}])


if __name__ == "__main__":
    unittest.main()
