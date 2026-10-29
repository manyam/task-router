from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import openpyxl

from task_value_mvp.evals.graders import grade_bfcl, grade_gaia, grade_spreadsheet
from task_value_mvp.evals.manifest import load_manifest, validate_tasks
from task_value_mvp.evals.reporting import generate_report
from task_value_mvp.evals.runner import OpenAICompatibleClient, load_models_config
from task_value_mvp.evals.schema import EvalResult, EvalTask, ModelConfig


def task(**overrides):
    values = {
        "eval_id": "fixture:1",
        "source": "bfcl",
        "source_revision": "abc123",
        "source_task_id": "1",
        "prompt": "Use the weather function for Boston.",
        "category": "simple",
        "capability_tags": ["tool_calling"],
        "grader": "fixture",
        "split": "calibration",
        "access_policy": "public",
    }
    values.update(overrides)
    return EvalTask(**values)


class ManifestTests(unittest.TestCase):
    def test_manifest_counts_and_locked_splits(self) -> None:
        manifest = load_manifest()
        expected = {"bfcl": (64, 16), "spreadsheetbench": (24, 6), "gaia": (12, 3)}
        for source, (selected_count, holdout_count) in expected.items():
            source_manifest = manifest["sources"][source]
            strata = source_manifest["strata"].values()
            selected = [item for value in strata for item in value["selected_ids"]]
            holdout = [item for value in strata for item in value["holdout_ids"]]
            self.assertEqual(len(selected), selected_count)
            self.assertEqual(len(set(selected)), selected_count)
            self.assertEqual(len(holdout), holdout_count)
            self.assertTrue(set(holdout) <= set(selected))
            for stratum, value in source_manifest["strata"].items():
                hashes = [
                    hashlib.sha256(
                        source_manifest["selection_key"].format(stratum=stratum, source_task_id=source_id).encode()
                    ).hexdigest()
                    for source_id in value["selected_ids"]
                ]
                self.assertEqual(hashes, sorted(hashes))
        self.assertEqual(manifest["selection_seed"], 42)
        self.assertIn("SHA-256", manifest["selection_method"])
        self.assertEqual(manifest["profiles"]["local-core"], ["bfcl", "spreadsheetbench"])
        for source in expected:
            self.assertTrue(manifest["sources"][source]["file_sha256"])

    def test_duplicate_and_near_duplicate_prompts_are_rejected(self) -> None:
        first = task()
        duplicate = task(eval_id="fixture:2", source_task_id="2")
        with self.assertRaisesRegex(ValueError, "duplicate prompts"):
            validate_tasks([first, duplicate], 0.92)


class GraderTests(unittest.TestCase):
    def test_bfcl_good_wrong_and_irrelevant_calls(self) -> None:
        fixture = task(grader_payload={"expected_calls": [{"weather.get": {"city": ["Boston"]}}]})
        good = grade_bfcl(fixture, [{"name": "weather_get", "arguments": '{"city":"Boston"}'}])
        self.assertTrue(good.passed)
        self.assertFalse(grade_bfcl(fixture, [{"name": "weather_get", "arguments": "{}"}]).passed)
        self.assertFalse(grade_bfcl(fixture, [{"name": "weather_get", "arguments": "not-json"}]).passed)
        irrelevant = task(category="irrelevance", grader_payload={"expected_calls": []})
        self.assertTrue(grade_bfcl(irrelevant, []).passed)
        self.assertFalse(grade_bfcl(irrelevant, [{"name": "weather_get", "arguments": "{}"}]).passed)

    def test_spreadsheet_good_wrong_and_corrupt_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            answer = root / "answer.xlsx"
            good = root / "good.xlsx"
            wrong = root / "wrong.xlsx"
            corrupt = root / "corrupt.xlsx"
            for path, value in ((answer, 42), (good, 42), (wrong, 41)):
                workbook = openpyxl.Workbook()
                workbook.active.title = "Unrelated"
                workbook.create_sheet("Results")
                workbook["Results"]["B2"] = value
                workbook.save(path)
            corrupt.write_text("not an xlsx", encoding="utf-8")
            fixture = task(
                source="spreadsheetbench",
                category="Cell-Level",
                grader_payload={"answer_files": [str(answer)], "answer_position": "Results!B2", "answer_sheet": "Results"},
            )
            self.assertTrue(grade_spreadsheet(fixture, [good]).passed)
            self.assertFalse(grade_spreadsheet(fixture, [wrong]).passed)
            corrupt_result = grade_spreadsheet(fixture, [corrupt])
            self.assertFalse(corrupt_result.passed)
            self.assertTrue(corrupt_result.severe_failure)

    def test_gaia_normalized_exact_answer(self) -> None:
        fixture = task(source="gaia", category="level_1", grader_payload={"final_answer": "12,345"})
        self.assertTrue(grade_gaia(fixture, "Final answer: 12,345.").passed)
        self.assertFalse(grade_gaia(fixture, "12,346").passed)


class _FakeApiResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class RunnerClientTests(unittest.TestCase):
    def test_example_config_has_exactly_three_catalog_models_and_env_credentials(self) -> None:
        credentials = {
            "TASK_ROUTER_LOW_API_KEY": "low-secret",
            "TASK_ROUTER_MIDDLE_API_KEY": "middle-secret",
            "TASK_ROUTER_PREMIUM_API_KEY": "premium-secret",
        }
        with patch.dict(os.environ, credentials):
            config, models = load_models_config(Path(__file__).parents[1] / "eval_models.example.yaml")
        self.assertEqual(len(models), 3)
        self.assertEqual({model.role for model in models}, {"low", "middle", "premium"})
        self.assertEqual(config.max_tool_output_chars, 20_000)

    def test_openai_client_retries_rate_limit_and_parses_usage(self) -> None:
        body = {
            "choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "weather_get", "arguments": '{"city":"Boston"}'}}]}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 3}},
        }
        rate_limit = HTTPError("http://fake/chat/completions", 429, "rate limit", {}, None)
        model = ModelConfig("cheap", "low", "catalog-id", "api-id", "http://fake", "FIXTURE_KEY", 1.0, 2.0)
        with patch("urllib.request.urlopen", side_effect=[rate_limit, _FakeApiResponse(body)]) as urlopen:
            with patch("time.sleep"), patch.dict(os.environ, {"FIXTURE_KEY": "secret"}):
                response = OpenAICompatibleClient(model, 2).complete([{"role": "user", "content": "hello"}], None, 128)
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(response.tool_calls[0]["name"], "weather_get")
        self.assertEqual((response.input_tokens, response.output_tokens, response.cached_input_tokens), (17, 5, 3))
        payload = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 42)

    def test_openai_client_accepts_missing_usage_fields(self) -> None:
        body = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        model = ModelConfig("cheap", "low", "catalog-id", "api-id", "http://fake", "FIXTURE_KEY", 1.0, 2.0)
        with patch("urllib.request.urlopen", return_value=_FakeApiResponse(body)), patch.dict(os.environ, {"FIXTURE_KEY": "secret"}):
            response = OpenAICompatibleClient(model, 2).complete([{"role": "user", "content": "hello"}], None, 128)
        self.assertEqual((response.input_tokens, response.output_tokens, response.cached_input_tokens), (0, 0, 0))


class ReportingTests(unittest.TestCase):
    def test_synthetic_matrix_generates_all_baselines_and_gate(self) -> None:
        models = [
            {"alias": "low", "role": "low", "catalog_model_id": "gpt-4.1-nano", "api_model": "low-api", "base_url": "http://local", "input_per_million_usd": 0.1, "output_per_million_usd": 0.4},
            {"alias": "mid", "role": "middle", "catalog_model_id": "gpt-4.1-mini", "api_model": "mid-api", "base_url": "http://local", "input_per_million_usd": 0.4, "output_per_million_usd": 1.6},
            {"alias": "premium", "role": "premium", "catalog_model_id": "gpt-4.1", "api_model": "premium-api", "base_url": "http://local", "input_per_million_usd": 2.0, "output_per_million_usd": 8.0},
        ]
        rows = []
        specs = [
            ("bfcl:cal", "bfcl", "simple", "calibration", {"low": True, "mid": True, "premium": True}),
            ("sheet:cal", "spreadsheetbench", "Cell-Level", "calibration", {"low": False, "mid": True, "premium": True}),
            ("bfcl:hold", "bfcl", "simple", "holdout", {"low": True, "mid": True, "premium": True}),
            ("sheet:hold", "spreadsheetbench", "Cell-Level", "holdout", {"low": False, "mid": True, "premium": True}),
        ]
        costs = {"low": 0.01, "mid": 0.03, "premium": 0.12}
        for eval_id, source, category, split, passes in specs:
            for model in models:
                passed = passes[model["alias"]]
                rows.append(EvalResult(
                    run_id="fixture-run", eval_id=eval_id, source=source, category=category, split=split,
                    model_alias=model["alias"], catalog_model_id=model["catalog_model_id"], trial=0,
                    native_score=float(passed), normalized_score=float(passed), passed=passed,
                    severe_failure=False, latency_ms=100, input_tokens=10, output_tokens=2,
                    cached_input_tokens=0, cost_usd=costs[model["alias"]], artifact_paths=[], error=None,
                    grader_details={}, provenance={"capability_tags": [source]},
                ))
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            metadata = {
                "run_id": "fixture-run", "profile": "local-core", "status": "complete",
                "started_at": "2026-01-01T00:00:00+00:00", "completed_at": "2026-01-01T00:01:00+00:00",
                "task_count": 4, "models": models, "run_config": {"max_cost_per_task_usd": 1.0},
            }
            (run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
            (run_dir / "results.jsonl").write_text("\n".join(json.dumps(row.to_dict()) for row in rows) + "\n", encoding="utf-8")
            report = generate_report(run_dir)
            self.assertTrue((run_dir / "report.json").exists())
        self.assertEqual(set(report["baselines"]), {"cheapest_only", "premium_only", "seeded_random", "current_heuristic", "evidence_router", "oracle"})
        self.assertEqual(set(report["per_source"]), {"bfcl", "spreadsheetbench"})
        self.assertIn("brier_score", report["router"])
        self.assertIn("p95_token_usage", report["overall"])
        self.assertEqual(report["overall"]["coverage"], 1.0)
        self.assertIn("confidence_interval_supports_quality", report["promotion_gate"]["checks"])


if __name__ == "__main__":
    unittest.main()
