from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = Path(
    os.environ.get(
        "GDPVAL_DATA_PATH",
        "/Users/manyam2025/src/hf-evals/gdpval/output_data.csv",
    )
)
ARTIFACT_DIR = Path(os.environ.get("TASK_VALUE_ARTIFACT_DIR", BASE_DIR / "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
FORECASTS_PATH = ARTIFACT_DIR / "forecasts.json"
FORECASTS_CSV_PATH = ARTIFACT_DIR / "forecasts.csv"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.10
TEST_FRACTION = 0.20
INTERVAL_COVERAGE = 0.80

EMPLOYER_LOAD_FACTOR = 1.35
REALIZABILITY_FACTOR = 0.65
FALLBACK_HOURLY_WAGE = 46.00

# Illustrative US hourly wage assumptions for the occupations in GDPVal.
# These are configuration inputs, not observed values from the task dataset.
OCCUPATION_WAGES = {
    "Accountants and Auditors": 42.50,
    "Administrative Services Managers": 55.00,
    "Audio and Video Technicians": 29.00,
    "Buyers and Purchasing Agents": 38.00,
    "Child, Family, and School Social Workers": 29.50,
    "Compliance Officers": 40.00,
    "Computer and Information Systems Managers": 84.00,
    "Concierges": 19.00,
    "Counter and Rental Clerks": 18.50,
    "Customer Service Representatives": 20.00,
    "Editors": 39.00,
    "Film and Video Editors": 39.50,
    "Financial and Investment Analysts": 51.00,
    "Financial Managers": 78.00,
    "First-Line Supervisors of Non-Retail Sales Workers": 48.00,
    "General and Operations Managers": 61.00,
    "Graphic Designers": 31.00,
    "Human Resources Specialists": 36.00,
    "Insurance Underwriters": 39.00,
    "Lawyers": 78.00,
    "Logisticians": 39.00,
    "Management Analysts": 50.00,
    "Market Research Analysts and Marketing Specialists": 38.00,
    "Meeting, Convention, and Event Planners": 30.00,
    "Paralegals and Legal Assistants": 31.00,
    "Personal Financial Advisors": 50.00,
    "Project Management Specialists": 48.00,
    "Property, Real Estate, and Community Association Managers": 37.00,
    "Public Relations Specialists": 36.00,
    "Purchasing Managers": 65.00,
    "Real Estate Brokers": 47.00,
    "Sales Managers": 68.00,
    "Sales Representatives of Services": 38.00,
    "Social and Community Service Managers": 40.00,
    "Software Developers": 65.00,
    "Training and Development Specialists": 34.00,
    "Transportation, Storage, and Distribution Managers": 53.00,
    "Wholesale and Manufacturing Sales Representatives": 37.00,
    "Writers and Authors": 39.00,
}

