"""data.py — Synthetic employee data for AttritionIQ.

Employee-level data with tenure (time-to-attrition), event indicator
(1=left, 0=still employed), and HR covariates (salary, department,
satisfaction, performance, commute). This mirrors real HR analytics data.

The outcome is time-to-attrition (survival), NOT binary classification.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


def make_synthetic(n: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Generate synthetic employee data with tenure-based survival outcomes.

    Low satisfaction, long commute, and low salary increase attrition hazard.
    High performers have slightly higher hazard (external opportunities).
    """
    rng = np.random.default_rng(seed)

    salary = rng.lognormal(10.5, 0.3, n).clip(30000, 200000).astype(int)
    department = rng.choice(["engineering", "sales", "marketing", "ops", "hr"], n,
                            p=[0.30, 0.25, 0.15, 0.20, 0.10])
    satisfaction = rng.beta(5, 3, n).round(3)
    performance = rng.beta(6, 3, n).round(3)
    commute_mins = rng.exponential(30, n).clip(5, 120).astype(int)
    age = rng.integers(22, 60, n)

    log_hazard = (
        -5.5
        - 0.5 * np.log(salary / 50000)
        + 0.3 * (1 - satisfaction)
        + 0.5 * performance  # high performers leave more
        + 0.01 * commute_mins
        - 0.02 * age
        + rng.normal(0, 0.3, n)
    )
    hazard = np.exp(log_hazard)
    tenure = rng.exponential(1.0 / hazard).clip(1, 365 * 10)

    # Right censoring (employees still employed)
    max_tenure = tenure.max() * 0.7
    censor_time = rng.uniform(0, max_tenure, n)
    observed_tenure = np.minimum(tenure, censor_time)
    event = (tenure <= censor_time).astype(int)

    df = pd.DataFrame({
        "salary": salary, "department": department, "satisfaction": satisfaction,
        "performance": performance, "commute_mins": commute_mins, "age": age,
        "tenure_days": observed_tenure.round(0), "attrition": event,
    })

    return {
        "df": df,
        "features": ["salary", "satisfaction", "performance", "commute_mins", "age"],
        "categorical_features": ["department"],
        "numerical_features": ["salary", "satisfaction", "performance", "commute_mins", "age"],
        "time_col": "tenure_days",
        "event_col": "attrition",
        "n_samples": n,
        "attrition_rate": float(event.mean()),
    }

IBM_HR_URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"


def load_ibm_hr(cache_dir: str | None = None) -> dict[str, Any]:
    """The IBM HR Analytics dataset (1,470 employees), mapped to the survival schema.

    Tenure is YearsAtCompany in days (coarse — the dataset only reports whole
    years), and everyone still employed at survey time is right-censored.
    """
    from pathlib import Path

    cache = Path(cache_dir or Path(__file__).parent.parent / "data") / "ibm_hr_attrition.csv"
    if cache.exists():
        raw = pd.read_csv(cache)
    else:
        raw = pd.read_csv(IBM_HR_URL)
        cache.parent.mkdir(exist_ok=True)
        raw.to_csv(cache, index=False)

    event = raw["Attrition"].eq("Yes").astype(int)
    df = pd.DataFrame({
        "monthly_income": raw["MonthlyIncome"],
        "department": raw["Department"],
        "job_satisfaction": raw["JobSatisfaction"],
        "performance_rating": raw["PerformanceRating"],
        "distance_from_home": raw["DistanceFromHome"],
        "age": raw["Age"],
        "tenure_days": (raw["YearsAtCompany"] * 365).clip(lower=1),
        "attrition": event,
    })
    feats = ["monthly_income", "job_satisfaction", "performance_rating",
             "distance_from_home", "age"]
    return {
        "df": df,
        "features": feats,
        "categorical_features": ["department"],
        "numerical_features": feats,
        "time_col": "tenure_days",
        "event_col": "attrition",
        "n_samples": len(df),
        "attrition_rate": float(event.mean()),
    }
