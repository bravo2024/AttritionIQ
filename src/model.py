"""model.py — Cox PH attrition risk model for AttritionIQ (Multiplier).

Fits a Cox proportional hazards model to predict employee attrition risk
based on HR features. Unlike ClinPredict (clinical survival), this uses
HR-specific covariates and reports department-level risk comparison.

References
----------
Cox (1972); Morita et al. (1993) "Survival analysis of employee turnover."
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from src.core import concordance_index, retention_rate, hazard_rate, median_tenure


def _cox_neg_loglik(beta, X, times, events):
    """Negative partial likelihood for Cox model."""
    beta = np.asarray(beta, dtype=float)
    risk = X @ beta
    ll = 0.0
    for i in range(len(times)):
        if events[i] != 1:
            continue
        risk_set = times >= times[i]
        ll += risk[i] - np.log(np.sum(np.exp(risk[risk_set])))
    return -ll


def fit_and_evaluate(data, seed=42):
    """Fit Cox attrition model and compute retention metrics."""
    df = data["df"]
    features = data["numerical_features"]
    time_col = data["time_col"]
    event_col = data["event_col"]

    X = df[features].values.astype(float)
    times = df[time_col].values.astype(float)
    events = df[event_col].values.astype(int)

    result = minimize(
        _cox_neg_loglik, np.zeros(len(features)),
        args=(X, times, events), method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-6},
    )
    beta = result.x
    risk_scores = X @ beta
    c_idx = concordance_index(times, events, risk_scores)
    hazard_ratios = np.exp(beta)

    # Retention rate at 1 year (365 days)
    ret_1yr = retention_rate(times, events, 365)
    ret_6mo = retention_rate(times, events, 180)
    med_ten = median_tenure(times, events)
    hz = hazard_rate(times, events)

    model = {
        "beta": beta, "hazard_ratios": hazard_ratios.tolist(),
        "feature_names": features, "risk_scores": risk_scores,
    }
    metrics = {
        "n_employees": data["n_samples"],
        "attrition_rate": data["attrition_rate"],
        "c_index": c_idx,
        "retention_6mo": ret_6mo,
        "retention_1yr": ret_1yr,
        "median_tenure_days": med_ten,
        "hazard_rate": hz,
        "hazard_ratios": dict(zip(features, hazard_ratios.tolist())),
    }
    return model, metrics