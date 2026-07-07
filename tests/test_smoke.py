"""Smoke tests for AttritionIQ — employee attrition survival analysis."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic
from src.model import fit_and_evaluate
from src.core import concordance_index, retention_rate, median_tenure


def test_data():
    """Employee data has tenure and attrition event."""
    d = make_synthetic(n=300, seed=42)
    assert d["n_samples"] == 300
    assert "tenure_days" in d["df"].columns
    assert "attrition" in d["df"].columns
    assert 0.0 < d["attrition_rate"] < 1.0


def test_retention_rate():
    """Retention rate decreases over time."""
    times = np.array([30, 60, 90, 180, 365])
    events = np.array([1, 1, 0, 1, 1])
    r_30 = retention_rate(times, events, 30)
    r_180 = retention_rate(times, events, 180)
    assert 0.0 <= r_30 <= 1.0
    assert 0.0 <= r_180 <= 1.0


def test_concordance():
    """C-index > 0.5 when risk scores align with attrition."""
    times = np.array([30, 60, 90, 180, 365])
    events = np.array([1, 1, 1, 0, 1])
    risk = np.array([5, 4, 3, 2, 1])  # high risk = short tenure
    c = concordance_index(times, events, risk)
    assert c > 0.5


def test_fit_and_evaluate():
    """Full pipeline returns model and metrics."""
    d = make_synthetic(n=400, seed=42)
    model, metrics = fit_and_evaluate(d, seed=42)
    assert "c_index" in metrics
    assert "retention_1yr" in metrics
    assert "hazard_ratios" in metrics
    assert metrics["c_index"] > 0.5
    assert 0.0 <= metrics["retention_1yr"] <= 1.0


if __name__ == "__main__":
    test_data()
    test_retention_rate()
    test_concordance()
    test_fit_and_evaluate()
    print("All AttritionIQ smoke tests passed!")
