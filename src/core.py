"""core.py — Employee retention / attrition survival metrics for AttritionIQ.

NOT generic classification. Uses survival analysis adapted for HR:
  * **C-index** — concordance for tenure-based risk models.
  * **Retention rate** — fraction still employed at time t.
  * **Hazard rate** — instantaneous attrition risk at time t.
  * **Median tenure** — time at which 50% of employees have left.

References
----------
Cox (1972); Harrell (2015) "Regression Modeling Strategies."
Morita et al. (1993), "Survival analysis of employee turnover."
"""
from __future__ import annotations
import numpy as np


def retention_rate(tenures, events, eval_time) -> float:
    """Fraction of employees still employed at eval_time (Kaplan-Meier based)."""
    t = np.asarray(tenures, dtype=float)
    e = np.asarray(events, dtype=int)
    order = np.argsort(t)
    t, e = t[order], e[order]
    survival = 1.0
    for i in range(len(t)):
        if t[i] > eval_time:
            break
        at_risk = (t >= t[i]).sum()
        if at_risk > 0:
            survival *= (1 - e[i] / at_risk)
    return float(survival)


def hazard_rate(tenures, events, window=30) -> float:
    """Average daily hazard (attrition rate) over the observation period."""
    t = np.asarray(tenures, dtype=float)
    e = np.asarray(events, dtype=int)
    total_events = e.sum()
    total_time = t.sum()
    return float(total_events / total_time) if total_time > 0 else 0.0


def concordance_index(tenures, events, risk_scores) -> float:
    """Harrell's C-index for tenure/risk concordance."""
    t = np.asarray(tenures, dtype=float)
    e = np.asarray(events, dtype=int)
    r = np.asarray(risk_scores, dtype=float)
    n = len(t)
    concordant, permissible = 0, 0
    for i in range(n):
        if e[i] != 1:
            continue
        for j in range(n):
            if t[j] > t[i]:
                permissible += 1
                if r[i] > r[j]:
                    concordant += 1
                elif r[i] == r[j]:
                    concordant += 0.5
    return concordant / permissible if permissible > 0 else 0.5


def median_tenure(tenures, events) -> float:
    """Time at which 50% of employees have left (KM-based)."""
    t = np.asarray(tenures, dtype=float)
    e = np.asarray(events, dtype=int)
    order = np.argsort(t)
    t, e = t[order], e[order]
    survival = 1.0
    for i in range(len(t)):
        at_risk = (t >= t[i]).sum()
        if at_risk > 0:
            survival *= (1 - e[i] / at_risk)
        if survival <= 0.5:
            return float(t[i])
    return float(t[-1])