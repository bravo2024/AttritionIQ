"""
AttritionIQ — IBM HR Analytics Attrition Prediction Dashboard
Production-grade Streamlit app. Dependencies: numpy, pandas, matplotlib, scipy, streamlit only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import streamlit as st

matplotlib.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#f1f5f9",
    "grid.color": "#334155",
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})

PALETTE = {
    "stayed": "#22c55e",
    "attrited": "#f43f5e",
    "accent": "#38bdf8",
    "warn": "#fbbf24",
    "purple": "#a78bfa",
    "bg": "#0f172a",
    "card": "#1e293b",
}

DEPT_MAP = {"Sales": 0.30, "R&D": 0.55, "HR": 0.15}
TRAVEL_MAP = {"Non-Travel": 0.40, "Travel_Rarely": 0.40, "Travel_Frequently": 0.20}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttritionIQ | HR Analytics",
    layout="wide",
    page_icon="👥",
)

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    threshold = st.slider("Classification Threshold", 0.10, 0.90, 0.50, 0.05)
    cost_multiplier = st.slider("Cost Multiplier (× avg salary)", 0.3, 3.0, 0.5, 0.1)
    dept_filter = st.multiselect("Department Filter", ["Sales", "R&D", "HR"], default=["Sales", "R&D", "HR"])
    intervention_eff = st.slider("Intervention Effectiveness %", 5, 50, 20, 5)
    st.markdown("---")
    st.caption("AttritionIQ | IBM HR Analytics | v2.0")

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
IBM_HR_URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"


@st.cache_data(show_spinner="Loading IBM HR attrition data…")
def load_ibm_hr() -> pd.DataFrame:
    """The real IBM HR Analytics attrition dataset (1,470 employees).

    Cached to data/ after the first download. Column names are mapped to
    the snake_case schema the rest of the app expects.
    """
    from pathlib import Path

    cache = Path(__file__).parent / "data" / "ibm_hr_attrition.csv"
    if cache.exists():
        raw = pd.read_csv(cache)
    else:
        raw = pd.read_csv(IBM_HR_URL)
        cache.parent.mkdir(exist_ok=True)
        raw.to_csv(cache, index=False)

    df = pd.DataFrame({
        "emp_id": raw["EmployeeNumber"].map("EMP{:05d}".format),
        "age": raw["Age"],
        "department": raw["Department"].map(
            {"Research & Development": "R&D", "Human Resources": "HR", "Sales": "Sales"}),
        "job_level": raw["JobLevel"],
        "years_at_company": raw["YearsAtCompany"],
        "years_with_manager": raw["YearsWithCurrManager"],
        "monthly_income": raw["MonthlyIncome"],
        "percent_salary_hike": raw["PercentSalaryHike"],
        "job_satisfaction": raw["JobSatisfaction"],
        "environment_satisfaction": raw["EnvironmentSatisfaction"],
        "work_life_balance": raw["WorkLifeBalance"],
        "overtime": raw["OverTime"].eq("Yes"),
        "business_travel": raw["BusinessTravel"],
        "distance_from_home": raw["DistanceFromHome"],
        "num_companies_worked": raw["NumCompaniesWorked"],
        "training_times_last_year": raw["TrainingTimesLastYear"],
        "stock_option_level": raw["StockOptionLevel"],
        "attrition": raw["Attrition"].eq("Yes"),
    })
    return df


@st.cache_data(show_spinner="Generating synthetic HR dataset…")
def make_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dept = rng.choice(list(DEPT_MAP.keys()), n, p=list(DEPT_MAP.values()))
    travel = rng.choice(list(TRAVEL_MAP.keys()), n, p=list(TRAVEL_MAP.values()))
    job_level = rng.integers(1, 6, n)
    age = rng.integers(22, 61, n)
    years_at_company = np.clip(rng.integers(0, 41, n), 0, age - 22)
    years_with_manager = np.clip(rng.integers(0, years_at_company + 1, n), 0, years_at_company)
    monthly_income = np.clip(
        np.exp(rng.normal(np.log(6000), 0.7, n)) * (0.6 + 0.4 * job_level / 5), 3000, 20000
    ).astype(int)
    percent_salary_hike = rng.integers(5, 26, n)
    job_satisfaction = rng.integers(1, 5, n)
    environment_satisfaction = rng.integers(1, 5, n)
    work_life_balance = rng.integers(1, 5, n)
    overtime = rng.choice([True, False], n, p=[0.25, 0.75])
    distance_from_home = rng.integers(1, 31, n)
    num_companies_worked = rng.integers(0, 10, n)
    training_times_last_year = rng.integers(0, 7, n)
    stock_option_level = rng.integers(0, 4, n)

    # Logistic attrition model targeting ~16% rate
    logit = (
        -2.4
        + 1.4 * overtime.astype(float)
        + 0.9 * (travel == "Travel_Frequently").astype(float)
        + 0.5 * (travel == "Travel_Rarely").astype(float)
        - 0.5 * (job_satisfaction - 1) / 3
        - 0.4 * (work_life_balance - 1) / 3
        - 0.6 * (job_level - 1) / 4
        - 0.003 * monthly_income / 1000
        + 0.3 * (num_companies_worked / 9)
        - 0.02 * years_at_company
        + 0.4 * (distance_from_home / 30)
        - 0.2 * (stock_option_level / 3)
    )
    prob = 1 / (1 + np.exp(-logit))
    attrition = rng.binomial(1, prob).astype(bool)

    df = pd.DataFrame({
        "emp_id": [f"EMP{i+1:05d}" for i in range(n)],
        "age": age,
        "department": dept,
        "job_level": job_level,
        "years_at_company": years_at_company,
        "years_with_manager": years_with_manager,
        "monthly_income": monthly_income,
        "percent_salary_hike": percent_salary_hike,
        "job_satisfaction": job_satisfaction,
        "environment_satisfaction": environment_satisfaction,
        "work_life_balance": work_life_balance,
        "overtime": overtime,
        "business_travel": travel,
        "distance_from_home": distance_from_home,
        "num_companies_worked": num_companies_worked,
        "training_times_last_year": training_times_last_year,
        "stock_option_level": stock_option_level,
        "attrition": attrition,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL (Logistic Regression via NumPy SGD)
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _encode(df: pd.DataFrame):
    num_cols = [
        "age", "years_at_company", "years_with_manager", "monthly_income",
        "percent_salary_hike", "distance_from_home", "num_companies_worked",
        "training_times_last_year",
    ]
    ord_cols = ["job_satisfaction", "environment_satisfaction", "work_life_balance", "job_level", "stock_option_level"]
    bool_cols = ["overtime"]

    dept_dummies = pd.get_dummies(df["department"], prefix="dept", drop_first=False)
    travel_dummies = pd.get_dummies(df["business_travel"], prefix="travel", drop_first=False)
    dept_dummies = dept_dummies.astype(float)
    travel_dummies = travel_dummies.astype(float)

    X_parts = []
    col_names = []

    for c in num_cols:
        X_parts.append(df[c].values.astype(float))
        col_names.append(c)
    for c in ord_cols:
        X_parts.append(df[c].values.astype(float))
        col_names.append(c)
    for c in bool_cols:
        X_parts.append(df[c].astype(float).values)
        col_names.append(c)
    for c in dept_dummies.columns:
        X_parts.append(dept_dummies[c].values)
        col_names.append(c)
    for c in travel_dummies.columns:
        X_parts.append(travel_dummies[c].values)
        col_names.append(c)

    X = np.column_stack(X_parts)
    return X, col_names


def _standardize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


@st.cache_data(show_spinner="Loading data…")
def get_data() -> pd.DataFrame:
    """Real IBM data when reachable, synthetic fallback offline."""
    try:
        return load_ibm_hr()
    except Exception:
        st.sidebar.warning("Could not fetch the IBM HR dataset — using synthetic data.")
        return make_dataset(5000)


@st.cache_resource(show_spinner="Training logistic regression…")
def train_model():
    df = get_data()
    rng = np.random.default_rng(0)

    X_raw, col_names = _encode(df)
    y = df["attrition"].values.astype(float)

    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    X_tr_raw, X_te_raw = X_raw[tr], X_raw[te]
    y_tr, y_te = y[tr], y[te]

    X_tr, X_te, mu, sigma = _standardize(X_tr_raw, X_te_raw)

    n_feat = X_tr.shape[1]
    w = np.zeros(n_feat)
    b_bias = 0.0

    # Class weights to handle imbalance
    pos_rate = y_tr.mean()
    w_pos = (1 - pos_rate) / pos_rate
    sample_weights = np.where(y_tr == 1, w_pos, 1.0)

    lr = 0.05
    n_epochs = 200
    batch_size = 128
    n_tr = len(y_tr)

    for epoch in range(n_epochs):
        order = rng.permutation(n_tr)
        for start in range(0, n_tr, batch_size):
            batch = order[start: start + batch_size]
            Xb, yb, wb = X_tr[batch], y_tr[batch], sample_weights[batch]
            pred = sigmoid(Xb @ w + b_bias)
            err = (pred - yb) * wb
            grad_w = Xb.T @ err / len(batch)
            grad_b = err.mean()
            w -= lr * grad_w
            b_bias -= lr * grad_b
        lr *= 0.995

    proba_tr = sigmoid(X_tr @ w + b_bias)
    proba_te = sigmoid(X_te @ w + b_bias)

    return {
        "w": w, "b": b_bias, "mu": mu, "sigma": sigma,
        "col_names": col_names,
        "X_tr": X_tr, "y_tr": y_tr, "proba_tr": proba_tr,
        "X_te": X_te, "y_te": y_te, "proba_te": proba_te,
        "tr_idx": tr, "te_idx": te,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRICS (manual, no sklearn)
# ─────────────────────────────────────────────────────────────────────────────
def roc_curve_manual(y_true, y_score):
    thresholds = np.linspace(1, 0, 300)
    fprs, tprs = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fpr = fp / (fp + tn + 1e-9)
        tpr = tp / (tp + fn + 1e-9)
        fprs.append(fpr); tprs.append(tpr)
    return np.array(fprs), np.array(tprs), thresholds


def auc_manual(x, y):
    order = np.argsort(x)
    return np.trapz(y[order], x[order])


def pr_curve_manual(y_true, y_score):
    thresholds = np.linspace(1, 0, 300)
    precisions, recalls = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        precisions.append(prec); recalls.append(rec)
    return np.array(precisions), np.array(recalls), thresholds


def compute_metrics(y_true, y_prob, thresh):
    pred = (y_prob >= thresh).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    logloss = -np.mean(y_true * np.log(y_prob + 1e-9) + (1 - y_true) * np.log(1 - y_prob + 1e-9))
    fprs, tprs, _ = roc_curve_manual(y_true, y_prob)
    roc_auc = auc_manual(fprs, tprs)
    precs, recs, _ = pr_curve_manual(y_true, y_prob)
    pr_auc = auc_manual(recs, precs)
    return {
        "AUC-ROC": roc_auc, "PR-AUC": pr_auc,
        "Log Loss": logloss, "F1": f1,
        "Precision": prec, "Recall": rec,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# KAPLAN-MEIER
# ─────────────────────────────────────────────────────────────────────────────
def kaplan_meier(durations, events):
    times = np.sort(np.unique(durations))
    n = len(durations)
    S = 1.0
    surv = [1.0]
    t_out = [0]
    for t in times:
        at_risk = (durations >= t).sum()
        died = ((durations == t) & (events == 1)).sum()
        if at_risk > 0:
            S *= (1 - died / at_risk)
        surv.append(S)
        t_out.append(t)
    return np.array(t_out), np.array(surv)


# ─────────────────────────────────────────────────────────────────────────────
# ODDS RATIO
# ─────────────────────────────────────────────────────────────────────────────
def odds_ratio(df: pd.DataFrame, feature: str, positive_val):
    exposed_att = ((df[feature] == positive_val) & df["attrition"]).sum()
    exposed_stay = ((df[feature] == positive_val) & ~df["attrition"]).sum()
    unexposed_att = ((df[feature] != positive_val) & df["attrition"]).sum()
    unexposed_stay = ((df[feature] != positive_val) & ~df["attrition"]).sum()
    a, b, c, d = exposed_att + 0.5, exposed_stay + 0.5, unexposed_att + 0.5, unexposed_stay + 0.5
    OR = (a / b) / (c / d)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo = np.exp(np.log(OR) - 1.96 * se)
    hi = np.exp(np.log(OR) + 1.96 * se)
    return OR, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# RISK TIER
# ─────────────────────────────────────────────────────────────────────────────
def assign_tier(p):
    if p >= 0.70: return "Critical"
    if p >= 0.50: return "High"
    if p >= 0.30: return "Medium"
    if p >= 0.15: return "Watch"
    return "Low"


TIER_COLORS = {
    "Critical": "#f43f5e",
    "High": "#fb923c",
    "Medium": "#fbbf24",
    "Watch": "#a3e635",
    "Low": "#22c55e",
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────
df_full = get_data()
model = train_model()

# Score all employees
X_all_raw, _ = _encode(df_full)
X_all_scaled = (X_all_raw - model["mu"]) / (model["sigma"])
df_full["risk_score"] = sigmoid(X_all_scaled @ model["w"] + model["b"])
df_full["risk_tier"] = df_full["risk_score"].apply(assign_tier)
df_full["pred_attrition"] = (df_full["risk_score"] >= threshold).astype(bool)

df = df_full[df_full["department"].isin(dept_filter)].copy()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#38bdf8;margin-bottom:4px;'>"
    "👥 AttritionIQ — HR Analytics Platform</h1>"
    "<p style='text-align:center;color:#64748b;margin-top:0;'>IBM HR Analytics | 5,000 Employees | Logistic Regression</p>",
    unsafe_allow_html=True,
)

att_rate = df["attrition"].mean()
avg_salary = df["monthly_income"].mean() * 12
ann_cost = att_rate * avg_salary * cost_multiplier * len(df)
n_att = df["attrition"].sum()
n_critical = (df["risk_tier"] == "Critical").sum()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Employees", f"{len(df):,}")
k2.metric("Attrition Rate", f"{att_rate:.1%}")
k3.metric("Annual Leavers", f"{int(n_att):,}")
k4.metric("Annual Cost ($M)", f"${ann_cost/1e6:.2f}M")
k5.metric("Critical Risk", f"{n_critical:,}")
k6.metric("Avg Salary", f"${avg_salary:,.0f}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👥 Workforce Explorer",
    "🔍 Attrition Risk Factors",
    "🤖 Attrition Prediction Model",
    "⚠ At-Risk Employee Identification",
    "💰 Financial Impact & ROI",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — WORKFORCE EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Workforce Overview")

    # Department attrition bar chart
    c_left, c_right = st.columns(2)
    with c_left:
        dept_att = df.groupby("department")["attrition"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(dept_att.index, dept_att.values * 100,
                      color=[PALETTE["attrited"], PALETTE["warn"], PALETTE["accent"]])
        for bar, val in zip(bars, dept_att.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=9, color="white")
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_title("Attrition Rate by Department", pad=10)
        ax.grid(axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Age distribution stayers vs leavers
    with c_right:
        stayed = df.loc[~df["attrition"], "age"]
        left = df.loc[df["attrition"], "age"]
        fig, ax = plt.subplots(figsize=(5, 3))
        bins = np.arange(20, 65, 3)
        ax.hist(stayed, bins=bins, alpha=0.6, color=PALETTE["stayed"], label="Stayed", density=True)
        ax.hist(left, bins=bins, alpha=0.7, color=PALETTE["attrited"], label="Left", density=True)
        ax.set_xlabel("Age")
        ax.set_ylabel("Density")
        ax.set_title("Age Distribution: Stayers vs Leavers", pad=10)
        ax.legend()
        ax.grid(axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Income distribution by attrition
    c_left2, c_right2 = st.columns(2)
    with c_left2:
        fig, ax = plt.subplots(figsize=(5, 3))
        income_stayed = df.loc[~df["attrition"], "monthly_income"] / 1000
        income_left = df.loc[df["attrition"], "monthly_income"] / 1000
        bins_inc = np.linspace(2, 21, 30)
        ax.hist(income_stayed, bins=bins_inc, alpha=0.6, color=PALETTE["stayed"],
                label="Stayed", density=True)
        ax.hist(income_left, bins=bins_inc, alpha=0.7, color=PALETTE["attrited"],
                label="Left", density=True)
        ax.set_xlabel("Monthly Income ($K)")
        ax.set_ylabel("Density")
        ax.set_title("Income Distribution by Attrition", pad=10)
        ax.legend()
        ax.grid(axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Job satisfaction heatmap
    with c_right2:
        heat_data = df.groupby(["department", "job_satisfaction"])["attrition"].mean().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(heat_data.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.4)
        ax.set_xticks(range(len(heat_data.columns)))
        ax.set_xticklabels([f"Sat {c}" for c in heat_data.columns])
        ax.set_yticks(range(len(heat_data.index)))
        ax.set_yticklabels(heat_data.index)
        ax.set_title("Attrition Rate: Dept × Job Satisfaction", pad=10)
        plt.colorbar(im, ax=ax, label="Attrition Rate")
        for i in range(len(heat_data.index)):
            for j in range(len(heat_data.columns)):
                ax.text(j, i, f"{heat_data.values[i, j]:.1%}", ha="center",
                        va="center", fontsize=8, color="white")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Sample Employee Records")
    st.dataframe(
        df[["emp_id", "department", "age", "job_level", "monthly_income",
            "overtime", "business_travel", "job_satisfaction",
            "attrition", "risk_score", "risk_tier"]].head(100),
        use_container_width=True, height=280,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK FACTORS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Attrition Risk Factors")

    st.markdown("#### Odds Ratio Analysis")
    st.latex(r"\text{OR} = \frac{a/b}{c/d}, \quad 95\%\,\text{CI} = \exp\!\left(\ln(\text{OR}) \pm 1.96\sqrt{\tfrac{1}{a}+\tfrac{1}{b}+\tfrac{1}{c}+\tfrac{1}{d}}\right)")

    factor_specs = [
        ("overtime", True, "Overtime=Yes"),
        ("business_travel", "Travel_Frequently", "Travel Frequently"),
        ("business_travel", "Travel_Rarely", "Travel Rarely"),
        ("job_satisfaction", 1, "Job Sat=1 (Low)"),
        ("job_satisfaction", 4, "Job Sat=4 (High)"),
        ("work_life_balance", 1, "WLB=1 (Low)"),
        ("job_level", 1, "Job Level 1"),
        ("stock_option_level", 0, "Stock Option=0"),
    ]

    or_rows = []
    for feat, val, label in factor_specs:
        OR, lo, hi = odds_ratio(df, feat, val)
        or_rows.append({"Factor": label, "OR": OR, "CI_lo": lo, "CI_hi": hi})
    or_df = pd.DataFrame(or_rows).sort_values("OR", ascending=True).reset_index(drop=True)

    c_forest, c_table = st.columns([3, 2])
    with c_forest:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        y_pos = np.arange(len(or_df))
        colors = [PALETTE["attrited"] if r > 1 else PALETTE["stayed"] for r in or_df["OR"]]
        ax.barh(y_pos, or_df["OR"], xerr=[or_df["OR"] - or_df["CI_lo"], or_df["CI_hi"] - or_df["OR"]],
                color=colors, alpha=0.8, height=0.5, error_kw={"ecolor": "#94a3b8", "capsize": 4, "linewidth": 1.5})
        ax.axvline(x=1, color="#94a3b8", linestyle="--", linewidth=1.5, label="OR=1 (no effect)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(or_df["Factor"], fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Odds Ratio (log scale)")
        ax.set_title("Forest Plot — Attrition Odds Ratios", pad=10)
        ax.legend()
        ax.grid(axis="x")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c_table:
        st.markdown("**OR Summary Table**")
        display_or = or_df.copy()
        display_or["OR"] = display_or["OR"].map("{:.2f}".format)
        display_or["95% CI"] = display_or.apply(lambda r: f"[{r['CI_lo']:.2f}, {r['CI_hi']:.2f}]", axis=1)
        st.dataframe(display_or[["Factor", "OR", "95% CI"]].sort_values("OR", ascending=False), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Kaplan-Meier Survival Curve")
    st.latex(r"S(t) = \prod_{t_i \le t}\!\left(1 - \frac{d_i}{n_i}\right)")

    km_left, km_right = st.columns(2)
    with km_left:
        dur_all = df["years_at_company"].values
        ev_all = df["attrition"].values.astype(int)
        t_all, s_all = kaplan_meier(dur_all, ev_all)
        t_hi, s_hi = kaplan_meier(
            df.loc[df["risk_tier"].isin(["Critical", "High"]), "years_at_company"].values,
            df.loc[df["risk_tier"].isin(["Critical", "High"]), "attrition"].values.astype(int),
        )
        t_lo, s_lo = kaplan_meier(
            df.loc[~df["risk_tier"].isin(["Critical", "High"]), "years_at_company"].values,
            df.loc[~df["risk_tier"].isin(["Critical", "High"]), "attrition"].values.astype(int),
        )
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.step(t_all, s_all, color=PALETTE["accent"], linewidth=2, label="All Employees")
        ax.step(t_hi, s_hi, color=PALETTE["attrited"], linewidth=1.8, label="High Risk")
        ax.step(t_lo, s_lo, color=PALETTE["stayed"], linewidth=1.8, label="Low Risk")
        ax.set_xlabel("Years at Company (Tenure)")
        ax.set_ylabel("Survival Probability")
        ax.set_title("KM Survival Curve by Risk Tier", pad=10)
        ax.legend()
        ax.grid()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with km_right:
        st.markdown("#### Point-Biserial Correlation with Attrition")
        num_feats = ["age", "monthly_income", "years_at_company", "distance_from_home",
                     "job_satisfaction", "work_life_balance", "environment_satisfaction",
                     "percent_salary_hike", "num_companies_worked", "training_times_last_year"]
        corrs = {}
        for f in num_feats:
            r, _ = stats.pointbiserialr(df["attrition"].astype(float), df[f])
            corrs[f] = r
        corr_s = pd.Series(corrs).sort_values()
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        colors_corr = [PALETTE["attrited"] if v > 0 else PALETTE["stayed"] for v in corr_s.values]
        ax.barh(corr_s.index, corr_s.values, color=colors_corr, alpha=0.85)
        ax.axvline(0, color="#94a3b8", linewidth=1.2)
        ax.set_xlabel("Point-Biserial r")
        ax.set_title("Feature Correlation with Attrition", pad=10)
        ax.grid(axis="x")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("#### Flight Risk Segment Characteristics")
    seg_tiers = ["Critical", "High", "Medium", "Watch", "Low"]
    seg_rows = []
    for tier in seg_tiers:
        seg = df[df["risk_tier"] == tier]
        if len(seg) == 0:
            continue
        seg_rows.append({
            "Tier": tier,
            "Count": len(seg),
            "Actual Attrition": f"{seg['attrition'].mean():.1%}",
            "Avg Age": f"{seg['age'].mean():.1f}",
            "Avg Salary": f"${seg['monthly_income'].mean():,.0f}",
            "Overtime %": f"{seg['overtime'].mean():.1%}",
            "Avg Job Sat": f"{seg['job_satisfaction'].mean():.2f}",
            "Avg WLB": f"{seg['work_life_balance'].mean():.2f}",
        })
    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Logistic Regression — Attrition Prediction Model")
    st.latex(r"P(\text{Attrition}) = \sigma\!\left(\mathbf{w}^\top \mathbf{x} + b\right), \quad \sigma(z) = \frac{1}{1+e^{-z}}")
    st.markdown("Trained with **NumPy SGD** (class-weighted for 16% minority class, 80/20 split). No sklearn used.")

    y_te = model["y_te"]
    proba_te = model["proba_te"]
    metrics = compute_metrics(y_te, proba_te, threshold)

    # Metrics KPI row
    mk1, mk2, mk3, mk4, mk5, mk6 = st.columns(6)
    mk1.metric("AUC-ROC", f"{metrics['AUC-ROC']:.4f}")
    mk2.metric("PR-AUC", f"{metrics['PR-AUC']:.4f}")
    mk3.metric("Log Loss", f"{metrics['Log Loss']:.4f}")
    mk4.metric("F1", f"{metrics['F1']:.4f}")
    mk5.metric("Precision", f"{metrics['Precision']:.4f}")
    mk6.metric("Recall", f"{metrics['Recall']:.4f}")

    c_roc, c_pr = st.columns(2)
    with c_roc:
        fprs, tprs, _ = roc_curve_manual(y_te, proba_te)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(fprs, tprs, color=PALETTE["accent"], linewidth=2,
                label=f"AUC = {metrics['AUC-ROC']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="#475569", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", pad=10)
        ax.legend()
        ax.grid()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c_pr:
        precs, recs, _ = pr_curve_manual(y_te, proba_te)
        baseline = y_te.mean()
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(recs, precs, color=PALETTE["purple"], linewidth=2,
                label=f"PR-AUC = {metrics['PR-AUC']:.4f}")
        ax.axhline(baseline, color="#475569", linestyle="--", linewidth=1, label=f"Baseline = {baseline:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve", pad=10)
        ax.legend()
        ax.grid()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    c_cm, c_coef = st.columns(2)
    with c_cm:
        st.markdown(f"**Confusion Matrix** (threshold = {threshold})")
        cm = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Stay", "Pred Leave"])
        ax.set_yticklabels(["Actual Stay", "Actual Leave"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=14, color="white" if cm[i, j] > cm.max() * 0.5 else "#0f172a")
        ax.set_title("Confusion Matrix", pad=10)
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c_coef:
        col_names = model["col_names"]
        w = model["w"]
        coef_df = pd.DataFrame({"Feature": col_names, "Coefficient": w})
        coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=True).index)
        top_coef = coef_df.tail(15)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        colors_c = [PALETTE["attrited"] if v > 0 else PALETTE["stayed"] for v in top_coef["Coefficient"]]
        ax.barh(top_coef["Feature"], top_coef["Coefficient"], color=colors_c, alpha=0.85)
        ax.axvline(0, color="#94a3b8", linewidth=1.2)
        ax.set_xlabel("Coefficient Weight")
        ax.set_title("Feature Coefficients (Top 15)", pad=10)
        ax.grid(axis="x")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — AT-RISK IDENTIFICATION
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("At-Risk Employee Identification")

    tier_order = ["Critical", "High", "Medium", "Watch", "Low"]
    tier_counts = df["risk_tier"].value_counts().reindex(tier_order, fill_value=0)

    c4a, c4b = st.columns([2, 3])
    with c4a:
        st.markdown("#### Risk Tier Distribution")
        fig, ax = plt.subplots(figsize=(4.5, 3))
        colors_t = [TIER_COLORS[t] for t in tier_order]
        bars = ax.bar(tier_order, tier_counts.values, color=colors_t, alpha=0.9)
        for bar, val in zip(bars, tier_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha="center", va="bottom", fontsize=9, color="white")
        ax.set_ylabel("Employee Count")
        ax.set_title("Risk Tier Headcount", pad=10)
        ax.grid(axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c4b:
        st.markdown("#### Department Risk Distribution")
        dept_tier = df.groupby(["department", "risk_tier"]).size().unstack(fill_value=0)
        dept_tier = dept_tier.reindex(columns=tier_order, fill_value=0)
        fig, ax = plt.subplots(figsize=(5.5, 3))
        x = np.arange(len(dept_tier))
        width = 0.15
        for i, tier in enumerate(tier_order):
            ax.bar(x + i * width, dept_tier[tier].values,
                   width, label=tier, color=TIER_COLORS[tier], alpha=0.85)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(dept_tier.index)
        ax.set_ylabel("Count")
        ax.set_title("Department Risk Distribution", pad=10)
        ax.legend(fontsize=7)
        ax.grid(axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("#### At-Risk Employee Register")

    def top3_risk_factors(row):
        factors = []
        if row["overtime"]: factors.append("Overtime")
        if row["job_satisfaction"] == 1: factors.append("Low Job Satisfaction")
        if row["business_travel"] == "Travel_Frequently": factors.append("Frequent Travel")
        if row["monthly_income"] < 4000: factors.append("Low Income")
        if row["job_level"] == 1: factors.append("Entry Level")
        if row["work_life_balance"] == 1: factors.append("Poor WLB")
        if row["num_companies_worked"] >= 7: factors.append("Job Hopper")
        if row["distance_from_home"] > 25: factors.append("Far Commute")
        return ", ".join(factors[:3]) if factors else "—"

    high_risk = df[df["risk_tier"].isin(["Critical", "High"])].copy()
    high_risk["top_risk_factors"] = high_risk.apply(top3_risk_factors, axis=1)
    display_hr = high_risk[[
        "emp_id", "department", "job_level", "risk_score", "risk_tier", "top_risk_factors"
    ]].sort_values("risk_score", ascending=False).head(200)
    display_hr["risk_score"] = display_hr["risk_score"].map("{:.1%}".format)
    st.dataframe(display_hr, use_container_width=True, height=280)

    st.markdown("---")
    st.markdown("#### Retention Intervention Recommendations")

    def get_recommendation(row):
        if row["overtime"] and row["work_life_balance"] <= 2:
            return "Offer flexible work arrangements / remote options"
        if row["percent_salary_hike"] <= 8 and row["job_level"] >= 3:
            return "Benchmark compensation against market; consider raise"
        if row["job_satisfaction"] <= 2 and row["years_at_company"] >= 5:
            return "Initiate career development discussion / promotion path"
        if row["business_travel"] == "Travel_Frequently" and row["age"] <= 30:
            return "Reduce travel frequency; offer travel stipend"
        if row["environment_satisfaction"] <= 2:
            return "Conduct team environment survey; address team dynamics"
        return "Scheduled 1-on-1 check-in; engagement survey"

    high_risk["recommendation"] = high_risk.apply(get_recommendation, axis=1)
    rec_display = high_risk[[
        "emp_id", "department", "risk_tier", "risk_score", "recommendation"
    ]].sort_values("risk_score", ascending=False).head(100)
    rec_display["risk_score"] = rec_display["risk_score"].map("{:.1%}".format)
    st.dataframe(rec_display, use_container_width=True, height=260)

    st.markdown("---")
    st.markdown("#### Counterfactual: Expected vs. Improved Attrition")
    st.latex(r"\text{Attrition}_{\text{improved}} = \text{Attrition}_{\text{current}} \times \left(1 - \frac{\text{Effectiveness}}{100}\right)_{\text{targeted}}")

    current_att = df["attrition"].mean()
    targeted_mask = df["risk_tier"].isin(["Critical", "High"])
    n_targeted = targeted_mask.sum()
    eff_frac = intervention_eff / 100
    improved_rate = (
        (df.loc[~targeted_mask, "attrition"].sum() + df.loc[targeted_mask, "attrition"].sum() * (1 - eff_frac))
        / len(df)
    )
    cf_cols = st.columns(3)
    cf_cols[0].metric("Current Attrition Rate", f"{current_att:.2%}")
    cf_cols[1].metric("Projected Rate (post-intervention)", f"{improved_rate:.2%}",
                       delta=f"{(improved_rate - current_att):.2%}")
    cf_cols[2].metric("Targeted Employees", f"{n_targeted:,}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — FINANCIAL IMPACT & ROI
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Financial Impact & ROI Analysis")

    st.latex(r"\text{Replacement Cost} = \begin{cases} 0.5 \times \text{Salary} & \text{Junior (Level 1-2)} \\ 1.0 \times \text{Salary} & \text{Mid (Level 3)} \\ 2.0 \times \text{Salary} & \text{Senior (Level 4-5)} \end{cases}")

    def replacement_cost_factor(level):
        if level <= 2: return 0.5
        if level == 3: return 1.0
        return 2.0

    df_cost = df.copy()
    df_cost["annual_salary"] = df_cost["monthly_income"] * 12
    df_cost["cost_factor"] = df_cost["job_level"].apply(replacement_cost_factor)
    df_cost["replacement_cost"] = df_cost["annual_salary"] * df_cost["cost_factor"]
    df_cost["attrition_cost"] = df_cost["attrition"].astype(float) * df_cost["replacement_cost"]

    total_attrition_cost = df_cost["attrition_cost"].sum()
    n_leavers = df_cost["attrition"].sum()
    avg_cost_per_leaver = total_attrition_cost / max(n_leavers, 1)

    f1c, f2c, f3c, f4c = st.columns(4)
    f1c.metric("Total Annual Attrition Cost", f"${total_attrition_cost/1e6:.2f}M")
    f2c.metric("Actual Leavers", f"{n_leavers:,}")
    f3c.metric("Avg Cost per Leaver", f"${avg_cost_per_leaver:,.0f}")
    f4c.metric("Attrition Rate", f"{df_cost['attrition'].mean():.1%}")

    st.markdown("---")
    c5a, c5b = st.columns(2)

    with c5a:
        st.markdown("#### Replacement Cost Waterfall by Department")
        dept_cost = df_cost.groupby("department")["attrition_cost"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        running = 0
        for dept, cost in dept_cost.items():
            bar = ax.bar(dept, cost / 1e6, bottom=0, color=PALETTE["attrited"], alpha=0.85)
            ax.text(list(dept_cost.index).index(dept), cost / 1e6 + 0.02,
                    f"${cost/1e6:.2f}M", ha="center", va="bottom", fontsize=9, color="white")
        ax.set_ylabel("Cost ($M)")
        ax.set_title("Attrition Replacement Cost by Department", pad=10)
        ax.grid(axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c5b:
        st.markdown("#### ROI Sensitivity Analysis")
        st.latex(r"\text{ROI} = \frac{\text{Cost Avoided} - \text{Intervention Cost}}{\text{Intervention Cost}} \times 100\%")
        reduction_rates = [0.10, 0.20, 0.30]
        intervention_cost_per_emp = 2000
        sensitivity_rows = []
        for red in reduction_rates:
            avoided = total_attrition_cost * red
            n_targeted_for_roi = int(n_leavers * red)
            interv_cost = n_targeted_for_roi * intervention_cost_per_emp
            net_benefit = avoided - interv_cost
            roi = (net_benefit / max(interv_cost, 1)) * 100
            sensitivity_rows.append({
                "Reduction": f"{red:.0%}",
                "Cost Avoided ($M)": f"${avoided/1e6:.2f}M",
                "Intervention Cost ($K)": f"${interv_cost/1e3:.0f}K",
                "Net Benefit ($M)": f"${net_benefit/1e6:.2f}M",
                "ROI": f"{roi:.0f}%",
            })
        st.dataframe(pd.DataFrame(sensitivity_rows), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 12-Month Headcount Projection")
    st.latex(r"H_t = H_0 \times (1 - r_{\text{att}})^t + \text{Hires}_t")

    months = np.arange(0, 13)
    H0 = len(df)
    monthly_att_current = att_rate / 12
    monthly_att_improved = improved_rate / 12
    monthly_hire_rate = monthly_att_current * 0.85  # assume 85% backfill

    H_current = [H0]
    H_improved = [H0]
    for _ in months[1:]:
        H_current.append(H_current[-1] * (1 - monthly_att_current) + H_current[-1] * monthly_hire_rate)
        H_improved.append(H_improved[-1] * (1 - monthly_att_improved) + H_improved[-1] * monthly_att_improved * 0.85)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(months, H_current, color=PALETTE["attrited"], linewidth=2.5, marker="o", markersize=4,
            label="Current Attrition Trajectory")
    ax.plot(months, H_improved, color=PALETTE["stayed"], linewidth=2.5, marker="s", markersize=4,
            label=f"After {intervention_eff}% Intervention")
    ax.fill_between(months, H_improved, H_current, alpha=0.15, color=PALETTE["stayed"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Headcount")
    ax.set_title("12-Month Headcount Projection: Current vs Improved", pad=10)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("#### Detailed Cost Breakdown by Level")
    level_cost = df_cost.groupby("job_level").agg(
        Employees=("attrition", "count"),
        Leavers=("attrition", "sum"),
        Total_Cost=("attrition_cost", "sum"),
        Avg_Cost_Per_Leaver=("replacement_cost", "mean"),
    ).reset_index()
    level_cost["Attrition Rate"] = (level_cost["Leavers"] / level_cost["Employees"]).map("{:.1%}".format)
    level_cost["Total Cost"] = level_cost["Total_Cost"].map("${:,.0f}".format)
    level_cost["Avg Cost/Leaver"] = level_cost["Avg_Cost_Per_Leaver"].map("${:,.0f}".format)
    st.dataframe(
        level_cost[["job_level", "Employees", "Leavers", "Attrition Rate", "Total Cost", "Avg Cost/Leaver"]]
        .rename(columns={"job_level": "Job Level"}),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### Investment vs. Return: Intervention Budget Simulator")
    budget_m = st.slider("Retention Budget ($K)", 50, 2000, 500, 50)
    budget = budget_m * 1_000
    cost_per_emp_intv = st.slider("Cost per Employee Intervention ($)", 500, 5000, 2000, 250)
    n_can_treat = int(budget / max(cost_per_emp_intv, 1))
    high_risk_leavers = df_cost[df_cost["risk_tier"].isin(["Critical", "High"]) & df_cost["attrition"]]
    n_saved = min(n_can_treat, len(high_risk_leavers))
    cost_avoided = high_risk_leavers.nlargest(n_saved, "replacement_cost")["replacement_cost"].sum()
    net_return = cost_avoided - budget
    roi_budget = (net_return / max(budget, 1)) * 100

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Employees Treatable", f"{n_can_treat:,}")
    b2.metric("Estimated Saved", f"{n_saved:,}")
    b3.metric("Cost Avoided", f"${cost_avoided:,.0f}")
    b4.metric("Net ROI", f"{roi_budget:.0f}%", delta=f"${net_return:,.0f}")
