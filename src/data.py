from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["satisfaction_level","last_evaluation_score","number_projects","avg_monthly_hours","tenure_years","work_accident","promotion_last_5yrs","salary_level","overtime_flag"]
CATEGORICAL_FEATURES = ["salary_level"]
NUMERICAL_FEATURES = ["satisfaction_level","last_evaluation_score","number_projects","avg_monthly_hours","tenure_years","work_accident","promotion_last_5yrs","overtime_flag"]
TARGET_NAME = "attrition"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "satisfaction_level": rng.uniform(0.1,1.0,size=n).round(3),
        "last_evaluation_score": rng.uniform(0.3,1.0,size=n).round(3),
        "number_projects": rng.poisson(lam=4,size=n).clip(1,9),
        "avg_monthly_hours": rng.normal(loc=180,scale=45,size=n).clip(60,320).astype(int),
        "tenure_years": rng.exponential(scale=4,size=n).clip(0,25).round(1),
        "work_accident": rng.choice([0,1],size=n,p=[0.85,0.15]),
        "promotion_last_5yrs": rng.choice([0,1],size=n,p=[0.88,0.12]),
        "salary_level": rng.choice(["low","medium","high"],size=n,p=[0.40,0.35,0.25]),
        "overtime_flag": rng.choice([0,1],size=n,p=[0.70,0.30]),
    })
    sat=df["satisfaction_level"]; eval_s=df["last_evaluation_score"]/10; proj=df["number_projects"]/10
    hrs=np.clip(df["avg_monthly_hours"]/320,0,1); ten=np.clip(df["tenure_years"]/10,0,1)
    acc=df["work_accident"]; prom=df["promotion_last_5yrs"]; salary=df["salary_level"].map({"low":1,"medium":0.5,"high":0})
    over=df["overtime_flag"]
    log_odds = -1.0 - 2.0*sat + 0.5*eval_s + 0.3*proj + 0.4*hrs + 0.3*ten + 0.4*acc - 0.3*prom + 0.3*salary + 0.5*over + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,85)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(attrition=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
