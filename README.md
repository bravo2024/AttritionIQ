# AttritionIQ

> Employee attrition prediction with multi-model comparison and cost-based intervention planning.

Trains four classifiers (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) on synthetic HR data engineered to mirror IBM HR Analytics dataset patterns. The dashboard provides department-level attrition breakdowns, threshold tuning, and cost-benefit analysis for retention interventions.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.872 |
| Gini | 0.744 |
| KS Statistic | 0.664 |
| F1 Score | 0.500 |
| Accuracy | 0.776 |

5-fold CV AUC: 0.873 ± 0.051. Four models compared with full ROC curves and calibration plots.

## Features

| Component | What it does |
|---|---|
| **Controls** | Classification threshold, cost multiplier, department filter, intervention effectiveness |
| **Data Explorer** | Attrition distribution, feature correlations, department-level breakdowns |
| **Model Lab** | Multi-model comparison, ROC/PR curves, calibration, confusion matrix |
| **Cost-Benefit** | Intervention ROI calculator based on salary, turnover cost, and effectiveness rate |
| **Attrition Drivers** | Feature importance, partial dependence plots for top risk factors |

## Repo Structure

```
AttritionIQ/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic HR dataset matching IBM HR Analytics patterns: department, job level, overtime, satisfaction scores, tenure, income, and travel frequency features.

## License

MIT
