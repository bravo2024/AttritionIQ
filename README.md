# AttritionIQ

Predicts which employees are likely to leave, and weighs what an intervention is worth against what it costs.

Runs on the real **IBM HR Analytics attrition dataset** (1,470 employees, 16.1% attrition), downloaded automatically on first run and cached under `data/`. If the download isn't possible, both the app and the trainer fall back to a synthetic generator with the same schema.

There are two models here, on purpose:

- **The dashboard** (`app.py`) fits a class-weighted logistic regression written in plain NumPy (SGD, no sklearn) — because everything downstream of the score needs a probability: threshold tuning, cost curves, intervention ROI.
- **The CLI trainer** (`train.py`) fits a Cox proportional-hazards model, treating attrition as a *time-to-event* problem: tenure is the clock, employees still on payroll are right-censored. This answers "who leaves soonest," not just "who leaves."

## Results on the IBM data

Logistic regression, 80/20 holdout, threshold 0.5:

| AUC-ROC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| 0.847 | 0.672 | 0.55 | 0.41 | 0.82 |

Recall is deliberately favoured (class weighting): missing a leaver costs a multiple of salary, while a false alarm costs one retention conversation. The threshold slider in the app lets you move along that trade-off and see the cost impact directly.

Cox model: concordance index **0.782**. The hazard ratios are the interesting part — each additional point of job satisfaction (1–4 scale) cuts the attrition hazard by ~20%, distance from home increases it ~2% per km, and income and age are both protective.

## Run it

```bash
pip install -r requirements.txt
python train.py              # Cox model on the IBM data (downloads once)
python train.py --synthetic  # offline pipeline check
pytest -q
streamlit run app.py
```

## What's in the dashboard

- **Data Explorer** — attrition split, feature correlations, department breakdowns
- **Model Lab** — ROC/PR curves, calibration, confusion matrix, all computed by hand (no sklearn)
- **Cost-Benefit** — ROI calculator: threshold × salary multiple × intervention effectiveness
- **Attrition Drivers** — coefficients and partial-dependence views of the top risk factors

## Layout

```
src/         data loading (IBM HR + synthetic fallback), Cox model, metrics
train.py     Cox survival trainer
app.py       Streamlit dashboard (self-contained logistic model)
tests/       smoke tests
data/        cached IBM HR csv (created on first run)
models/      saved model + metrics (gitignored)
```

MIT licensed.
