"""Fit the Cox attrition model and save metrics."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_ibm_hr, make_synthetic
from src.evaluate import save_metrics, print_report
from src.model import fit_and_evaluate
from src.persist import save_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                   help="skip the IBM HR download and use generated data")
    p.add_argument("--n", type=int, default=1000, help="rows for --synthetic")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.synthetic:
        data = make_synthetic(n=args.n, seed=args.seed)
    else:
        try:
            data = load_ibm_hr()
        except Exception as e:
            print(f"IBM HR download failed ({e}), falling back to synthetic")
            data = make_synthetic(n=args.n, seed=args.seed)

    print(f"{data['n_samples']:,} employees, attrition rate {data['attrition_rate']:.1%}")

    model, metrics = fit_and_evaluate(data, seed=args.seed)
    print_report({"cox": metrics})
    save_model(model)
    save_metrics(metrics)
    print("Saved model + metrics under models/")


if __name__ == "__main__":
    main()
