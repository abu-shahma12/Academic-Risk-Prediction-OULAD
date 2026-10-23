"""
main.py - Auditable Multi-Agent Early Warning & Adaptive Intervention System (OULAD)

Usage:
    python main.py                         # data in ./, horizons 15/30/45 %
    python main.py --data-dir /path/to/oulad --fast
    python main.py --horizons 0.30 --tol 0.02 --gap-metric eo
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from coordinator_fusion import METRICS, cross_validate_all
from data_pipeline import HORIZONS, SEED, build_horizon_dataset, load_raw
from fairness_mitigation import run_fairness_audit
from feature_engineering import feature_groups
from specialist_agents import ModelConfig

warnings.filterwarnings("ignore")
MODEL_ORDER = ["Logistic Regression", "Random Forest", "HGB", "Stacking Coordinator"]
AGENT_ORDER = ["Agent 1: Diagnostic", "Agent 2: Progress", "Agent 3: Context"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def fmt(v):
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def mean_std_table(fold_df: pd.DataFrame, models) -> pd.DataFrame:
    rows = []
    for m in models:
        d = fold_df[fold_df["model"] == m]
        rows.append({"Model": m, **{k: f"{d[k].mean():.4f} ± {d[k].std(ddof=1):.4f}" for k in METRICS}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--horizons", type=float, nargs="+", default=list(HORIZONS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fast", action="store_true", help="fewer trees / inner folds (quick run)")
    ap.add_argument("--tol", type=float, default=0.02, help="max allowed FPR gap after mitigation")
    ap.add_argument(
    "--gap-metric",
    choices=["fpr", "eo"],
    default="eo"
)
    ap.add_argument("--context-model", choices=["rf", "lr"], default="rf")
    ap.add_argument("--keep-early-withdrawals", action="store_true",
                    help="do not exclude students who already withdrew / not yet registered at cutoff")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    np.random.seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = ModelConfig(fast=args.fast, seed=SEED, context_model=args.context_model)

    print("# Auditable Multi-Agent Early Warning System - OULAD\n")
    t0 = time.time()
    print(f"Loading CSVs from '{args.data_dir}' ...", flush=True)
    raw = load_raw(args.data_dir)
    print(f"Loaded in {time.time()-t0:.0f}s\n", flush=True)

    fairness_rows, detail_rows = [], []
    for pct in args.horizons:
        tag = f"{int(round(pct*100))}%"
        print(f"\n## Horizon t = {tag} of presentation length\n", flush=True)
        df = build_horizon_dataset(raw, pct, exclude_early_withdrawals=not args.keep_early_withdrawals)
        fg = feature_groups(df)
        print(f"Rows: {len(df):,} | unique students: {df['id_student'].nunique():,} | "
              f"at-risk rate: {df['y'].mean():.3f} | features A/B/C: "
              f"{len(fg['diagnostic'])}/{len(fg['progress'])}/{len(fg['context'])}\n", flush=True)

        fold_df, oof = cross_validate_all(df, cfg, n_splits=args.folds)
        fold_df.insert(0, "horizon", tag)
        fold_df.to_csv(os.path.join(args.out_dir, f"fold_metrics_{tag.strip('%')}.csv"), index=False)
        oof.assign(id_student=df["id_student"].to_numpy(), y=df["y"].to_numpy(),
                   gender=df["gender"].to_numpy(), disability=df["disability"].to_numpy()
                   ).to_csv(os.path.join(args.out_dir, f"oof_predictions_{tag.strip('%')}.csv"), index=False)

        print(f"\n### Baselines vs. Stacking Coordinator (t = {tag}; mean ± std over "
              f"{args.folds} group folds, threshold 0.5)\n")
        print(md_table(mean_std_table(fold_df, MODEL_ORDER)))
        print("\nSpecialist agents (calibrated, standalone):\n")
        print(md_table(mean_std_table(fold_df, AGENT_ORDER)))
        print(f"\nCoordinator - per-fold metrics (t = {tag}):\n")
        pf = fold_df[fold_df["model"] == "Stacking Coordinator"][["fold"] + METRICS]
        print(md_table(pf))

        summ, detail, thr = run_fairness_audit(oof, df, "Stacking Coordinator",
                                               tol=args.tol, gap_metric=args.gap_metric, seed=SEED)
        summ.insert(0, "Horizon", tag); detail.insert(0, "Horizon", tag)
        fairness_rows.append(summ); detail_rows.append(detail)

    fair, det = pd.concat(fairness_rows, ignore_index=True), pd.concat(detail_rows, ignore_index=True)
    fair.to_csv(os.path.join(args.out_dir, "fairness_summary.csv"), index=False)
    det.to_csv(os.path.join(args.out_dir, "fairness_subgroup_detail.csv"), index=False)

    print(f"\n## Fairness: Pre- vs Post-Mitigation (Coordinator OOF; tol={args.tol}, "
          f"gap={args.gap_metric.upper()}; post = cross-fitted on held-out students)\n")
    print(md_table(fair[["Horizon", "Attribute", "FPR gap (pre)", "FPR gap (post)", "TPR gap (pre)",
                         "TPR gap (post)", "F1 (pre)", "F1 (post)", "theta_g"]]))
    print("\n### Subgroup audit detail\n")
    print(md_table(det[["Horizon", "attribute", "group", "stage", "n", "TPR", "FPR", "Precision"]]))
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min. Results saved to '{args.out_dir}/'.")


if __name__ == "__main__":
    main()
