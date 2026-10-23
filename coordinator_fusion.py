"""
coordinator_fusion.py
=====================
XGBoost stacking coordinator over the specialist agents + standalone baselines,
evaluated with 5-fold Stratified Group CV (grouped by id_student, seed 42).

Leakage-safe stacking: inside each outer training fold the coordinator is trained
on *inner* group-aware OOF specialist probabilities; at test time specialists
(refit on the full outer-train fold) score the held-out fold.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import get_cv_splits
from feature_engineering import feature_groups
from specialist_agents import AGENT_NAMES, AgentEnsemble, ModelConfig, oof_agent_probs

META_RAW = ["a_mean_score", "b_total_clicks"]
METRICS = ["Accuracy", "F1", "ROC-AUC", "PR-AUC", "Brier"]


def build_meta_features(P: np.ndarray, raw: pd.DataFrame) -> pd.DataFrame:
    P = np.asarray(P)
    conf = 2.0 * np.abs(P - 0.5)                       # c_i = 2|p_i - 0.5|
    meta = pd.DataFrame(P, columns=AGENT_NAMES, index=raw.index)
    meta["disagreement_std"] = P.std(axis=1)           # sigma(p_diag, p_prog, p_context)
    for i, n in enumerate(AGENT_NAMES):
        meta["conf_" + n[2:]] = conf[:, i]
    for c in META_RAW:
        meta[c] = raw[c].to_numpy(float)
    return meta


class StackingCoordinator:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def _make_meta_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=80 if self.cfg.fast else 200, max_depth=3,
                             learning_rate=0.1 if self.cfg.fast else 0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=10, reg_lambda=2.0,
                             eval_metric="logloss", tree_method="hist",
                             random_state=self.cfg.seed, n_jobs=-1)

    def fit(self, df, y, groups):
        fg = feature_groups(df)
        P_oof = oof_agent_probs(df, y, groups, fg, self.cfg)
        self.ensemble_ = AgentEnsemble(fg, self.cfg).fit(df, y, groups)   # for test time
        M = build_meta_features(P_oof, df)
        self.imp_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(M)  # train only
        self.meta_ = self._make_meta_model().fit(self.imp_.transform(M), y)
        return self

    def predict_proba(self, df, return_agents=False):
        P = self.ensemble_.predict_proba(df)
        p = self.meta_.predict_proba(self.imp_.transform(build_meta_features(P, df)))[:, 1]
        return (p, P) if return_agents else p


def make_baselines(cfg: ModelConfig) -> dict:

    n = 150 if cfg.fast else 400

    def pre(scale=False):

        steps = [
            (
                "imp",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            )
        ]

        if scale:

            steps.append(
                ("scale", StandardScaler())
            )

        return steps

    return {

        "HGB": Pipeline(
            pre() + [
                (
                    "m",
                    HistGradientBoostingClassifier(
                        max_iter=n,
                        learning_rate=0.05,
                        random_state=cfg.seed
                    )
                )
            ]
        ),

        "Random Forest": Pipeline(
            pre() + [
                (
                    "m",
                    RandomForestClassifier(
                        n_estimators=n,
                        min_samples_leaf=10,
                        n_jobs=-1,
                        random_state=cfg.seed
                    )
                )
            ]
        ),

        "Logistic Regression": Pipeline(
            pre(scale=True) + [
                (
                    "m",
                    LogisticRegression(
                        max_iter=3000,
                        C=1.0
                    )
                )
            ]
        )
    }


def compute_metrics(y, p, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "F1": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p)}


def cross_validate_all(df: pd.DataFrame, cfg: ModelConfig, n_splits: int = 5, verbose=True):
    """
    Returns (fold_metrics, oof):
      fold_metrics : long DataFrame [model, fold, <metrics>]
      oof          : DataFrame aligned to df.index with per-model OOF probabilities + fold id
    """
    y = df["y"].to_numpy()
    groups = df["id_student"].to_numpy()
    fg = feature_groups(df)
    all_feats = (
    fg["diagnostic"]
    + fg["progress"]
    + fg["context"]
)
    splits = get_cv_splits(df, n_splits, cfg.seed)

    rows, oof = [], pd.DataFrame(index=df.index, dtype=float)
    oof["fold"] = -1
    for k, (tr, te) in enumerate(splits):
        t0 = time.time()
        dtr, dte, ytr, yte = df.iloc[tr], df.iloc[te], y[tr], y[te]

        coord = StackingCoordinator(cfg).fit(dtr, ytr, groups[tr])
        p, P = coord.predict_proba(dte, return_agents=True)
        preds = {"Stacking Coordinator": p,
                 "Agent 1: Diagnostic": P[:, 0], "Agent 2: Progress": P[:, 1],
                 "Agent 3: Context": P[:, 2]}
        for name, model in make_baselines(cfg).items():
            model.fit(dtr[all_feats], ytr)
            preds[name] = model.predict_proba(dte[all_feats])[:, 1]

        for name, pr in preds.items():
            rows.append({"model": name, "fold": k + 1, **compute_metrics(yte, pr)})
            if name not in oof:
                oof[name] = np.nan
            oof.iloc[te, oof.columns.get_loc(name)] = pr
        oof.iloc[te, oof.columns.get_loc("fold")] = k + 1
        if verbose:
            c = [r for r in rows if r["model"] == "Stacking Coordinator" and r["fold"] == k + 1][0]
            print(f"   fold {k+1}/{n_splits}  coordinator AUC={c['ROC-AUC']:.4f} "
                  f"F1={c['F1']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), oof
