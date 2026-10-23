"""
fairness_mitigation.py
======================
Subgroup audit (TPR / FPR / Precision) and active mitigation via group-specific
decision thresholds theta_g (equalized-odds post-processing, custom exact grid search).

Positive class = At-Risk (1).  FPR = share of NON-risk students flagged at-risk.

Objective per protected attribute (2 groups):
    maximise overall F1   s.t.   gap(theta_a, theta_b) <= tol
where gap = |FPR_a - FPR_b|  (gap_metric="fpr", default) or
            max(|dFPR|, |dTPR|)  (gap_metric="eo", full equalized odds).
If no feasible pair exists, the pair with the smallest gap is used.

Honest evaluation: `cross_fit_mitigation` learns thresholds on some students and
applies them to held-out students (grouped by id_student), so post-mitigation numbers
are not fit on the data they are scored on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def _rates(tp, fp, fn, tn):
    tpr = tp / np.maximum(tp + fn, 1)
    fpr = fp / np.maximum(fp + tn, 1)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    return tpr, fpr, prec


def subgroup_audit(y, pred, s, attr: str = "group") -> pd.DataFrame:
    y, pred, s = np.asarray(y), np.asarray(pred), np.asarray(s)
    out = []
    for g in sorted(pd.unique(s)):
        m = s == g
        tp = int(((pred == 1) & (y == 1) & m).sum()); fp = int(((pred == 1) & (y == 0) & m).sum())
        fn = int(((pred == 0) & (y == 1) & m).sum()); tn = int(((pred == 0) & (y == 0) & m).sum())
        tpr, fpr, prec = _rates(tp, fp, fn, tn)
        out.append({"attribute": attr, "group": g, "n": int(m.sum()), "TPR": float(tpr),
                    "FPR": float(fpr), "Precision": float(prec), "Flag rate": (tp + fp) / max(m.sum(), 1)})
    return pd.DataFrame(out)


def gap_summary(audit: pd.DataFrame) -> dict:
    return {"FPR gap": float(audit["FPR"].max() - audit["FPR"].min()),
            "TPR gap": float(audit["TPR"].max() - audit["TPR"].min())}


class GroupThresholdOptimizer:
    """Learns one threshold per level of a binary sensitive attribute."""

    def __init__(self, tol: float = 0.02, gap_metric: str = "fpr", grid=GRID):
        assert gap_metric in ("fpr", "eo")
        self.tol, self.gap_metric, self.grid = tol, gap_metric, np.asarray(grid)

    def _counts(self, p, y):
        pred = p[:, None] >= self.grid[None, :]
        pos = (y == 1)[:, None]
        tp = (pred & pos).sum(0); fp = (pred & ~pos).sum(0)
        return tp, fp, int(pos.sum()) - tp, int((~pos).sum()) - fp   # tp, fp, fn, tn

    def fit(self, p, y, s):
        p, y, s = np.asarray(p, float), np.asarray(y), np.asarray(s)
        self.levels_ = sorted(pd.unique(s))
        if len(self.levels_) != 2:
            raise ValueError("GroupThresholdOptimizer supports binary attributes (M/F, Y/N).")
        a, b = self.levels_
        ca, cb = self._counts(p[s == a], y[s == a]), self._counts(p[s == b], y[s == b])
        tpr_a, fpr_a, _ = _rates(*ca); tpr_b, fpr_b, _ = _rates(*cb)

        TP = ca[0][:, None] + cb[0][None, :]; FP = ca[1][:, None] + cb[1][None, :]
        FN = ca[2][:, None] + cb[2][None, :]
        F1 = 2 * TP / np.maximum(2 * TP + FP + FN, 1)
        gap = np.abs(fpr_a[:, None] - fpr_b[None, :])
        if self.gap_metric == "eo":
            gap = np.maximum(gap, np.abs(tpr_a[:, None] - tpr_b[None, :]))

        feasible = gap <= self.tol
        if feasible.any():
            score = np.where(feasible, F1 - 1e-6 * gap, -np.inf)      # max F1, tie -> smaller gap
        else:
            score = -gap + 1e-6 * F1                                    # fallback: min gap
        i, j = np.unravel_index(np.argmax(score), score.shape)
        self.thresholds_ = {a: float(self.grid[i]), b: float(self.grid[j])}
        self.feasible_ = bool(feasible.any())
        return self

    def predict(self, p, s):
        thr = np.array([self.thresholds_.get(g, 0.5) for g in np.asarray(s)])
        return (np.asarray(p) >= thr).astype(int)


def cross_fit_mitigation(y, p, s, groups, tol=0.02, gap_metric="fpr", n_splits=5, seed=42):
    """Thresholds fit on training students, applied to held-out students (grouped CV)."""
    y, p, s, groups = map(np.asarray, (y, p, s, groups))
    post = np.zeros(len(y), int)
    skf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(p, y, groups):
        opt = GroupThresholdOptimizer(tol, gap_metric).fit(p[tr], y[tr], s[tr])
        post[te] = opt.predict(p[te], s[te])
    final = GroupThresholdOptimizer(tol, gap_metric).fit(p, y, s)     # deployable thresholds
    return post, final


def run_fairness_audit(oof: pd.DataFrame, info: pd.DataFrame, prob_col="Stacking Coordinator",
                       tol=0.02, gap_metric="fpr", seed=42):
    """
    oof: OOF probabilities aligned with info; info has y, id_student, gender, disability.
    Returns (summary_df, detail_df, thresholds dict).
    """
    from sklearn.metrics import accuracy_score, f1_score
    y, p, groups = info["y"].to_numpy(), oof[prob_col].to_numpy(), info["id_student"].to_numpy()
    pre = (p >= 0.5).astype(int)
    summary, detail, thresholds = [], [], {}
    for attr, col in [("Gender", "gender"), ("Disability", "disability")]:
        s = info[col].to_numpy()
        post, opt = cross_fit_mitigation(y, p, s, groups, tol, gap_metric, seed=seed)
        thresholds[attr] = opt.thresholds_
        a_pre, a_post = subgroup_audit(y, pre, s, attr), subgroup_audit(y, post, s, attr)
        g_pre, g_post = gap_summary(a_pre), gap_summary(a_post)
        summary.append({"Attribute": attr,
                        "FPR gap (pre)": g_pre["FPR gap"], "FPR gap (post)": g_post["FPR gap"],
                        "TPR gap (pre)": g_pre["TPR gap"], "TPR gap (post)": g_post["TPR gap"],
                        "F1 (pre)": f1_score(y, pre), "F1 (post)": f1_score(y, post),
                        "Acc (pre)": accuracy_score(y, pre), "Acc (post)": accuracy_score(y, post),
                        "theta_g": ", ".join(f"{k}={v:.2f}" for k, v in opt.thresholds_.items())})
        a_pre["stage"], a_post["stage"] = "Pre (theta=0.50)", "Post (theta_g)"
        detail += [a_pre, a_post]
    return pd.DataFrame(summary), pd.concat(detail, ignore_index=True), thresholds
