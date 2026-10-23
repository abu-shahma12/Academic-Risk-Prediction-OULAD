"""
feature_engineering.py
======================
Feature engine for the OULAD early-warning pipeline.

Every function receives data that has ALREADY been truncated to the temporal
cutoff (see data_pipeline.py), so nothing in here can see post-cutoff events.

Column prefixes define the specialist agents' views:
    a_*  Group A - Diagnostic / academic state
    b_*  Group B - Progress & engagement behaviour
    c_*  Group C - Demographic / background context
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

KEYS = ["code_module", "code_presentation", "id_student"]
VLE_TYPES = ["forumng", "oucontent", "quiz", "resource", "homepage"]
PASS_MARK = 40.0      # OULAD pass threshold per assessment
EWM_ALPHA = 0.5       # smoothing for the sequential mastery proxy


def _fix_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["code_module"] = df["code_module"].astype(str)
    df["code_presentation"] = df["code_presentation"].astype(str)
    df["id_student"] = df["id_student"].astype("int64")
    return df


# ----------------------------------------------------------------------------
# Group A - Diagnostic / academic state
# ----------------------------------------------------------------------------
A_COLS = ["a_n_assessments", "a_mean_score", "a_min_score", "a_max_score",
          "a_weighted_timeliness", "a_late_rate", "a_score_ewm", "a_score_slope",
          "a_score_last_minus_first", "a_pass_rate", "a_improvement_rate"]


def academic_features(sub: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
    """sub: pre-cutoff submissions with date_submitted, date_due, weight, score."""
    if sub.empty:
        out = pd.DataFrame(np.nan, index=index, columns=A_COLS)
        out["a_n_assessments"] = 0.0
        return out

    sub = _fix_keys(sub)
    # positive = submitted early, negative = late
    sub["days_early"] = sub["date_due"] - sub["date_submitted"]
    # +1 so zero-weight (e.g. CMA) assessments still contribute to timeliness
    sub["w"] = sub["weight"].fillna(0.0) + 1.0
    valid = sub["days_early"].notna()
    sub["wd"] = sub["days_early"] * sub["w"]
    sub["w_valid"] = sub["w"].where(valid)
    sub["late"] = (sub["days_early"] < 0).astype(float).where(valid)

    g = sub.groupby(KEYS)
    base = g.agg(a_n_assessments=("date_submitted", "size"),
                 a_mean_score=("score", "mean"),
                 a_min_score=("score", "min"),
                 a_max_score=("score", "max"),
                 a_late_rate=("late", "mean"),
                 _wd=("wd", "sum"), _w=("w_valid", "sum"))
    base["a_weighted_timeliness"] = np.where(base["_w"] > 0, base["_wd"] / base["_w"], np.nan)
    base = base.drop(columns=["_wd", "_w"])

    # Sequential mastery proxies over chronologically ordered scores
    rows, keys = [], []
    scored = sub.dropna(subset=["score"]).sort_values(KEYS + ["date_submitted"])
    for key, s in scored.groupby(KEYS, sort=False)["score"]:
        v = s.to_numpy(float)
        n = len(v)
        ewm = v[0]
        for x in v[1:]:
            ewm = EWM_ALPHA * x + (1 - EWM_ALPHA) * ewm
        if n >= 2:
            xs = np.arange(n) - (n - 1) / 2.0
            slope = float((xs * (v - v.mean())).sum() / (xs ** 2).sum())
            lmf = float(v[-1] - v[0])
            imp = float((np.diff(v) > 0).mean())
        else:
            slope = lmf = imp = np.nan
        rows.append((ewm, slope, lmf, float((v >= PASS_MARK).mean()), imp))
        keys.append(key)
    mastery = pd.DataFrame(rows, columns=["a_score_ewm", "a_score_slope",
                                          "a_score_last_minus_first", "a_pass_rate",
                                          "a_improvement_rate"],
                           index=pd.MultiIndex.from_tuples(keys, names=KEYS))
    out = base.join(mastery, how="left").reindex(index)
    out["a_n_assessments"] = out["a_n_assessments"].fillna(0.0)
    return out[A_COLS]


# ----------------------------------------------------------------------------
# Group B - Progress & engagement behaviour
# ----------------------------------------------------------------------------
def _type_cols():
    cols = []
    for t in VLE_TYPES:
        cols += [f"b_{t}_clicks", f"b_{t}_clicks_per_day", f"b_{t}_active_days"]
    cols += ["b_other_clicks"]
    return cols


B_COLS = (_type_cols() +
          ["b_total_clicks", "b_active_days", "b_clicks_per_active_day",
           "b_active_day_std", "b_active_gap_std", "b_days_since_last_vle",
           "b_click_decay_rate", "b_late_early_click_ratio",
           "b_mean_sub_gap", "b_max_sub_gap"])
B_ENG_COLS = [c for c in B_COLS if c not in ("b_mean_sub_gap", "b_max_sub_gap")]


def submission_gap_features(sub: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=index, columns=["b_mean_sub_gap", "b_max_sub_gap"])
    if sub.empty:
        return out
    sub = _fix_keys(sub).sort_values(KEYS + ["date_submitted"])
    sub["gap"] = sub.groupby(KEYS)["date_submitted"].diff()
    agg = sub.groupby(KEYS)["gap"].agg(b_mean_sub_gap="mean", b_max_sub_gap="max")
    return agg.reindex(index)


def engagement_features(vle: pd.DataFrame, index: pd.MultiIndex,
                        cutoff: pd.Series) -> pd.DataFrame:
    """
    vle: pre-cutoff click logs [KEYS, activity, date, sum_click].
    cutoff: float cutoff day per row of `index` (same order).
    """
    out = pd.DataFrame(index=index)
    window = cutoff.clip(lower=1.0)

    if vle.empty:
        for c in B_ENG_COLS:
            out[c] = 0.0 if c.endswith(("clicks", "per_day", "days")) else np.nan
        return out[B_ENG_COLS]

    daily = (vle.groupby(KEYS + ["activity", "date"], observed=True, sort=False)["sum_click"]
                .sum().reset_index())
    daily = _fix_keys(daily)

    # ---- per activity type: totals, per-day frequency, active days
    tp = (daily.groupby(KEYS + ["activity"])
               .agg(clicks=("sum_click", "sum"), days=("date", "nunique"))
               .unstack("activity"))
    for t in VLE_TYPES:
        clicks = tp["clicks"][t].reindex(index).fillna(0.0) if t in tp["clicks"] else 0.0
        days = tp["days"][t].reindex(index).fillna(0.0) if t in tp["days"] else 0.0
        out[f"b_{t}_clicks"] = clicks
        out[f"b_{t}_clicks_per_day"] = clicks / window
        out[f"b_{t}_active_days"] = days
    out["b_other_clicks"] = (tp["clicks"]["other"].reindex(index).fillna(0.0)
                             if "other" in tp["clicks"] else 0.0)

    # ---- all-activity daily series
    tot = (daily.groupby(KEYS + ["date"])["sum_click"].sum().reset_index()
                .sort_values(KEYS + ["date"]))
    tot["gap"] = tot.groupby(KEYS)["date"].diff()
    g = tot.groupby(KEYS)
    s = pd.DataFrame({
        "b_total_clicks": g["sum_click"].sum(),
        "b_active_days": g["date"].nunique(),
        "b_active_day_std": g["date"].std(ddof=0),       # spread of active study days
        "b_active_gap_std": g["gap"].std(ddof=0),        # irregularity of study rhythm
        "_last": g["date"].max(),
    }).reindex(index)
    out["b_total_clicks"] = s["b_total_clicks"].fillna(0.0)
    out["b_active_days"] = s["b_active_days"].fillna(0.0)
    out["b_clicks_per_active_day"] = np.where(out["b_active_days"] > 0,
                                              out["b_total_clicks"] / out["b_active_days"].replace(0, np.nan),
                                              np.nan)
    out["b_active_day_std"] = s["b_active_day_std"]
    out["b_active_gap_std"] = s["b_active_gap_std"]
    out["b_days_since_last_vle"] = cutoff.to_numpy() - s["_last"].to_numpy()

    # ---- temporal click decay (weekly series, masked to each row's own window)
    K = int(np.ceil(cutoff.max() / 7.0)) + 1
    tot["week"] = (np.floor(tot["date"].clip(lower=0) / 7.0).astype(int)).clip(upper=K - 1)
    wk = (tot.groupby(KEYS + ["week"])["sum_click"].sum()
             .unstack("week", fill_value=0)
             .reindex(index=index, columns=range(K), fill_value=0).fillna(0))
    Y = wk.to_numpy(float)
    nweeks = np.maximum(np.ceil(cutoff.to_numpy() / 7.0), 1).astype(int)
    x = np.arange(K)[None, :].astype(float)
    mask = (x < nweeks[:, None]).astype(float)
    n = mask.sum(1)
    xm = (x * mask).sum(1) / n
    ym = (Y * mask).sum(1) / n
    cov = ((x - xm[:, None]) * (Y - ym[:, None]) * mask).sum(1)
    var = (((x - xm[:, None]) ** 2) * mask).sum(1)
    slope = np.where(var > 0, cov / np.where(var > 0, var, 1), 0.0)
    out["b_click_decay_rate"] = -slope / (ym + 1.0)      # >0 => disengaging
    half = (nweeks // 2)[:, None]
    early = (Y * ((x < half) * mask)).sum(1)
    late = (Y * ((x >= half) * mask)).sum(1)
    out["b_late_early_click_ratio"] = (late + 1.0) / (early + 1.0)
    return out[B_ENG_COLS]


# ----------------------------------------------------------------------------
# Group C - Demographic & background context
# ----------------------------------------------------------------------------
C_COLS = ["c_studied_credits", "c_prev_attempts", "c_registration_timing",
          "c_presentation_length"]


def context_features(base: pd.DataFrame) -> pd.DataFrame:
    """base indexed by KEYS. Gender/disability are intentionally NOT features."""
    return pd.DataFrame({
        "c_studied_credits": base["studied_credits"].astype(float),
        "c_prev_attempts": base["num_of_prev_attempts"].astype(float),
        "c_registration_timing": base["date_registration"].astype(float),  # days vs. start
        "c_presentation_length": base["module_presentation_length"].astype(float),
    }, index=base.index)


class ModulePriorTransformer(BaseEstimator, TransformerMixin):
    """
    Module-difficulty prior = smoothed at-risk rate per `code_module`, estimated
    ONLY from the data passed to fit() (i.e. the training fold). Consumes the
    `code_module` column and emits `c_module_prior`.
    """

    def __init__(self, smoothing: float = 50.0):
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        self.global_ = float(y.mean())
        d = pd.DataFrame({"m": X["code_module"].to_numpy(), "y": y}).groupby("m")["y"].agg(["sum", "count"])
        self.prior_ = ((d["sum"] + self.smoothing * self.global_) / (d["count"] + self.smoothing)).to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X["c_module_prior"] = X["code_module"].map(self.prior_).fillna(self.global_).astype(float)
        return X.drop(columns=["code_module"])


def feature_groups(df: pd.DataFrame) -> dict:
    cols = df.columns
    return {"diagnostic": [c for c in cols if c.startswith("a_")],
            "progress": [c for c in cols if c.startswith("b_")],
            "context": [c for c in cols if c.startswith("c_")]}
