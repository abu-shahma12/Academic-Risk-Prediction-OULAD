"""
data_pipeline.py
================
Loads raw OULAD CSVs, builds leakage-safe multi-horizon datasets and provides
the grouped cross-validation splitter.

Leakage rules enforced here
---------------------------
* Label (final_result) is used ONLY to create `y`, then dropped.
* `date_unregistration` (withdrawal date) is used ONLY to exclude students whose
  outcome is already known at the cutoff, then dropped (never a feature).
* Assessment submissions with date_submitted >= cutoff are removed.
* VLE click rows with date >= cutoff are removed.
* Exams and banked (carried-over) assessments are excluded.
* Imputation medians / module priors are computed inside training folds only
  (see specialist_agents.py / coordinator_fusion.py).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from feature_engineering import (KEYS, VLE_TYPES, academic_features, engagement_features,
                                 submission_gap_features, context_features)

SEED = 42
HORIZONS = (0.15, 0.30, 0.45)
LABEL_MAP = {"Pass": 0, "Distinction": 0, "Fail": 1, "Withdrawn": 1}


def load_raw(data_dir: str = ".") -> dict:
    """Read all OULAD CSVs from disk (large studentVle.csv is read with compact dtypes)."""
    p = lambda f: os.path.join(data_dir, f)
    courses = pd.read_csv(p("courses.csv"))
    student_info = pd.read_csv(p("studentInfo.csv"))
    registration = pd.read_csv(p("studentRegistration.csv"))
    assessments = pd.read_csv(p("assessments.csv"))

    sa = pd.read_csv(p("studentAssessment.csv"))
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    for c in ["date", "weight"]:
        assessments[c] = pd.to_numeric(assessments[c], errors="coerce")
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    sa = sa.merge(assessments.rename(columns={"date": "date_due"}),
                  on="id_assessment", how="left")
    sa = sa[(sa["assessment_type"] != "Exam") & (sa["is_banked"] != 1)].copy()

    vle_meta = (pd.read_csv(p("vle.csv"), usecols=["id_site", "activity_type"])
                  .drop_duplicates("id_site").set_index("id_site")["activity_type"])
    sv = pd.read_csv(p("studentVle.csv"),
                     usecols=["code_module", "code_presentation", "id_student",
                              "id_site", "date", "sum_click"],
                     dtype={"code_module": "category", "code_presentation": "category",
                            "id_student": "int32", "id_site": "int32",
                            "date": "int16", "sum_click": "int32"})
    act = sv["id_site"].map(vle_meta)
    sv["activity"] = pd.Categorical(np.where(act.isin(VLE_TYPES), act, "other"),
                                    categories=VLE_TYPES + ["other"])
    sv = sv.drop(columns="id_site")

    # per-row course length via composite integer code (avoids 10M-row string merges)
    mcat, pcat = sv["code_module"].cat.categories, sv["code_presentation"].cat.categories
    lut = np.full((len(mcat) + 1) * 100, np.nan)
    for _, r in courses.iterrows():
        if r["code_module"] in mcat and r["code_presentation"] in pcat:
            lut[mcat.get_loc(r["code_module"]) * 100 + pcat.get_loc(r["code_presentation"])] = \
                r["module_presentation_length"]
    code = sv["code_module"].cat.codes.to_numpy().astype(np.int32) * 100 + \
        sv["code_presentation"].cat.codes.to_numpy()
    sv["course_length"] = lut[code].astype("float32")
    return dict(courses=courses, student_info=student_info, registration=registration,
                sub=sa, vle=sv)


def build_horizon_dataset(raw: dict, pct: float,
                          exclude_early_withdrawals: bool = True) -> pd.DataFrame:
    """One row per (module, presentation, student) using only info observed before pct*length."""
    si = raw["student_info"][KEYS + ["gender", "disability", "final_result",
                                    "studied_credits", "num_of_prev_attempts"]]
    reg = raw["registration"][KEYS + ["date_registration", "date_unregistration"]]
    df = (si.merge(reg, on=KEYS, how="left")
            .merge(raw["courses"], on=["code_module", "code_presentation"], how="left"))
    df["y"] = df["final_result"].map(LABEL_MAP)          # Pass/Distinction=0, Fail/Withdrawn=1
    df = df.dropna(subset=["y"]).drop_duplicates(KEYS)
    # OULAD encodes missing dates as '?', so force numeric before any comparison.
    for c in ["date_registration", "date_unregistration", "module_presentation_length"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["cutoff_day"] = np.floor(
    pct * df["module_presentation_length"]
)
    df["date_unregistration"] = pd.to_numeric(df["date_unregistration"], errors="coerce").astype(float)

    if exclude_early_withdrawals:
        # Outcome already known (withdrew) / not yet enrolled at time t -> nothing to predict.
        gone = df["date_unregistration"].notna() & (df["date_unregistration"] <= df["cutoff_day"])
        late = df["date_registration"].notna() & (df["date_registration"] > df["cutoff_day"])
        df = df[~(gone | late)]
    df = df.drop(columns=["date_unregistration", "final_result"])    # leakage rule
    df["y"] = df["y"].astype(int)
    df["id_student"] = df["id_student"].astype("int64")
    df = df.set_index(KEYS)
    cutoff = df["cutoff_day"]

    # ---- assessments strictly before cutoff
    sub = raw["sub"].merge(df.reset_index()[KEYS + ["cutoff_day"]], on=KEYS, how="inner")
    sub = sub[sub["date_submitted"] < sub["cutoff_day"]]
    assert (
    sub["date_submitted"] < sub["cutoff_day"]
).all(), "Temporal leakage detected in assessment data"

    # ---- VLE clicks strictly before cutoff
    v = raw["vle"]
    vle = v[
    v["date"] < np.floor(
        pct * v["course_length"]
    )
]
    assert (
    vle["date"] < np.floor(
        pct * vle["course_length"]
    )
).all(), "Temporal leakage detected in VLE data"
    feat = pd.concat([
        academic_features(sub, df.index),
        submission_gap_features(sub, df.index),
        engagement_features(vle, df.index, cutoff),
        context_features(df),
    ], axis=1)

    out = pd.concat([df[["y", "gender", "disability"]], feat], axis=1).reset_index()
    return out


def get_cv_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = SEED):
    """5-fold Stratified Group CV, grouped strictly by id_student."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df, df["y"].to_numpy(), df["id_student"].to_numpy()))
