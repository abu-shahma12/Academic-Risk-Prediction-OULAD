# ============================================================
# PAIRED STUDENT-CLUSTERED BOOTSTRAP
# HGB vs FIXED 50:50 HYBRID
#
# No model retraining required.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    roc_auc_score
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "hybrid_results"

OUTPUT_DIR = BASE_DIR / "hybrid_bootstrap_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [15, 30, 45]

N_BOOTSTRAP = 2000

SEED = 42

THRESHOLD = 0.5

ALPHA = 0.05

BASELINE = "HGB"

HYBRID = "Hybrid_50_50"


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(y, probabilities):

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return {
        "F1": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y,
            probabilities
        )
    }


# ============================================================
# PAIRED STUDENT-CLUSTERED BOOTSTRAP
# ============================================================

def analyze_horizon(horizon):

    print("\n" + "=" * 65)

    print(
        f"HYBRID BOOTSTRAP ANALYSIS — {horizon}%"
    )

    print("=" * 65)

    path = (
        INPUT_DIR /
        f"hybrid_oof_predictions_{horizon}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "id_student",
        "fold",
        "y",
        BASELINE,
        HYBRID
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    if not set(df["y"].unique()).issubset({0, 1}):
        raise ValueError(
            "Labels must be binary (0 or 1)."
        )

    for model in [BASELINE, HYBRID]:

        values = df[model].to_numpy(
            dtype=float
        )

        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 1)
        ):
            raise ValueError(
                f"Invalid probabilities in {model}"
            )

    fold_counts = (
        df.groupby("id_student")["fold"].nunique()
    )

    if not (fold_counts == 1).all():
        raise ValueError(
            "One or more students occur in "
            "multiple outer validation folds."
        )

    y = df["y"].to_numpy(dtype=int)

    p_hgb = df[BASELINE].to_numpy(
        dtype=float
    )

    p_hybrid = df[HYBRID].to_numpy(
        dtype=float
    )

    students = df["id_student"].to_numpy()

    unique_students, inverse = np.unique(
        students,
        return_inverse=True
    )

    n_students = len(unique_students)

    # Store all enrollment rows for each student.

    student_rows = [
        np.flatnonzero(inverse == i)
        for i in range(n_students)
    ]

    # Point estimates from all pooled OOF records.

    original_hgb = calculate_metrics(
        y,
        p_hgb
    )

    original_hybrid = calculate_metrics(
        y,
        p_hybrid
    )

    rng = np.random.default_rng(
        SEED + horizon
    )

    differences = {
        "F1": [],
        "ROC-AUC": []
    }

    for iteration in range(N_BOOTSTRAP):

        # Resample STUDENTS, not individual rows.

        sampled_students = rng.integers(
            0,
            n_students,
            size=n_students
        )

        indices = np.concatenate([
            student_rows[i]
            for i in sampled_students
        ])

        y_boot = y[indices]

        if np.unique(y_boot).size < 2:
            continue

        hgb_scores = calculate_metrics(
            y_boot,
            p_hgb[indices]
        )

        hybrid_scores = calculate_metrics(
            y_boot,
            p_hybrid[indices]
        )

        for metric in differences:

            differences[metric].append(
                hybrid_scores[metric]
                - hgb_scores[metric]
            )

        if (iteration + 1) % 500 == 0:

            print(
                f"Completed {iteration + 1}/"
                f"{N_BOOTSTRAP} resamples"
            )

    rows = []

    for metric, samples in differences.items():

        samples = np.asarray(
            samples,
            dtype=float
        )

        observed_difference = (
            original_hybrid[metric]
            - original_hgb[metric]
        )

        lower = np.quantile(
            samples,
            ALPHA / 2
        )

        upper = np.quantile(
            samples,
            1 - ALPHA / 2
        )

        rows.append({
            "Horizon": horizon,
            "Metric": metric,

            "HGB": original_hgb[metric],

            "Hybrid 50:50":
                original_hybrid[metric],

            "Difference":
                observed_difference,

            "CI Lower": lower,

            "CI Upper": upper,

            "CI Excludes Zero":
                bool(
                    lower > 0 or upper < 0
                ),

            "Bootstrap Samples":
                len(samples)
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print("HGB vs HYBRID 50:50 — PAIRED BOOTSTRAP")

    print("=" * 70)

    all_results = []

    for horizon in HORIZONS:

        result = analyze_horizon(horizon)

        all_results.append(result)

    final = pd.concat(
        all_results,
        ignore_index=True
    )

    output_path = (
        OUTPUT_DIR /
        "hybrid_paired_bootstrap.csv"
    )

    final.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)

    print("BOOTSTRAP ANALYSIS COMPLETED")

    print("=" * 70)

    print(
        final.round(6).to_string(
            index=False
        )
    )

    print("\nResults saved to:")

    print(output_path)


if __name__ == "__main__":
    main()