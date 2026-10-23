# ============================================================
# HGB + XGBOOST STACKING COORDINATOR HYBRID EXPERIMENT
#
# Research paper:
# Auditable Multi-Agent Early Warning System (OULAD)
#
# Uses corrected out-of-fold predictions.
# NO MODEL RETRAINING REQUIRED.
#
# Primary hybrid: 50% HGB + 50% Coordinator
# Sensitivity analysis: 25% and 75% HGB weights
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "results_corrected"

OUTPUT_DIR = BASE_DIR / "hybrid_results"

FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [15, 30, 45]

THRESHOLD = 0.50

PRIMARY_WEIGHT = 0.50

EXPLORATORY_WEIGHTS = [0.25, 0.75]


# ============================================================
# 2. METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, probabilities):

    y_true = np.asarray(y_true, dtype=int)

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    result = {
        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "PR-AUC": average_precision_score(
            y_true,
            probabilities
        ),

        "Brier": brier_score_loss(
            y_true,
            probabilities
        )
    }

    if len(np.unique(y_true)) == 2:

        result["ROC-AUC"] = roc_auc_score(
            y_true,
            probabilities
        )

    else:

        result["ROC-AUC"] = np.nan

    return result


# ============================================================
# 3. VALIDATE DATA
# ============================================================

def validate_data(df, horizon):

    required = [
        "id_student",
        "fold",
        "y",
        "HGB",
        "Stacking Coordinator"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Horizon {horizon}: "
            f"Missing columns: {missing}"
        )

    if df[required].isnull().any().any():

        raise ValueError(
            f"Horizon {horizon}: "
            "Missing values detected"
        )

    if not set(df["y"].unique()).issubset({0, 1}):

        raise ValueError(
            f"Horizon {horizon}: "
            "Outcome labels must be 0 or 1"
        )

    for model in [
        "HGB",
        "Stacking Coordinator"
    ]:

        p = df[model].to_numpy(dtype=float)

        if not np.isfinite(p).all():

            raise ValueError(
                f"Invalid probability in {model}"
            )

        if np.any((p < 0) | (p > 1)):

            raise ValueError(
                f"Probabilities outside [0,1] "
                f"in {model}"
            )

    if df["fold"].nunique() != 5:

        raise ValueError(
            "Expected five validation folds"
        )

    # A student must not appear in multiple
    # saved outer validation folds.

    fold_counts = (
        df.groupby("id_student")["fold"]
        .nunique()
    )

    if not (fold_counts == 1).all():

        raise ValueError(
            "Student appears in multiple folds"
        )

    print(
        f"Horizon {horizon}%: "
        f"{len(df):,} records, "
        f"{df['id_student'].nunique():,} students"
    )


# ============================================================
# 4. DEFINE HYBRID MODELS
# ============================================================

def make_predictions(df):

    hgb = df["HGB"].to_numpy(dtype=float)

    coordinator = df[
        "Stacking Coordinator"
    ].to_numpy(dtype=float)

    predictions = {
        "HGB": hgb,

        "Stacking Coordinator": coordinator
    }

    # Predefined primary hybrid.

    predictions["Hybrid_50_50"] = (
        PRIMARY_WEIGHT * hgb
        + (1 - PRIMARY_WEIGHT) * coordinator
    )

    # Exploratory sensitivity analysis.

    for weight in EXPLORATORY_WEIGHTS:

        name = f"Hybrid_HGB_{int(weight * 100)}"

        predictions[name] = (
            weight * hgb
            + (1 - weight) * coordinator
        )

    return predictions


# ============================================================
# 5. PROCESS EACH HORIZON
# ============================================================

def process_horizon(horizon):

    print("\n" + "=" * 65)

    print(
        f"HYBRID ENSEMBLE ANALYSIS — {horizon}%"
    )

    print("=" * 65)

    path = (
        INPUT_DIR /
        f"oof_predictions_{horizon}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing corrected prediction file: {path}"
        )

    df = pd.read_csv(path)

    validate_data(df, horizon)

    y = df["y"].to_numpy(dtype=int)

    predictions = make_predictions(df)

    # --------------------------------------------------------
    # Overall pooled OOF performance
    # --------------------------------------------------------

    overall_rows = []

    for model, probabilities in predictions.items():

        metrics = calculate_metrics(
            y,
            probabilities
        )

        overall_rows.append({
            "Horizon": horizon,
            "Model": model,
            **metrics
        })

    overall = pd.DataFrame(overall_rows)

    # --------------------------------------------------------
    # Five-fold metrics
    # --------------------------------------------------------

    fold_rows = []

    for fold in sorted(df["fold"].unique()):

        mask = (
            df["fold"].to_numpy() == fold
        )

        y_fold = y[mask]

        for model, probabilities in predictions.items():

            scores = calculate_metrics(
                y_fold,
                probabilities[mask]
            )

            fold_rows.append({
                "Horizon": horizon,
                "Fold": fold,
                "Model": model,
                **scores
            })

    fold_metrics = pd.DataFrame(fold_rows)

    # --------------------------------------------------------
    # Save probabilities for auditability
    # --------------------------------------------------------

    prediction_columns = [
        "id_student",
        "fold",
        "y"
    ]

    # Preserve enrollment-identifying columns
    # if they are present in the original output.

    for extra in [
        "code_module",
        "code_presentation"
    ]:

        if extra in df.columns:

            prediction_columns.append(extra)

    saved_predictions = df[
        prediction_columns
    ].copy()

    for name, probabilities in predictions.items():

        saved_predictions[name] = probabilities

    saved_predictions.to_csv(
        OUTPUT_DIR /
        f"hybrid_oof_predictions_{horizon}.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print overall results
    # --------------------------------------------------------

    print(
        "\nPooled out-of-fold performance:"
    )

    print(
        overall[
            [
                "Model",
                "F1",
                "ROC-AUC",
                "PR-AUC",
                "Brier"
            ]
        ].round(5).to_string(index=False)
    )

    return overall, fold_metrics


# ============================================================
# 6. GENERATE FIGURES
# ============================================================

def plot_metric(summary, metric, filename):

    plt.figure(figsize=(9, 5.5))

    models = [
        "HGB",
        "Stacking Coordinator",
        "Hybrid_50_50",
        "Hybrid_HGB_25",
        "Hybrid_HGB_75"
    ]

    for model in models:

        subset = summary[
            summary["Model"] == model
        ].sort_values("Horizon")

        plt.plot(
            subset["Horizon"],
            subset[metric],
            marker="o",
            label=model
        )

    plt.xlabel(
        "Course Completion Horizon (%)"
    )

    plt.ylabel(metric)

    plt.title(
        f"HGB-Coordinator Hybrid: {metric}"
    )

    plt.xticks(HORIZONS)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)

    print(
        "HGB + STACKING COORDINATOR HYBRID EXPERIMENT"
    )

    print("=" * 70)

    all_overall = []

    all_folds = []

    for horizon in HORIZONS:

        overall, folds = process_horizon(
            horizon
        )

        all_overall.append(overall)

        all_folds.append(folds)

    # --------------------------------------------------------
    # Combine all horizons
    # --------------------------------------------------------

    overall_results = pd.concat(
        all_overall,
        ignore_index=True
    )

    fold_results = pd.concat(
        all_folds,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save individual outputs
    # --------------------------------------------------------

    overall_results.to_csv(
        OUTPUT_DIR /
        "hybrid_overall_metrics.csv",
        index=False
    )

    fold_results.to_csv(
        OUTPUT_DIR /
        "hybrid_fold_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Mean and standard deviation across folds
    # --------------------------------------------------------

    metrics = [
        "Accuracy",
        "F1",
        "ROC-AUC",
        "PR-AUC",
        "Brier"
    ]

    summary = (
        fold_results.groupby(
            ["Horizon", "Model"]
        )[metrics]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        OUTPUT_DIR /
        "hybrid_summary.csv"
    )

    # --------------------------------------------------------
    # Compare against HGB
    # --------------------------------------------------------

    comparison_rows = []

    for horizon in HORIZONS:

        subset = overall_results[
            overall_results["Horizon"] == horizon
        ]

        hgb_row = subset[
            subset["Model"] == "HGB"
        ].iloc[0]

        for _, row in subset.iterrows():

            if row["Model"] == "HGB":
                continue

            comparison_rows.append({
                "Horizon": horizon,
                "Model": row["Model"],

                "F1 Improvement vs HGB":
                    row["F1"] - hgb_row["F1"],

                "ROC-AUC Improvement vs HGB":
                    row["ROC-AUC"] - hgb_row["ROC-AUC"],

                "PR-AUC Improvement vs HGB":
                    row["PR-AUC"] - hgb_row["PR-AUC"],

                "Brier Difference vs HGB":
                    row["Brier"] - hgb_row["Brier"]
            })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        OUTPUT_DIR /
        "hybrid_vs_hgb.csv",
        index=False
    )

    # --------------------------------------------------------
    # Generate graphs
    # --------------------------------------------------------

    plot_metric(
        overall_results,
        "F1",
        "hybrid_f1_comparison.png"
    )

    plot_metric(
        overall_results,
        "ROC-AUC",
        "hybrid_roc_auc_comparison.png"
    )

    # --------------------------------------------------------
    # Display important final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("HYBRID EXPERIMENT COMPLETED")

    print("=" * 70)

    print(
        "\nPRIMARY HYBRID vs HGB:"
    )

    primary = comparison_df[
        comparison_df["Model"] == "Hybrid_50_50"
    ]

    print(
        primary.round(5).to_string(
            index=False
        )
    )

    print(
        "\nAll output files saved in:"
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()