import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results_corrected"

OUTPUT_DIR = BASE_DIR / "verification_corrected"

OUTPUT_DIR.mkdir(exist_ok=True)

HORIZONS = [15, 30, 45]

METRICS = [
    "Accuracy",
    "F1",
    "ROC-AUC",
    "PR-AUC",
    "Brier"
]

# Model names used in your existing experiment code

MODELS = [
    "Logistic Regression",
    "Random Forest",
    "HGB",
    "Stacking Coordinator",
    "Agent 1: Diagnostic",
    "Agent 2: Progress",
    "Agent 3: Context"
]

# Values currently reported in your IEEE manuscript.
# These are reference values, NOT verified results.

PAPER_RESULTS = {
    15: {
        "F1": 0.6442,
        "ROC-AUC": 0.7792
    },
    30: {
        "F1": 0.6932,
        "ROC-AUC": 0.8409
    },
    45: {
        "F1": 0.7421,
        "ROC-AUC": 0.8828
    }
}

# Numerical tolerance for comparisons.
# CSV values may contain small rounding differences.

TOLERANCE = 1e-4

verification_rows = []

manuscript_rows = []

summary_rows = []


# ==========================================================
# FUNCTION: CALCULATE PERFORMANCE METRICS
# ==========================================================

def calculate_metrics(y, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    results = {
        "Accuracy": accuracy_score(y, predictions),

        "F1": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y,
            probabilities
        ),

        "PR-AUC": average_precision_score(
            y,
            probabilities
        ),

        "Brier": brier_score_loss(
            y,
            probabilities
        )
    }

    return results


# ==========================================================
# VERIFY EACH HORIZON
# ==========================================================

for horizon in HORIZONS:

    print("\n" + "=" * 65)

    print(f"VERIFYING {horizon}% COURSE HORIZON")

    print("=" * 65)

    fold_file = (
        RESULTS_DIR /
        f"fold_metrics_{horizon}.csv"
    )

    predictions_file = (
        RESULTS_DIR /
        f"oof_predictions_{horizon}.csv"
    )

    if not fold_file.exists():
        raise FileNotFoundError(
            f"Missing file: {fold_file}"
        )

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Missing file: {predictions_file}"
        )

    folds = pd.read_csv(fold_file)

    oof = pd.read_csv(predictions_file)

    print(f"\nLoaded {len(oof):,} enrollment records")

    print(
        "Unique students:",
        oof["id_student"].nunique()
    )

    print(
        "At-risk percentage:",
        round(oof["y"].mean() * 100, 2),
        "%"
    )

    # ------------------------------------------------------
    # VERIFY GROUPED CROSS-VALIDATION
    # ------------------------------------------------------

    print("\nChecking student-group separation...")

    student_fold_counts = (
        oof.groupby("id_student")["fold"]
        .nunique()
    )

    group_separation_ok = bool(
        (student_fold_counts == 1).all()
    )

    if group_separation_ok:
        print(
            "PASS: Each student appears in only one "
            "validation fold."
        )
    else:
        print(
            "FAIL: Some students appear in multiple "
            "validation folds!"
        )

    # ------------------------------------------------------
    # VERIFY EACH MODEL
    # ------------------------------------------------------

    for model in MODELS:

        print(f"\nChecking model: {model}")

        if model not in oof.columns:
            raise ValueError(
                f"Missing prediction column: {model}"
            )

        model_fold_results = []

        for fold_number in sorted(
            oof["fold"].unique()
        ):

            subset = oof[
                oof["fold"] == fold_number
            ]

            y = subset["y"].to_numpy()

            probabilities = (
                subset[model].to_numpy()
            )

            calculated = calculate_metrics(
                y,
                probabilities
            )

            model_fold_results.append(
                calculated
            )

            reported = folds[
                (folds["model"] == model) &
                (folds["fold"] == fold_number)
            ]

            if len(reported) != 1:
                raise ValueError(
                    f"Expected one saved row for "
                    f"{model}, fold {fold_number}; "
                    f"found {len(reported)}"
                )

            for metric in METRICS:

                actual = calculated[metric]

                saved = float(
                    reported.iloc[0][metric]
                )

                difference = abs(
                    actual - saved
                )

                status = (
                    "PASS"
                    if difference <= TOLERANCE
                    else "FAIL"
                )

                verification_rows.append({
                    "Horizon": horizon,
                    "Model": model,
                    "Fold": fold_number,
                    "Metric": metric,
                    "Calculated": actual,
                    "Saved": saved,
                    "Absolute Difference": difference,
                    "Status": status
                })

        # --------------------------------------------------
        # MODEL MEAN PERFORMANCE
        # --------------------------------------------------

        model_results = pd.DataFrame(
            model_fold_results
        )

        summary = {
            "Horizon": horizon,
            "Model": model
        }

        for metric in METRICS:

            summary[
                f"{metric} Mean"
            ] = model_results[metric].mean()

            summary[
                f"{metric} Std"
            ] = model_results[metric].std(
                ddof=1
            )

        summary_rows.append(summary)

        print(
            "Mean F1:",
            round(summary["F1 Mean"], 4)
        )

        print(
            "Mean ROC-AUC:",
            round(summary["ROC-AUC Mean"], 4)
        )

        # --------------------------------------------------
        # MANUSCRIPT COMPARISON
        # --------------------------------------------------

        if model == "Stacking Coordinator":

            for metric, expected in (
                PAPER_RESULTS[horizon].items()
            ):

                calculated_mean = (
                    model_results[metric].mean()
                )

                difference = abs(
                    calculated_mean - expected
                )

                manuscript_rows.append({
                    "Horizon": horizon,
                    "Model": model,
                    "Metric": metric,
                    "Paper Value": expected,
                    "Calculated Value": calculated_mean,
                    "Absolute Difference": difference,
                    "Status": (
                        "UNCHANGED"
                        if difference <= TOLERANCE
                        else "CHANGED"
                    )
                })


# ==========================================================
# SAVE VERIFICATION REPORTS
# ==========================================================

verification_df = pd.DataFrame(
    verification_rows
)

manuscript_df = pd.DataFrame(
    manuscript_rows
)

summary_df = pd.DataFrame(
    summary_rows
)

verification_df.to_csv(
    OUTPUT_DIR / "numerical_verification.csv",
    index=False
)

manuscript_df.to_csv(
    OUTPUT_DIR / "manuscript_comparison.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "verified_model_summary.csv",
    index=False
)


# ==========================================================
# DISPLAY FINAL RESULTS
# ==========================================================

print("\n" + "=" * 65)

print("FINAL NUMERICAL VERIFICATION REPORT")

print("=" * 65)

print("\nSaved prediction vs fold-metric checks:")

print(
    verification_df["Status"].value_counts()
)

print("\nManuscript comparison:")

print(
    manuscript_df.to_string(
        index=False
    )
)

print("\nReports saved in:")

print(OUTPUT_DIR)

print("\nFILES GENERATED:")

print("1. numerical_verification.csv")
print("2. manuscript_comparison.csv")
print("3. verified_model_summary.csv")

print("\nVerification finished.")