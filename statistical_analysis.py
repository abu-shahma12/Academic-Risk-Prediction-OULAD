# ============================================================
# STATISTICAL ANALYSIS FOR IEEE RESEARCH PAPER
#
# Auditable Multi-Agent Early Warning System (OULAD)
#
# Analyses:
# 1. Overall OOF performance
# 2. Student-clustered bootstrap confidence intervals
# 3. Paired model comparisons
# 4. Fold-level descriptive comparisons
# 5. Performance comparison figures
#
# No model retraining required.
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

RESULTS_DIR = BASE_DIR / "results_corrected"

OUTPUT_DIR = BASE_DIR / "statistical_corrected"

FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)

FIGURE_DIR.mkdir(exist_ok=True)

HORIZONS = [15, 30, 45]

SEED = 42

# Number of student-clustered bootstrap repetitions.
# Increase to 5000 for a more stable final analysis.

N_BOOTSTRAP = 2000

CONFIDENCE_LEVEL = 0.95

ALPHA = 1 - CONFIDENCE_LEVEL

THRESHOLD = 0.5


MODELS = [
    "Logistic Regression",
    "Random Forest",
    "HGB",
    "Stacking Coordinator",
    "Agent 1: Diagnostic",
    "Agent 2: Progress",
    "Agent 3: Context"
]


BASELINES = [
    "HGB",
    "Random Forest",
    "Logistic Regression"
]


METRICS = [
    "Accuracy",
    "F1",
    "ROC-AUC",
    "PR-AUC",
    "Brier"
]


# ============================================================
# 2. METRIC CALCULATION
# ============================================================

def calculate_metrics(y, probabilities):

    y = np.asarray(y, dtype=int)

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y,
            predictions
        ),

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


# ============================================================
# 3. VALIDATE INPUT DATA
# ============================================================

def validate_data(oof, horizon):

    required_columns = (
        ["id_student", "y", "fold"] + MODELS
    )

    missing = [
        c for c in required_columns
        if c not in oof.columns
    ]

    if missing:
        raise ValueError(
            f"Horizon {horizon}: Missing columns: {missing}"
        )

    if oof[required_columns].isna().any().any():
        raise ValueError(
            f"Horizon {horizon}: Missing prediction data"
        )

    if not set(oof["y"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Horizon {horizon}: Labels must be 0 or 1"
        )

    for model in MODELS:

        p = oof[model].to_numpy(dtype=float)

        if not np.isfinite(p).all():
            raise ValueError(
                f"Horizon {horizon}: "
                f"Non-finite predictions in {model}"
            )

        if np.any((p < 0) | (p > 1)):
            raise ValueError(
                f"Horizon {horizon}: "
                f"Invalid probabilities in {model}"
            )

    # Check whether all records belonging to the same
    # student appear in the same saved validation fold.

    fold_counts = (
        oof.groupby("id_student")["fold"]
        .nunique()
    )

    if not (fold_counts == 1).all():
        raise ValueError(
            f"Horizon {horizon}: "
            "Student appears in multiple validation folds"
        )

    if oof["fold"].nunique() != 5:
        raise ValueError(
            f"Horizon {horizon}: Expected 5 folds"
        )

    print(
        f"Horizon {horizon}%: "
        f"{len(oof):,} enrollments, "
        f"{oof['id_student'].nunique():,} students"
    )


# ============================================================
# 4. STUDENT-CLUSTERED BOOTSTRAP
# ============================================================

def generate_cluster_bootstrap_indices(
    student_ids,
    n_bootstrap,
    seed
):

    rng = np.random.default_rng(seed)

    # Each unique student is one sampling unit.
    # All enrollment records for a selected student
    # are retained together.

    group_codes, unique_students = pd.factorize(
        student_ids,
        sort=False
    )

    n_students = len(unique_students)

    # Efficiently store row positions for each student.

    order = np.argsort(group_codes, kind="stable")

    sorted_codes = group_codes[order]

    counts = np.bincount(
        sorted_codes,
        minlength=n_students
    )

    starts = np.concatenate((
        [0],
        np.cumsum(counts[:-1])
    ))

    for _ in range(n_bootstrap):

        sampled_students = rng.integers(
            0,
            n_students,
            size=n_students
        )

        # Repeat the complete set of enrollment rows
        # each time a student is sampled.

        sampled_positions = np.concatenate([
            order[
                starts[g]:
                starts[g] + counts[g]
            ]
            for g in sampled_students
        ])

        yield sampled_positions


# ============================================================
# 5. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_analysis(oof, horizon):

    y = oof["y"].to_numpy(dtype=int)

    students = oof["id_student"].to_numpy()

    probabilities = {
        model: oof[model].to_numpy(dtype=float)
        for model in MODELS
    }

    # Calculate point estimates from all saved OOF rows.

    point_estimates = {
        model: calculate_metrics(
            y,
            probabilities[model]
        )
        for model in MODELS
    }

    # Store bootstrap metrics.

    bootstrap_results = {
        model: {
            metric: []
            for metric in METRICS
        }
        for model in MODELS
    }

    # Resample students, not individual enrollment rows.
    # Identical samples are used for all models,
    # preserving paired model comparisons.

    for iteration, indices in enumerate(
        generate_cluster_bootstrap_indices(
            students,
            N_BOOTSTRAP,
            SEED + horizon
        ),
        start=1
    ):

        y_boot = y[indices]

        # ROC-AUC requires both classes.

        if np.unique(y_boot).size < 2:
            continue

        for model in MODELS:

            p_boot = probabilities[model][indices]

            scores = calculate_metrics(
                y_boot,
                p_boot
            )

            for metric in METRICS:

                bootstrap_results[model][metric].append(
                    scores[metric]
                )

        if iteration % 500 == 0:

            print(
                f"  Bootstrap iteration "
                f"{iteration}/{N_BOOTSTRAP}"
            )

    # --------------------------------------------------------
    # Confidence intervals for individual models
    # --------------------------------------------------------

    summary_rows = []

    for model in MODELS:

        for metric in METRICS:

            samples = np.asarray(
                bootstrap_results[model][metric]
            )

            lower = np.quantile(
                samples,
                ALPHA / 2
            )

            upper = np.quantile(
                samples,
                1 - ALPHA / 2
            )

            summary_rows.append({
                "Horizon": horizon,
                "Model": model,
                "Metric": metric,
                "OOF Estimate":
                    point_estimates[model][metric],
                "CI Lower": lower,
                "CI Upper": upper,
                "Bootstrap SE": samples.std(ddof=1),
                "Bootstrap Samples": len(samples)
            })

    summary_df = pd.DataFrame(summary_rows)

    # --------------------------------------------------------
    # Paired differences against the Stacking Coordinator
    # --------------------------------------------------------

    comparison_rows = []

    coordinator = "Stacking Coordinator"

    for baseline in BASELINES:

        for metric in METRICS:

            coord_samples = np.asarray(
                bootstrap_results[coordinator][metric]
            )

            baseline_samples = np.asarray(
                bootstrap_results[baseline][metric]
            )

            differences = (
                coord_samples - baseline_samples
            )

            point_difference = (
                point_estimates[coordinator][metric]
                - point_estimates[baseline][metric]
            )

            lower = np.quantile(
                differences,
                ALPHA / 2
            )

            upper = np.quantile(
                differences,
                1 - ALPHA / 2
            )

            comparison_rows.append({
                "Horizon": horizon,
                "Coordinator": coordinator,
                "Baseline": baseline,
                "Metric": metric,
                "Difference": point_difference,
                "CI Lower": lower,
                "CI Upper": upper,
                "CI Excludes Zero":
                    bool(lower > 0 or upper < 0)
            })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    return summary_df, comparison_df


# ============================================================
# 6. DESCRIPTIVE PAIRED FOLD ANALYSIS
# ============================================================

def paired_fold_analysis(folds, horizon):

    rows = []

    for baseline in BASELINES:

        coord = (
            folds[
                folds["model"] == "Stacking Coordinator"
            ]
            .set_index("fold")
            .sort_index()
        )

        other = (
            folds[
                folds["model"] == baseline
            ]
            .set_index("fold")
            .sort_index()
        )

        if not coord.index.equals(other.index):
            raise ValueError(
                f"Horizon {horizon}: "
                f"Fold mismatch for {baseline}"
            )

        for metric in METRICS:

            differences = (
                coord[metric] - other[metric]
            )

            rows.append({
                "Horizon": horizon,
                "Baseline": baseline,
                "Metric": metric,
                "Mean Fold Difference":
                    differences.mean(),
                "Std Fold Difference":
                    differences.std(ddof=1),
                "Min Fold Difference":
                    differences.min(),
                "Max Fold Difference":
                    differences.max(),
                "Coordinator Better Folds":
                    int((differences > 0).sum()),
                "Baseline Better Folds":
                    int((differences < 0).sum()),
                "Tied Folds":
                    int((differences == 0).sum())
            })

    return pd.DataFrame(rows)


# ============================================================
# 7. PERFORMANCE FIGURES
# ============================================================

def plot_metric_comparison(
    summary_df,
    metric,
    filename
):

    subset = summary_df[
        summary_df["Metric"] == metric
    ]

    plt.figure(figsize=(9, 5.5))

    plot_models = [
        "Stacking Coordinator",
        "HGB",
        "Random Forest",
        "Logistic Regression"
    ]

    for model in plot_models:

        d = subset[
            subset["Model"] == model
        ].sort_values("Horizon")

        x = d["Horizon"].to_numpy()

        y = d["OOF Estimate"].to_numpy()

        lower = d["CI Lower"].to_numpy()

        upper = d["CI Upper"].to_numpy()

        yerr = np.vstack([
            y - lower,
            upper - y
        ])

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=model
        )

    plt.xlabel("Course Completion Horizon (%)")

    plt.ylabel(metric)

    plt.title(
        f"{metric}: Performance with 95% "
        "Student-Clustered Bootstrap CIs"
    )

    plt.xticks(HORIZONS)

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 70)

    print("IEEE RESEARCH PAPER - STATISTICAL ANALYSIS")

    print("=" * 70)

    all_summaries = []

    all_comparisons = []

    all_fold_analyses = []

    for horizon in HORIZONS:

        print(
            f"\n========== HORIZON {horizon}% =========="
        )

        prediction_path = (
            RESULTS_DIR /
            f"oof_predictions_{horizon}.csv"
        )

        fold_path = (
            RESULTS_DIR /
            f"fold_metrics_{horizon}.csv"
        )

        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Missing file: {prediction_path}"
            )

        if not fold_path.exists():
            raise FileNotFoundError(
                f"Missing file: {fold_path}"
            )

        oof = pd.read_csv(
            prediction_path
        )

        folds = pd.read_csv(
            fold_path
        )

        validate_data(
            oof,
            horizon
        )

        summary, comparisons = bootstrap_analysis(
            oof,
            horizon
        )

        fold_analysis = paired_fold_analysis(
            folds,
            horizon
        )

        all_summaries.append(summary)

        all_comparisons.append(comparisons)

        all_fold_analyses.append(fold_analysis)

        print(
            f"Completed statistical analysis "
            f"for horizon {horizon}%"
        )

    # --------------------------------------------------------
    # Combine all results
    # --------------------------------------------------------

    summary_df = pd.concat(
        all_summaries,
        ignore_index=True
    )

    comparison_df = pd.concat(
        all_comparisons,
        ignore_index=True
    )

    fold_df = pd.concat(
        all_fold_analyses,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    summary_df.to_csv(
        OUTPUT_DIR / "bootstrap_confidence_intervals.csv",
        index=False
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "paired_model_comparisons.csv",
        index=False
    )

    fold_df.to_csv(
        OUTPUT_DIR / "paired_fold_analysis.csv",
        index=False
    )

    # --------------------------------------------------------
    # Create compact IEEE-style statistical table
    # --------------------------------------------------------

    ieee_rows = []

    for _, row in summary_df.iterrows():

        ieee_rows.append({
            "Horizon": f"{int(row['Horizon'])}%",
            "Model": row["Model"],
            "Metric": row["Metric"],
            "Estimate (95% CI)":
                f"{row['OOF Estimate']:.4f} "
                f"[{row['CI Lower']:.4f}, "
                f"{row['CI Upper']:.4f}]"
        })

    ieee_table = pd.DataFrame(
        ieee_rows
    )

    ieee_table.to_csv(
        OUTPUT_DIR / "ieee_statistical_table.csv",
        index=False
    )

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    plot_metric_comparison(
        summary_df,
        "F1",
        "f1_confidence_intervals.png"
    )

    plot_metric_comparison(
        summary_df,
        "ROC-AUC",
        "roc_auc_confidence_intervals.png"
    )

    # --------------------------------------------------------
    # Display important results
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("STATISTICAL ANALYSIS COMPLETED")

    print("=" * 70)

    print(
        "\nStacking Coordinator confidence intervals:"
    )

    coordinator_results = summary_df[
        (
            summary_df["Model"]
            == "Stacking Coordinator"
        )
        &
        (
            summary_df["Metric"]
            .isin(["F1", "ROC-AUC"])
        )
    ]

    print(
        coordinator_results[
            [
                "Horizon",
                "Metric",
                "OOF Estimate",
                "CI Lower",
                "CI Upper"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nCoordinator vs HGB comparisons:"
    )

    hgb_results = comparison_df[
        (
            comparison_df["Baseline"] == "HGB"
        )
        &
        (
            comparison_df["Metric"].isin(
                ["F1", "ROC-AUC"]
            )
        )
    ]

    print(
        hgb_results[
            [
                "Horizon",
                "Metric",
                "Difference",
                "CI Lower",
                "CI Upper",
                "CI Excludes Zero"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nResults saved in:"
    )

    print(OUTPUT_DIR)

    print("\nGenerated files:")

    print("1. bootstrap_confidence_intervals.csv")
    print("2. paired_model_comparisons.csv")
    print("3. paired_fold_analysis.csv")
    print("4. ieee_statistical_table.csv")
    print("5. figures/f1_confidence_intervals.png")
    print("6. figures/roc_auc_confidence_intervals.png")


if __name__ == "__main__":

    main()