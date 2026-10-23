"""
Nested fairness-utility evaluation for OULAD.

For each outer fold:

1. Generate group-aware OOF coordinator probabilities
   inside the outer training data.

2. Select fairness thresholds using only those
   training-side predictions and outcomes.

3. Fit a coordinator using the outer training data.

4. Evaluate fixed thresholds on the untouched
   outer test fold.

Gender and disability are evaluated separately.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from coordinator_fusion import StackingCoordinator

from data_pipeline import (
    load_raw,
    build_horizon_dataset,
    get_cv_splits
)

from fairness_mitigation import (
    subgroup_audit,
    gap_summary
)

from specialist_agents import ModelConfig


SEED = 42

GRID = np.round(
    np.arange(0.05, 0.951, 0.02),
    2
)

MAX_ACCURACY_LOSS = 0.02


def fairness_gap(y, predictions, sensitive):

    audit = subgroup_audit(
        y,
        predictions,
        sensitive
    )

    gaps = gap_summary(audit)

    return max(
        gaps["FPR gap"],
        gaps["TPR gap"]
    )


def select_thresholds(
    y,
    probabilities,
    sensitive,
    max_accuracy_loss=0.02
):

    """
    Select thresholds on training-side
    out-of-fold probabilities only.

    Minimize empirical EO gap while requiring
    accuracy within a fixed tolerance of
    the unmitigated threshold.
    """

    y = np.asarray(y)

    probabilities = np.asarray(probabilities)

    sensitive = np.asarray(sensitive)

    levels = sorted(pd.unique(sensitive))

    if len(levels) != 2:
        raise ValueError(
            "This implementation requires exactly two groups"
        )

    baseline = (
        probabilities >= 0.5
    ).astype(int)

    baseline_accuracy = accuracy_score(
        y,
        baseline
    )

    best = None
    best_score = None

    for theta_a in GRID:

        for theta_b in GRID:

            thresholds = {
                levels[0]: float(theta_a),
                levels[1]: float(theta_b)
            }

            threshold_vector = np.where(
                sensitive == levels[0],
                theta_a,
                theta_b
            )

            predictions = (
                probabilities >= threshold_vector
            ).astype(int)

            accuracy = accuracy_score(
                y,
                predictions
            )

            if accuracy < (
                baseline_accuracy - max_accuracy_loss
            ):
                continue

            gap = fairness_gap(
                y,
                predictions,
                sensitive
            )

            f1 = f1_score(
                y,
                predictions,
                zero_division=0
            )

            # Primary objective: minimum EO gap.
            # Secondary objective: maximum F1.
            # Tertiary objective: higher accuracy.

            score = (
                gap,
                -f1,
                -accuracy
            )

            if best_score is None or score < best_score:

                best_score = score

                best = thresholds

    # Threshold 0.5 for both groups is always
    # an available feasible fallback.

    if best is None:

        best = {
            levels[0]: 0.5,
            levels[1]: 0.5
        }

    return best


def apply_thresholds(
    probabilities,
    sensitive,
    thresholds
):

    values = np.asarray([
        thresholds[group]
        for group in sensitive
    ])

    return (
        np.asarray(probabilities) >= values
    ).astype(int)


def inner_coordinator_oof(
    train,
    cfg,
    n_inner_folds=3
):

    """
    Produce OOF coordinator probabilities
    using students only from the outer
    training fold.
    """

    y = train["y"].to_numpy()

    groups = train["id_student"].to_numpy()

    predictions = np.zeros(
        len(train),
        dtype=float
    )

    splits = get_cv_splits(
        train,
        n_splits=n_inner_folds,
        seed=SEED + 100
    )

    for inner_train, inner_valid in splits:

        model = StackingCoordinator(cfg)

        model.fit(
            train.iloc[inner_train],
            y[inner_train],
            groups[inner_train]
        )

        predictions[inner_valid] = (
            model.predict_proba(
                train.iloc[inner_valid]
            )
        )

    return predictions


def evaluate_horizon(
    raw,
    horizon,
    cfg,
    output_dir
):

    print(
        f"\nFAIRNESS STUDY — {horizon * 100:.0f}%",
        flush=True
    )

    df = build_horizon_dataset(raw, horizon)

    splits = get_cv_splits(
        df,
        n_splits=5,
        seed=SEED
    )

    rows = []

    for fold_number, (tr, te) in enumerate(
        splits,
        start=1
    ):

        start_time = time.time()

        train = df.iloc[tr]

        test = df.iloc[te]

        y_train = train["y"].to_numpy()

        y_test = test["y"].to_numpy()

        groups_train = (
            train["id_student"].to_numpy()
        )

        print(
            f"Outer fold {fold_number}/5",
            flush=True
        )

        # Training-side OOF calibration probabilities

        p_calibration = inner_coordinator_oof(
            train,
            cfg
        )

        # Fit final outer model

        model = StackingCoordinator(cfg)

        model.fit(
            train,
            y_train,
            groups_train
        )

        # Untouched outer-test probabilities

        p_test = model.predict_proba(test)

        baseline_predictions = (
            p_test >= 0.5
        ).astype(int)

        for attribute in [
            "gender",
            "disability"
        ]:

            s_train = train[attribute].to_numpy()

            s_test = test[attribute].to_numpy()

            thresholds = select_thresholds(
                y_train,
                p_calibration,
                s_train,
                MAX_ACCURACY_LOSS
            )

            adjusted_predictions = apply_thresholds(
                p_test,
                s_test,
                thresholds
            )

            pre_audit = subgroup_audit(
                y_test,
                baseline_predictions,
                s_test,
                attribute
            )

            post_audit = subgroup_audit(
                y_test,
                adjusted_predictions,
                s_test,
                attribute
            )

            pre_gaps = gap_summary(pre_audit)

            post_gaps = gap_summary(post_audit)

            rows.append({
                "Horizon": int(horizon * 100),
                "Fold": fold_number,
                "Attribute": attribute,

                "Pre Accuracy": accuracy_score(
                    y_test,
                    baseline_predictions
                ),

                "Post Accuracy": accuracy_score(
                    y_test,
                    adjusted_predictions
                ),

                "Pre F1": f1_score(
                    y_test,
                    baseline_predictions
                ),

                "Post F1": f1_score(
                    y_test,
                    adjusted_predictions
                ),

                "Pre FPR Gap":
                    pre_gaps["FPR gap"],

                "Post FPR Gap":
                    post_gaps["FPR gap"],

                "Pre TPR Gap":
                    pre_gaps["TPR gap"],

                "Post TPR Gap":
                    post_gaps["TPR gap"],

                "Pre EO Gap": max(
                    pre_gaps["FPR gap"],
                    pre_gaps["TPR gap"]
                ),

                "Post EO Gap": max(
                    post_gaps["FPR gap"],
                    post_gaps["TPR gap"]
                ),

                "Thresholds": str(thresholds)
            })

        pd.DataFrame(rows).to_csv(
            output_dir /
            f"fairness_progress_{int(horizon*100)}.csv",
            index=False
        )

        print(
            f"Fold completed in "
            f"{time.time()-start_time:.1f}s",
            flush=True
        )

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="dataset"
    )

    parser.add_argument(
        "--horizons",
        nargs="+",
        type=float,
        default=[0.15, 0.30, 0.45]
    )

    parser.add_argument(
        "--fast",
        action="store_true"
    )

    args = parser.parse_args()

    output_dir = Path(
        "fairness_nested_results"
    )

    output_dir.mkdir(exist_ok=True)

    raw = load_raw(args.data_dir)

    cfg = ModelConfig(
        fast=args.fast,
        seed=SEED
    )

    all_results = []

    for horizon in args.horizons:

        result = evaluate_horizon(
            raw,
            horizon,
            cfg,
            output_dir
        )

        all_results.append(result)

    final = pd.concat(
        all_results,
        ignore_index=True
    )

    final.to_csv(
        output_dir / "nested_fairness_fold_metrics.csv",
        index=False
    )

    numeric_columns = [
        "Pre Accuracy",
        "Post Accuracy",
        "Pre F1",
        "Post F1",
        "Pre FPR Gap",
        "Post FPR Gap",
        "Pre TPR Gap",
        "Post TPR Gap",
        "Pre EO Gap",
        "Post EO Gap"
    ]

    summary = (
        final.groupby(
            ["Horizon", "Attribute"]
        )[numeric_columns]
        .mean()
        .reset_index()
    )

    summary.to_csv(
        output_dir / "nested_fairness_summary.csv",
        index=False
    )

    print("\nNESTED FAIRNESS STUDY COMPLETED")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()