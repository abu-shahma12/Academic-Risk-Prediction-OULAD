"""
Full specialist-agent ablation study.

Uses the corrected OULAD pipeline.

Each outer fold:
1. Fits specialist models on inner training folds.
2. Generates group-aware OOF specialist probabilities.
3. Trains separate fusion models for every agent combination.
4. Evaluates configurations on the untouched outer test fold.

All configurations share the same outer validation splits.
"""

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from coordinator_fusion import (
    StackingCoordinator,
    compute_metrics
)

from data_pipeline import (
    load_raw,
    build_horizon_dataset,
    get_cv_splits
)

from feature_engineering import feature_groups

from specialist_agents import (
    ModelConfig,
    AgentEnsemble,
    oof_agent_probs
)


SEED = 42

NAMES = [
    "Diagnostic",
    "Progress",
    "Context"
]


def all_combinations():

    configurations = []

    for size in (1, 2, 3):

        for indices in combinations(range(3), size):

            configurations.append(
                (
                    "+".join(NAMES[i] for i in indices),
                    list(indices)
                )
            )

    return configurations


def fusion_features(P):

    """
    Constructs fusion features from the
    selected specialist probabilities only.

    No protected demographic attributes
    or raw features are included.
    """

    P = np.asarray(P, dtype=float)

    features = pd.DataFrame(
        P,
        columns=[
            f"agent_{i}" for i in range(P.shape[1])
        ]
    )

    features["mean_probability"] = P.mean(axis=1)

    features["disagreement"] = P.std(axis=1)

    return features


def train_fusion(P_train, y_train, cfg):

    """
    Train an XGBoost coordinator using
    inner out-of-fold specialist predictions.
    """

    from xgboost import XGBClassifier

    X_train = fusion_features(P_train)

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(X_train)

    model = XGBClassifier(
        n_estimators=80 if cfg.fast else 200,
        max_depth=3,
        learning_rate=0.1 if cfg.fast else 0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, imputer


def evaluate_horizon(raw, horizon, cfg, output_dir):

    print(f"\n{'=' * 60}")
    print(f"ABLATION STUDY — {horizon * 100:.0f}%")
    print("=" * 60, flush=True)

    df = build_horizon_dataset(raw, horizon)

    y = df["y"].to_numpy()

    student_ids = df["id_student"].to_numpy()

    feature_sets = feature_groups(df)

    splits = get_cv_splits(
        df,
        n_splits=5,
        seed=SEED
    )

    results = []

    for fold_number, (train_idx, test_idx) in enumerate(
        splits,
        start=1
    ):

        start_time = time.time()

        print(
            f"\nTraining outer fold {fold_number}/5",
            flush=True
        )

        train = df.iloc[train_idx]

        test = df.iloc[test_idx]

        y_train = y[train_idx]

        y_test = y[test_idx]

        groups_train = student_ids[train_idx]

        # Generate inner OOF specialist probabilities.
        # No outer-test outcomes are used.

        P_train = oof_agent_probs(
            train,
            y_train,
            groups_train,
            feature_sets,
            cfg
        )

        # Train specialists on the complete outer
        # training fold for outer-test predictions.

        ensemble = AgentEnsemble(
            feature_sets,
            cfg
        )

        ensemble.fit(
            train,
            y_train,
            groups_train
        )

        P_test = ensemble.predict_proba(test)

        # Evaluate all seven agent combinations.

        for name, selected in all_combinations():

            p_train = P_train[:, selected]

            p_test = P_test[:, selected]

            fusion, imputer = train_fusion(
                p_train,
                y_train,
                cfg
            )

            X_test = fusion_features(p_test)

            X_test = imputer.transform(X_test)

            probabilities = fusion.predict_proba(
                X_test
            )[:, 1]

            metrics = compute_metrics(
                y_test,
                probabilities
            )

            results.append({
                "Horizon": int(horizon * 100),
                "Fold": fold_number,
                "Configuration": name,
                **metrics
            })

        # Mean-probability fusion baseline

        mean_probabilities = P_test.mean(axis=1)

        metrics = compute_metrics(
            y_test,
            mean_probabilities
        )

        results.append({
            "Horizon": int(horizon * 100),
            "Fold": fold_number,
            "Configuration": "MeanFusion_All",
            **metrics
        })

        # Existing full coordinator for reference.
        # This preserves the original meta-features.

        original = StackingCoordinator(cfg)

        original.fit(
            train,
            y_train,
            groups_train
        )

        original_probabilities = original.predict_proba(
            test
        )

        metrics = compute_metrics(
            y_test,
            original_probabilities
        )

        results.append({
            "Horizon": int(horizon * 100),
            "Fold": fold_number,
            "Configuration": "OriginalCoordinator",
            **metrics
        })

        print(
            f"Fold {fold_number} completed in "
            f"{time.time()-start_time:.1f} seconds",
            flush=True
        )

        # Save progress after each fold.

        pd.DataFrame(results).to_csv(
            output_dir /
            f"ablation_progress_{int(horizon*100)}.csv",
            index=False
        )

    return pd.DataFrame(results)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="dataset"
    )

    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=[0.15, 0.30, 0.45]
    )

    parser.add_argument(
        "--fast",
        action="store_true"
    )

    args = parser.parse_args()

    output_dir = Path("ablation_results")

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
        output_dir / "ablation_fold_metrics.csv",
        index=False
    )

    metrics = [
        "Accuracy",
        "F1",
        "ROC-AUC",
        "PR-AUC",
        "Brier"
    ]

    summary = (
        final.groupby(
            ["Horizon", "Configuration"]
        )[metrics]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        output_dir / "ablation_summary.csv"
    )

    print("\nABLATION STUDY COMPLETED")

    print("\nMean F1 and ROC-AUC:")

    print(
        final.groupby(
            ["Horizon", "Configuration"]
        )[["F1", "ROC-AUC"]]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nResults saved in ablation_results/")


if __name__ == "__main__":
    main()