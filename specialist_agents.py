"""
specialist_agents.py
====================
Three calibrated specialist agents:
    Agent 1 Diagnostic       - XGBoost            on Group A (academic state)
    Agent 2 Progress Monitor - LightGBM/CatBoost  on Group B (VLE / behaviour)
    Agent 3 Contextual Risk  - Calibrated RF / LR on Group C (+ module prior)

Each agent:  [module prior] -> median imputer (train-fold only) -> [scaler]
             -> estimator with group-aware Platt/isotonic calibration.
Out-of-fold probability vectors [p_diag, p_prog, p_context] come from
`oof_agent_probs` (StratifiedGroupKFold by id_student).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import ModulePriorTransformer

AGENT_NAMES = ["p_diag", "p_prog", "p_context"]


@dataclass
class ModelConfig:
    fast: bool = False
    seed: int = 42
    context_model: str = "rf"          # "rf" (calibrated Random Forest) or "lr"
    calibration: str = "sigmoid"       # "sigmoid" (Platt) or "isotonic"

    @property
    def n_trees(self): return 80 if self.fast else 300
    @property
    def lr(self): return 0.1 if self.fast else 0.05
    @property
    def inner_splits(self): return 3 if self.fast else 5


# ---------------------------------------------------------------- estimators
def make_diagnostic_estimator(cfg: ModelConfig):
    from xgboost import XGBClassifier
    return lambda: XGBClassifier(n_estimators=cfg.n_trees, max_depth=4, learning_rate=cfg.lr,
                                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                 eval_metric="logloss", tree_method="hist",
                                 random_state=cfg.seed, n_jobs=-1)


def make_progress_estimator(cfg: ModelConfig):
    try:
        from lightgbm import LGBMClassifier
        return lambda: LGBMClassifier(n_estimators=cfg.n_trees, learning_rate=cfg.lr,
                                      num_leaves=31, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, min_child_samples=30,
                                      random_state=cfg.seed, n_jobs=-1, verbose=-1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        return lambda: CatBoostClassifier(iterations=cfg.n_trees, learning_rate=cfg.lr, depth=6,
                                          random_seed=cfg.seed, verbose=0, thread_count=-1)
    except ImportError:
        return lambda: HistGradientBoostingClassifier(max_iter=cfg.n_trees, learning_rate=cfg.lr,
                                                      random_state=cfg.seed)


def make_context_estimator(cfg: ModelConfig):
    if cfg.context_model == "lr":
        return lambda: LogisticRegression(max_iter=2000, C=1.0)
    return lambda: RandomForestClassifier(n_estimators=200 if cfg.fast else 400,
                                          min_samples_leaf=25, max_features="sqrt",
                                          random_state=cfg.seed, n_jobs=-1)


# --------------------------------------------------------------- calibration
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class GroupCalibratedClassifier:
    """
    Group-aware probability calibration.

    Each internal training fold fits its own
    preprocessing pipeline.

    This prevents preprocessing information from
    leaking into the internal validation fold.
    """

    def __init__(
        self,
        make_estimator,
        make_preprocessor,
        method="sigmoid",
        n_splits=3,
        seed=42
    ):
        self.make_estimator = make_estimator
        self.make_preprocessor = make_preprocessor
        self.method = method
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X, y, groups):

        X = X.reset_index(drop=True).copy()

        y = np.asarray(y)
        groups = np.asarray(groups)

        oof = np.zeros(len(y), dtype=float)

        skf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed
        )

        for tr, va in skf.split(X, y, groups):

            # Create a NEW preprocessor for this fold.
            preprocessor = self.make_preprocessor()

            X_train = preprocessor.fit_transform(
                X.iloc[tr],
                y[tr]
            )

            X_valid = preprocessor.transform(
                X.iloc[va]
            )

            # Train only on inner-training students.
            model = self.make_estimator()

            model.fit(
                X_train,
                y[tr]
            )

            # Predict the held-out students.
            oof[va] = model.predict_proba(
                X_valid
            )[:, 1]

        # Fit calibration using only internal
        # out-of-fold predictions.
        if self.method == "isotonic":

            self.cal_ = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0,
                y_max=1
            )

            self.cal_.fit(oof, y)

        elif self.method == "sigmoid":

            self.cal_ = LogisticRegression(
                C=1e6,
                max_iter=1000
            )

            self.cal_.fit(
                _logit(oof)[:, None],
                y
            )

        else:
            raise ValueError(
                "Calibration must be sigmoid or isotonic"
            )

        # Refit preprocessing and base model
        # on the complete outer-training data.
        self.pre_ = self.make_preprocessor()

        X_full = self.pre_.fit_transform(
            X,
            y
        )

        self.est_ = self.make_estimator()

        self.est_.fit(
            X_full,
            y
        )

        return self

    def predict_proba1(self, X):

        X = X.reset_index(drop=True)

        X_processed = self.pre_.transform(X)

        raw = self.est_.predict_proba(
            X_processed
        )[:, 1]

        if self.method == "isotonic":

            return self.cal_.predict(raw)

        return self.cal_.predict_proba(
            _logit(raw)[:, None]
        )[:, 1]

# -------------------------------------------------------------------- agents
class SpecialistAgent:

    def __init__(
        self,
        name,
        columns,
        make_estimator,
        cfg,
        use_prior=False,
        scale=False,
        calibrate=True
    ):

        self.name = name

        self.columns = list(columns)

        self.make_estimator = make_estimator

        self.cfg = cfg

        self.use_prior = use_prior

        self.scale = scale

        self.calibrate = calibrate

    def _preprocessor(self):

        steps = []

        if self.use_prior:

            steps.append(
                ("prior", ModulePriorTransformer())
            )

        steps.append(
            (
                "imp",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            )
        )

        if self.scale:

            steps.append(
                ("scale", StandardScaler())
            )

        return Pipeline(steps)

    def fit(self, df, y, groups):

        cols = list(self.columns)

        if self.use_prior:
            cols.append("code_module")

        X = df[cols].copy()

        y = np.asarray(y)

        groups = np.asarray(groups)

        if self.calibrate:

            self.model_ = GroupCalibratedClassifier(
                make_estimator=self.make_estimator,
                make_preprocessor=self._preprocessor,
                method=self.cfg.calibration,
                n_splits=3,
                seed=self.cfg.seed
            )

            self.model_.fit(
                X,
                y,
                groups
            )

        else:

            self.pre_ = self._preprocessor()

            X_processed = self.pre_.fit_transform(
                X,
                y
            )

            self.model_ = self.make_estimator()

            self.model_.fit(
                X_processed,
                y
            )

        return self

    def predict_proba(self, df):

        cols = list(self.columns)

        if self.use_prior:
            cols.append("code_module")

        X = df[cols].copy()

        if self.calibrate:

            return self.model_.predict_proba1(X)

        X_processed = self.pre_.transform(X)

        return self.model_.predict_proba(
            X_processed
        )[:, 1]


class AgentEnsemble:
    """Agents 1-3 -> probability matrix [p_diag, p_prog, p_context]."""

    def __init__(self, groups: dict, cfg: ModelConfig):
        lr_ctx = cfg.context_model == "lr"
        self.agents = [
            SpecialistAgent("Diagnostic (XGBoost)", groups["diagnostic"],
                            make_diagnostic_estimator(cfg), cfg),
            SpecialistAgent("Progress Monitor (LightGBM)", groups["progress"],
                            make_progress_estimator(cfg), cfg),
            SpecialistAgent("Contextual Risk (" + ("LR" if lr_ctx else "Calibrated RF") + ")",
                            groups["context"], make_context_estimator(cfg), cfg,
                            use_prior=True, scale=lr_ctx, calibrate=not lr_ctx),
        ]

    def fit(self, df, y, groups):
        for a in self.agents:
            a.fit(df, y, groups)
        return self

    def predict_proba(self, df) -> np.ndarray:
        return np.column_stack([a.predict_proba(df) for a in self.agents])


def oof_agent_probs(df: pd.DataFrame, y, groups, feat_groups: dict, cfg: ModelConfig,
                    n_splits=None) -> np.ndarray:
    """Calibrated out-of-fold specialist probabilities (group-aware inner CV)."""
    n_splits = n_splits or cfg.inner_splits
    P = np.zeros((len(df), 3))
    skf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=cfg.seed)
    for tr, va in skf.split(df, y, groups):
        ens = AgentEnsemble(feat_groups, cfg).fit(df.iloc[tr], y[tr], groups[tr])
        P[va] = ens.predict_proba(df.iloc[va])
    return P
