from __future__ import annotations
from typing import Dict, Any, Tuple
import time, os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import get_scorer

from core.models import get_models

def build_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline([("pre", preprocessor), ("est", estimator)])

def get_cv(task_type: str, folds: int = 5, stratified: bool = True):
    if task_type == "classification" and stratified:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return KFold(n_splits=folds, shuffle=True, random_state=42)

def fit_one(
    model_name: str,
    estimator,
    param_dist: Dict[str, Any],
    X_train, y_train,
    cv, scoring: str,
    n_iter: int = 30,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict[str, Any]]:
    pipe = Pipeline([("pre", None), ("est", estimator)])  # pre set at outer build
    # NOTE: We'll build the full pipeline outside to avoid re-fitting the preprocessor in search.
    # Simpler approach: include preprocessor here; it's fine for clarity:
    from sklearn.compose import ColumnTransformer
    # but we expect caller to pass a pipeline already built with preprocessor.
    raise RuntimeError("fit_one is called on full pipeline in run_all; do not call directly.")

def run_all(
    X_train, y_train, X_test, y_test,
    task_type: str,
    picked_models: list[str],
    preprocessor,
    n_iter: int = 30,
    cv_folds: int = 5,
    primary_scoring: str | None = None,
    artifacts_dir: str = "artifacts"
):
    """
    Train all picked models with RandomizedSearchCV and return a leaderboard and artifacts.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    if primary_scoring is None:
        primary_scoring = "f1_macro" if task_type == "classification" else "neg_root_mean_squared_error"

    cv = get_cv(task_type, folds=cv_folds, stratified=True)

    models = get_models(task_type, picked_models)

    rows = []
    artifacts = {}

    for name, cfg in models.items():
        est = cfg["estimator"]
        param_dist = cfg["param_dist"]

        pipe = Pipeline([("pre", preprocessor), ("est", est)])

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions={f"est__{k}": v for k, v in param_dist.items()},
            n_iter=n_iter,
            scoring=primary_scoring,
            n_jobs=-1,
            cv=cv,
            random_state=42,
            verbose=0,
            refit=True,
        )

        t0 = time.time()
        search.fit(X_train, y_train)
        fit_s = time.time() - t0

        best = search.best_estimator_
        # predict time
        t1 = time.time()
        y_pred = best.predict(X_test)
        pred_s = time.time() - t1

        # Pack results for later evaluation (we'll compute full metric set in evaluate.py)
        artifacts[name] = {
            "best_estimator": best,
            "cv_best_score": float(search.best_score_),
            "cv_params": search.best_params_,
            "fit_time_s": fit_s,
            "predict_time_s": pred_s,
            "y_pred": y_pred,  # temporary (evaluate will recompute if needed)
        }

        rows.append({
            "model": name,
            "cv_score(primary)": float(search.best_score_),
            "fit_s": round(fit_s, 3),
            "predict_s": round(pred_s, 6),
            "params": search.best_params_,
        })

    leaderboard = pd.DataFrame(rows).sort_values("cv_score(primary)", ascending=(task_type!="classification"))
    # For regression, neg RMSE higher is better; we can leave sorted descending by default for clarity:
    leaderboard = leaderboard.sort_values("cv_score(primary)", ascending=False).reset_index(drop=True)

    # Save leaderboard
    lb_path = os.path.join(artifacts_dir, "leaderboard.csv")
    leaderboard.to_csv(lb_path, index=False)

    return leaderboard, artifacts
