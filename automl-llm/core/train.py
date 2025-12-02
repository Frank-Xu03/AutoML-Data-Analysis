from __future__ import annotations
from typing import Dict, Any, Tuple
import time, os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
import joblib

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

        # Compute test metrics
        test_metrics: Dict[str, Any] = {}
        if task_type == "classification":
            # 支持二分类/多分类的通用指标
            try:
                test_metrics["acc"] = float(accuracy_score(y_test, y_pred))
                # 宏平均 F1 适用于类别不平衡
                test_metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))
                test_metrics["precision_macro"] = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
                test_metrics["recall_macro"] = float(recall_score(y_test, y_pred, average="macro"))
            except Exception:
                pass
            # 若可获得 predict_proba，计算 ROC-AUC（仅二分类/多标签需特别处理，这里以二分类为主）
            try:
                if hasattr(best, "predict_proba"):
                    proba = best.predict_proba(X_test)
                    if proba.ndim == 2 and proba.shape[1] == 2:
                        test_metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
            except Exception:
                pass
        else:
            try:
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))
                r2 = float(r2_score(y_test, y_pred))
                test_metrics.update({"rmse": rmse, "mae": mae, "r2": r2})
            except Exception:
                pass

        # Persist best model
        model_path = os.path.join(artifacts_dir, f"best_model__{name}.pkl")
        try:
            joblib.dump(best, model_path)
        except Exception:
            model_path = None

        # Pack results for later consumption
        artifacts[name] = {
            "best_estimator": best,
            "cv_best_score": float(search.best_score_),
            "cv_params": search.best_params_,
            "fit_time_s": fit_s,
            "predict_time_s": pred_s,
            "y_pred": y_pred,
            "test_metrics": test_metrics,
            "model_path": model_path,
        }

        row = {
            "model": name,
            "cv_score(primary)": float(search.best_score_),
            "fit_s": round(fit_s, 3),
            "predict_s": round(pred_s, 6),
            "params": search.best_params_,
        }
        # Merge key test metrics into leaderboard row (compact)
        if task_type == "classification":
            for k in ["acc", "f1_macro", "roc_auc"]:
                if k in test_metrics:
                    row[k] = round(float(test_metrics[k]), 6)
        else:
            for k in ["rmse", "mae", "r2"]:
                if k in test_metrics:
                    row[k] = round(float(test_metrics[k]), 6)

        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values("cv_score(primary)", ascending=(task_type!="classification"))
    # For regression, neg RMSE higher is better; we can leave sorted descending by default for clarity:
    leaderboard = leaderboard.sort_values("cv_score(primary)", ascending=False).reset_index(drop=True)

    # Save leaderboard
    lb_path = os.path.join(artifacts_dir, "leaderboard.csv")
    leaderboard.to_csv(lb_path, index=False)

    return leaderboard, artifacts
