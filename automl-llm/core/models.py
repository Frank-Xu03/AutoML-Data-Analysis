from __future__ import annotations
from typing import Dict, Tuple, Any
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

def get_models(task_type: str, names: list[str]) -> Dict[str, Dict[str, Any]]:
    """
    Return a dict: {name: {"estimator": est, "param_dist": dist}}
    """
    all_cls = {
        "logreg": {
            "estimator": LogisticRegression(max_iter=2000, n_jobs=None),
            "param_dist": {"C": loguniform(1e-3, 1e3), "penalty": ["l2"], "solver": ["lbfgs"]}
        },
        "rf": {
            "estimator": RandomForestClassifier(),
            "param_dist": {
                "n_estimators": randint(150, 600),
                "max_depth": randint(3, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", None]
            }
        },
        "xgb": {
            "estimator": XGBClassifier(
                n_estimators=300, tree_method="hist", eval_metric="logloss", use_label_encoder=False
            ),
            "param_dist": {
                "n_estimators": randint(150, 500),
                "max_depth": randint(3, 12),
                "learning_rate": loguniform(1e-2, 3e-1),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "reg_lambda": loguniform(1e-2, 10)
            }
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "param_dist": {"n_neighbors": randint(3, 51), "weights": ["uniform", "distance"], "p": [1, 2]}
        },
        "mlp": {
            "estimator": MLPClassifier(max_iter=500),
            "param_dist": {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "alpha": loguniform(1e-5, 1e-2),
                "learning_rate_init": loguniform(1e-4, 1e-2)
            }
        },
    }

    all_reg = {
        "linreg": {
            "estimator": LinearRegression(),
            "param_dist": {}  # no tuning
        },
        "ridge": {
            "estimator": Ridge(),
            "param_dist": {"alpha": loguniform(1e-3, 1e2)}
        },
        "rf": {
            "estimator": RandomForestRegressor(),
            "param_dist": {
                "n_estimators": randint(150, 600),
                "max_depth": randint(3, 20),
                "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", "log2", None]
            }
        },
        "xgb": {
            "estimator": XGBRegressor(n_estimators=300, tree_method="hist"),
            "param_dist": {
                "n_estimators": randint(150, 500),
                "max_depth": randint(3, 12),
                "learning_rate": loguniform(1e-2, 3e-1),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "reg_lambda": loguniform(1e-2, 10)
            }
        },
        "knn": {
            "estimator": KNeighborsRegressor(),
            "param_dist": {"n_neighbors": randint(3, 51), "weights": ["uniform", "distance"], "p": [1, 2]}
        },
        "mlp": {
            "estimator": MLPRegressor(max_iter=1000),
            "param_dist": {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "alpha": loguniform(1e-5, 1e-2),
                "learning_rate_init": loguniform(1e-4, 1e-2)
            }
        },
    }

    base = all_cls if task_type == "classification" else all_reg
    out = {}
    for n in names:
        if n in base:
            out[n] = base[n]
    if not out:
        # safe default
        out = {"rf": base["rf"], "xgb": base["xgb"]}
    return out
