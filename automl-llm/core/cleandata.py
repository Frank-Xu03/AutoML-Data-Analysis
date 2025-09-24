from __future__ import annotations
from typing import Dict, Tuple, Any, List
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Reuse column analyzer from ingest to keep logic consistent.
from core.ingest import analyze_columns

# ----------------------------- helpers -----------------------------

def _expand_datetime(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Expand datetime-like columns into multiple numeric features."""
    df = df.copy()
    for c in cols:
        s = pd.to_datetime(df[c], errors="coerce")
        df[f"{c}__year"] = s.dt.year
        df[f"{c}__month"] = s.dt.month
        df[f"{c}__day"] = s.dt.day
        df[f"{c}__dow"] = s.dt.dayofweek
        df[f"{c}__is_month_start"] = s.dt.is_month_start.astype("float")
        df[f"{c}__is_month_end"] = s.dt.is_month_end.astype("float")
    # drop raw datetime columns after expansion
    return df.drop(columns=cols, errors="ignore")

def _iqr_clip(df: pd.DataFrame, cols: List[str], whisker: float = 1.5) -> pd.DataFrame:
    """Clip numeric columns using IQR rule to reduce extreme outliers."""
    df = df.copy()
    for c in cols:
        s = df[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - whisker * iqr, q3 + whisker * iqr
        df[c] = s.clip(lower=low, upper=high)
    return df

def _drop_constant_and_high_missing(df: pd.DataFrame, high_missing_thresh: float = 0.98) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop all-NA columns, constant columns, and columns with too many missings."""
    info = {"dropped_all_na": [], "dropped_constant": [], "dropped_high_missing": []}
    df = df.copy()
    # all NA
    all_na = [c for c in df.columns if df[c].isna().all()]
    if all_na: 
        df = df.drop(columns=all_na); info["dropped_all_na"] = all_na
    # constant
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        df = df.drop(columns=const_cols); info["dropped_constant"] = const_cols
    # high missing
    miss = df.isna().mean()
    high_miss = [c for c in df.columns if miss[c] >= high_missing_thresh]
    if high_miss:
        df = df.drop(columns=high_miss); info["dropped_high_missing"] = high_miss
    return df, info

def _leakage_guard(df: pd.DataFrame, target: str, task_type: str, name_based: bool = True, corr_based: bool = True, corr_thr: float = 0.98) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Crude leakage guard: drop columns that look like target leaks."""
    info = {"dropped_name_suspects": [], "dropped_corr_suspects": []}
    X = df.drop(columns=[target])
    y = df[target]

    suspects = []
    if name_based:
        pat = re.compile(rf"{re.escape(target)}|label|target|outcome|answer", re.I)
        suspects += [c for c in X.columns if pat.search(c)]

    # correlation-based only for numeric columns
    if corr_based:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        try:
            # for classification, coerce y to numeric labels
            yy = y
            if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
                yy = pd.factorize(y.astype(str))[0]
            corr = pd.DataFrame(X[num_cols]).corrwith(pd.Series(yy))
            corr_sus = list(corr[abs(corr) >= corr_thr].index)
            suspects += corr_sus
            info["dropped_corr_suspects"] = corr_sus
        except Exception:
            pass

    suspects = sorted(set(suspects))
    if suspects:
        info["dropped_name_suspects"] = [c for c in suspects if c in X.columns]
        df = df.drop(columns=suspects, errors="ignore")
    return df, info

def _class_imbalance_stats(y: np.ndarray) -> Dict[str, Any]:
    vals, cnts = np.unique(y, return_counts=True)
    total = cnts.sum()
    maj = cnts.max()
    ratio = float(maj / total) if total else 0.0
    return {"is_imbalanced": bool(ratio >= 0.8), "majority_ratio": ratio, "labels": vals.tolist(), "counts": cnts.tolist()}

# ----------------------------- main -----------------------------

def prepare(
    df: pd.DataFrame,
    target: str,
    task_type: str,  # "classification" | "regression"
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    enable_datetime_expand: bool = True,
    enable_iqr_clip: bool = False,
    iqr_whisker: float = 1.5,
    drop_high_missing_thresh: float = 0.98,
    enable_leakage_guard: bool = True,
    rare_category_min_freq: float = 0.01,  # OneHot 内置 min_frequency
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, Dict[str, Any]]:
    """
    Split data, clean, build preprocessing, and encode target if classification.
    Returns: X_train, X_test, y_train, y_test, preprocessor, col_info
    """
    df = df.copy()
    assert target in df.columns, f"Target '{target}' not found."

    # 0) Drop missing target
    df = df.loc[~df[target].isna()].reset_index(drop=True)

    # 1) Optional leakage guard BEFORE split
    leak_info = {}
    if enable_leakage_guard:
        df, leak_info = _leakage_guard(df, target, task_type, name_based=True, corr_based=True, corr_thr=0.98)

    # 2) Optional datetime expand on features (not target)
    X_all = df.drop(columns=[target])
    y_raw = df[target]
    col_groups_all = analyze_columns(X_all)
    if enable_datetime_expand and col_groups_all["date_cols"]:
        X_all = _expand_datetime(X_all, col_groups_all["date_cols"])

    # 3) Optional drop constant / high-missing cols
    X_all, drop_info = _drop_constant_and_high_missing(X_all, high_missing_thresh=drop_high_missing_thresh)

    # 4) Optional IQR clip (numeric only)
    col_groups_all = analyze_columns(X_all)
    if enable_iqr_clip and col_groups_all["num_cols"]:
        X_all = _iqr_clip(X_all, col_groups_all["num_cols"], whisker=iqr_whisker)

    # 5) Split
    # Target encoding for classification
    target_encoder = None
    if task_type == "classification":
        if not pd.api.types.is_integer_dtype(y_raw) and not pd.api.types.is_bool_dtype(y_raw) and not pd.api.types.is_float_dtype(y_raw):
            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
            target_encoder = le
        else:
            y = y_raw.values
    else:
        # For regression, try to convert to float
        # If conversion fails, it might be a misclassified task
        try:
            y = y_raw.astype(float).values
        except (ValueError, TypeError) as e:
            # If target contains non-numeric values, suggest it might be classification
            unique_values = y_raw.unique()
            non_numeric = [v for v in unique_values if pd.isna(v) or (isinstance(v, str) and not str(v).replace('.', '').replace('-', '').isdigit())]
            
            if non_numeric:
                raise ValueError(
                    f"无法将目标变量转换为数值类型进行回归分析。"
                    f"发现非数值值: {non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}。"
                    f"请检查:\n"
                    f"1. 目标列是否选择正确\n"
                    f"2. 任务类型是否应该设置为 'classification' 而不是 'regression'\n"
                    f"3. 数据是否需要清洗"
                ) from e
            else:
                raise e

    # 对于分类任务，检查是否可以进行分层采样
    stratify_arg = None
    if task_type == "classification":
        # 检查每个类别是否至少有2个样本
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        num_classes = len(unique)
        total_samples = len(y)
        test_samples = int(total_samples * test_size)
        
        # 检查测试集是否足够大来包含所有类别
        if min_class_count >= 2 and test_samples >= num_classes:
            stratify_arg = y
        else:
            # 给出详细的警告信息
            if min_class_count < 2:
                single_sample_classes = unique[counts == 1]
                print(f"警告: 发现 {len(single_sample_classes)} 个类别只有1个样本，跳过分层采样")
            elif test_samples < num_classes:
                print(f"警告: 测试集样本数 ({test_samples}) 小于类别数 ({num_classes})，跳过分层采样")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    # 6) Preprocessor on TRAIN schema
    col_groups = analyze_columns(X_train)
    num_cols = col_groups["num_cols"]
    cat_cols = col_groups["cat_cols"]

    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore", min_frequency=rare_category_min_freq))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

    # 7) Imbalance stats (classification only)
    imb_stats = _class_imbalance_stats(y) if task_type == "classification" else None

    col_info = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target": target,
        "task_type": task_type,
        "target_encoder_present": target_encoder is not None,
        "target_classes": (list(target_encoder.classes_) if target_encoder else None),
        "datetime_expanded": enable_datetime_expand,
        "iqr_clip": enable_iqr_clip,
        "rare_category_min_freq": rare_category_min_freq,
        "drop_info": drop_info,
        "leakage_guard": leak_info,
        "imbalance_stats": imb_stats
    }
    if target_encoder is not None:
        col_info["target_encoder"] = target_encoder

    return X_train, X_test, y_train, y_test, preprocessor, col_info
