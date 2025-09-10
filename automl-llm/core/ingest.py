from __future__ import annotations
import io, re, json
from typing import Any, Dict, Iterable, Tuple, Union
import pandas as pd
import numpy as np

# ------------ 读取 & 抽样 ------------

def read_table(path_or_file: Union[str, io.BytesIO, io.StringIO],
               max_rows: int | None = None) -> pd.DataFrame:
    """
    读取 CSV / Parquet；自动识别；可限制最大行数。
    """
    if hasattr(path_or_file, "name"):
        name = getattr(path_or_file, "name", "")
    else:
        name = str(path_or_file)

    if name.lower().endswith(".parquet"):
        df = pd.read_parquet(path_or_file)  # 需要 pyarrow/fastparquet 时再装
    else:
        df = pd.read_csv(path_or_file)
    if max_rows:
        df = df.head(max_rows)
    return df

def sample_df(df: pd.DataFrame, n: int = 5000, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state)

# ------------ 列类型识别 ------------

def analyze_columns(df: pd.DataFrame) -> Dict[str, list]:
    num_cols, cat_cols, date_cols = [], [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            # 尝试将可解析日期的字符串判为日期
            if s.dropna().astype(str).head(50).map(_looks_like_datetime).mean() > 0.8:
                date_cols.append(c)
            else:
                cat_cols.append(c)
    return {"num_cols": num_cols, "cat_cols": cat_cols, "date_cols": date_cols}

def _looks_like_datetime(x: str) -> bool:
    try:
        pd.to_datetime(x)
        return True
    except Exception:
        return False

# ------------ PII 脱敏（只示例：邮件/电话/身份证样式） ------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")
_ID_RE = re.compile(r"\b\d{15,18}[0-9Xx]?\b")

def mask_pii_values(df: pd.DataFrame, max_rows: int = 5) -> pd.DataFrame:
    """
    返回一份最多 max_rows 的样例，并将疑似 PII 的字符串替换为 ***
    仅用于发给 LLM 的“示例几行”展示。
    """
    sample = df.head(max_rows).copy()
    for c in sample.columns:
        if pd.api.types.is_string_dtype(sample[c]):
            sample[c] = (sample[c].astype(str)
                         .str.replace(_EMAIL_RE, "***", regex=True)
                         .str.replace(_PHONE_RE, "***", regex=True)
                         .str.replace(_ID_RE, "***", regex=True))
    return sample

# ------------ 概要画像（给 LLM 的摘要） ------------

def profile(df: pd.DataFrame, example_rows: int = 5) -> Dict[str, Any]:
    """
    只返回：列名、dtype、缺失率、唯一值数、描述统计（数值/类别），以及脱敏后的少量样例行。
    不包含完整数据。
    """
    df = df.copy()
    # 尝试解析日期
    for c in df.columns:
        if df[c].dtype == object:
            # 轻量探测
            try:
                parsed = pd.to_datetime(df[c], errors="raise", utc=False, format=None)
                # 如果可解析且有意义，替换
                if parsed.notna().mean() > 0.9:
                    df[c] = parsed
            except Exception:
                pass

    col_types = analyze_columns(df)

    # 缺失与唯一值
    missing = (df.isna().mean() * 100).round(2).to_dict()
    nunique = df.nunique(dropna=True).to_dict()

    # 数值&类别描述
    num_desc = df[col_types["num_cols"]].describe(percentiles=[.05,.25,.5,.75,.95]).T.round(4).to_dict(orient="index") \
                if col_types["num_cols"] else {}
    cat_desc = {}
    for c in col_types["cat_cols"]:
        vc = df[c].value_counts(dropna=True).head(10)
        cat_desc[c] = {"top_values": vc.to_dict(), "nunique": int(df[c].nunique())}

    # 样例行（脱敏）
    sample_rows = mask_pii_values(df, max_rows=example_rows).astype(str).to_dict(orient="records")

    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": [
            {"name": c, "dtype": str(df[c].dtype), "missing_pct": float(missing.get(c, 0.0)),
             "nunique": int(nunique.get(c, 0))}
            for c in df.columns
        ],
        "column_groups": col_types,
        "numeric_summary": num_desc,
        "categorical_summary": cat_desc,
        "examples_sanitized": sample_rows
    }

# 小型自测
if __name__ == "__main__":
    dff = read_table("examples/titanic_small.csv")
    prof = profile(dff)
    print(json.dumps(prof, ensure_ascii=False, indent=2)[:1500], "...\nOK")
