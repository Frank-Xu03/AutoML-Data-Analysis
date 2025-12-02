
from __future__ import annotations
import os, json
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError, model_validator

# OpenAI 官方 SDK（支持自定义 base_url，亦可兼容各类 OpenAI-Compatible 服务）
from openai import OpenAI

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    # 从项目根目录加载 .env 文件
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
except ImportError:
    # 如果没有安装 python-dotenv，给出提示
    print("提示: 安装 python-dotenv 可以自动加载 .env 文件中的环境变量")

# -------- LLM 配置（可通过环境变量覆盖） --------
# 说明：
# - 支持通过 OPENAI_BASE_URL 指向 OpenAI 兼容的推理服务（如本地代理/第三方兼容服务）
# - 可设置 LLM_OFFLINE=1 关闭外网调用，仅使用本地启发式回退
# - 可分别设置两个任务的模型与温度
LLM_OFFLINE = os.getenv("LLM_OFFLINE", "0") == "1"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # e.g. https://api.openai.com/v1 或第三方兼容地址

DEFAULT_TASK_MODEL = os.getenv("LLM_TASK_MODEL", "gpt-4o-mini")
DEFAULT_RESEARCH_MODEL = os.getenv("LLM_RESEARCH_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

ALGO_WHITELIST = {
    "logistic_regression","random_forest","xgboost","knn","mlp",
    "linear_regression","ridge","lasso","svm","kmeans","gmm","agglomerative"
}
METRIC_WHITELIST = {
    # classification
    "accuracy","f1_macro","f1_weighted","roc_auc_ovr","roc_auc","precision","recall","log_loss",
    # regression
    "rmse","mae","r2","mape",
    # clustering
    "silhouette","calinski_harabasz","davies_bouldin"
}

class CVSpec(BaseModel):
    folds: int = 5
    stratified: bool = True

class Plan(BaseModel):
    task_type: str = Field(pattern="^(classification|regression|clustering)$")
    target_candidates: List[str] = Field(default_factory=list)
    imbalance: Dict[str, Optional[float] | Optional[bool]] = Field(default_factory=lambda: {"is_imbalanced": None, "ratio": None})
    algorithms: List[str]
    metrics: List[str]
    cv: CVSpec

    @model_validator(mode="after")
    def clamp_whitelists(self):
        self.algorithms = [a for a in self.algorithms if a in ALGO_WHITELIST][:5]
        self.metrics = [m for m in self.metrics if m in METRIC_WHITELIST][:3]
        # sane defaults if model drifted
        if not self.algorithms:
            self.algorithms = (["random_forest","xgboost","logistic_regression"]
                               if self.task_type=="classification"
                               else ["xgboost","ridge","knn"] if self.task_type=="regression"
                               else ["kmeans","gmm"])
        if not self.metrics:
            self.metrics = (["f1_macro","roc_auc_ovr","accuracy"]
                            if self.task_type=="classification"
                            else ["rmse","mae","r2"] if self.task_type=="regression"
                            else ["silhouette"])
        return self

def _client() -> OpenAI:
    """
    构建 OpenAI 客户端：
    - 读取 OPENAI_API_KEY（必需，除非 LLM_OFFLINE=1）
    - 若配置了 OPENAI_BASE_URL，则指向兼容的服务端点
    - 可在上层通过 client.with_options(timeout=...) 指定超时、重试等
    """
    if LLM_OFFLINE:
        # 离线模式下不创建真实客户端
        raise RuntimeError("LLM_OFFLINE=1：已禁用外部 LLM 调用")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    if OPENAI_BASE_URL:
        return OpenAI(api_key=key, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=key)


def _read_prompt(prompt_filename: str) -> Optional[str]:
    """优先从项目根目录 prompts/ 读取 prompt；读取失败返回 None。"""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        p = os.path.join(project_root, "prompts", prompt_filename)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return None

def _extract_profile_columns(profile: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """统一读取 profile 中的列信息与样本总行数。
    兼容不同键：missing/missing_pct, unique/nunique/unique_count。
    返回 (columns, n_rows)。columns 每项至少包含 name, dtype, nunique(可为None), missing(可为None)。"""
    cols = []
    n_rows = None
    # rows from profile["shape"]
    shape = profile.get("shape")
    if isinstance(shape, dict):
        n_rows = shape.get("rows")
    elif isinstance(shape, (list, tuple)) and len(shape) >= 1:
        try:
            n_rows = int(shape[0])
        except Exception:
            n_rows = None

    for c in profile.get("columns", []) or []:
        if isinstance(c, dict):
            name = c.get("name") or c.get("col") or c.get("column")
            if name is None:
                continue
            dtype = str(c.get("dtype", ""))
            # normalize uniques
            nunique = c.get("nunique")
            if nunique is None:
                nunique = c.get("unique")
            if nunique is None:
                nunique = c.get("unique_count")
            try:
                nunique = int(nunique) if nunique is not None else None
            except Exception:
                nunique = None
            # normalize missing
            missing = c.get("missing")
            if missing is None:
                mp = c.get("missing_pct")
                if mp is not None and n_rows:
                    try:
                        missing = int(round(float(mp) * 0.01 * n_rows))
                    except Exception:
                        missing = None
            try:
                missing = int(missing) if missing is not None else None
            except Exception:
                missing = None
            cols.append({"name": str(name), "dtype": dtype, "nunique": nunique, "missing": missing})
    return cols, n_rows

def _fallback_target_and_features(profile: Dict[str, Any]) -> Dict[str, Any]:
    """在离线/失败时，基于启发式推荐目标列与保留/删除列。"""
    cols, n_rows = _extract_profile_columns(profile)
    names = [c["name"] for c in cols]
    lower_names = [n.lower() for n in names]

    # 1) 目标列候选
    keyword_rank = [
        "target","label","y","class","clicked","survived","churn","default","fraud"
    ]
    target_candidates = []
    for kw in keyword_rank:
        if kw in lower_names:
            target_candidates.append(names[lower_names.index(kw)])
    # also consider binary-like columns
    for c in cols:
        if c.get("nunique") in (2, 3):
            # Avoid duplicates
            if c["name"] not in target_candidates:
                target_candidates.append(c["name"])

    target = target_candidates[0] if target_candidates else None

    # 2) 推测任务类型
    task_type = None
    if target is not None:
        tcol = next((c for c in cols if c["name"] == target), None)
        if tcol:
            dtype = (tcol.get("dtype") or "").lower()
            nunique = tcol.get("nunique") or 0
            # 用近似规则：数值且唯一值较多 => 回归
            if any(k in dtype for k in ["int","float","double","long","decimal"]):
                if nunique is not None and nunique >= 20:
                    task_type = "regression"
                else:
                    task_type = "classification"
            else:
                task_type = "classification"

    # 3) 删除列候选
    drop_reasons = []
    def add_drop(name: str, reason: str):
        if name in names and not any(d["name"] == name for d in drop_reasons):
            drop_reasons.append({"name": name, "reason": reason})

    # name-based id like
    id_like_patterns = ["id","uuid","guid","ssn","phone","email","mobile","passport","zipcode","post\u0000", "address"]
    for n in names:
        ln = n.lower()
        if any(p in ln for p in id_like_patterns):
            add_drop(n, "identifier_like")

    # constant columns
    for c in cols:
        if c.get("nunique") is not None and c.get("nunique") <= 1:
            add_drop(c["name"], "constant_column")

    # high-missing if we know rows or missing_pct provided
    if n_rows and n_rows > 0:
        for c in cols:
            miss = c.get("missing")
            if isinstance(miss, int) and miss >= 0.98 * n_rows:
                add_drop(c["name"], "too_many_missing_values")

    # 4) 保留列 = 其他非目标且未标记删除的列
    drop_set = {d["name"] for d in drop_reasons}
    keep_columns = [n for n in names if n not in drop_set and n != target]

    notes = "离线启发式：基于列名、唯一值数量、缺失值与常见 ID 模式进行判断。建议在 UI 中确认。"
    confidence = 0.6 if target else 0.4

    return {
        "target": target,
        "target_candidates": target_candidates,
        "task_type": task_type,
        "keep_columns": keep_columns,
        "drop_columns": drop_reasons,
        "notes": notes,
        "confidence": confidence,
    }

def suggest_target_and_features(profile: Dict[str, Any], user_goal: Optional[str] = None) -> Dict[str, Any]:
    """
    依据数据概要画像，建议最可能的训练目标列，并给出应保留与应删除的列清单（带理由）。
    返回 JSON 字典，字段包含：
      - target: str | null
      - target_candidates: List[str]
      - task_type: 'classification'|'regression'|'clustering'|null
      - keep_columns: List[str]
      - drop_columns: List[{name, reason}]
      - notes: str
      - confidence: float [0,1]
    """
    # 离线/禁用时使用启发式
    if LLM_OFFLINE:
        return _fallback_target_and_features(profile)

    system_prompt = _read_prompt("target_and_features.txt") or (
        "你是 AutoML 助手。根据给定的数据集概要（仅含列名/类型/缺失率/唯一值等，不含原始数据），\n"
        "请选择最可能的监督学习目标列，并推荐应保留与应删除的特征列。\n"
        "- 若无法确定目标列，可给出若干候选；\n"
        "- 删除列示例原因：identifier_like, too_many_missing_values, constant_column, leakage_risk, quasi_unique, free_text_long;\n"
        "- 输出严格 JSON：{target, target_candidates, task_type, keep_columns, drop_columns:[{name,reason}], notes, confidence}；\n"
        "- 不要将 target 放入 keep_columns；所有列名必须来自输入 profile；仅基于提供的统计判断，禁止臆测。"
    )

    try:
        client = _client().with_options(timeout=30.0)
        resp = client.chat.completions.create(
            model=DEFAULT_TASK_MODEL,
            temperature=max(0.0, min(1.0, DEFAULT_TEMPERATURE)),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({
                    "dataset_profile": profile,
                    "user_goal": user_goal or ""
                }, ensure_ascii=False)}
            ],
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

        # 最低限度校验与清洗
        cols, _ = _extract_profile_columns(profile)
        name_set = {c["name"] for c in cols}
        target = data.get("target")
        if target is not None and target not in name_set:
            target = None
        keep_columns = [c for c in (data.get("keep_columns") or []) if c in name_set]
        drop_columns = []
        for d in data.get("drop_columns") or []:
            if isinstance(d, dict) and d.get("name") in name_set:
                drop_columns.append({"name": d.get("name"), "reason": d.get("reason") or "unknown"})
        # 目标不能在 keep 列表中
        if target in keep_columns:
            keep_columns = [c for c in keep_columns if c != target]

        return {
            "target": target,
            "target_candidates": [c for c in (data.get("target_candidates") or []) if c in name_set],
            "task_type": data.get("task_type"),
            "keep_columns": keep_columns,
            "drop_columns": drop_columns,
            "notes": data.get("notes") or "",
            "confidence": float(data.get("confidence") or 0.5),
        }
    except Exception:
        return _fallback_target_and_features(profile)


def suggest_cleaning_suggestions(profile: Dict[str, Any], user_goal: Optional[str] = None) -> Dict[str, Any]:
    """
    给出数据清洗/预处理建议：删除列、缺失值填充、类型转换/日期解析、缩放/编码、异常值处理等。
    返回 JSON 字典：
      - drop_columns: List[{name, reason}]
      - imputations: List[{name, strategy}]  # strategy: median|most_frequent|mean 等
      - type_casts: List[{name, to_dtype, reason}]
      - parse_dates: List[str]
      - scaling: {apply: bool, columns: List[str]}
      - outliers: {apply: bool, columns: List[str], method: str}
      - text_processing: List[{name, suggestion}]
      - leakage_risk: List[str]
      - notes: str
    """
    cols, n_rows = _extract_profile_columns(profile)

    def is_numeric_dtype_str(dt: str) -> bool:
        dt = (dt or "").lower()
        return any(k in dt for k in ["int", "float", "double", "long", "decimal"]) and ("datetime" not in dt)

    def is_categorical_dtype_str(dt: str) -> bool:
        dt = (dt or "").lower()
        return ("object" in dt) or ("category" in dt) or ("string" in dt)

    name_set = [c["name"] for c in cols]

    drop_columns: List[Dict[str, str]] = []
    imputations: List[Dict[str, str]] = []
    type_casts: List[Dict[str, str]] = []
    parse_dates: List[str] = []
    scaling_columns: List[str] = []
    outlier_columns: List[str] = []
    text_processing: List[Dict[str, str]] = []
    leakage_risk: List[str] = []

    # Heuristics
    id_like_patterns = ["id","uuid","guid","ssn","phone","email","mobile","passport","zipcode","postcode","address"]

    # pass 1: column-wise checks
    for c in cols:
        name = c["name"]
        dtype = (c.get("dtype") or "").lower()
        nunique = c.get("nunique")
        missing = c.get("missing")

        # Identify ID-like for drop
        ln = name.lower()
        if any(p in ln for p in id_like_patterns):
            drop_columns.append({"name": name, "reason": "identifier_like"})
            continue

        # Drop constant
        if isinstance(nunique, int) and nunique <= 1:
            drop_columns.append({"name": name, "reason": "constant_column"})
            continue

        # High missing
        if n_rows and isinstance(missing, int) and missing >= 0.98 * n_rows:
            drop_columns.append({"name": name, "reason": "too_many_missing_values"})
            continue

        # Missing imputations
        if n_rows and isinstance(missing, int) and 0 < missing <= 0.3 * n_rows:
            if is_numeric_dtype_str(dtype):
                imputations.append({"name": name, "strategy": "median"})
            elif is_categorical_dtype_str(dtype):
                imputations.append({"name": name, "strategy": "most_frequent"})

        # Parse dates by dtype or name hint
        if ("datetime" in dtype) or any(k in ln for k in ["date","time","timestamp"]):
            parse_dates.append(name)

        # scaling & outliers for numeric features
        if is_numeric_dtype_str(dtype):
            scaling_columns.append(name)
            outlier_columns.append(name)

        # text processing for high-cardinality object columns
        if is_categorical_dtype_str(dtype) and isinstance(nunique, int) and n_rows:
            if nunique > 0.5 * n_rows and nunique >= 50:
                text_processing.append({"name": name, "suggestion": "high_cardinality_text; consider hashing or drop"})

    # leakage risk by name
    for c in cols:
        name = c["name"]
        ln = name.lower()
        if any(k in ln for k in ["label","target","outcome","answer", "y_"]):
            leakage_risk.append(name)

    notes = "以上建议基于启发式：ID/常量/高缺失列建议删除；数值列建议标准化与 IQR 裁剪；类别空值用众数填充；日期字段建议解析。"

    # If online LLM allowed, try boost with prompt for nuanced decisions
    if not LLM_OFFLINE:
        system_prompt = _read_prompt("cleaning_suggestions.txt") or (
            "你是 AutoML 数据清洗助手。根据数据画像给出清洗建议（删除、填充、类型转换、日期解析、缩放、异常值处理、文本处理与泄露风险）。\n"
            "输出严格 JSON，字段：drop_columns, imputations, type_casts, parse_dates, scaling, outliers, text_processing, leakage_risk, notes。\n"
            "所有列名必须来自输入 profile，不得臆测不存在的字段。"
        )
        try:
            client = _client().with_options(timeout=30.0)
            resp = client.chat.completions.create(
                model=DEFAULT_TASK_MODEL,
                temperature=max(0.0, min(1.0, DEFAULT_TEMPERATURE)),
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({
                        "dataset_profile": profile,
                        "user_goal": user_goal or ""
                    }, ensure_ascii=False)}
                ],
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)

            # minimal validation & merge with heuristics as fallback defaults
            name_valid = set(name_set)
            def _filter_name_list(lst):
                return [x for x in lst if isinstance(x, str) and x in name_valid]

            d_drop = []
            for d in data.get("drop_columns", []) or []:
                if isinstance(d, dict) and d.get("name") in name_valid:
                    d_drop.append({"name": d.get("name"), "reason": d.get("reason") or "unknown"})

            d_imp = []
            for d in data.get("imputations", []) or []:
                if isinstance(d, dict) and d.get("name") in name_valid:
                    d_imp.append({"name": d.get("name"), "strategy": d.get("strategy") or "most_frequent"})

            d_casts = []
            for d in data.get("type_casts", []) or []:
                if isinstance(d, dict) and d.get("name") in name_valid:
                    d_casts.append({"name": d.get("name"), "to_dtype": d.get("to_dtype") or "auto", "reason": d.get("reason") or ""})

            d_parse = _filter_name_list(data.get("parse_dates", []) or [])
            d_scaling = data.get("scaling") or {}
            if isinstance(d_scaling, dict):
                d_scaling_cols = _filter_name_list(d_scaling.get("columns", []) or [])
                d_scaling = {"apply": bool(d_scaling.get("apply", True)), "columns": d_scaling_cols}
            else:
                d_scaling = {"apply": True, "columns": scaling_columns}

            d_outliers = data.get("outliers") or {}
            if isinstance(d_outliers, dict):
                d_out_cols = _filter_name_list(d_outliers.get("columns", []) or [])
                d_outliers = {"apply": bool(d_outliers.get("apply", True)), "columns": d_out_cols, "method": d_outliers.get("method") or "iqr_clip"}
            else:
                d_outliers = {"apply": True, "columns": outlier_columns, "method": "iqr_clip"}

            d_text = []
            for d in data.get("text_processing", []) or []:
                if isinstance(d, dict) and d.get("name") in name_valid:
                    d_text.append({"name": d.get("name"), "suggestion": d.get("suggestion") or "hashing_trick"})

            d_leak = _filter_name_list(data.get("leakage_risk", []) or [])

            return {
                "drop_columns": d_drop or drop_columns,
                "imputations": d_imp or imputations,
                "type_casts": d_casts or type_casts,
                "parse_dates": d_parse or parse_dates,
                "scaling": d_scaling,
                "outliers": d_outliers,
                "text_processing": d_text or text_processing,
                "leakage_risk": d_leak or leakage_risk,
                "notes": data.get("notes") or notes,
            }
        except Exception:
            pass

    # offline or error path: return heuristics
    return {
        "drop_columns": drop_columns,
        "imputations": imputations,
        "type_casts": type_casts,
        "parse_dates": parse_dates,
        "scaling": {"apply": True, "columns": scaling_columns},
        "outliers": {"apply": True, "columns": outlier_columns, "method": "iqr_clip"},
        "text_processing": text_processing,
        "leakage_risk": leakage_risk,
        "notes": notes,
    }

def suggest_research_questions(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析数据集并建议可以研究的问题和应用场景
    """
    # 读取问题建议 prompt（优先从项目根目录 prompts/）
    system_prompt = _read_prompt("research_questions.txt")
    if not system_prompt:
                # 内置备选（英文，保证 UI 一致性）
                system_prompt = """
You are a senior data science consultant. Based on the dataset profile (column names, dtypes, uniqueness, missingness, row count, etc.), propose valuable research questions and applications.

Output strictly in English. Use only the columns provided in the profile. Respond with a strict JSON object using the following schema (keys must match exactly):
{
    "research_questions": [
        {
            "question": "Clear, actionable question in English",
            "type": "prediction|analysis|exploration|classification|regression",
            "target_column": "target column name if applicable, else null",
            "difficulty": "Easy|Medium|Hard",
            "business_value": "Short description of business/decision value in English",
            "required_methods": ["e.g., linear regression, decision tree, random forest, collaborative filtering"]
        }
    ],
    "application_scenarios": ["Concise scenario 1", "Concise scenario 2"],
    "key_insights_potential": ["Potential insight 1", "Potential insight 2"],
    "dataset_strengths": ["Strength 1", "Strength 2"],
    "limitations": ["Limitation 1"],
    "recommendations": {
        "priority_questions": ["Top question 1", "Top question 2"],
        "next_steps": ["Concrete next step 1", "Concrete next step 2"],
        "additional_data": ["Optional additional data if any"]
    }
}
"""
    # 离线模式直接回退
    if LLM_OFFLINE:
        return {
            "research_questions": [
                {
                    "question": "Exploratory data analysis: understand distributions and basic patterns",
                    "type": "exploration",
                    "target_column": None,
                    "difficulty": "Easy",
                    "business_value": "Provides a baseline understanding for further modeling",
                    "required_methods": ["descriptive statistics", "histograms", "correlation analysis"]
                }
            ],
            "application_scenarios": ["Data science prototyping", "Business analytics"],
            "key_insights_potential": ["Identify missing values", "Spot skewed distributions"],
            "dataset_strengths": [],
            "limitations": [],
            "recommendations": {
                "priority_questions": ["Baseline EDA"],
                "next_steps": ["Profile columns", "Check target availability"],
                "additional_data": []
            }
        }
    
    # 在线调用
    try:
        client = _client().with_options(timeout=30.0)
        resp = client.chat.completions.create(
            model=DEFAULT_RESEARCH_MODEL,
            temperature=max(0.0, min(1.0, DEFAULT_TEMPERATURE)),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({
                    "dataset_profile": profile
                }, ensure_ascii=False)}
            ],
        )
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        # 简单的回退策略（英文）
        return {
            "research_questions": [
                {
                    "question": "Exploratory data analysis: understand distributions and basic patterns",
                    "type": "exploration",
                    "target_column": None,
                    "difficulty": "Easy",
                    "business_value": "Provides a baseline understanding for further modeling",
                    "required_methods": ["descriptive statistics", "histograms", "correlation analysis"]
                }
            ],
            "application_scenarios": ["Data science prototyping", "Business analytics"],
            "key_insights_potential": ["Identify missing values", "Spot skewed distributions"],
            "dataset_strengths": [],
            "limitations": [],
            "recommendations": {
                "priority_questions": ["Baseline EDA"],
                "next_steps": ["Profile columns", "Check target availability"],
                "additional_data": []
            }
        }

def detect_task(profile: Dict[str, Any], user_question: str) -> Dict[str, Any]:
    """
    Calls OpenAI once to decide task type, target candidates, algorithms, metrics, CV.
    Sends ONLY a small profile (no raw full data).
    """
    # 读取系统提示词
    system = _read_prompt("task_detection.txt") or (
        "你是 AutoML 助手。根据用户问题与数据结构，产出 JSON 计划，包含任务类型/候选目标/算法/指标/CV 设置。"
    )

    # 离线或缺少密钥时走回退
    if LLM_OFFLINE:
        return _fallback_plan(profile)

    try:
        client = _client().with_options(timeout=30.0)
        resp = client.chat.completions.create(
            model=DEFAULT_TASK_MODEL,
            temperature=max(0.0, min(1.0, DEFAULT_TEMPERATURE)),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({
                    "user_question": user_question,
                    "dataset_profile": profile
                }, ensure_ascii=False)}
            ],
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        plan = Plan.model_validate(data)
        return plan.model_dump()
    except (ValidationError, ValueError, KeyError, Exception):
        # 无论解析/网络错误，统一回退
        return _fallback_plan(profile)


def _fallback_plan(profile: Dict[str, Any]) -> Dict[str, Any]:
    """基于启发式的稳健回退方案。"""
    cols = [c.get("name") if isinstance(c, dict) else str(c) for c in profile.get("columns", [])]
    lname = [str(c).lower() for c in cols]
    target_guess = None
    for k in ["target","label","y","survived","churn","default","clicked","class"]:
        if k in lname:
            target_guess = cols[lname.index(k)]
            break
    task = "classification" if target_guess else "regression"
    fallback = Plan(
        task_type=task,
        target_candidates=[target_guess] if target_guess else [],
        algorithms=(
            ["random_forest","xgboost","logistic_regression"] if task=="classification" else ["xgboost","ridge","knn"]
        ),
        metrics=(
            ["f1_macro","roc_auc_ovr","accuracy"] if task=="classification" else ["rmse","mae","r2"]
        ),
        cv=CVSpec(folds=5, stratified=(task=="classification"))
    )
    return fallback.model_dump()

def write_report(bundle: Dict[str, Any]) -> str:
    """
    用 OpenAI 生成总结报告（Markdown）。
    输入 bundle 可包含：
      - profile: 简要数据画像
      - plan: 任务判定结果
      - cleaning_suggest: 数据清洗建议
      - research_suggestions: 研究问题与应用场景
      - leaderboard: 训练排行榜（若已训练）
      - artifacts: 训练产物元数据（可选）
      - meta: 其他元信息（如数据集名称等）

    返回 Markdown 字符串。
    """
    # 离线或无密钥时：生成简要报告
    def _offline_report(b: Dict[str, Any]) -> str:
        lines = ["# AutoML 总结报告", ""]
        meta = b.get("meta", {}) or {}
        if meta:
            lines.append("**数据集**: " + str(meta.get("dataset_name", "未知")))
            lines.append("")

        plan = b.get("plan") or {}
        if plan:
            lines.append("## 任务判定")
            lines.append(f"- 类型: {plan.get('task_type','unknown')}")
            if plan.get("target_candidates"):
                lines.append("- 目标候选: " + ", ".join(plan.get("target_candidates") or []))
            if plan.get("algorithms"):
                lines.append("- 推荐算法: " + ", ".join(plan.get("algorithms") or []))
            if plan.get("metrics"):
                lines.append("- 评估指标: " + ", ".join(plan.get("metrics") or []))
            lines.append("")

        cs = b.get("cleaning_suggest") or {}
        if cs:
            lines.append("## 数据清洗建议")
            drops = [d.get("name") for d in (cs.get("drop_columns") or [])]
            if drops:
                lines.append("- 删除列: " + ", ".join(drops))
            imps = [f"{d.get('name')}=>{d.get('strategy')}" for d in (cs.get("imputations") or [])]
            if imps:
                lines.append("- 缺失值填充: " + ", ".join(imps))
            casts = [f"{d.get('name')}->{d.get('to_dtype')}" for d in (cs.get("type_casts") or [])]
            if casts:
                lines.append("- 类型转换: " + ", ".join(casts))
            lines.append("")

        rs = b.get("research_suggestions") or {}
        if rs:
            lines.append("## 可研究的问题与应用")
            rqs = rs.get("research_questions") or []
            for i, q in enumerate(rqs[:5]):
                lines.append(f"- 问题{i+1}: {q.get('question')}")
            scenes = rs.get("application_scenarios") or []
            if scenes:
                lines.append("- 应用场景: " + ", ".join(scenes[:5]))
            lines.append("")
            # 追加研究问题分析结论
            try:
                research_analysis = analyze_research_questions(rs, b.get("profile"))
                lines.append("## 研究问题分析结论")
                lines.append(research_analysis.get("markdown", "(分析生成失败)") )
                lines.append("")
            except Exception:
                pass

        lb = b.get("leaderboard")  # 可能是 DataFrame 或 None
        try:
            import pandas as pd  # 局部导入，避免上层未装导致全局失败
        except ImportError:
            pd = None  # type: ignore

        training_analysis_md = None
        if pd is not None and isinstance(lb, pd.DataFrame):
            lines.append("## 训练结果摘要")
            try:
                show_cols = [c for c in ["model","cv_score(primary)","acc","f1_macro","roc_auc","rmse","mae","r2"] if c in lb.columns]
                if show_cols:
                    head = lb[show_cols].head(5).to_markdown(index=False)
                    lines.append(head)
                else:
                    lines.append("- 可展示的评估列为空，原始数据请在 UI 查看。")
            except Exception:
                lines.append("- 训练排行榜解析失败，原始数据请在 UI 查看。")
            lines.append("")
            # 深度分析（启发式或后续在线 LLM 增强）
            try:
                task_type = (b.get("plan") or {}).get("task_type") or "classification"
                artifacts = b.get("artifacts") or {}
                training_analysis_md = analyze_training_results(lb, artifacts, task_type, b.get("plan")).get("markdown")
            except Exception:
                training_analysis_md = None
        elif lb is not None:
            # 非 DataFrame 情况（例如序列化后的对象或者其它结构）
            lines.append("## 训练结果摘要")
            lines.append("- 训练排行榜可在 UI 中查看。")
            lines.append("")
        if training_analysis_md:
            lines.append("## 训练结果分析")
            lines.append(training_analysis_md)
            lines.append("")
        lines.append("> 提示：若配置 OPENAI_API_KEY，可生成更详细的自然语言报告。")
        return "\n".join(lines)

    if LLM_OFFLINE:
        return _offline_report(bundle)

    # 在线调用 OpenAI 生成 Markdown 报告
    # 预生成训练分析供 LLM 参考
    pre_training_md = None
    pre_research_md = None
    try:
        import pandas as pd
        lb_obj = bundle.get("leaderboard")
        if isinstance(lb_obj, pd.DataFrame) and not lb_obj.empty:
            task_type = (bundle.get("plan") or {}).get("task_type") or "classification"
            artifacts = bundle.get("artifacts") or {}
            pre_training_md = analyze_training_results(lb_obj, artifacts, task_type, bundle.get("plan")).get("markdown")
        rs_obj = bundle.get("research_suggestions")
        if isinstance(rs_obj, dict):
            pre_research_md = analyze_research_questions(rs_obj, bundle.get("profile")).get("markdown")
    except Exception:
        pre_training_md = None
        pre_research_md = None

    system_prompt = (
        "你是数据科学报告生成器。基于提供的结构化信息生成专业中文 Markdown 报告。\n"
        "结构：1) 数据概览 2) 任务判定 3) 数据清洗建议 4) 研究问题与应用场景 5) 训练结果摘要与分析 6) 下一步行动建议。\n"
        "规则：\n- 关键点用精简项目符号；\n- 不虚构未提供的指标或模型；\n- 若训练结果缺失需明确说明；\n- 将提供的预生成训练分析加以提炼，避免重复原文逐字粘贴。\n"
        "输出必须是纯 Markdown。"
    )

    try:
        client = _client().with_options(timeout=45.0)
        resp = client.chat.completions.create(
            model=DEFAULT_RESEARCH_MODEL,
            temperature=max(0.0, min(1.0, DEFAULT_TEMPERATURE)),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({
                    "bundle": bundle,
                    "pre_training_analysis": pre_training_md,
                    "pre_research_analysis": pre_research_md
                }, ensure_ascii=False)}
            ],
        )
        content = resp.choices[0].message.content or ""
        # 简单兜底：若返回空，则用离线报告
        return content.strip() or _offline_report(bundle)
    except Exception:
        return _offline_report(bundle)


def analyze_training_results(leaderboard, artifacts: Dict[str, Any], task_type: str, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """分析训练结果并返回结构化总结，同时包含可读Markdown。

    返回字典字段：
      - best_model: str | None
      - primary_metric: str
      - metrics_summary: Dict[str, Any]（包含范围与均值）
      - time_stats: {fit_avg, predict_avg, fastest_model, slowest_model}
      - potential_issues: List[str]
      - recommendations: List[str]
      - markdown: 汇总后的 Markdown 文本
    在线模式：调用 OpenAI 根据原始表格与启发式初稿进行润色与深入建议。
    离线模式：仅使用启发式。"""
    import pandas as pd
    if leaderboard is None or not isinstance(leaderboard, pd.DataFrame) or leaderboard.empty:
        return {
            "best_model": None,
            "primary_metric": "cv_score(primary)",
            "metrics_summary": {},
            "time_stats": {},
            "potential_issues": ["Leaderboard 为空，无法分析"],
            "recommendations": ["确认已成功训练并生成 leaderboard.csv"],
            "markdown": "# 训练结果分析\n\n未检测到有效的训练结果数据。"
        }

    lb = leaderboard.copy()
    primary_metric = "cv_score(primary)"
    # 找到最佳模型（按 primary metric 排序后第一行）
    if primary_metric in lb.columns:
        best_row = lb.sort_values(primary_metric, ascending=False).iloc[0]
        best_model = best_row.get("model")
        best_score = float(best_row.get(primary_metric))
    else:
        best_row = lb.iloc[0]
        best_model = best_row.get("model")
        best_score = None

    # 收集关键指标列
    cls_metrics = [c for c in ["acc","f1_macro","roc_auc"] if c in lb.columns]
    reg_metrics = [c for c in ["rmse","mae","r2"] if c in lb.columns]
    metric_cols = cls_metrics if task_type == "classification" else reg_metrics

    metrics_summary: Dict[str, Any] = {}
    for m in metric_cols + [primary_metric]:
        if m in lb.columns:
            col = lb[m].astype(float)
            metrics_summary[m] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "std": float(col.std(ddof=0)),
                "best": float(best_row.get(m)) if m in best_row else None,
            }

    # 时间统计
    time_stats = {}
    try:
        if "fit_s" in lb.columns and "predict_s" in lb.columns:
            time_stats = {
                "fit_avg": float(lb["fit_s"].mean()),
                "predict_avg": float(lb["predict_s"].mean()),
                "fastest_model": str(lb.sort_values("predict_s", ascending=True).iloc[0]["model"]),
                "slowest_model": str(lb.sort_values("predict_s", ascending=False).iloc[0]["model"]),
            }
    except Exception:
        pass

    potential_issues: List[str] = []
    recommendations: List[str] = []

    # 启发式问题检测
    if task_type == "classification":
        if "f1_macro" in metrics_summary and metrics_summary["f1_macro"]["best"] < 0.7:
            potential_issues.append("最佳 F1_macro < 0.7，类别区分能力可能不足")
            recommendations.append("尝试调参或引入更复杂模型（如提升迭代次数、使用集成方法）")
        if "roc_auc" in metrics_summary and metrics_summary["roc_auc"]["best"] and metrics_summary["roc_auc"]["best"] < 0.75:
            potential_issues.append("ROC_AUC < 0.75，正负类分离效果一般")
            recommendations.append("尝试特征工程：添加交叉特征或目标编码")
    else:
        if "rmse" in metrics_summary and metrics_summary["rmse"]["std"] < 1e-9:
            potential_issues.append("所有模型 RMSE 几乎相同，可能是数据泄露或特征单一")
            recommendations.append("检查是否对训练/测试使用了完全相同的派生特征或数据泄露")
        if "r2" in metrics_summary and metrics_summary["r2"]["best"] < 0.3:
            potential_issues.append("最佳 R2 < 0.3，解释度较低")
            recommendations.append("引入更多相关特征或尝试非线性模型")

    if not potential_issues:
        recommendations.append("继续进行模型解释（SHAP/Permutation Importance）以验证特征贡献")
    recommendations.append("对最佳模型进行持久化与版本记录")
    recommendations.append("尝试减少拟合时间：剔除最慢模型或降低搜索迭代")

    # 初步 Markdown
    md_lines = ["# 训练结果分析", ""]
    md_lines.append(f"**任务类型**: {task_type}")
    md_lines.append(f"**最佳模型**: `{best_model}`  (主指标={best_score:.4f} if best_score is not None else 'N/A')")
    if metric_cols:
        md_lines.append("\n## 关键指标范围")
        for m in metric_cols:
            ms = metrics_summary.get(m)
            if ms:
                md_lines.append(f"- {m}: min={ms['min']:.4f}, max={ms['max']:.4f}, mean={ms['mean']:.4f}, best={ms['best']:.4f}")
    if time_stats:
        md_lines.append("\n## 时间表现")
        md_lines.append(f"- 平均训练秒: {time_stats.get('fit_avg'):.3f}")
        md_lines.append(f"- 平均预测秒: {time_stats.get('predict_avg'):.6f}")
        md_lines.append(f"- 最快预测模型: {time_stats.get('fastest_model')}")
        md_lines.append(f"- 最慢预测模型: {time_stats.get('slowest_model')}")
    if potential_issues:
        md_lines.append("\n## 潜在问题")
        for p in potential_issues:
            md_lines.append(f"- {p}")
    md_lines.append("\n## 建议下一步")
    for r in recommendations:
        md_lines.append(f"- {r}")
    heuristic_markdown = "\n".join(md_lines)

    result = {
        "best_model": best_model,
        "primary_metric": primary_metric,
        "metrics_summary": metrics_summary,
        "time_stats": time_stats,
        "potential_issues": potential_issues,
        "recommendations": recommendations,
        "markdown": heuristic_markdown,
    }

    # 离线模式直接返回启发式
    if LLM_OFFLINE:
        return result

    # 在线调用 OpenAI 对启发式进行增强
    enhancement_prompt = (
        "你是资深 AutoML 顾问。根据下方启发式初稿与原始表格数据（仅关键列），"
        "生成更精炼、结构化的中文训练结果分析 Markdown：\n"
        "- 保留最佳模型与关键数值；\n"
        "- 对潜在问题进行验证语气说明；\n"
        "- 给出 3-6 条优先级排序的可执行建议；\n"
        "- 不要虚构不存在的指标。"
    )

    # 压缩表格（避免 token 过多）
    slim_cols = [c for c in ["model", primary_metric, *metric_cols, "fit_s", "predict_s"] if c in lb.columns]
    slim_table = lb[slim_cols].head(50).to_dict(orient="records")

    try:
        client = _client().with_options(timeout=40.0)
        resp = client.chat.completions.create(
            model=DEFAULT_TASK_MODEL,
            temperature=max(0.0, min(1.0, DEFAULT_TEMPERATURE)),
            messages=[
                {"role": "system", "content": enhancement_prompt},
                {"role": "user", "content": json.dumps({
                    "slim_table": slim_table,
                    "heuristic_markdown": heuristic_markdown,
                    "task_type": task_type,
                    "plan": plan or {}
                }, ensure_ascii=False)}
            ],
        )
        refined = resp.choices[0].message.content or ""
        if refined.strip():
            result["markdown"] = refined.strip()
    except Exception:
        pass

    return result


def check_research_alignment(
    leaderboard,
    artifacts: Dict[str, Any],
    task_type: str,
    research_suggestions: Optional[Dict[str, Any]] = None,
    trained_target: Optional[str] = None,
    picked_models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """检查训练结果是否覆盖/支持推荐研究问题，返回结构化结论与 Markdown。

    返回字段：
      - per_question: List[{
            question, status: covered|partially_covered|not_covered,
            reasons: List[str], metrics_present: List[str], algos_present: List[str], evidence: Dict
        }]
      - summary: {covered, partial, not_covered, total}
      - markdown: 汇总说明
    """
    import pandas as pd
    per_question: List[Dict[str, Any]] = []

    # 基础可用性检查
    if research_suggestions is None or not isinstance(research_suggestions, dict):
        return {
            "per_question": [],
            "summary": {"covered": 0, "partial": 0, "not_covered": 0, "total": 0},
            "markdown": "# 一致性检查\n\n尚未提供推荐研究问题，无法对齐分析。",
        }

    try:
        lb_ok = isinstance(leaderboard, pd.DataFrame) and not leaderboard.empty
    except Exception:
        lb_ok = False

    if not lb_ok:
        return {
            "per_question": [],
            "summary": {"covered": 0, "partial": 0, "not_covered": 0, "total": len(research_suggestions.get("research_questions", []) or [])},
            "markdown": "# 一致性检查\n\n训练排行榜为空，无法判断是否覆盖研究问题。",
        }

    # 归一化：可用指标集合（来自 leaderboard 列和 artifacts.test_metrics）
    available_metric_names: set = set(map(str, leaderboard.columns))
    for mname, meta in (artifacts or {}).items():
        tm = (meta or {}).get("test_metrics") or {}
        available_metric_names.update(list(tm.keys()))

    # 指标同义映射
    metric_alias = {
        "accuracy": {"accuracy", "acc"},
        "f1_macro": {"f1_macro", "f1"},
        "roc_auc": {"roc_auc", "roc_auc_ovr", "auc"},
        "rmse": {"rmse"},
        "mae": {"mae"},
        "r2": {"r2"},
    }

    def is_metric_supported(name: str) -> bool:
        name_l = (name or "").lower()
        for k, aliases in metric_alias.items():
            if name_l == k or name_l in aliases:
                return any(a in available_metric_names for a in aliases)
        # 直接匹配原名
        return name in available_metric_names

    # 归一化：可用算法集合（来自 leaderboard 的 model 列 + 训练时选择）
    algo_seen = set()
    if "model" in leaderboard.columns:
        for v in leaderboard["model"].astype(str).tolist():
            algo_seen.add(v.lower())
    for v in (picked_models or []):
        algo_seen.add(str(v).lower())

    # 算法同义映射（研究问题中的 required_methods 可能是自然语言）
    algo_alias = {
        "random_forest": {"random_forest", "rf"},
        "xgboost": {"xgboost", "xgb"},
        "logistic_regression": {"logistic_regression", "logistic", "logreg"},
        "linear_regression": {"linear_regression", "linear", "linreg"},
        "ridge": {"ridge"},
        "knn": {"knn", "k-nearest", "k nearest"},
        "mlp": {"mlp", "neural network", "multilayer perceptron"},
    }

    def method_supported(m: str) -> bool:
        ml = (m or "").lower()
        for _, aliases in algo_alias.items():
            if ml in aliases and any(a in algo_seen for a in aliases):
                return True
        return False

    def normalize_type(t: Optional[str]) -> str:
        tl = (t or "").lower()
        if tl in ("classification", "regression"):
            return tl
        if tl in ("prediction",):
            return task_type  # 预测问题按当前训练任务类型解释
        return "analysis"  # 其余视为探索/分析类

    questions = research_suggestions.get("research_questions", []) or []
    covered = partial = not_covered = 0

    # 取一个最佳模型证据（第一名）
    try:
        best_row = leaderboard.sort_values("cv_score(primary)", ascending=False).iloc[0]
    except Exception:
        best_row = leaderboard.iloc[0]

    for q in questions:
        qtext = q.get("question") or "(未命名问题)"
        qtype = normalize_type(q.get("type"))
        tcol = q.get("target_column")
        required_methods = q.get("required_methods") or []

        reasons: List[str] = []
        metrics_present: List[str] = []
        algos_present: List[str] = []

        # 任务类型匹配
        task_match = (qtype == task_type) or (qtype == "analysis")
        if not task_match:
            reasons.append(f"任务类型不匹配：问题类型={qtype}，训练任务={task_type}")

        # 目标列匹配（若问题声明了目标列，且我们知道训练目标）
        target_match = True
        if tcol and trained_target:
            target_match = (str(tcol) == str(trained_target))
            if not target_match:
                reasons.append(f"目标列不一致：问题要求 `{tcol}`，训练使用 `{trained_target}`")

        # 指标支持（按问题类型推断一组代表指标）
        expected_metrics = []
        if qtype == "classification":
            expected_metrics = ["accuracy", "f1_macro", "roc_auc"]
        elif qtype == "regression":
            expected_metrics = ["rmse", "mae", "r2"]
        else:
            expected_metrics = ["descriptive stats", "correlation"]

        for em in expected_metrics:
            if is_metric_supported(em):
                metrics_present.append(em)

        # 算法支持（若问题列出了 required_methods，用别名做模糊映射）
        for m in required_methods:
            if method_supported(m):
                algos_present.append(m)

        # 计算状态
        if task_match and target_match:
            if metrics_present:
                status = "covered"
                covered += 1
            else:
                status = "partially_covered"
                partial += 1
                if not reasons:
                    reasons.append("已匹配任务与目标，但缺少相应评估指标")
        else:
            status = "not_covered"
            not_covered += 1
            if not reasons:
                reasons.append("问题与当前训练配置不一致或信息不足")

        # 证据与建议
        evidence = {
            "best_model": str(best_row.get("model")),
            "primary_score": float(best_row.get("cv_score(primary)")) if "cv_score(primary)" in best_row else None,
        }

        # 下一步建议（小而明确）
        next_steps: List[str] = []
        if status != "covered":
            if qtype != task_type and qtype in ("classification", "regression"):
                next_steps.append(f"为该问题单独发起 {qtype} 训练流程")
            if tcol and trained_target and tcol != trained_target:
                next_steps.append(f"更换训练目标列为 `{tcol}` 或在 UI 中选择对应目标后重训")
            if not metrics_present and qtype in ("classification", "regression"):
                next_steps.append("在排行榜中添加/展示该问题关键评估指标")
            if required_methods and not algos_present:
                next_steps.append("加入问题所需的代表性算法（如 XGBoost/RandomForest）")

        per_question.append({
            "question": qtext,
            "status": status,
            "reasons": reasons,
            "metrics_present": metrics_present,
            "algos_present": algos_present,
            "expected_type": qtype,
            "expected_target": tcol,
            "evidence": evidence,
            "next_steps": next_steps,
        })

    summary = {"covered": covered, "partial": partial, "not_covered": not_covered, "total": len(questions)}

    # 生成 Markdown 汇总
    lines = ["# 训练与研究问题一致性检查", ""]
    lines.append(f"总计 {summary['total']} 个问题：\n- 覆盖: {covered}\n- 部分覆盖: {partial}\n- 未覆盖: {not_covered}")
    lines.append("")
    for i, item in enumerate(per_question, 1):
        status_emoji = {"covered": "✅", "partially_covered": "🟨", "not_covered": "❌"}.get(item["status"], "•")
        lines.append(f"## {status_emoji} 问题 {i}: {item['question']}")
        lines.append(f"- 期望类型: {item['expected_type']}；训练类型: {task_type}")
        if item.get("expected_target"):
            lines.append(f"- 期望目标列: `{item['expected_target']}`；训练目标: `{trained_target or '未知'}`")
        if item.get("metrics_present"):
            lines.append(f"- 指标支持: {', '.join(item['metrics_present'])}")
        if item.get("algos_present"):
            lines.append(f"- 算法覆盖: {', '.join(item['algos_present'])}")
        if item.get("reasons"):
            lines.append("- 备注/原因: " + "; ".join(item["reasons"]))
        if item.get("next_steps"):
            lines.append("- 下一步: " + "; ".join(item["next_steps"]))
        lines.append("")

    return {
        "per_question": per_question,
        "summary": summary,
        "markdown": "\n".join(lines),
    }

def answer_research_questions(
    research_suggestions: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
    leaderboard,
    artifacts: Dict[str, Any],
    task_type: str,
    trained_target: Optional[str] = None,
) -> Dict[str, Any]:
    """基于训练结果，尝试“作答”推荐研究问题并给出可验证的依据。

    逻辑：
    - 若问题类型与当前训练任务一致，且目标列匹配（若提供），则用最佳模型的测试指标进行回答；
    - 若类型一致但目标不匹配，给出原因与行动建议；
    - 对探索/分析类问题，基于 profile 给出可执行的分析步骤建议；
    - 产出 per_question 列表与汇总 Markdown。
    """
    import pandas as pd
    if not isinstance(research_suggestions, dict) or "research_questions" not in research_suggestions:
        return {
            "answers": [],
            "markdown": "# 研究问题作答\n\n未提供有效的研究问题对象。",
        }

    qs = research_suggestions.get("research_questions", []) or []
    answers: List[Dict[str, Any]] = []

    # 准备排行榜与最佳模型证据
    best_row = None
    best_model_name = None
    best_metrics = {}
    try:
        import pandas as pd  # 确保存在
        if isinstance(leaderboard, pd.DataFrame) and not leaderboard.empty:
            if "cv_score(primary)" in leaderboard.columns:
                best_row = leaderboard.sort_values("cv_score(primary)", ascending=False).iloc[0]
            else:
                best_row = leaderboard.iloc[0]
            best_model_name = str(best_row.get("model")) if "model" in best_row else None
            if best_model_name and isinstance(artifacts, dict) and best_model_name in artifacts:
                best_metrics = (artifacts[best_model_name] or {}).get("test_metrics") or {}
    except Exception:
        pass

    # 指标映射，便于稳定输出
    display_order_cls = ["accuracy", "f1_macro", "roc_auc"]
    display_order_reg = ["rmse", "mae", "r2"]
    alias_map = {"acc": "accuracy", "f1": "f1_macro", "auc": "roc_auc"}

    def format_metrics(task: str, metrics: Dict[str, Any]) -> List[str]:
        if not isinstance(metrics, dict) or not metrics:
            return []
        # 归一化 key
        norm = {}
        for k, v in metrics.items():
            kk = alias_map.get(k, k)
            norm[kk] = v
        ordered = display_order_cls if task == "classification" else display_order_reg
        out = []
        for k in ordered:
            if k in norm:
                try:
                    out.append(f"{k}: {float(norm[k]):.4f}")
                except Exception:
                    out.append(f"{k}: {norm[k]}")
        # 追加其余指标
        for k, v in norm.items():
            if k not in ordered:
                try:
                    out.append(f"{k}: {float(v):.4f}")
                except Exception:
                    out.append(f"{k}: {v}")
        return out

    # 基于 profile 提供探索类回答
    def exploration_hints(profile: Optional[Dict[str, Any]]) -> List[str]:
        hints: List[str] = []
        if not isinstance(profile, dict):
            return ["执行基础 EDA：分布、缺失、相关性热力图"]
        cols = profile.get("columns") or []
        if isinstance(cols, list):
            hints.append(f"列数量: {len(cols)}；示例: " + ", ".join([str((c or {}).get('name')) for c in cols[:5]]))
        hints.extend([
            "绘制数值列直方图与箱线图，检查偏态与异常值",
            "对类别列统计 Top-N 频次并检查长尾",
            "计算相关系数矩阵并展示热力图",
        ])
        return hints

    lines = ["# 研究问题作答", ""]

    for i, q in enumerate(qs, 1):
        qtext = q.get("question") or f"问题{i}"
        qtype = (q.get("type") or "").lower()
        tcol = q.get("target_column")
        answer_item: Dict[str, Any] = {"question": qtext, "type": qtype, "target": tcol}

        # 分类/回归问题尝试直接给出结论
        if qtype in ("classification", "regression", "prediction", "regression"):
            matches_task = (task_type == ("classification" if qtype == "classification" else "regression"))
            matches_target = True
            if tcol and trained_target:
                matches_target = (str(tcol) == str(trained_target))

            if matches_task and matches_target and best_model_name:
                # 直接引用最佳模型的测试指标
                met_lines = format_metrics(task_type, best_metrics)
                answer_item["status"] = "answered"
                answer_item["best_model"] = best_model_name
                answer_item["metrics"] = best_metrics
                lines.append(f"## ✅ {qtext}")
                if trained_target:
                    lines.append(f"- 目标列: `{trained_target}`")
                lines.append(f"- 最佳模型: `{best_model_name}`")
                if met_lines:
                    lines.append("- 测试集指标: " + "; ".join(met_lines))
                else:
                    lines.append("- 测试集指标: (未记录)")
                lines.append("- 结论: 模型已能对该问题给出可量化的答案；建议结合业务阈值进一步评审。")
                lines.append("")
            else:
                answer_item["status"] = "not_answered"
                reasons = []
                if not matches_task:
                    reasons.append(f"当前训练任务为 {task_type}，与问题类型不一致")
                if tcol and trained_target and tcol != trained_target:
                    reasons.append(f"问题目标列 `{tcol}` 与训练目标 `{trained_target}` 不一致")
                answer_item["reasons"] = reasons
                lines.append(f"## ❌ {qtext}")
                if reasons:
                    for r in reasons:
                        lines.append(f"- {r}")
                lines.append("- 建议: 针对该问题重新选择目标列/任务类型后训练，并复核相应指标。\n")
        else:
            # 探索/分析类问题：提供操作性建议
            answer_item["status"] = "action_plan"
            steps = exploration_hints(profile)
            answer_item["steps"] = steps
            lines.append(f"## 📝 {qtext}")
            lines.append("- 建议的分析步骤：")
            for s in steps:
                lines.append(f"  - {s}")
            lines.append("")

        answers.append(answer_item)

    return {"answers": answers, "markdown": "\n".join(lines)}

def analyze_research_questions(research_suggestions: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """根据已有的研究问题建议生成结构化分析与结论（离线 + 在线增强）。

    返回字段：
      - prioritized: List[str] 根据难度与价值排序的前若干问题
      - feasibility: Dict[str, str] 每个问题的可行性说明
      - required_data_checks: List[str] 建议先执行的数据完备性/质量检查
      - recommended_metrics: Dict[str, List[str]] 问题 -> 建议评估指标
      - next_steps: List[str] 整体下一步行动
      - markdown: 汇总后的 Markdown
    在线模式：将启发式初稿与原始问题传给 LLM 增强。"""
    if not isinstance(research_suggestions, dict):
        return {
            "prioritized": [],
            "feasibility": {},
            "required_data_checks": ["研究问题对象无效，需重新生成"],
            "recommended_metrics": {},
            "next_steps": ["重新获取研究问题建议"],
            "markdown": "# 研究问题分析\n\n输入无效，无法生成分析。"
        }

    questions = research_suggestions.get("research_questions", []) or []
    # 提取难度与价值用于排序（启发式：Hard>Medium>Easy 但优先商业价值高）
    def _difficulty_score(d: str) -> int:
        d = (d or "").lower()
        return {"easy": 1, "medium": 2, "hard": 3}.get(d, 2)

    prioritized = []
    for q in questions:
        # 简单记录 (question, diff_score)
        diff = _difficulty_score(q.get("difficulty"))
        value = len(q.get("business_value", ""))
        prioritized.append((q.get("question"), diff, value))
    # 排序策略：商业价值长度降序 + 难度升序（先做价值高但相对容易的）
    prioritized.sort(key=lambda x: (-x[2], x[1]))
    prioritized_names = [p[0] for p in prioritized][:6]

    # 可行性与指标建议
    feasibility: Dict[str, str] = {}
    recommended_metrics: Dict[str, List[str]] = {}
    for q in questions:
        name = q.get("question")
        qtype = (q.get("type") or "").lower()
        tcol = q.get("target_column")
        diff = (q.get("difficulty") or "").lower()
        if not name:
            continue
        feas_notes = []
        if tcol:
            feas_notes.append(f"依赖目标列 `{tcol}` 的完整性")
        else:
            feas_notes.append("无需明确监督目标，可先做探索性分析")
        if diff == "hard":
            feas_notes.append("建议原型验证后再投入大量资源")
        elif diff == "easy":
            feas_notes.append("可快速启动，低实施成本")
        feasibility[name] = "; ".join(feas_notes)
        # 指标推荐
        metrics = []
        if qtype in ("prediction","classification"):
            metrics = ["accuracy","f1_macro","roc_auc"]
        elif qtype in ("regression",):
            metrics = ["rmse","mae","r2"]
        elif qtype in ("analysis","exploration"):
            metrics = ["distribution plots","correlation heatmap","feature importance"]
        else:
            metrics = ["custom domain metrics"]
        recommended_metrics[name] = metrics

    # 数据检查建议
    required_data_checks = []
    if profile and isinstance(profile, dict):
        cols = profile.get("columns") or []
        if isinstance(cols, list):
            num_cols = sum(1 for c in cols if str(c.get("dtype","")) in ("int64","float64","int32","float32"))
            cat_cols = sum(1 for c in cols if "object" in str(c.get("dtype","")) or "category" in str(c.get("dtype","")))
            required_data_checks.append(f"数值列数量: {num_cols}, 类别列数量: {cat_cols}")
    required_data_checks.extend([
        "确认目标列（若有）缺失率 < 5%",
        "检查高基数类别特征是否需要降维或编码",
        "验证日期字段是否已解析为 datetime",
    ])

    next_steps = [
        "锁定前 3 个高价值且中低难度的问题进入原型阶段",
        "为每个问题建立数据字典与特征清单",
        "制定指标计算脚本并评审",
    ]

    # Markdown 初稿
    md_lines = ["# 研究问题分析", ""]
    if prioritized_names:
        md_lines.append("## 优先级排序（前 6 项）")
        for i, n in enumerate(prioritized_names, 1):
            md_lines.append(f"{i}. {n}")
        md_lines.append("")
    if feasibility:
        md_lines.append("## 可行性速览")
        for k, v in feasibility.items():
            md_lines.append(f"- **{k}**: {v}")
        md_lines.append("")
    if recommended_metrics:
        md_lines.append("## 推荐指标 / 分析度量")
        for k, mets in recommended_metrics.items():
            md_lines.append(f"- **{k}**: {', '.join(mets)}")
        md_lines.append("")
    if required_data_checks:
        md_lines.append("## 前置数据检查")
        for c in required_data_checks:
            md_lines.append(f"- {c}")
        md_lines.append("")
    md_lines.append("## 下一步行动")
    for ns in next_steps:
        md_lines.append(f"- {ns}")
    heuristic_md = "\n".join(md_lines)

    result = {
        "prioritized": prioritized_names,
        "feasibility": feasibility,
        "required_data_checks": required_data_checks,
        "recommended_metrics": recommended_metrics,
        "next_steps": next_steps,
        "markdown": heuristic_md,
    }

    if LLM_OFFLINE:
        return result

    # 在线增强
    enhancement_prompt = (
        "你是资深数据咨询顾问。对给定研究问题启发式分析进行打磨：\n"
        "- 保留关键信息，结构清晰；\n- 若问题与数据画像存在明显风险请标注；\n"
        "- 给出 3 条更具战略性的建议替换或补充原始下一步；\n"
        "- 输出纯 Markdown。"
    )
    try:
        client = _client().with_options(timeout=35.0)
        resp = client.chat.completions.create(
            model=DEFAULT_RESEARCH_MODEL,
            temperature=max(0.0, min(1.0, DEFAULT_TEMPERATURE)),
            messages=[
                {"role": "system", "content": enhancement_prompt},
                {"role": "user", "content": json.dumps({
                    "raw_questions": questions,
                    "heuristic_markdown": heuristic_md,
                    "profile_head": (profile or {})
                }, ensure_ascii=False)}
            ],
        )
        refined = resp.choices[0].message.content or ""
        if refined.strip():
            result["markdown"] = refined.strip()
    except Exception:
        pass
    return result


def analyze_single_research_question(question: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """对单个研究问题生成更具体的分析结论（离线 + 可选在线增强）。

    输入 question 结构示例：{
        'question': str,
        'type': 'prediction|analysis|exploration|classification|regression|clustering',
        'target_column': str | None,
        'difficulty': 'Easy|Medium|Hard',
        'business_value': str,
        'required_methods': [..]
    }

    返回：{
      'summary': str 概述
      'data_requirements': List[str]
      'modeling_approach': List[str]
      'metrics': List[str]
      'risks': List[str]
      'next_steps': List[str]
      'markdown': Markdown 汇总
    }
    """
    if not isinstance(question, dict):
        return {
            'summary': '问题格式无效',
            'data_requirements': [],
            'modeling_approach': [],
            'metrics': [],
            'risks': ['无法解析问题对象'],
            'next_steps': ['重新生成研究问题'],
            'markdown': '# 单题分析\n\n输入无效。'
        }

    qtext = question.get('question') or '未命名问题'
    qtype = (question.get('type') or '').lower()
    target = question.get('target_column')
    difficulty = (question.get('difficulty') or 'Medium')
    methods = question.get('required_methods') or []

    # 领域特定启发式：电影评分 / 推荐 类问题辨识
    domain_signals = {
        'movie': '电影', 'rating': '评分', 'user': '用户', 'recommend': '推荐', 'item': '物品', 'click': '点击'
    }
    is_recommendation = any(k in qtext.lower() for k in ['movie', 'rating', 'recommend', 'user', 'item'])

    data_requirements = []
    if target:
        data_requirements.append(f'目标列 `{target}` 的完整性与异常值检查')
    if is_recommendation:
        data_requirements.extend([
            '用户-物品交互（评分/点击）稀疏度统计 (稀疏率、活跃用户分布)',
            '用户侧特征：年龄、性别、地域（若可用）',
            '物品侧特征：类别、标签、发布时间、聚合统计',
            '时间戳/序列信息用于考察概念漂移',
        ])
    else:
        data_requirements.append('确认关键特征缺失率 < 10% 且类型已正确解析')

    # 建模方法启发
    modeling_approach = []
    if is_recommendation:
        modeling_approach.extend([
            '协同过滤：矩阵分解 (SVD, ALS)',
            '隐语义模型：Embedding + MLP (Neural CF)',
            '特征融合：用户画像 + 电影元数据的回归/排序模型 (XGBoost)',
            '混合推荐：召回 (协同/相似度) + 排序 (GBDT/深度网络)'
        ])
    elif qtype in ('prediction','regression','classification'):
        modeling_approach.extend([
            '基线模型：线性/逻辑回归作为效果下界',
            '非线性提升：XGBoost / RandomForest 比较增益',
            '正则化与特征选择：减少过拟合与冗余特征'
        ])
    elif qtype in ('clustering', 'exploration'):
        modeling_approach.extend([
            '基线聚类：KMeans (肘部法/轮廓系数确定 k)',
            '密度/概率：GMM 或 DBSCAN 捕捉非球形结构',
            '降维：PCA / UMAP 可视化群组分布'
        ])
    else:
        modeling_approach.append('根据数据分布先做 EDA 再细化技术栈')

    # 指标建议
    metrics = []
    if is_recommendation:
        metrics = ['RMSE', 'MAE', 'MAP@K', 'NDCG@K', 'Coverage', 'Cold-start hit rate']
    elif qtype in ('prediction','regression'):
        metrics = ['RMSE','MAE','R2']
    elif qtype in ('classification',):
        metrics = ['Accuracy','F1_macro','ROC_AUC','Precision','Recall']
    elif qtype in ('clustering',):
        metrics = ['Silhouette','Calinski-Harabasz','Davies-Bouldin']
    else:
        metrics = ['Descriptive stats','Correlation','Feature importance']

    # 风险/挑战
    risks = []
    if is_recommendation:
        risks.extend([
            '数据稀疏导致矩阵分解初期拟合不稳定',
            '冷启动：新用户/新电影无交互记录',
            '热门偏置：热门电影被过度推荐影响多样性',
            '时间漂移：流行度随时间变化需建时序特征'
        ])
    if difficulty == 'Hard':
        risks.append('高复杂度可能需要更多迭代与计算资源')
    if qtype in ('regression','prediction') and target:
        risks.append('目标可能存在长尾分布需做分桶或对数变换')

    # 下一步行动
    next_steps = []
    if is_recommendation:
        next_steps.extend([
            '统计交互矩阵稀疏度与冷启动用户/物品比例',
            '训练 SVD 基线并记录 RMSE/MAE 作为初始 benchmark',
            '提取用户/电影特征并构建特征矩阵用于混合排序模型',
            '评估多样性指标 (Coverage) 与准确性指标平衡'
        ])
    else:
        next_steps.extend([
            '构建基线模型并记录核心指标作为对照',
            '绘制目标变量与主特征的分布关系/相关性',
            '迭代增加非线性模型比较增益'
        ])

    summary = f"问题: {qtext}\n类型: {question.get('type','unknown')} 难度: {difficulty} 候选方法: {', '.join(methods) if methods else '未指定'}"

    md = [f"### 单题分析：{qtext}", '', f"**概要**: {summary}", '']
    md.append('**数据需求**:')
    for d in data_requirements: md.append(f"- {d}")
    md.append('')
    md.append('**建模思路**:')
    for m in modeling_approach: md.append(f"- {m}")
    md.append('')
    md.append('**评估指标**:')
    for met in metrics: md.append(f"- {met}")
    md.append('')
    if risks:
        md.append('**潜在风险**:')
        for r in risks: md.append(f"- {r}")
        md.append('')
    md.append('**下一步行动**:')
    for s in next_steps: md.append(f"- {s}")
    heuristic_md = '\n'.join(md)

    result = {
        'summary': summary,
        'data_requirements': data_requirements,
        'modeling_approach': modeling_approach,
        'metrics': metrics,
        'risks': risks,
        'next_steps': next_steps,
        'markdown': heuristic_md,
    }

    if LLM_OFFLINE:
        return result

    # 在线增强
    prompt = (
        '你是资深数据科学顾问。对下列针对单个研究问题的启发式分析进行优化：'\
        '补充更精准的评估指标（不杜撰），简化冗余描述，保留列表结构，输出 Markdown。'
    )
    try:
        client = _client().with_options(timeout=30.0)
        resp = client.chat.completions.create(
            model=DEFAULT_RESEARCH_MODEL,
            temperature=max(0.0, min(1.0, DEFAULT_TEMPERATURE)),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({
                    'question': question,
                    'heuristic_markdown': heuristic_md,
                    'profile_head': (profile or {})
                }, ensure_ascii=False)}
            ],
        )
        refined = resp.choices[0].message.content or ''
        if refined.strip():
            result['markdown'] = refined.strip()
    except Exception:
        pass
    return result
