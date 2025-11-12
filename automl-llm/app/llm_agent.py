
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
        # 内置备选
        system_prompt = """
你是一个数据分析专家。基于数据集的结构信息，分析这个数据集可以用来研究什么问题。

请从以下角度分析：
1. 预测性问题：基于现有特征可以预测什么？
2. 描述性问题：数据中有什么有趣的模式和关系？
3. 应用场景：在哪些实际业务场景中可以应用？
4. 研究价值：对学术研究或商业决策有什么价值？

返回 JSON 格式：
{
  "research_questions": [
    {
      "question": "问题描述",
      "type": "预测/分析/探索",
      "target": "目标变量（如果有）",
      "difficulty": "简单/中等/困难",
      "business_value": "商业价值描述"
    }
  ],
  "application_scenarios": ["场景1", "场景2", "场景3"],
  "key_insights": ["洞察1", "洞察2"],
  "recommendations": "使用建议"
}
"""
    # 离线模式直接回退
    if LLM_OFFLINE:
        return {
            "research_questions": [
                {
                    "question": "数据探索性分析：了解数据的基本特征和分布",
                    "type": "探索",
                    "target": "无",
                    "difficulty": "简单",
                    "business_value": "为后续深入分析提供基础"
                }
            ],
            "application_scenarios": ["数据科学研究", "业务分析"],
            "key_insights": ["需要进一步分析确定"],
            "recommendations": "LLM 处于离线模式（LLM_OFFLINE=1），请手动探索数据特征"
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
        # 简单的回退策略
        return {
            "research_questions": [
                {
                    "question": "数据探索性分析：了解数据的基本特征和分布",
                    "type": "探索",
                    "target": "无",
                    "difficulty": "简单",
                    "business_value": "为后续深入分析提供基础"
                }
            ],
            "application_scenarios": ["数据科学研究", "业务分析"],
            "key_insights": ["需要进一步分析确定"],
            "recommendations": f"由于分析出错({str(e)}), 建议手动探索数据特征"
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
    Report writing will be implemented in Part 7; keep placeholder for now.
    """
    return "# 报告\n\n（报告生成功能将于后续步骤接入 OpenAI）"
