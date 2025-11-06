
from __future__ import annotations
import os, json
from typing import Any, Dict, List, Optional
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
