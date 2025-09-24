
from __future__ import annotations
import os, json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError, model_validator
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

def _client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

def suggest_research_questions(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析数据集并建议可以研究的问题和应用场景
    """
    try:
        # 读取问题建议的 prompt
        with open("prompts/research_questions.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        # 如果文件不存在，使用内置的 prompt
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
    
    client = _client()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
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
            "recommendations": f"由于分析出错({str(e)})，建议手动探索数据特征"
        }

def detect_task(profile: Dict[str, Any], user_question: str) -> Dict[str, Any]:
    """
    Calls OpenAI once to decide task type, target candidates, algorithms, metrics, CV.
    Sends ONLY a small profile (no raw full data).
    """
    system = open("prompts/task_detection.txt", "r", encoding="utf-8").read()
    client = _client()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
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
    except (ValidationError, ValueError, KeyError) as e:
        # Fallback — robust defaults by simple heuristics
        # Heuristic: if any column is non-numeric with 2-10 unique values and named like 'y|label|target|survived|churn'
        cols = [c["name"] for c in profile.get("columns", [])]
        lname = [c.lower() for c in cols]
        target_guess = None
        for k in ["target","label","y","survived","churn","default","clicked","class"]:
            if k in lname:
                target_guess = cols[lname.index(k)]
                break
        task = "classification" if target_guess else "regression"
        fallback = Plan(
            task_type=task,
            target_candidates=[target_guess] if target_guess else [],
            algorithms=(["random_forest","xgboost","logistic_regression"] if task=="classification"
                        else ["xgboost","ridge","knn"]),
            metrics=(["f1_macro","roc_auc_ovr","accuracy"] if task=="classification"
                     else ["rmse","mae","r2"]),
            cv=CVSpec(folds=5, stratified=(task=="classification"))
        )
        return fallback.model_dump()

def write_report(bundle: Dict[str, Any]) -> str:
    """
    Report writing will be implemented in Part 7; keep placeholder for now.
    """
    return "# 报告\n\n（报告生成功能将于后续步骤接入 OpenAI）"
