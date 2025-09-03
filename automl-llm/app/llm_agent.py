import os, json
from typing import Dict, Any
from openai import OpenAI

def _client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def detect_task(profile: Dict[str, Any], user_question: str) -> Dict[str, Any]:
    # 先放一个占位实现，后续我们替换为真实 OpenAI 调用
    return {
        "task_type": "classification",
        "target_candidates": [],
        "algorithms": ["logistic_regression","random_forest","xgboost"],
        "metrics": ["f1_macro","roc_auc_ovr"],
        "cv": {"folds": 5, "stratified": True}
    }

def write_report(bundle: Dict[str, Any]) -> str:
    # 占位实现：返回简单 Markdown
    return "# 自动化报告\n\n即将由 OpenAI 生成..."
