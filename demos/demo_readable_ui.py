#!/usr/bin/env python3
"""
演示改进后的 AI 判定结果可读化显示
"""

import json

def demonstrate_readable_display():
    """演示可读化的判定结果显示"""
    
    # 模拟一个 AI 判定结果（类似你附件中的 JSON）
    sample_plan = {
        "task_type": "regression",
        "target_candidates": ["price", "salary"],
        "imbalance": {
            "is_imbalanced": None,
            "ratio": None
        },
        "algorithms": ["xgboost", "ridge", "knn"],
        "metrics": ["rmse", "mae", "r2"],
        "cv": {
            "folds": 5,
            "stratified": False
        }
    }
    
    print("=== 原始 JSON 格式 ===")
    print(json.dumps(sample_plan, indent=2, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("=== 改进后的可读化显示 ===")
    print("="*60)
    
    # 可读化显示函数（模拟 Streamlit 显示效果）
    def display_readable_plan_console(plan):
        """在控制台中模拟 Streamlit 的可读化显示"""
        
        # 任务类型
        task_type = plan.get("task_type", "未知")
        task_type_cn = {
            "classification": "分类任务", 
            "regression": "回归任务", 
            "clustering": "聚类任务"
        }.get(task_type, task_type)
        
        print(f"📊 **任务类型**: {task_type_cn}")
        if task_type == "classification":
            print("   🎯 这是一个分类任务，目标是预测离散的类别标签")
        elif task_type == "regression":
            print("   📈 这是一个回归任务，目标是预测连续的数值")
        elif task_type == "clustering":
            print("   🔍 这是一个聚类任务，目标是发现数据中的隐藏模式")
        
        # 目标候选列
        targets = plan.get("target_candidates", [])
        if targets:
            print(f"\n🎯 **推荐目标列**")
            for i, target in enumerate(targets):
                print(f"   {i+1}. `{target}`")
        else:
            print(f"\n⚠️  未找到明确的目标列，请手动选择")
        
        # 推荐算法
        algorithms = plan.get("algorithms", [])
        if algorithms:
            print(f"\n🤖 **推荐算法**")
            algo_names = {
                "xgboost": "XGBoost (极端梯度提升)",
                "ridge": "Ridge 回归 (岭回归)",
                "knn": "K-近邻算法",
                "random_forest": "随机森林",
                "linear_regression": "线性回归",
                "logistic_regression": "逻辑回归",
                "svm": "支持向量机",
                "mlp": "多层感知机"
            }
            
            for i, algo in enumerate(algorithms):
                algo_display = algo_names.get(algo, algo.replace('_', ' ').title())
                print(f"   {i+1}. {algo_display}")
        
        # 评估指标
        metrics = plan.get("metrics", [])
        if metrics:
            print(f"\n📏 **评估指标**")
            metric_names = {
                "rmse": "RMSE (均方根误差)",
                "mae": "MAE (平均绝对误差)", 
                "r2": "R² (决定系数)",
                "accuracy": "准确率",
                "f1": "F1 分数",
                "precision": "精确率",
                "recall": "召回率",
                "auc": "AUC (曲线下面积)"
            }
            
            for metric in metrics:
                metric_display = metric_names.get(metric, metric.upper())
                print(f"   • {metric_display}")
        
        # 交叉验证设置
        cv_info = plan.get("cv", {})
        if cv_info:
            print(f"\n✅ **交叉验证设置**")
            folds = cv_info.get("folds", 5)
            stratified = cv_info.get("stratified", False)
            
            print(f"   交叉验证折数: {folds} 折")
            stratify_text = "是" if stratified else "否"
            print(f"   分层采样: {stratify_text}")
        
        # 类别不平衡信息
        imbalance = plan.get("imbalance", {})
        if imbalance and imbalance.get("is_imbalanced"):
            print(f"\n⚖️  **数据不平衡警告**")
            ratio = imbalance.get("ratio")
            if ratio:
                print(f"   检测到数据不平衡，主要类别占比: {ratio:.1%}")
                print(f"   建议考虑使用类别权重平衡或采样技术")
    
    display_readable_plan_console(sample_plan)
    
    print("\n" + "="*60)
    print("🎉 **改进效果对比**")
    print("="*60)
    print("✅ 原始 JSON：技术性强，难以理解")
    print("✅ 可读化显示：")
    print("   • 中文标题和说明")
    print("   • 图标和视觉分组")
    print("   • 算法和指标的完整名称")
    print("   • 智能建议和警告")
    print("   • 分层次的信息展示")
    print("   • 可折叠的详细 JSON")

def demonstrate_auto_application():
    """演示自动应用 AI 判定结果到训练设置"""
    
    print("\n" + "="*60)
    print("=== 智能应用 AI 判定结果 ===")
    print("="*60)
    
    sample_plan = {
        "task_type": "classification",
        "target_candidates": ["survived"],
        "algorithms": ["xgboost", "random_forest", "knn"],
        "cv": {"folds": 5, "stratified": True}
    }
    
    print("🤖 **AI 判定结果自动应用**:")
    print(f"   ✅ 自动选择目标列: {sample_plan['target_candidates'][0]}")
    print(f"   ✅ 自动设置任务类型: {sample_plan['task_type']}")
    print(f"   ✅ 自动推荐算法: {', '.join(sample_plan['algorithms'])}")
    print(f"   ✅ 自动设置 CV 折数: {sample_plan['cv']['folds']}")
    
    print("\n💡 **用户体验改进**:")
    print("   • 减少手动配置的工作量")
    print("   • 降低错误配置的可能性") 
    print("   • 提供智能默认值")
    print("   • 保留手动调整的灵活性")

if __name__ == "__main__":
    demonstrate_readable_display()
    demonstrate_auto_application()
    
    print("\n🎯 **下一步**:")
    print("1. 启动 Streamlit 应用")
    print("2. 上传数据并点击 '智能判定（OpenAI）'")
    print("3. 查看美化后的判定结果")
    print("4. 观察训练设置如何自动应用 AI 推荐")