#!/usr/bin/env python3
"""
测试研究问题建议功能
"""

import sys
import os
import pandas as pd

# 动态导入 llm_agent 以避免 Pylance 错误
def import_llm_agent():
    """动态导入 llm_agent 模块"""
    import importlib.util
    llm_agent_path = os.path.join(os.path.dirname(__file__), '..', 'automl-llm', 'app', 'llm_agent.py')
    spec = importlib.util.spec_from_file_location("llm_agent", llm_agent_path)
    llm_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_agent)
    return llm_agent.suggest_research_questions

def test_titanic_research_questions():
    """测试 Titanic 数据集的研究问题建议"""
    
    print("=== Titanic 数据集研究问题建议测试 ===\n")
    
    # 加载 Titanic 数据
    try:
        df = pd.read_csv('examples/titanic_small.csv')
        print(f"数据加载成功: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 创建数据概览
        profile = {
            "shape": list(df.shape),
            "columns": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "missing_pct": round(df[col].isnull().mean() * 100, 2),
                    "unique_count": df[col].nunique(),
                    "sample_values": df[col].dropna().head(3).tolist()
                }
                for col in df.columns
            ],
            "dataset_description": "泰坦尼克号乘客数据集，包含乘客的基本信息和生存状况"
        }
        
        print("\n正在调用 OpenAI 分析研究问题...")
        
        # 动态导入并调用研究问题建议
        suggest_research_questions = import_llm_agent()
        suggestions = suggest_research_questions(profile)
        
        print("\n" + "="*60)
        print("🔍 AI 研究问题建议结果")
        print("="*60)
        
        # 显示研究问题
        questions = suggestions.get("research_questions", [])
        if questions:
            print("\n💡 推荐研究问题:")
            for i, q in enumerate(questions):
                print(f"\n{i+1}. {q.get('question', '未知问题')}")
                print(f"   类型: {q.get('type', '未知')}")
                print(f"   难度: {q.get('difficulty', '未知')}")
                print(f"   目标: {q.get('target_column', '无')}")
                print(f"   价值: {q.get('business_value', '未提供')}")
        
        # 显示应用场景
        scenarios = suggestions.get("application_scenarios", [])
        if scenarios:
            print(f"\n🎯 应用场景:")
            for i, scenario in enumerate(scenarios):
                print(f"   {i+1}. {scenario}")
        
        # 显示关键洞察
        insights = suggestions.get("key_insights_potential", [])
        if insights:
            print(f"\n🔮 可能发现的洞察:")
            for insight in insights:
                print(f"   • {insight}")
        
        # 显示建议
        recommendations = suggestions.get("recommendations", {})
        if recommendations:
            priority = recommendations.get("priority_questions", [])
            if priority:
                print(f"\n🔥 优先研究问题:")
                for p in priority:
                    print(f"   • {p}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def test_tags_research_questions():
    """测试 Tags 数据集的研究问题建议"""
    
    print("\n\n=== Tags 数据集研究问题建议测试 ===\n")
    
    try:
        df = pd.read_csv('examples/tags.csv')
        print(f"数据加载成功: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 创建数据概览
        profile = {
            "shape": list(df.shape),
            "columns": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "missing_pct": round(df[col].isnull().mean() * 100, 2),
                    "unique_count": df[col].nunique(),
                    "sample_values": df[col].dropna().head(3).tolist()
                }
                for col in df.columns
            ],
            "dataset_description": "电影标签数据集，包含用户对电影的标签评价"
        }
        
        print("\n正在调用 OpenAI 分析研究问题...")
        
        suggest_research_questions = import_llm_agent()
        suggestions = suggest_research_questions(profile)
        
        print("\n" + "="*60)
        print("🔍 AI 研究问题建议结果")
        print("="*60)
        
        # 简化显示主要结果
        questions = suggestions.get("research_questions", [])
        if questions:
            print(f"\n💡 发现 {len(questions)} 个研究问题:")
            for i, q in enumerate(questions):
                print(f"\n{i+1}. {q.get('question', '未知问题')}")
                print(f"   价值: {q.get('business_value', '未提供')[:100]}...")
        
        scenarios = suggestions.get("application_scenarios", [])
        if scenarios:
            print(f"\n🎯 应用场景 ({len(scenarios)} 个):")
            for scenario in scenarios[:3]:  # 只显示前3个
                print(f"   • {scenario}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def demo_without_openai():
    """演示没有 OpenAI 时的回退机制"""
    
    print("\n\n=== 回退机制演示 ===\n")
    
    # 模拟没有 API Key 的情况
    os.environ.pop('OPENAI_API_KEY', None)
    
    profile = {
        "shape": [100, 5],
        "columns": [
            {"name": "age", "dtype": "int64", "missing_pct": 5.0, "unique_count": 50},
            {"name": "income", "dtype": "float64", "missing_pct": 2.0, "unique_count": 80},
            {"name": "category", "dtype": "object", "missing_pct": 0.0, "unique_count": 3}
        ]
    }
    
    try:
        suggest_research_questions = import_llm_agent()
        suggestions = suggest_research_questions(profile)
        print("✅ 回退机制正常工作")
        print(f"问题数量: {len(suggestions.get('research_questions', []))}")
        print(f"建议: {suggestions.get('recommendations', '无')}")
    except Exception as e:
        print(f"❌ 回退机制失败: {str(e)}")

if __name__ == "__main__":
    print("开始测试研究问题建议功能...\n")
    
    # 测试不同数据集
    success1 = test_titanic_research_questions()
    success2 = test_tags_research_questions()
    
    # 测试回退机制
    demo_without_openai()
    
    print(f"\n{'='*60}")
    print("测试总结:")
    print(f"Titanic 测试: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"Tags 测试: {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1 or success2:
        print("\n🎉 研究问题建议功能已准备就绪！")
        print("现在可以在 Streamlit 应用中使用 '🔍 发现研究问题' 功能")
    else:
        print("\n⚠️ 请检查 OpenAI API 设置和网络连接")