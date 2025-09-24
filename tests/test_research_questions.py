#!/usr/bin/env python3#!/usr/bin/env python3

""""""

测试研究问题建议功能测试研究问题建议功能

""""""

        print("正在调用 OpenAI 分析研究问题...")

import sys        

import os        # 动态导入并调用研究问题建议

import pandas as pd        suggest_research_questions = import_llm_agent()

        suggestions = suggest_research_questions(profile)port sys

# 动态导入 llm_agent 以避免 Pylance 错误import os

def import_llm_agent():import pandas as pd

    """动态导入 llm_agent 模块"""

    import importlib.util# 动态导入 llm_agent 以避免 Pylance 错误

    llm_agent_path = os.path.join(os.path.dirname(__file__), '..', 'automl-llm', 'app', 'llm_agent.py')def import_llm_agent():

    spec = importlib.util.spec_from_file_location("llm_agent", llm_agent_path)    """动态导入 llm_agent 模块"""

    llm_agent = importlib.util.module_from_spec(spec)    import importlib.util

    spec.loader.exec_module(llm_agent)    llm_agent_path = os.path.join(os.path.dirname(__file__), '..', 'automl-llm', 'app', 'llm_agent.py')

    return llm_agent.suggest_research_questions    spec = importlib.util.spec_from_file_location("llm_agent", llm_agent_path)

    llm_agent = importlib.util.module_from_spec(spec)

def create_sample_profile():    spec.loader.exec_module(llm_agent)

    """创建测试用的数据profile"""    return llm_agent.suggest_research_questions

    # 创建示例数据

    data = {def test_titanic_research_questions():

        'age': [25, 30, 35, 40, 45],    """测试 Titanic 数据集的研究问题建议"""

        'income': [50000, 60000, 70000, 80000, 90000],    

        'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],    print("=== Titanic 数据集研究问题建议测试 ===\n")

        'satisfaction': [7, 8, 9, 6, 8]    

    }    # 加载 Titanic 数据

    df = pd.DataFrame(data)    try:

            df = pd.read_csv('examples/titanic_small.csv')

    # 创建简化的profile        print(f"数据加载成功: {df.shape}")

    profile = {        print(f"列名: {list(df.columns)}")

        "shape": list(df.shape),        

        "columns": list(df.columns),        # 创建数据概览

        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},        profile = {

        "head": df.head().to_dict('records'),            "shape": list(df.shape),

        "missing": df.isnull().sum().to_dict(),            "columns": [

        "nunique": df.nunique().to_dict()                {

    }                    "name": col,

                        "dtype": str(df[col].dtype),

    return df, profile                    "missing_pct": round(df[col].isnull().mean() * 100, 2),

                    "unique_count": df[col].nunique(),

def test_research_questions():                    "sample_values": df[col].dropna().head(3).tolist()

    """测试研究问题建议功能"""                }

    print("=== 测试研究问题建议功能 ===\n")                for col in df.columns

                ],

    try:            "dataset_description": "泰坦尼克号乘客数据集，包含乘客的基本信息和生存状况"

        # 创建测试数据        }

        df, profile = create_sample_profile()        

        print("✅ 测试数据创建成功")        print("\n正在调用 OpenAI 分析研究问题...")

        print(f"数据形状: {df.shape}")        

                # 调用研究问题建议

        print("正在调用 OpenAI 分析研究问题...")        suggestions = suggest_research_questions(profile)

                

        # 动态导入并调用研究问题建议        print("\n" + "="*60)

        suggest_research_questions = import_llm_agent()        print("🔍 AI 研究问题建议结果")

        suggestions = suggest_research_questions(df)        print("="*60)

                

        print("✅ OpenAI 调用成功")        # 显示研究问题

        print(f"建议类型: {type(suggestions)}")        questions = suggestions.get("research_questions", [])

                if questions:

        # 显示结果            print("\n💡 推荐研究问题:")

        if isinstance(suggestions, dict):            for i, q in enumerate(questions):

            questions = suggestions.get('questions', [])                print(f"\n{i+1}. {q.get('question', '未知问题')}")

            print(f"\n📋 发现 {len(questions)} 个研究问题:")                print(f"   类型: {q.get('type', '未知')}")

            for i, q in enumerate(questions, 1):                print(f"   难度: {q.get('difficulty', '未知')}")

                print(f"{i}. {q}")                print(f"   目标: {q.get('target_column', '无')}")

                            print(f"   价值: {q.get('business_value', '未提供')}")

            recommendations = suggestions.get('recommendations', '')        

            if recommendations:        # 显示应用场景

                print(f"\n💡 分析建议:")        scenarios = suggestions.get("application_scenarios", [])

                if isinstance(recommendations, dict):        if scenarios:

                    for key, value in recommendations.items():            print(f"\n🎯 应用场景:")

                        print(f"  {key}: {value}")            for i, scenario in enumerate(scenarios):

                else:                print(f"   {i+1}. {scenario}")

                    print(f"  {recommendations}")        

        else:        # 显示关键洞察

            print(f"响应格式: {suggestions}")        insights = suggestions.get("key_insights_potential", [])

                if insights:

        return True            print(f"\n🔮 可能发现的洞察:")

                    for insight in insights:

    except Exception as e:                print(f"   • {insight}")

        print(f"❌ 测试失败: {e}")        

        return False        # 显示建议

        recommendations = suggestions.get("recommendations", {})

def main():        if recommendations:

    """主函数"""            priority = recommendations.get("priority_questions", [])

    print("🚀 开始测试研究问题建议功能...\n")            if priority:

                    print(f"\n🔥 优先研究问题:")

    success = test_research_questions()                for p in priority:

                        print(f"   • {p}")

    print("\n" + "="*50)        

    if success:        return True

        print("🎉 测试完成! 研究问题建议功能正常工作")        

    else:    except Exception as e:

        print("⚠️  测试未完全通过，请检查配置和网络连接")        print(f"❌ 测试失败: {str(e)}")

            return False

    print("\n提示:")

    print("- 确保 OpenAI API Key 已正确配置")def test_tags_research_questions():

    print("- 检查网络连接是否正常")    """测试 Tags 数据集的研究问题建议"""

    print("- 确认账户有足够的API调用余额")    

    print("\n\n=== Tags 数据集研究问题建议测试 ===\n")

if __name__ == "__main__":    

    main()    try:
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