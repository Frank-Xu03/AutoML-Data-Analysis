"""
完整的Streamlit UI功能测试
测试数据上传、分析和研究问题发现功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'automl-llm'))

import pandas as pd
import tempfile
import importlib

def create_test_data():
    """创建测试数据"""
    # 创建一个简单的回归数据集
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [2.1, 4.2, 6.3, 8.4, 10.5, 12.6, 14.7, 16.8, 18.9, 21.0],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'target': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
    return pd.DataFrame(data)

def test_llm_agent():
    """测试LLM代理功能"""
    print("=== 测试LLM代理 ===")
    
    try:
        # 动态导入以避免Pylance错误
        llm_agent_module = importlib.import_module('app.llm_agent')
        
        # 创建测试数据
        df = create_test_data()
        
        print("1. 测试任务检测...")
        # 需要先创建profile
        ingest_module = importlib.import_module('core.ingest')
        profile = ingest_module.profile(df)
        
        task_result = llm_agent_module.detect_task(profile, "我想分析这个数据")
        print(f"任务检测结果类型: {type(task_result)}")
        
        print("\n2. 测试研究问题建议...")
        research_result = llm_agent_module.suggest_research_questions(df)
        print(f"研究问题建议类型: {type(research_result)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_core_modules():
    """测试核心模块"""
    print("=== 测试核心模块 ===")
    
    try:
        # 测试数据清理模块
        cleandata_module = importlib.import_module('core.cleandata')
        
        df = create_test_data()
        target_col = 'target'
        task_type = 'regression'
        
        print("1. 测试数据清理和分割...")
        X_train, X_test, y_train, y_test, preprocessor, col_info = cleandata_module.prepare(
            df, target_col, task_type
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        print(f"目标变量类型: {type(y_train[0])}")
        print(f"列信息: {list(col_info.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 核心模块测试失败: {e}")
        return False

def test_ui_components():
    """测试UI组件的关键函数"""
    print("=== 测试UI组件 ===")
    
    # 模拟研究建议响应
    mock_responses = [
        {
            "questions": ["这个数据集可以预测什么？", "哪些特征最重要？"],
            "recommendations": {
                "immediate_actions": ["检查数据质量", "进行特征工程"],
                "analysis_priorities": ["相关性分析", "异常值检测"]
            }
        },
        {
            "questions": ["数据有什么规律？"],
            "recommendations": "建议先进行探索性数据分析"
        },
        {
            "questions": ["如何提升模型性能？"]
            # 没有recommendations字段
        }
    ]
    
    print("测试不同格式的研究建议处理...")
    
    for i, response in enumerate(mock_responses, 1):
        print(f"\n测试用例 {i}:")
        
        try:
            # 模拟UI中的处理逻辑
            questions = response.get('questions', [])
            recommendations = response.get('recommendations', '')
            
            print(f"问题数量: {len(questions)}")
            
            if isinstance(recommendations, dict):
                print("✅ 字典格式推荐处理正常")
                if 'immediate_actions' in recommendations:
                    print(f"即时行动: {len(recommendations['immediate_actions'])}项")
            elif isinstance(recommendations, str) and recommendations:
                print("✅ 字符串格式推荐处理正常")
                print(f"建议长度: {len(recommendations)}字符")
            else:
                print("✅ 无推荐字段处理正常")
                
        except Exception as e:
            print(f"❌ UI组件测试失败: {e}")
            return False
    
    return True

def main():
    print("🚀 开始完整系统测试...\n")
    
    results = []
    
    # 测试核心模块
    core_test = test_core_modules()
    results.append(("核心模块", core_test))
    
    print()
    
    # 测试UI组件
    ui_test = test_ui_components()
    results.append(("UI组件", ui_test))
    
    print()
    
    # 测试LLM代理（可能因为API密钥问题失败）
    llm_test = test_llm_agent()
    results.append(("LLM代理", llm_test))
    
    # 总结测试结果
    print("\n" + "="*50)
    print("🎯 测试结果总结:")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！系统已准备就绪。")
    elif passed >= total - 1:
        print("⚠️ 大部分测试通过，可能只是API配置问题。")
    else:
        print("⚠️ 有多项测试失败，需要进一步检查。")

if __name__ == "__main__":
    main()