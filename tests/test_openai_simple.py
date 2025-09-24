#!/usr/bin/env python3
"""
简化的 OpenAI 设置测试脚本 - 无 Pylance 导入错误
"""

import os
import sys
import json

def test_basic_setup():
    """测试基本的 OpenAI 设置"""
    print("=== OpenAI 基础设置检查 ===\n")
    
    # 检查 .env 文件
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        print("✅ .env 文件存在")
        
        # 尝试加载 dotenv
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            print("✅ python-dotenv 已安装并加载")
        except ImportError:
            print("⚠️  python-dotenv 未安装，请运行: pip install python-dotenv")
            return False
    else:
        print("❌ .env 文件不存在")
        return False
    
    # 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return False
    elif api_key == "your_openai_api_key_here":
        print("⚠️  请替换默认的 API Key 占位符")
        return False
    elif not api_key.startswith('sk-'):
        print("⚠️  API Key 格式可能不正确")
        return False
    else:
        print(f"✅ OPENAI_API_KEY 已正确设置 ({api_key[:10]}...)")
        return True

def test_openai_library():
    """测试 OpenAI 库"""
    print("\n=== OpenAI 库测试 ===\n")
    
    try:
        from openai import OpenAI
        print("✅ OpenAI 库已安装")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ API Key 未设置，跳过连接测试")
            return False
            
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI 客户端创建成功")
        return True
        
    except ImportError:
        print("❌ OpenAI 库未安装，请运行: pip install openai")
        return False
    except Exception as e:
        print(f"❌ OpenAI 客户端创建失败: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n=== 文件结构检查 ===\n")
    
    base_dir = os.path.dirname(__file__)
    required_files = [
        '../automl-llm/app/llm_agent.py',
        '../automl-llm/app/ui_streamlit.py',
        'prompts/task_detection.txt',
        'prompts/research_questions.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            all_exist = False
    
    return all_exist

def test_simple_api_call():
    """测试简单的 API 调用"""
    print("\n=== API 连接测试 ===\n")
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ API Key 未设置")
            return False
            
        client = OpenAI(api_key=api_key)
        
        print("正在测试 API 连接...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "请回复'测试成功'"}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ API 连接成功！响应: {result}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ API 调用失败: {error_msg}")
        
        # 提供具体的解决建议
        if "authentication" in error_msg.lower():
            print("💡 建议: 检查 API Key 是否正确")
        elif "quota" in error_msg.lower():
            print("💡 建议: 检查 OpenAI 账户余额")
        elif "rate" in error_msg.lower():
            print("💡 建议: API 调用频率过高，请稍后再试")
        else:
            print("💡 建议: 检查网络连接或 OpenAI 服务状态")
            
        return False

def create_sample_profile():
    """创建示例数据配置文件用于测试"""
    return {
        "shape": [150, 5],
        "columns": [
            {
                "name": "sepal_length",
                "dtype": "float64",
                "missing_pct": 0.0,
                "unique_count": 35,
                "sample_values": [5.1, 4.9, 4.7]
            },
            {
                "name": "sepal_width", 
                "dtype": "float64",
                "missing_pct": 0.0,
                "unique_count": 23,
                "sample_values": [3.5, 3.0, 3.2]
            },
            {
                "name": "species",
                "dtype": "object", 
                "missing_pct": 0.0,
                "unique_count": 3,
                "sample_values": ["setosa", "versicolor", "virginica"]
            }
        ],
        "dataset_description": "鸢尾花数据集 - 经典分类数据"
    }

def test_research_questions_workflow():
    """测试研究问题发现工作流"""
    print("\n=== 研究问题发现功能测试 ===\n")
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ 需要 API Key 进行此测试")
            return False
            
        # 读取 research_questions.txt prompt
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', 'research_questions.txt')
        if not os.path.exists(prompt_path):
            print("❌ research_questions.txt 文件不存在")
            return False
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        client = OpenAI(api_key=api_key)
        sample_profile = create_sample_profile()
        
        print("正在测试研究问题发现功能...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({
                    "dataset_profile": sample_profile
                }, ensure_ascii=False)}
            ],
        )
        
        result = json.loads(response.choices[0].message.content)
        
        print("✅ 研究问题发现功能测试成功！")
        
        # 显示部分结果
        questions = result.get("research_questions", [])
        if questions:
            print(f"\n发现 {len(questions)} 个研究问题:")
            for i, q in enumerate(questions[:2]):  # 只显示前2个
                print(f"  {i+1}. {q.get('question', '未知')}")
        
        scenarios = result.get("application_scenarios", [])
        if scenarios:
            print(f"\n应用场景: {scenarios[0] if scenarios else '无'}")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 OpenAI 设置完整性测试\n")
    
    # 运行所有测试
    tests = [
        ("基础设置", test_basic_setup),
        ("OpenAI 库", test_openai_library), 
        ("文件结构", test_file_structure),
        ("API 连接", test_simple_api_call),
        ("问题发现功能", test_research_questions_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n通过率: {passed}/{len(results)} ({passed/len(results)*100:.0f}%)")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！你的 AutoML 系统已准备就绪！")
        print("\n🚀 现在可以运行:")
        print("   streamlit run automl-llm/app/ui_streamlit.py")
    else:
        print("\n⚠️ 部分测试失败，请检查上述错误并修复")

if __name__ == "__main__":
    main()