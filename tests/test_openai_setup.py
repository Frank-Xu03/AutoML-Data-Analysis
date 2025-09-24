#!/usr/bin/env python3
"""
验证 OpenAI API Key 设置的测试脚本
"""

import os
import sys

# 添加 llm_agent 模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'automl-llm', 'app'))

def test_env_loading():
    """测试环境变量加载"""
    print("=== OpenAI API Key 设置验证 ===\n")
    
    # 尝试加载 .env 文件
    try:
        from dotenv import load_dotenv
        
        # 从项目根目录加载 .env 文件
        project_root = os.path.dirname(os.path.dirname(__file__))
        env_path = os.path.join(project_root, '.env')
        
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"✅ 找到 .env 文件: {env_path}")
        else:
            print(f"⚠️  .env 文件不存在: {env_path}")
            
    except ImportError:
        print("❌ python-dotenv 未安装")
        return False
    
    # 检查 OPENAI_API_KEY 环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 环境变量未设置")
        print("\n请按照以下步骤设置:")
        print("1. 打开 .env 文件")
        print("2. 将 'your_openai_api_key_here' 替换为你的实际 API Key")
        print("3. 保存文件后重新运行此脚本")
        return False
    
    if api_key == "your_openai_api_key_here":
        print("⚠️  检测到默认占位符，请设置你的实际 API Key")
        print("\n请按照以下步骤设置:")
        print("1. 打开 .env 文件") 
        print("2. 将 'your_openai_api_key_here' 替换为你的实际 API Key")
        print("3. API Key 格式应该类似: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return False
    
    # 检查 API Key 格式
    if not api_key.startswith('sk-'):
        print(f"⚠️  API Key 格式可能不正确: {api_key[:10]}...")
        print("OpenAI API Key 通常以 'sk-' 开头")
        return False
    
    print(f"✅ OPENAI_API_KEY 已设置")
    print(f"   Key 前缀: {api_key[:10]}...")
    print(f"   Key 长度: {len(api_key)} 字符")
    
    return True

def test_openai_connection():
    """测试 OpenAI API 连接"""
    print("\n=== OpenAI API 连接测试 ===\n")
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ 无法测试连接：API Key 未设置")
            return False
            
        client = OpenAI(api_key=api_key)
        
        # 尝试一个简单的 API 调用
        print("正在测试 API 连接...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello! Just testing the connection. Please respond with 'OK'."}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ OpenAI API 连接成功!")
        print(f"   响应: {result}")
        return True
        
    except ImportError:
        print("❌ OpenAI 库未安装")
        return False
    except Exception as e:
        print(f"❌ OpenAI API 连接失败: {str(e)}")
        
        # 提供常见错误的解决建议
        error_str = str(e).lower()
        if "authentication" in error_str or "invalid api key" in error_str:
            print("\n💡 建议:")
            print("1. 检查 API Key 是否正确")
            print("2. 确认 API Key 是否有效且未过期")
            print("3. 检查 OpenAI 账户是否有足够的余额")
        elif "quota" in error_str or "billing" in error_str:
            print("\n💡 建议:")
            print("1. 检查 OpenAI 账户余额")
            print("2. 确认账户的使用限额")
        elif "network" in error_str or "connection" in error_str:
            print("\n💡 建议:")
            print("1. 检查网络连接")
            print("2. 确认是否需要代理设置")
            
        return False

def test_llm_agent_import():
    """测试 llm_agent 模块导入"""
    print("\n=== LLM Agent 模块测试 ===\n")
    
    try:
        # 动态导入以避免 Pylance 检查错误
        import importlib.util
        llm_agent_path = os.path.join(os.path.dirname(__file__), '..', 'automl-llm', 'app', 'llm_agent.py')
        
        if not os.path.exists(llm_agent_path):
            print(f"❌ llm_agent.py 文件不存在: {llm_agent_path}")
            return False
            
        spec = importlib.util.spec_from_file_location("llm_agent", llm_agent_path)
        llm_agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llm_agent)
        
        detect_task = llm_agent.detect_task
        print("✅ llm_agent 模块导入成功")
        
        # 测试一个简单的检测任务
        sample_profile = {
            "shape": [100, 5],
            "columns": ["feature1", "feature2", "feature3", "target"],
            "dtypes": {"feature1": "float64", "target": "object"},
            "target_candidates": ["target"]
        }
        
        # 注意：这会实际调用 OpenAI API
        print("正在测试任务检测功能...")
        try:
            result = detect_task(sample_profile, "This is a test")
            print("✅ 任务检测功能正常")
            return True
        except Exception as e:
            if "OPENAI_API_KEY not set" in str(e):
                print("❌ API Key 未正确设置")
            else:
                print(f"⚠️  任务检测测试失败: {str(e)}")
            return False
            
    except ImportError as e:
        print(f"❌ 无法导入 llm_agent: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始验证 OpenAI 设置...\n")
    
    # 测试环境变量加载
    env_ok = test_env_loading()
    
    if env_ok:
        # 测试 API 连接
        api_ok = test_openai_connection()
        
        if api_ok:
            # 测试 LLM Agent
            agent_ok = test_llm_agent_import()
    
    print("\n" + "="*50)
    print("验证完成!")
    print("\n如果所有测试都通过，你现在可以使用智能判定功能了。")
    print("在 Streamlit 应用中点击 '智能判定（OpenAI）' 按钮即可。")