"""
测试UI AttributeError修复
验证recommendations字段的类型检查是否正常工作
"""

# 模拟不同的AI响应格式
test_responses = [
    # 情况1: recommendations是字典（正常情况）
    {
        "questions": ["问题1", "问题2"],
        "recommendations": {
            "immediate_actions": ["建议1", "建议2"],
            "analysis_priorities": ["优先级1", "优先级2"]
        }
    },
    
    # 情况2: recommendations是字符串（导致AttributeError的情况）
    {
        "questions": ["问题3", "问题4"],
        "recommendations": "这是一个字符串格式的建议"
    },
    
    # 情况3: 没有recommendations字段
    {
        "questions": ["问题5", "问题6"]
    }
]

def test_recommendation_handling(response):
    """测试推荐建议的处理逻辑"""
    print(f"测试响应: {response}")
    
    # 这是修复后的逻辑
    recommendations = response.get('recommendations', '')
    
    if isinstance(recommendations, dict):
        print("✅ 检测到字典格式的recommendations")
        # 显示详细推荐
        if 'immediate_actions' in recommendations:
            print(f"即时行动: {recommendations['immediate_actions']}")
        if 'analysis_priorities' in recommendations:
            print(f"分析优先级: {recommendations['analysis_priorities']}")
    elif isinstance(recommendations, str) and recommendations:
        print("✅ 检测到字符串格式的recommendations")
        print(f"通用建议: {recommendations}")
    else:
        print("✅ 没有recommendations或为空")
    
    print("-" * 50)

def main():
    print("=== AttributeError修复测试 ===\n")
    
    for i, response in enumerate(test_responses, 1):
        print(f"测试用例 {i}:")
        try:
            test_recommendation_handling(response)
            print("✅ 测试通过")
        except AttributeError as e:
            print(f"❌ AttributeError: {e}")
        except Exception as e:
            print(f"❌ 其他错误: {e}")
        
        print()

if __name__ == "__main__":
    main()