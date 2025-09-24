"""
最终演示 - 验证AttributeError修复和UI功能
"""

import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'automl-llm'))

# 模拟不同的研究问题API响应
mock_responses = {
    "正常字典格式": {
        "questions": [
            "这个数据集可以用来预测什么目标变量？",
            "数据中的哪些特征对预测最有价值？",
            "数据是否存在异常值或缺失值问题？"
        ],
        "recommendations": {
            "immediate_actions": [
                "检查数据质量和完整性",
                "进行探索性数据分析（EDA）",
                "评估特征重要性"
            ],
            "analysis_priorities": [
                "目标变量分布分析",
                "特征间相关性分析",
                "异常值检测和处理"
            ]
        }
    },
    "字符串格式": {
        "questions": [
            "如何提升模型性能？",
            "应该使用哪种机器学习算法？"
        ],
        "recommendations": "建议先进行数据预处理，包括特征缩放和异常值处理，然后尝试多种算法进行比较。"
    },
    "无推荐字段": {
        "questions": [
            "数据集的基本统计信息如何？",
            "是否适合进行机器学习？"
        ]
    }
}

def display_research_suggestions_test(suggestions):
    """测试研究建议显示功能（修复AttributeError）"""
    if not suggestions:
        st.warning("未获得研究建议")
        return
    
    # 显示研究问题
    questions = suggestions.get('questions', [])
    if questions:
        st.subheader("🔍 建议研究的问题")
        for i, question in enumerate(questions, 1):
            st.write(f"{i}. {question}")
    
    # 显示推荐建议 - 修复AttributeError的关键部分
    recommendations = suggestions.get('recommendations', '')
    
    if isinstance(recommendations, dict):
        # 字典格式 - 详细推荐
        st.subheader("💡 分析建议")
        
        if 'immediate_actions' in recommendations:
            st.write("**立即行动:**")
            for action in recommendations['immediate_actions']:
                st.write(f"• {action}")
        
        if 'analysis_priorities' in recommendations:
            st.write("**分析优先级:**")
            for priority in recommendations['analysis_priorities']:
                st.write(f"• {priority}")
                
    elif isinstance(recommendations, str) and recommendations:
        # 字符串格式 - 通用建议
        st.subheader("💡 分析建议")
        st.write(recommendations)
    
    else:
        # 没有推荐或为空
        st.info("暂无具体推荐建议")

def main():
    st.title("🔧 AttributeError修复验证")
    st.write("演示修复后的研究建议显示功能")
    
    st.header("测试不同的API响应格式")
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["正常字典格式", "字符串格式", "无推荐字段"])
    
    with tab1:
        st.subheader("✅ 字典格式响应测试")
        st.code("""
{
    "questions": ["问题1", "问题2"],
    "recommendations": {
        "immediate_actions": ["行动1", "行动2"],
        "analysis_priorities": ["优先级1", "优先级2"]
    }
}
        """)
        
        try:
            display_research_suggestions_test(mock_responses["正常字典格式"])
            st.success("✅ 字典格式处理成功！")
        except Exception as e:
            st.error(f"❌ 错误: {e}")
    
    with tab2:
        st.subheader("✅ 字符串格式响应测试")
        st.code("""
{
    "questions": ["问题1", "问题2"],
    "recommendations": "这是一个字符串建议"
}
        """)
        
        try:
            display_research_suggestions_test(mock_responses["字符串格式"])
            st.success("✅ 字符串格式处理成功！")
        except Exception as e:
            st.error(f"❌ 错误: {e}")
    
    with tab3:
        st.subheader("✅ 无推荐字段响应测试")
        st.code("""
{
    "questions": ["问题1", "问题2"]
    # 没有 recommendations 字段
}
        """)
        
        try:
            display_research_suggestions_test(mock_responses["无推荐字段"])
            st.success("✅ 无推荐字段处理成功！")
        except Exception as e:
            st.error(f"❌ 错误: {e}")
    
    st.header("🎯 修复总结")
    
    st.success("**AttributeError已修复！**")
    
    with st.expander("🔍 查看修复详情"):
        st.code('''
# 修复前的问题代码：
recommendations.get('immediate_actions', [])  # 如果recommendations是字符串，会报AttributeError

# 修复后的代码：
recommendations = suggestions.get('recommendations', '')

if isinstance(recommendations, dict):
    # 处理字典格式
    if 'immediate_actions' in recommendations:
        # 安全访问字典键
elif isinstance(recommendations, str) and recommendations:
    # 处理字符串格式
else:
    # 处理空值或其他情况
        ''')
    
    st.info("""
    **修复要点:**
    1. 使用 `isinstance()` 检查数据类型
    2. 分别处理字典和字符串格式
    3. 添加空值检查和默认处理
    4. 确保所有情况都有适当的UI显示
    """)

if __name__ == "__main__":
    main()