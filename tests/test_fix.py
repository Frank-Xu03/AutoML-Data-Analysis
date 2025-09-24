#!/usr/bin/env python3
"""
测试脚本：验证 cleandata.py 修复是否有效
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'automl-llm'))

from core import cleandata

def test_string_target_with_regression():
    """测试：当目标列包含字符串但任务类型设为回归时的处理"""
    print("测试1: 字符串目标列 + 回归任务类型")
    
    # 创建包含字符串目标的测试数据
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': ['funny', 'sad', 'happy', 'angry', 'funny']
    }
    df = pd.DataFrame(data)
    
    try:
        X_train, X_test, y_train, y_test, pre, col_info = cleandata.prepare(
            df, 'target', 'regression'
        )
        print("❌ 应该抛出错误但没有")
    except ValueError as e:
        print(f"✅ 正确抛出错误: {str(e)}")
    except Exception as e:
        print(f"❌ 抛出了意外错误: {str(e)}")

def test_string_target_with_classification():
    """测试：字符串目标列 + 分类任务类型"""
    print("\n测试2: 字符串目标列 + 分类任务类型")
    
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': ['funny', 'sad', 'happy', 'angry', 'funny']
    }
    df = pd.DataFrame(data)
    
    try:
        X_train, X_test, y_train, y_test, pre, col_info = cleandata.prepare(
            df, 'target', 'classification'
        )
        print(f"✅ 成功处理分类任务")
        print(f"   训练集大小: {X_train.shape}")
        print(f"   目标编码后唯一值: {len(set(y_train))}")
    except Exception as e:
        print(f"❌ 处理分类任务时出错: {str(e)}")

def test_numeric_target_with_regression():
    """测试：数值目标列 + 回归任务类型"""
    print("\n测试3: 数值目标列 + 回归任务类型")
    
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [1.5, 2.7, 3.1, 4.8, 5.9]
    }
    df = pd.DataFrame(data)
    
    try:
        X_train, X_test, y_train, y_test, pre, col_info = cleandata.prepare(
            df, 'target', 'regression'
        )
        print(f"✅ 成功处理回归任务")
        print(f"   训练集大小: {X_train.shape}")
        print(f"   目标值范围: {y_train.min():.2f} - {y_train.max():.2f}")
    except Exception as e:
        print(f"❌ 处理回归任务时出错: {str(e)}")

def test_tags_csv_example():
    """测试：使用实际的 tags.csv 数据"""
    print("\n测试4: 使用 tags.csv 数据")
    
    try:
        # 加载数据
        df = pd.read_csv('examples/tags.csv')
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"tag列唯一值数量: {df['tag'].nunique()}")
        print(f"tag列样例值: {df['tag'].head().tolist()}")
        
        # 测试用 tag 作为目标的分类任务
        print("\n尝试 tag 列作为分类目标...")
        X_train, X_test, y_train, y_test, pre, col_info = cleandata.prepare(
            df, 'tag', 'classification'
        )
        print(f"✅ 成功处理 tag 分类任务")
        print(f"   训练集大小: {X_train.shape}")
        print(f"   类别数量: {len(set(y_train))}")
        
    except Exception as e:
        print(f"❌ 处理 tags.csv 时出错: {str(e)}")

if __name__ == "__main__":
    print("开始测试 cleandata.py 修复...")
    test_string_target_with_regression()
    test_string_target_with_classification()
    test_numeric_target_with_regression()
    test_tags_csv_example()
    print("\n测试完成!")