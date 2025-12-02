#!/usr/bin/env python3
"""
使用示例：如何避免和处理 cleandata 中的常见错误

这个脚本展示了如何正确选择任务类型和目标列，以避免 ValueError。
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'automl-llm'))

def auto_detect_task_type(df, target_col):
    """自动检测任务类型"""
    target_series = df[target_col].dropna()
    
    # 检查是否为数值类型
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    unique_count = target_series.nunique()
    total_count = len(target_series)
    
    if is_numeric and unique_count > 20 and unique_count / total_count > 0.05:
        return "regression", f"数值类型，{unique_count} 个唯一值"
    else:
        if not is_numeric:
            return "classification", f"非数值类型，{unique_count} 个类别"
        else:
            return "classification", f"数值类型但只有 {unique_count} 个唯一值，可能是分类"

def demonstrate_correct_usage():
    """演示正确的使用方法"""
    print("=== 正确使用 AutoML 系统的演示 ===\n")
    
    # 示例1：使用 titanic 数据
    print("示例1: Titanic 生存预测")
    try:
        df_titanic = pd.read_csv('examples/titanic_small.csv')
        print(f"数据形状: {df_titanic.shape}")
        print(f"列名: {list(df_titanic.columns)}")
        
        # 假设有一个 'survived' 列
        if 'Survived' in df_titanic.columns:
            target = 'Survived'
            task_type, reason = auto_detect_task_type(df_titanic, target)
            print(f"推荐任务类型: {task_type} ({reason})")
            
            from core import cleandata
            X_train, X_test, y_train, y_test, pre, col_info = cleandata.prepare(
                df_titanic, target, task_type
            )
            print(f"✅ 成功处理: 训练集大小 {X_train.shape}")
        
    except FileNotFoundError:
        print("titanic_small.csv 文件未找到")
    except Exception as e:
        print(f"处理 Titanic 数据时出错: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例2：使用 tags 数据（演示大量类别的分类）
    print("示例2: 电影标签分类（演示如何处理大量类别）")
    try:
        df_tags = pd.read_csv('examples/tags.csv')
        
        # 使用 tag 作为目标，但先过滤掉只出现一次的标签
        target = 'tag'
        print(f"原始数据: {df_tags.shape}")
        print(f"原始标签数量: {df_tags[target].nunique()}")
        
        # 只保留出现至少2次的标签
        tag_counts = df_tags[target].value_counts()
        popular_tags = tag_counts[tag_counts >= 2].index
        df_filtered = df_tags[df_tags[target].isin(popular_tags)].copy()
        
        print(f"过滤后数据: {df_filtered.shape}")
        print(f"过滤后标签数量: {df_filtered[target].nunique()}")
        
        task_type, reason = auto_detect_task_type(df_filtered, target)
        print(f"推荐任务类型: {task_type} ({reason})")
        
        from core import cleandata
        X_train, X_test, y_train, y_test, pre, col_info = cleandata.prepare(
            df_filtered, target, task_type
        )
        print(f"✅ 成功处理: 训练集大小 {X_train.shape}")
        
        # 进一步演示：运行训练并生成排行榜与模型文件
        try:
            from core import train
            leaderboard, artifacts = train.run_all(
                X_train, y_train, X_test, y_test,
                task_type=task_type,
                picked_models=["rf", "xgb"],
                preprocessor=pre,
                n_iter=10,  # 演示用较小迭代
                cv_folds=3,
                artifacts_dir="artifacts"
            )
            print("训练排行榜:\n", leaderboard.head())
            print("模型文件示例:", [v.get("model_path") for v in artifacts.values()])
        except Exception as e:
            print(f"训练运行时出错: {e}")
        
    except Exception as e:
        print(f"处理 Tags 数据时出错: {e}")
    
    print("\n" + "="*50 + "\n")
    
def demonstrate_error_scenarios():
    """演示常见错误场景和解决方法"""
    print("=== 常见错误场景和解决方法 ===\n")
    
    print("错误场景1: 字符串目标 + 回归任务")
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': ['funny', 'sad', 'happy', 'angry', 'funny']
    }
    df = pd.DataFrame(data)
    
    try:
        from core import cleandata
        cleandata.prepare(df, 'target', 'regression')
    except ValueError as e:
        print(f"❌ 错误信息: {str(e)}")
        print("✅ 解决方法: 将任务类型改为 'classification'\n")
    
    print("错误场景2: 正确的解决方案")
    try:
        from core import cleandata
        X_train, X_test, y_train, y_test, pre, col_info = cleandata.prepare(
            df, 'target', 'classification'
        )
        print(f"✅ 成功: 训练集大小 {X_train.shape}")
    except Exception as e:
        print(f"仍然出错: {e}")

if __name__ == "__main__":
    demonstrate_correct_usage()
    demonstrate_error_scenarios()
    
    print("\n=== 使用建议 ===")
    print("1. 在选择目标列后，使用自动检测功能确定任务类型")
    print("2. 对于分类任务，确保每个类别至少有2个样本")
    print("3. 对于回归任务，确保目标列是数值类型")
    print("4. 如果目标列包含字符串，通常应该选择分类任务")
    print("5. 如果遇到错误，仔细阅读错误消息中的建议")