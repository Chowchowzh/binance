#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征降维脚本
通过特征选择和降维技术减少内存使用
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def reduce_features_by_importance(input_path: str, output_path: str, 
                                max_features: int = 50, method: str = 'random_forest'):
    """
    通过特征重要性减少特征数量
    
    Args:
        input_path: 输入数据文件路径
        output_path: 输出数据文件路径
        max_features: 保留的最大特征数
        method: 特征选择方法 ('random_forest', 'mutual_info', 'f_test')
    """
    print(f"🔍 开始特征降维: {method}")
    print(f"   输入文件: {input_path}")
    print(f"   输出文件: {output_path}")
    print(f"   目标特征数: {max_features}")
    
    # 加载数据
    print("\n📁 加载数据...")
    df = pd.read_parquet(input_path)
    print(f"   原始数据形状: {df.shape}")
    
    # 分离特征和目标
    feature_columns = [col for col in df.columns 
                      if col not in ['target', 'future_return', 'open_time', 'close_time']]
    
    X = df[feature_columns].values
    y = df['target'].values
    
    print(f"   原始特征数: {len(feature_columns)}")
    print(f"   样本数: {len(X)}")
    
    # 处理NaN值
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 选择特征选择方法
    print(f"\n⚙️  使用 {method} 方法选择特征...")
    
    if method == 'random_forest':
        # 使用随机森林进行特征选择
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=6)
        rf.fit(X, y)
        
        feature_importance = rf.feature_importances_
        feature_names = np.array(feature_columns)
        
        # 按重要性排序
        sorted_indices = np.argsort(feature_importance)[::-1]
        selected_indices = sorted_indices[:max_features]
        selected_features = feature_names[selected_indices].tolist()
        
        print(f"   🌟 Top 10 重要特征:")
        for i in range(min(10, len(selected_indices))):
            idx = sorted_indices[i]
            print(f"      {i+1}. {feature_columns[idx]}: {feature_importance[idx]:.4f}")
    
    elif method == 'mutual_info':
        # 使用互信息进行特征选择
        selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_columns[i] for i, selected in enumerate(selected_mask) if selected]
        
        scores = selector.scores_
        selected_scores = scores[selected_mask]
        
        print(f"   🌟 Top 10 互信息得分:")
        sorted_pairs = sorted(zip(selected_features, selected_scores), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_pairs[:10]):
            print(f"      {i+1}. {feature}: {score:.4f}")
    
    elif method == 'xgboost':
        # 使用XGBoost进行特征选择
        print("   训练XGBoost模型...")
        
        # 对标签进行编码（XGBoost需要连续的标签）
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"   原始标签: {np.unique(y)} -> 编码后标签: {np.unique(y_encoded)}")
        
        # 创建XGBoost分类器
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=6,
            verbosity=0,  # 减少输出
            eval_metric='logloss'
        )
        
        # 训练模型
        xgb_model.fit(X, y_encoded)
        
        # 获取特征重要性
        feature_importance = xgb_model.feature_importances_
        feature_names = np.array(feature_columns)
        
        # 按重要性排序
        sorted_indices = np.argsort(feature_importance)[::-1]
        selected_indices = sorted_indices[:max_features]
        selected_features = feature_names[selected_indices].tolist()
        
        print(f"   🌟 Top 10 XGBoost重要特征:")
        for i in range(min(10, len(selected_indices))):
            idx = sorted_indices[i]
            print(f"      {i+1}. {feature_columns[idx]}: {feature_importance[idx]:.4f}")
    
    elif method == 'f_test':
        # 使用F统计量进行特征选择
        selector = SelectKBest(score_func=f_classif, k=max_features)
        X_selected = selector.fit_transform(X, y)
        
        selected_mask = selector.get_support()
        selected_features = [feature_columns[i] for i, selected in enumerate(selected_mask) if selected]
        
        scores = selector.scores_
        selected_scores = scores[selected_mask]
        
        print(f"   🌟 Top 10 F统计量得分:")
        sorted_pairs = sorted(zip(selected_features, selected_scores), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_pairs[:10]):
            print(f"      {i+1}. {feature}: {score:.4f}")
    
    # 创建降维后的数据集
    print(f"\n📦 创建降维数据集...")
    
    # 保留的列：目标变量 + 选择的特征
    keep_columns = ['target', 'future_return', 'open_time'] + selected_features
    reduced_df = df[keep_columns].copy()
    
    print(f"   降维后形状: {reduced_df.shape}")
    print(f"   特征减少: {len(feature_columns)} → {len(selected_features)}")
    print(f"   压缩比例: {len(selected_features)/len(feature_columns)*100:.1f}%")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    reduced_df.to_parquet(output_path, index=False)
    
    # 计算文件大小对比
    original_size = os.path.getsize(input_path) / (1024**2)  # MB
    reduced_size = os.path.getsize(output_path) / (1024**2)  # MB
    
    print(f"\n💾 文件大小对比:")
    print(f"   原始文件: {original_size:.1f} MB")
    print(f"   降维文件: {reduced_size:.1f} MB")
    print(f"   节省空间: {(1 - reduced_size/original_size)*100:.1f}%")
    
    # 保存特征列表
    feature_list_path = output_path.replace('.parquet', '_features.txt')
    with open(feature_list_path, 'w') as f:
        f.write(f"# 特征降维结果 - {datetime.now()}\n")
        f.write(f"# 方法: {method}\n")
        f.write(f"# 原始特征数: {len(feature_columns)}\n")
        f.write(f"# 选择特征数: {len(selected_features)}\n\n")
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    print(f"✅ 降维完成!")
    print(f"   输出文件: {output_path}")
    print(f"   特征列表: {feature_list_path}")
    
    return selected_features


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征降维工具')
    parser.add_argument('--input', type=str, 
                       default='processed_data/featured_data.parquet',
                       help='输入数据文件路径')
    parser.add_argument('--output', type=str,
                       default='processed_data/featured_data_reduced.parquet',
                       help='输出数据文件路径')
    parser.add_argument('--max-features', type=int, default=50,
                       help='保留的最大特征数 (默认: 50)')
    parser.add_argument('--method', type=str, default='random_forest',
                       choices=['random_forest', 'mutual_info', 'f_test', 'xgboost'],
                       help='特征选择方法')
    
    args = parser.parse_args()
    
    print("🎯 特征降维工具")
    print("=" * 50)
    
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    try:
        selected_features = reduce_features_by_importance(
            input_path=args.input,
            output_path=args.output,
            max_features=args.max_features,
            method=args.method
        )
        
        print(f"\n🎉 特征降维成功!")
        print(f"💡 现在可以使用减少后的数据进行训练:")
        print(f"   数据文件: {args.output}")
        print(f"   特征数量: {len(selected_features)}")
        
    except Exception as e:
        print(f"❌ 特征降维失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 