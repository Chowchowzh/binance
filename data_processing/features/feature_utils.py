# -*- coding: utf-8 -*-
"""
特征工程辅助工具模块
提供特征选择、变换、验证等工具函数
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class FeatureUtils:
    """特征工程工具类"""
    
    @staticmethod
    def remove_correlated_features(df: pd.DataFrame, target_col: str = 'target', 
                                 threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
        """
        移除高度相关的特征
        
        Args:
            df: 特征DataFrame
            target_col: 目标列名
            threshold: 相关性阈值
            
        Returns:
            (清理后的DataFrame, 被移除的特征列表)
        """
        # 分离特征和目标
        feature_cols = [col for col in df.columns if col != target_col]
        features_df = df[feature_cols]
        
        # 计算相关性矩阵
        corr_matrix = features_df.corr().abs()
        
        # 找到高度相关的特征对
        upper_tri = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # 标记要删除的特征
        to_drop = []
        for column in upper_tri.columns:
            if any(upper_tri[column] > threshold):
                to_drop.append(column)
        
        # 创建清理后的DataFrame
        cleaned_df = df.drop(columns=to_drop)
        
        print(f"移除了 {len(to_drop)} 个高度相关的特征")
        return cleaned_df, to_drop
    
    @staticmethod
    def select_k_best_features(df: pd.DataFrame, target_col: str = 'target', 
                              k: int = 100, method: str = 'f_classif') -> Tuple[pd.DataFrame, List[str]]:
        """
        使用统计方法选择K个最佳特征
        
        Args:
            df: 特征DataFrame
            target_col: 目标列名
            k: 选择的特征数量
            method: 选择方法 ('f_classif' 或 'mutual_info')
            
        Returns:
            (选择后的DataFrame, 被选中的特征列表)
        """
        # 分离特征和目标
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # 移除包含NaN的行
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        # 选择评分函数
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"未知的选择方法: {method}")
        
        # 特征选择
        selector = SelectKBest(score_func=score_func, k=min(k, len(feature_cols)))
        X_selected = selector.fit_transform(X_clean, y_clean)
        
        # 获取被选中的特征名
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(X_selected, columns=selected_features, index=X_clean.index)
        result_df[target_col] = y_clean
        
        print(f"使用 {method} 方法选择了 {len(selected_features)} 个特征")
        return result_df, selected_features
    
    @staticmethod
    def create_polynomial_features(df: pd.DataFrame, target_col: str = 'target',
                                 feature_groups: List[List[str]] = None, 
                                 degree: int = 2) -> pd.DataFrame:
        """
        创建多项式特征（交互特征）
        
        Args:
            df: 特征DataFrame
            target_col: 目标列名
            feature_groups: 特征分组，每组内创建交互特征
            degree: 多项式次数
            
        Returns:
            包含多项式特征的DataFrame
        """
        result_df = df.copy()
        
        if feature_groups is None:
            # 默认分组：价格特征、技术指标等
            feature_cols = [col for col in df.columns if col != target_col]
            feature_groups = [feature_cols[:10]]  # 只对前10个特征创建交互
        
        for group in feature_groups:
            # 过滤存在的特征
            valid_features = [f for f in group if f in df.columns]
            
            if len(valid_features) < 2:
                continue
            
            # 创建两两交互特征
            for i, feat1 in enumerate(valid_features):
                for feat2 in valid_features[i+1:]:
                    if degree >= 2:
                        result_df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                    
                    # 特征比值（避免除零）
                    mask = df[feat2] != 0
                    result_df[f'{feat1}_div_{feat2}'] = 0
                    result_df.loc[mask, f'{feat1}_div_{feat2}'] = df.loc[mask, feat1] / df.loc[mask, feat2]
        
        print(f"创建多项式特征后，特征数量: {len(result_df.columns) - 1}")
        return result_df
    
    @staticmethod
    def create_technical_combinations(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        创建技术指标组合特征
        
        Args:
            df: 特征DataFrame
            symbol: 交易对符号
            
        Returns:
            包含组合特征的DataFrame
        """
        result_df = df.copy()
        
        # RSI组合
        rsi_cols = [col for col in df.columns if f'{symbol}_rsi' in col]
        if len(rsi_cols) >= 2:
            result_df[f'{symbol}_rsi_spread'] = df[rsi_cols[0]] - df[rsi_cols[1]]
        
        # 移动平均组合
        sma_cols = [col for col in df.columns if f'{symbol}_sma' in col]
        if len(sma_cols) >= 2:
            result_df[f'{symbol}_sma_slope'] = df[sma_cols[0]] - df[sma_cols[1]]
        
        # 布林带相关组合
        bb_upper = f'{symbol}_bb_upper'
        bb_lower = f'{symbol}_bb_lower'
        close_col = f'{symbol}_close'
        
        if all(col in df.columns for col in [bb_upper, bb_lower, close_col]):
            # 布林带挤压
            result_df[f'{symbol}_bb_squeeze'] = (df[bb_upper] - df[bb_lower]) / df[close_col]
        
        # MACD组合
        macd_col = f'{symbol}_macd'
        macd_signal_col = f'{symbol}_macd_signal'
        
        if all(col in df.columns for col in [macd_col, macd_signal_col]):
            result_df[f'{symbol}_macd_divergence'] = df[macd_col] - df[macd_signal_col]
        
        return result_df
    
    @staticmethod
    def validate_features(df: pd.DataFrame, target_col: str = 'target') -> Dict[str, Any]:
        """
        验证特征质量
        
        Args:
            df: 特征DataFrame
            target_col: 目标列名
            
        Returns:
            验证结果字典
        """
        feature_cols = [col for col in df.columns if col != target_col]
        
        validation_results = {
            'total_features': len(feature_cols),
            'missing_values': {},
            'infinite_values': {},
            'constant_features': [],
            'low_variance_features': [],
            'data_types': {}
        }
        
        for col in feature_cols:
            # 检查缺失值
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                validation_results['missing_values'][col] = missing_count
            
            # 检查无限值
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:
                validation_results['infinite_values'][col] = infinite_count
            
            # 检查常量特征
            if df[col].nunique() <= 1:
                validation_results['constant_features'].append(col)
            
            # 检查低方差特征
            if df[col].var() < 1e-6:
                validation_results['low_variance_features'].append(col)
            
            # 数据类型
            validation_results['data_types'][col] = str(df[col].dtype)
        
        # 打印验证摘要
        print("特征验证结果:")
        print(f"- 总特征数: {validation_results['total_features']}")
        print(f"- 包含缺失值的特征: {len(validation_results['missing_values'])}")
        print(f"- 包含无限值的特征: {len(validation_results['infinite_values'])}")
        print(f"- 常量特征: {len(validation_results['constant_features'])}")
        print(f"- 低方差特征: {len(validation_results['low_variance_features'])}")
        
        return validation_results
    
    @staticmethod
    def scale_features(df: pd.DataFrame, target_col: str = 'target', 
                      method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
        """
        特征缩放
        
        Args:
            df: 特征DataFrame
            target_col: 目标列名
            method: 缩放方法 ('standard', 'robust', 'minmax')
            
        Returns:
            (缩放后的DataFrame, 缩放器对象)
        """
        feature_cols = [col for col in df.columns if col != target_col]
        
        # 选择缩放器
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"未知的缩放方法: {method}")
        
        # 缩放特征
        scaled_features = scaler.fit_transform(df[feature_cols])
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(
            scaled_features, 
            columns=feature_cols, 
            index=df.index
        )
        result_df[target_col] = df[target_col]
        
        print(f"使用 {method} 方法完成特征缩放")
        return result_df, scaler
    
    @staticmethod
    def get_feature_statistics(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """
        获取特征统计信息
        
        Args:
            df: 特征DataFrame
            target_col: 目标列名
            
        Returns:
            特征统计DataFrame
        """
        feature_cols = [col for col in df.columns if col != target_col]
        
        stats = []
        for col in feature_cols:
            stat = {
                'feature': col,
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurt(),
                'unique_values': df[col].nunique(),
                'missing_pct': df[col].isnull().mean() * 100
            }
            stats.append(stat)
        
        return pd.DataFrame(stats) 