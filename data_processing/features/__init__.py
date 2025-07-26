# -*- coding: utf-8 -*-
"""
特征工程子模块
包含技术指标计算、特征构建等功能
"""

__version__ = "1.0.0"
__author__ = "Binance Trading Strategy Team"

from .feature_builder import FeatureBuilder
from .technical_indicators import TechnicalIndicators
from .feature_utils import FeatureUtils
from .triple_barrier_labeling import TripleBarrierLabeler

# 导入主特征工程函数
try:
    from ..features import build_features_with_tbm, analyze_tbm_features_quality
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        from features import build_features_with_tbm, analyze_tbm_features_quality
    except ImportError:
        # 最后尝试直接导入
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "features_main", 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "features.py")
        )
        features_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(features_main)
        build_features_with_tbm = features_main.build_features_with_tbm
        analyze_tbm_features_quality = features_main.analyze_tbm_features_quality

def build_features(df, target_symbol, feature_symbols=None):
    """
    构建特征工程管道 - 兼容性包装函数
    
    Args:
        df: 原始K线数据
        target_symbol: 目标交易对
        feature_symbols: 用于特征构建的交易对列表
        
    Returns:
        处理后的特征DataFrame
    """
    builder = FeatureBuilder()
    return builder.build_features(df, target_symbol, feature_symbols)

def clean_features(df):
    """
    清理特征数据 - 兼容性包装函数
    
    Args:
        df: 特征DataFrame
        
    Returns:
        清理后的DataFrame
    """
    builder = FeatureBuilder()
    return builder._post_process_features(df)

def identify_symbols_from_columns(columns):
    """从列名识别交易对符号 - 兼容性函数"""
    builder = FeatureBuilder()
    import pandas as pd
    return builder._identify_symbols(pd.DataFrame(columns=columns))

__all__ = [
    'FeatureBuilder',
    'TechnicalIndicators', 
    'FeatureUtils',
    'TripleBarrierLabeler',
    'build_features',
    'clean_features',
    'identify_symbols_from_columns',
    'build_features_with_tbm',
    'analyze_tbm_features_quality'
] 