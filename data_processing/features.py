# -*- coding: utf-8 -*-
"""
特征工程模块 - 统一接口
提供数据特征构建功能的主要入口
"""

import pandas as pd
import numpy as np
from typing import List, Optional

try:
    # 尝试相对导入
    from .features.feature_builder import FeatureBuilder
    from .features.dollar_bar_features import DollarBarFeatures
    from .features.triple_barrier_labeling import TripleBarrierLabeler
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from data_processing.features.feature_builder import FeatureBuilder
    from data_processing.features.dollar_bar_features import DollarBarFeatures
    from data_processing.features.triple_barrier_labeling import TripleBarrierLabeler


def build_features(df: pd.DataFrame, target_symbol: str, 
                  feature_symbols: Optional[List[str]] = None,
                  data_type: str = 'minute_bars') -> pd.DataFrame:
    """
    构建特征工程管道 - 统一入口
    
    Args:
        df: 原始K线数据
        target_symbol: 目标交易对
        feature_symbols: 用于特征构建的交易对列表
        data_type: 数据类型 ('minute_bars' 或 'dollar_bars')
        
    Returns:
        处理后的特征DataFrame
    """
    print(f"\n开始特征工程... (数据类型: {data_type})")
    
    if data_type == 'dollar_bars':
        # 使用成交额K线特征构建器
        builder = DollarBarFeatures(use_fp16=False)  # 禁用fp16以兼容numba
        result_df = builder.build_comprehensive_features(
            df=df,
            target_symbol=target_symbol,
            feature_symbols=feature_symbols or [target_symbol]
        )
    else:
        # 使用传统的分钟K线特征构建器
        builder = FeatureBuilder()
        result_df = builder.build_features(df, target_symbol, feature_symbols)
    
    print(f"特征工程完成，最终特征数量: {result_df.shape[1]}")
    return result_df


def build_dollar_bar_features(df: pd.DataFrame, 
                            target_symbol: str = 'ETHUSDT',
                            feature_symbols: Optional[List[str]] = None,
                            future_periods: int = 15,
                            use_fp16: bool = False) -> pd.DataFrame:  # 默认关闭fp16以兼容numba
    """
    构建成交额K线特征工程 - 专用接口
    
    Args:
        df: 成交额K线数据
        target_symbol: 目标交易对
        feature_symbols: 用于特征构建的交易对列表
        future_periods: 未来收益预测期数
        use_fp16: 是否使用fp16格式
        
    Returns:
        处理后的特征DataFrame（fp16格式）
    """
    print(f"\n开始成交额K线特征工程... (fp16: {use_fp16})")
    
    # 使用成交额K线特征构建器
    builder = DollarBarFeatures(use_fp16=use_fp16)
    
    # 构建所有特征
    result_df = builder.build_comprehensive_features(
        df=df,
        target_symbol=target_symbol,
        feature_symbols=feature_symbols or [target_symbol],
        future_periods=future_periods
    )
    
    print(f"成交额K线特征工程完成，最终特征数量: {result_df.shape[1]}")
    print(f"内存使用: {result_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return result_df


def build_features_with_tbm(df: pd.DataFrame, 
                           target_symbol: str = 'ETHUSDT',
                           feature_symbols: Optional[List[str]] = None,
                           data_type: str = 'dollar_bars',
                           use_fp16: bool = False,  # 默认关闭fp16以兼容numba
                           # TBM参数
                           profit_factor: float = 2.0,
                           loss_factor: float = 1.0,
                           volatility_window: int = 20,
                           max_holding_period: int = 60,
                           min_return_threshold: float = 0.0001,
                           use_symmetric_barriers: bool = False,
                           volatility_method: str = 'log_return',
                           use_cusum_events: bool = False,
                           cusum_threshold: float = 0.01,
                           n_jobs: int = 1) -> pd.DataFrame:
    """
    使用三分类标签法构建特征工程数据
    
    Args:
        df: 原始K线数据
        target_symbol: 目标交易对
        feature_symbols: 用于特征构建的交易对列表
        data_type: 数据类型 ('minute_bars' 或 'dollar_bars')
        use_fp16: 是否使用fp16格式（仅对dollar_bars有效）
        
        # TBM参数
        profit_factor: 止盈因子，相对于波动率的倍数
        loss_factor: 止损因子，相对于波动率的倍数
        volatility_window: 滚动波动率计算窗口
        max_holding_period: 最大持仓期（信息时钟）
        min_return_threshold: 最小收益阈值，过滤噪声
        use_symmetric_barriers: 是否使用对称边界
        volatility_method: 波动率计算方法
        use_cusum_events: 是否使用CUSUM过滤器生成事件
        cusum_threshold: CUSUM阈值
        n_jobs: 并行作业数
        
    Returns:
        包含TBM标签的特征DataFrame
    """
    print(f"\n开始基于三分类标签法的特征工程... (数据类型: {data_type})")
    print(f"原始数据维度: {df.shape}")
    
    # 1. 首先构建基础特征（不包含目标变量）
    if data_type == 'dollar_bars':
        print("构建成交额K线特征...")
        builder = DollarBarFeatures(use_fp16=use_fp16)
        
        # 获取特征但不构建传统目标变量
        feature_df = df.copy()
        feature_df = builder._add_basic_features(feature_df)
        
        if feature_symbols is None:
            feature_symbols = [target_symbol]
            
        for symbol in feature_symbols:
            feature_df = builder._add_symbol_technical_features(feature_df, symbol)
        
        if len(feature_symbols) > 1:
            feature_df = builder._add_cross_asset_features(feature_df, feature_symbols)
        
        feature_df = builder._add_temporal_features(feature_df)
        feature_df = builder._add_advanced_combination_features(feature_df, feature_symbols)
        feature_df = builder._add_statistical_features(feature_df, feature_symbols)
        
    else:
        print("构建分钟K线特征...")
        builder = FeatureBuilder()
        feature_df = df.copy()
        
        if feature_symbols is None:
            feature_symbols = builder._identify_symbols(df)
        
        # 为每个交易对添加特征
        for symbol in feature_symbols:
            feature_df = builder._add_symbol_features(feature_df, symbol)
        
        # 添加跨资产特征
        if len(feature_symbols) > 1:
            feature_df = builder._add_cross_asset_features(feature_df, target_symbol, feature_symbols)
    
    print(f"基础特征构建完成，特征数量: {feature_df.shape[1]}")
    
    # 2. 使用三分类标签法生成目标变量
    print("\n开始三分类标签生成...")
    
    # 初始化TBM标签器
    tbm_labeler = TripleBarrierLabeler(
        profit_factor=profit_factor,
        loss_factor=loss_factor,
        volatility_window=volatility_window,
        max_holding_period=max_holding_period,
        min_return_threshold=min_return_threshold,
        use_symmetric_barriers=use_symmetric_barriers
    )
    
    # 获取价格序列
    price_col = f'{target_symbol}_close' if f'{target_symbol}_close' in feature_df.columns else 'close'
    if price_col not in feature_df.columns:
        raise ValueError(f"未找到价格列 {price_col}")
    
    prices = feature_df[price_col]
    
    # 生成事件
    if use_cusum_events:
        print("使用CUSUM过滤器生成事件...")
        event_indices = tbm_labeler.generate_cusum_events(prices, threshold=cusum_threshold)
    else:
        print("使用全部有效价格点作为事件...")
        # 使用所有有效点作为事件
        valid_start = max(volatility_window, 1)
        event_indices = list(range(valid_start, len(prices) - max_holding_period))
    
    # 生成三分类标签
    tbm_results = tbm_labeler.generate_triple_barrier_labels(
        prices=prices,
        event_indices=event_indices,
        volatility_method=volatility_method,
        n_jobs=n_jobs
    )
    
    print(f"TBM标签生成完成，事件数量: {len(tbm_results)}")
    
    # 3. 将TBM标签合并到特征数据中
    print("合并TBM标签到特征数据...")
    
    # 创建与原DataFrame索引对齐的标签Series
    tbm_label = pd.Series(np.nan, index=feature_df.index, name='tbm_label')
    tbm_return_pct = pd.Series(np.nan, index=feature_df.index, name='tbm_return_pct')
    tbm_holding_period = pd.Series(np.nan, index=feature_df.index, name='tbm_holding_period')
    tbm_touch_type = pd.Series(np.nan, index=feature_df.index, name='tbm_touch_type')
    
    # 填充TBM结果 - 修复索引问题
    for _, row in tbm_results.iterrows():
        event_idx = int(row['event_idx'])  # 确保是整数
        if 0 <= event_idx < len(feature_df):
            # 使用位置索引获取实际的DataFrame索引
            actual_index = feature_df.index[event_idx]
            tbm_label.loc[actual_index] = row['label']
            tbm_return_pct.loc[actual_index] = row['return_pct']
            tbm_holding_period.loc[actual_index] = row['holding_period']
            tbm_touch_type.loc[actual_index] = row['touch_type']
    
    # 添加到特征DataFrame
    feature_df['tbm_label'] = tbm_label
    feature_df['tbm_return_pct'] = tbm_return_pct
    feature_df['tbm_holding_period'] = tbm_holding_period
    feature_df['tbm_touch_type'] = tbm_touch_type
    
    # 设置主要目标变量
    feature_df['target'] = feature_df['tbm_label']
    feature_df['future_return'] = feature_df['tbm_return_pct']
    
    # 4. 数据清理和格式转换
    print("进行数据清理...")
    
    # 4.1 强制数据类型转换 - 解决float64和datetime64问题
    print("强制转换数据类型为float32...")
    
    # 识别需要处理的列
    exclude_cols = ['start_time', 'end_time', 'open_time', 'close_time']  # 排除明确的时间列
    time_related_cols = []
    
    for col in feature_df.columns:
        # 检查是否为时间相关列
        if (col in exclude_cols or 
            'time' in col.lower() or 
            'timestamp' in col.lower() or
            feature_df[col].dtype.name.startswith('datetime')):
            time_related_cols.append(col)
    
    # 移除时间相关列（避免原始时间被放入特征）
    if time_related_cols:
        print(f"移除时间相关列: {time_related_cols}")
        feature_df = feature_df.drop(columns=time_related_cols)
    
    # 强制转换所有数值列为float32
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if feature_df[col].dtype != np.float32:
            feature_df[col] = feature_df[col].astype(np.float32)
    
    # 移除非数值列（除了我们需要保留的目标列）
    non_numeric_cols = feature_df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"移除非数值列: {list(non_numeric_cols)}")
        feature_df = feature_df.drop(columns=non_numeric_cols)
    
    # 验证数据类型
    print("数据清理完成，最终数据类型分布：")
    dtype_counts = feature_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} 列")
    
    # 4.2 移除包含过多NaN的行和列
    print("移除无效数据...")
    
    # 移除NaN比例过高的列（>80%），但保护目标列
    nan_ratio_cols = feature_df.isnull().sum() / len(feature_df)
    protected_cols = ['target', 'tbm_label', 'future_return', 'tbm_return_pct', 'tbm_holding_period', 'tbm_touch_type']  # 保护所有TBM相关列
    cols_to_drop = nan_ratio_cols[nan_ratio_cols > 0.8].index
    # 从待删除列表中移除保护列
    cols_to_drop = [col for col in cols_to_drop if col not in protected_cols]
    if len(cols_to_drop) > 0:
        print(f"移除NaN比例过高的列 ({len(cols_to_drop)}个): {list(cols_to_drop[:5])}{'...' if len(cols_to_drop) > 5 else ''}")
        feature_df = feature_df.drop(columns=cols_to_drop)
    
    # 移除NaN比例过高的行（>50%）
    nan_ratio_rows = feature_df.isnull().sum(axis=1) / len(feature_df.columns)
    feature_df = feature_df[nan_ratio_rows <= 0.5]
    
    # 最终移除有NaN目标值的行
    target_cols = ['target', 'tbm_label', 'future_return', 'tbm_return_pct']
    available_target_cols = [col for col in target_cols if col in feature_df.columns]
    if available_target_cols:
        feature_df = feature_df.dropna(subset=available_target_cols, how='all')
    
    print(f"数据清理完成，最终维度: {feature_df.shape}")
    
    if len(feature_df) == 0:
        raise ValueError("数据清理后没有剩余数据，请检查输入数据质量")
    
    return feature_df


def analyze_tbm_features_quality(df: pd.DataFrame) -> dict:
    """
    分析基于TBM的特征质量
    
    Args:
        df: 包含TBM标签的特征DataFrame
        
    Returns:
        质量分析结果
    """
    print("分析TBM特征质量...")
    
    analysis = {}
    
    # 基本统计
    total_samples = len(df)
    
    # 寻找可用的目标列
    target_cols = ['target', 'tbm_label']
    available_target_col = None
    for col in target_cols:
        if col in df.columns:
            available_target_col = col
            break
    
    if available_target_col is None:
        print("警告: 没有找到目标列，跳过标签分析")
        analysis['total_samples'] = total_samples
        analysis['valid_labels'] = 0
        analysis['label_coverage'] = 0
        analysis['label_distribution'] = {}
        return analysis
    
    valid_labels = df[available_target_col].dropna()
    analysis['total_samples'] = total_samples
    analysis['valid_labels'] = len(valid_labels)
    analysis['label_coverage'] = len(valid_labels) / total_samples if total_samples > 0 else 0
    
    # 标签分布
    if len(valid_labels) > 0:
        analysis['label_distribution'] = valid_labels.value_counts().to_dict()
        
        # 按标签分组的收益率统计
        for label in [-1, 0, 1]:
            label_mask = df[available_target_col] == label
            # 寻找可用的收益率列
            return_cols = ['future_return', 'tbm_return_pct']
            available_return_col = None
            for ret_col in return_cols:
                if ret_col in df.columns:
                    available_return_col = ret_col
                    break
            
            if available_return_col is None:
                continue
                
            label_returns = df.loc[label_mask, available_return_col].dropna()
            
            if len(label_returns) > 0:
                analysis[f'returns_label_{int(label)}'] = {
                    'count': len(label_returns),
                    'mean': label_returns.mean(),
                    'std': label_returns.std(),
                    'median': label_returns.median(),
                    'min': label_returns.min(),
                    'max': label_returns.max()
                }
        
        # 持仓期分析
        valid_holding = df['tbm_holding_period'].dropna()
        if len(valid_holding) > 0:
            analysis['holding_period_stats'] = {
                'mean': valid_holding.mean(),
                'median': valid_holding.median(),
                'std': valid_holding.std(),
                'min': valid_holding.min(),
                'max': valid_holding.max()
            }
        
        # 触碰类型分析
        touch_types = df['tbm_touch_type'].dropna()
        if len(touch_types) > 0:
            analysis['touch_type_distribution'] = touch_types.value_counts().to_dict()
    
    # 特征统计
    feature_cols = [col for col in df.columns if col not in [
        'target', 'future_return', 'tbm_label', 'tbm_return_pct', 
        'tbm_holding_period', 'tbm_touch_type'
    ]]
    analysis['total_features'] = len(feature_cols)
    
    # 内存使用
    analysis['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    return analysis


def _remove_raw_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    移除原始OHLCV数据以避免前视偏差
    但为强化学习保留必要的价格和成交量信息
    """
    # 需要移除的原始数据列（会导致数据泄露）
    remove_columns = [
        # 原始价格数据 - 保留close用于强化学习
        'open', 'high', 'low',  # 移除但保留close
        # 原始成交量数据 - 保留volume用于强化学习  
        'quote_volume', 'trades',  # 移除但保留volume
        # 时间戳数据 - 在时间序列预测中会导致泄露
        'start_time', 'end_time', 'end_timestamp',
        # 其他原始数据
        'duration_minutes', 'bar_count', 'avg_price',
        # 基于原始价格的简单特征（保留技术指标）
        'body_size', 'upper_shadow', 'lower_shadow', 'total_range',
        'buy_pressure', 'sell_pressure', 'price_position'
    ]
    
    # 只移除存在的列
    columns_to_remove = [col for col in remove_columns if col in df.columns]
    
    if columns_to_remove:
        print(f"移除 {len(columns_to_remove)} 个原始数据列以避免数据泄露:")
        for col in columns_to_remove:
            print(f"  - {col}")
        df = df.drop(columns=columns_to_remove)
    
    # 确保强化学习必需的列存在
    required_for_rl = ['close', 'volume']
    missing_for_rl = [col for col in required_for_rl if col not in df.columns]
    if missing_for_rl:
        print(f"警告: 强化学习需要的列缺失: {missing_for_rl}")
        print("为强化学习保留必要的价格和成交量列...")
    
    print(f"清理后特征数量: {df.shape[1]}")
    print(f"保留强化学习必需列: {[col for col in required_for_rl if col in df.columns]}")
    return df


def identify_symbols_from_columns(columns) -> List[str]:
    """从列名识别交易对符号 - 兼容性函数"""
    from .features.feature_builder import FeatureBuilder
    builder = FeatureBuilder()
    return builder._identify_symbols(pd.DataFrame(columns=columns))


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理特征数据 - 兼容性函数
    
    Args:
        df: 特征DataFrame
        
    Returns:
        清理后的DataFrame
    """
    from .features.feature_builder import FeatureBuilder
    builder = FeatureBuilder()
    return builder._post_process_features(df) 