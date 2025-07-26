# -*- coding: utf-8 -*-
"""
特征构建器模块
整合多种特征工程方法，构建完整的特征集
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set
import warnings

from .technical_indicators import TechnicalIndicators
from .feature_utils import FeatureUtils

warnings.filterwarnings('ignore')


class FeatureBuilder:
    """特征构建器 - 整合所有特征工程功能"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.feature_utils = FeatureUtils()
    
    def build_features(self, df: pd.DataFrame, target_symbol: str, 
                      feature_symbols: List[str] = None, is_first_chunk: bool = True) -> pd.DataFrame:
        """
        构建特征
        
        Args:
            df: 输入数据
            target_symbol: 目标交易对
            feature_symbols: 特征交易对列表
            is_first_chunk: 是否是第一个数据块（第一个块需要清理NaN，后续块不需要）
        """
        print(f"开始构建特征... 目标交易对: {target_symbol}")
        
        if feature_symbols is None:
            # 自动识别所有交易对
            feature_symbols = self._identify_symbols(df)
        
        # 复制数据避免修改原始数据
        feature_df = df.copy()
        
        # 为每个交易对添加特征
        for symbol in feature_symbols:
            feature_df = self._add_symbol_features(feature_df, symbol)
        
        # 添加跨资产特征
        if len(feature_symbols) > 1:
            feature_df = self._add_cross_asset_features(feature_df, target_symbol, feature_symbols)
        
        # 构建目标变量
        feature_df = self._build_target_variable(feature_df, target_symbol)
        
        # 后处理 - 只有第一个块需要清理开头的NaN
        feature_df = self._post_process_features(feature_df, clean_initial_nans=is_first_chunk)
        
        # 最后移除原始k线数据，确保所有特征计算完成后再清理
        feature_df = self._remove_raw_ohlcv_data(feature_df)
        
        return feature_df
    
    def _identify_symbols(self, df: pd.DataFrame) -> List[str]:
        """从DataFrame列名中识别交易对符号"""
        symbols = set()
        
        # 标准的OHLCV字段后缀
        standard_suffixes = ['_open', '_high', '_low', '_close', '_volume']
        
        for col in df.columns:
            # 只使用标准的OHLCV字段来识别交易对
            for suffix in standard_suffixes:
                if col.endswith(suffix):
                    symbol = col[:-len(suffix)]
                    # 确保这个symbol确实有完整的OHLCV数据
                    has_all_ohlcv = all(f'{symbol}{s}' in df.columns for s in standard_suffixes)
                    if has_all_ohlcv:
                        symbols.add(symbol)
                    break
        
        return sorted(list(symbols))
    
    def _add_symbol_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """为单个交易对添加所有特征"""
        # 技术指标
        df = self.technical_indicators.add_price_features(df, symbol)
        df = self.technical_indicators.add_moving_averages(df, symbol)  # 添加移动平均差值特征
        df = self.technical_indicators.add_momentum_indicators(df, symbol)
        df = self.technical_indicators.add_volatility_indicators(df, symbol)
        df = self.technical_indicators.add_volume_indicators(df, symbol)
        
        # 订单流和市场微观结构特征
        df = self.technical_indicators.add_order_flow_features(df, symbol)
        df = self.technical_indicators.add_market_microstructure_features(df, symbol)
        
        # 滞后特征
        df = self._add_lag_features(df, symbol)
        
        # 滚动统计特征
        df = self._add_rolling_features(df, symbol)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, symbol: str, lags: List[int] = None) -> pd.DataFrame:
        """添加滞后特征 - 减少数量"""
        if lags is None:
            lags = [1, 3, 10, 30]  # 从8个减少到4个
        
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'
        
        # 价格滞后 - 只保留关键lag
        for lag in lags:
            # 改为对数收益率
            df[f'{symbol}_return_lag_{lag}'] = np.log(df[close_col] / df[close_col].shift(lag).replace(0, np.nan))
        
        # 成交量滞后 - 进一步减少
        for lag in [1, 10, 30]:  # 从5个减少到3个
            df[f'{symbol}_volume_change_lag_{lag}'] = np.log(df[volume_col] / df[volume_col].shift(lag).replace(0, np.nan))
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, symbol: str, 
                             windows: List[int] = None) -> pd.DataFrame:
        """添加滚动统计特征 - 减少数量"""
        if windows is None:
            windows = [10, 30]  # 从4个减少到2个
        
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'
        
        # 改为对数收益率
        returns = np.log(df[close_col] / df[close_col].shift(1).replace(0, np.nan))
        
        for window in windows:
            # 只保留最重要的统计特征
            df[f'{symbol}_return_std_{window}'] = returns.rolling(window).std()
            df[f'{symbol}_return_mean_{window}'] = returns.rolling(window).mean()
            
            # 价格范围特征
            df[f'{symbol}_close_max_{window}'] = df[close_col].rolling(window).max()
            df[f'{symbol}_close_min_{window}'] = df[close_col].rolling(window).min()
            df[f'{symbol}_close_range_{window}'] = df[f'{symbol}_close_max_{window}'] - df[f'{symbol}_close_min_{window}']
            
            # 成交量统计 - 简化
            df[f'{symbol}_volume_std_{window}'] = df[volume_col].rolling(window).std()
            
        return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame, target_symbol: str, 
                                 symbols: List[str], windows: List[int] = None) -> pd.DataFrame:
        """添加跨资产特征 - 减少数量"""
        if windows is None:
            windows = [10, 30]  # 从3个减少到2个
        
        target_close = f'{target_symbol}_close'
        
        for symbol in symbols:
            if symbol == target_symbol:
                continue
                
            close1 = target_close
            close2 = f'{symbol}_close'
            
            # 检查数据是否存在
            if close2 not in df.columns:
                print(f"  警告：缺少 {symbol} 数据，跳过跨资产特征")
                continue
            
            # 改为对数收益率
            returns1 = np.log(df[close1] / df[close1].shift(1).replace(0, np.nan))
            returns2 = np.log(df[close2] / df[close2].shift(1).replace(0, np.nan))
            
            for window in windows:
                # 只保留最重要的跨资产特征
                # 相关性
                corr = returns1.rolling(window).corr(returns2)
                df[f'{target_symbol}_{symbol}_corr_{window}'] = corr
                
                # 价格比率
                df[f'{target_symbol}_{symbol}_price_ratio_{window}'] = (
                    df[close1].rolling(window).mean() / df[close2].rolling(window).mean()
                )
        
        return df
    
    def _build_target_variable(self, df: pd.DataFrame, target_symbol: str, 
                              lookahead: int = 15) -> pd.DataFrame:
        """构建目标变量"""
        close_col = f'{target_symbol}_close'
        
        if close_col not in df.columns:
            print(f"警告：未找到 {close_col} 列，无法构建目标变量")
            return df
        
        # 改为对数收益率
        future_return = np.log(df[close_col].shift(-lookahead) / df[close_col].replace(0, np.nan))
        
        # 分类目标：上涨(1), 下跌(-1), 横盘(0)
        # 调整阈值为0.0005（0.05%），减少横盘数据比例，提高信号质量
        threshold = 0.0005  # 0.05%的阈值，比之前的0.1%更敏感
        target = pd.Series(0, index=df.index)
        target[future_return > threshold] = 1
        target[future_return < -threshold] = -1
        
        df['target'] = target
        df['future_return'] = future_return
        
        # 输出目标变量构建的统计信息
        target_counts = target.value_counts().sort_index()
        print(f"目标变量分布 (阈值={threshold}):")
        for val, count in target_counts.items():
            percentage = count / len(target) * 100
            label = {-1: "下跌", 0: "横盘", 1: "上涨"}[val]
            print(f"  {label}({val}): {count} ({percentage:.2f}%)")
        
        print(f"未来收益率统计: 均值={future_return.mean():.6f}, 标准差={future_return.std():.6f}")
        
        return df
    
    def _remove_raw_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        移除原始的k线数据（OHLCV），只保留特征工程后的特征
        
        Args:
            df: 包含原始数据和特征的DataFrame
            
        Returns:
            移除原始数据后的DataFrame
        """
        # 标准的OHLCV字段后缀
        ohlcv_suffixes = ['_open', '_high', '_low', '_close', '_volume']
        
        # 可能的交易数据字段
        additional_raw_suffixes = ['_trades', '_taker_buy_volume', '_taker_buy_count']
        
        all_raw_suffixes = ohlcv_suffixes + additional_raw_suffixes
        
        # 找出需要移除的列
        columns_to_remove = []
        for col in df.columns:
            for suffix in all_raw_suffixes:
                if col.endswith(suffix):
                    columns_to_remove.append(col)
                    break
        
        # 移除原始数据列
        if columns_to_remove:
            print(f"正在移除 {len(columns_to_remove)} 个原始k线数据列...")
            print(f"移除的列包括: {columns_to_remove[:5]}{'...' if len(columns_to_remove) > 5 else ''}")
            df_cleaned = df.drop(columns=columns_to_remove)
            
            # 统计信息
            original_cols = len(df.columns)
            cleaned_cols = len(df_cleaned.columns)
            print(f"列数变化: {original_cols} -> {cleaned_cols} (减少了 {original_cols - cleaned_cols} 列)")
            
            return df_cleaned
        else:
            print("未找到需要移除的原始k线数据列")
            return df
    
    def _post_process_features(self, df: pd.DataFrame, clean_initial_nans: bool = True) -> pd.DataFrame:
        """特征后处理 - 智能数据清理"""
        # 移除无限值和NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 记录清理前的形状
        original_shape = df.shape
        
        # 智能清理策略：
        if clean_initial_nans:
            # 第一个块：正常清理策略
            # 1. 对于滞后特征和滚动特征，允许开始的NaN值
            # 2. 找到目标变量列
            target_cols = [col for col in df.columns if col.endswith('_target')]
            
            if target_cols:
                # 如果有目标变量，以目标变量的有效性为准
                target_col = target_cols[0]
                valid_target_mask = df[target_col].notna()
                df = df[valid_target_mask]
            else:
                # 如果没有目标变量，使用更宽松的清理策略
                # 统计每行的NaN比例，只删除NaN比例过高的行（>50%）
                nan_ratio = df.isnull().sum(axis=1) / len(df.columns)
                valid_rows_mask = nan_ratio <= 0.5  # 保留NaN比例小于50%的行
                df_filtered = df[valid_rows_mask]
                
                # 如果还是没有数据，进一步放宽条件到70%
                if len(df_filtered) == 0:
                    valid_rows_mask = nan_ratio <= 0.7
                    df_filtered = df[valid_rows_mask]
                    
                # 如果仍然没有数据，保留至少80%的原始数据
                if len(df_filtered) == 0:
                    # 按NaN数量排序，保留前80%的行
                    nan_counts = df.isnull().sum(axis=1)
                    keep_rows = int(len(df) * 0.8)
                    sorted_indices = nan_counts.sort_values().index[:keep_rows]
                    df_filtered = df.loc[sorted_indices]
                    
                df = df_filtered
        else:
            # 后续块：只做最基本的清理，不删除因为lag特征产生的开头NaN
            # 只删除完全异常的行（>80% NaN）
            nan_ratio = df.isnull().sum(axis=1) / len(df.columns)
            valid_rows_mask = nan_ratio <= 0.8
            df = df[valid_rows_mask]
        
        # 删除完全为NaN的列
        df = df.dropna(axis=1, how='all')
        
        # 只在数据量变化显著时输出清理信息
        size_change_ratio = df.shape[0] / original_shape[0] if original_shape[0] > 0 else 0
        if size_change_ratio < 0.8:  # 只有当数据减少超过20%时才报告
            print(f"数据清理: {original_shape} -> {df.shape}")
        
        return df
    
    def get_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """获取特征分组"""
        feature_cols = [col for col in df.columns if col not in ['target', 'future_return']]
        
        groups = {
            'price': [],
            'volume': [],
            'technical': [],
            'lag': [],
            'rolling': [],
            'cross_asset': []
        }
        
        for col in feature_cols:
            if any(x in col for x in ['price', 'close', 'open', 'high', 'low', 'range']):
                groups['price'].append(col)
            elif 'volume' in col:
                groups['volume'].append(col)
            elif any(x in col for x in ['rsi', 'macd', 'bb', 'atr', 'ema', 'sma']):
                groups['technical'].append(col)
            elif 'lag' in col:
                groups['lag'].append(col)
            elif any(x in col for x in ['mean', 'std', 'skew', 'kurt', 'max', 'min']):
                groups['rolling'].append(col)
            elif any(x in col for x in ['corr', 'ratio', 'diff']):
                groups['cross_asset'].append(col)
        
        return groups 