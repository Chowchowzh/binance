# -*- coding: utf-8 -*-
"""
成交额K线特征工程模块
专门为成交额K线设计的特征工程，包含完整的交叉特征
"""

import numpy as np
import pandas as pd
import talib as ta
from typing import Dict, List, Tuple, Optional, Set
import warnings
from itertools import combinations
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings('ignore')


class DollarBarFeatures:
    """成交额K线特征工程器"""
    
    def __init__(self, use_fp16: bool = False):  # 默认关闭fp16以兼容numba
        """
        初始化成交额K线特征工程器
        
        Args:
            use_fp16: 是否使用fp16格式以节省内存
        """
        # 为了兼容numba，强制使用float32而不是float16
        self.use_fp16 = False  # 强制关闭float16以兼容numba
        self.float_dtype = np.float32  # 始终使用float32确保numba兼容性
        
    def build_comprehensive_features(self, df: pd.DataFrame, 
                                   target_symbol: str = 'ETHUSDT',
                                   feature_symbols: List[str] = None,
                                   future_periods: int = 15) -> pd.DataFrame:
        """
        构建完整的成交额K线特征工程
        
        Args:
            df: 成交额K线数据
            target_symbol: 目标交易对
            feature_symbols: 特征交易对列表
            future_periods: 未来收益预测期数
            
        Returns:
            完整的特征DataFrame（fp16格式）
        """
        print(f"开始构建成交额K线特征工程 (fp16: {self.use_fp16})")
        print(f"原始数据维度: {df.shape}")
        
        # 复制数据并转换为合适的数据类型
        feature_df = df.copy()
        
        if feature_symbols is None:
            feature_symbols = [target_symbol]
        
        # 1. 基础特征工程
        feature_df = self._add_basic_features(feature_df)
        
        # 2. 为每个交易对添加技术特征
        for symbol in feature_symbols:
            feature_df = self._add_symbol_technical_features(feature_df, symbol)
        
        # 3. 交叉特征（最重要的部分）
        if len(feature_symbols) > 1:
            feature_df = self._add_cross_asset_features(feature_df, feature_symbols)
        
        # 4. 时间特征
        feature_df = self._add_temporal_features(feature_df)
        
        # 5. 高级组合特征
        feature_df = self._add_advanced_combination_features(feature_df, feature_symbols)
        
        # 6. 统计学习特征
        feature_df = self._add_statistical_features(feature_df, feature_symbols)
        
        # 7. 构建目标变量
        feature_df = self._build_target_variables(feature_df, target_symbol, future_periods)
        
        # 8. 数据清理和格式转换
        feature_df = self._clean_and_convert_features(feature_df)
        
        print(f"特征工程完成，最终维度: {feature_df.shape}")
        print(f"内存使用: {feature_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return feature_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基础特征"""
        print("添加基础特征...")
        
        # 时间特征
        df['duration_log'] = np.log(df['duration_minutes'] + 1)
        df['bar_count_log'] = np.log(df['bar_count'] + 1)
        
        # K线形态特征
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # 避免除零
        safe_range = df['total_range'].replace(0, np.nan)
        df['body_ratio'] = df['body_size'] / safe_range
        df['upper_shadow_ratio'] = df['upper_shadow'] / safe_range
        df['lower_shadow_ratio'] = df['lower_shadow'] / safe_range
        
        # 价格位置
        df['price_position'] = (df['close'] - df['low']) / safe_range
        
        # 成交量特征
        df['volume_log'] = np.log(df['volume'] + 1)
        df['quote_volume_log'] = np.log(df['quote_volume'] + 1)
        df['avg_price'] = df['quote_volume'] / (df['volume'] + 1e-10)
        
        # 买卖压力
        df['buy_pressure'] = df['taker_buy_volume'] / (df['volume'] + 1e-10)
        df['sell_pressure'] = 1 - df['buy_pressure']
        df['buy_sell_ratio'] = df['taker_buy_volume'] / (df['volume'] - df['taker_buy_volume'] + 1e-10)
        
        # 订单流不平衡
        df['order_flow_imbalance'] = (df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])) / df['volume']
        
        return df
    
    def _add_symbol_technical_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """为单个交易对添加技术特征"""
        if symbol is None:
            # 使用默认列名
            close_col, high_col, low_col, open_col = 'close', 'high', 'low', 'open'
            volume_col = 'volume'
        else:
            # 使用符号前缀
            close_col = f'{symbol}_close'
            high_col = f'{symbol}_high'
            low_col = f'{symbol}_low'
            open_col = f'{symbol}_open'
            volume_col = f'{symbol}_volume'
        
        # 如果列不存在，使用默认列名
        if close_col not in df.columns:
            close_col, high_col, low_col, open_col = 'close', 'high', 'low', 'open'
            volume_col = 'volume'
        
        print(f"添加技术特征 (使用列: {close_col})")
        
        close = df[close_col].values
        high = df[high_col].values
        low = df[low_col].values
        open_vals = df[open_col].values
        volume = df[volume_col].values
        
        prefix = f"{symbol}_" if symbol else ""
        
        # 收益率特征
        returns = np.log(close[1:] / close[:-1])
        df[f'{prefix}log_return'] = np.concatenate([[np.nan], returns])
        
        # 移动平均线
        periods = [5, 10, 20, 50]
        for period in periods:
            sma = ta.SMA(close, timeperiod=period)
            ema = ta.EMA(close, timeperiod=period)
            df[f'{prefix}sma_{period}'] = sma
            df[f'{prefix}ema_{period}'] = ema
            df[f'{prefix}price_vs_sma_{period}'] = close / sma - 1
            df[f'{prefix}price_vs_ema_{period}'] = close / ema - 1
        
        # 移动平均线差值（重要的交叉特征）
        for i, p1 in enumerate(periods):
            for p2 in periods[i+1:]:
                df[f'{prefix}sma_diff_{p1}_{p2}'] = df[f'{prefix}sma_{p1}'] - df[f'{prefix}sma_{p2}']
                df[f'{prefix}ema_diff_{p1}_{p2}'] = df[f'{prefix}ema_{p1}'] - df[f'{prefix}ema_{p2}']
        
        # 动量指标
        df[f'{prefix}rsi_14'] = ta.RSI(close, timeperiod=14)
        df[f'{prefix}rsi_21'] = ta.RSI(close, timeperiod=21)
        df[f'{prefix}rsi_diff'] = df[f'{prefix}rsi_14'] - df[f'{prefix}rsi_21']
        
        # MACD
        macd, macd_signal, macd_hist = ta.MACD(close)
        df[f'{prefix}macd'] = macd
        df[f'{prefix}macd_signal'] = macd_signal
        df[f'{prefix}macd_hist'] = macd_hist
        df[f'{prefix}macd_cross'] = np.where(macd > macd_signal, 1, -1)
        
        # 布林带
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20)
        df[f'{prefix}bb_upper'] = bb_upper
        df[f'{prefix}bb_middle'] = bb_middle
        df[f'{prefix}bb_lower'] = bb_lower
        df[f'{prefix}bb_width'] = (bb_upper - bb_lower) / bb_middle
        df[f'{prefix}bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # ATR和波动率
        atr_14 = ta.ATR(high, low, close, timeperiod=14)
        atr_21 = ta.ATR(high, low, close, timeperiod=21)
        df[f'{prefix}atr_14'] = atr_14
        df[f'{prefix}atr_21'] = atr_21
        df[f'{prefix}atr_ratio'] = atr_14 / (atr_21 + 1e-10)
        
        # 成交量指标
        df[f'{prefix}obv'] = ta.OBV(close, volume)
        df[f'{prefix}mfi'] = ta.MFI(high, low, close, volume, timeperiod=14)
        
        # 滞后特征
        lag_periods = [1, 3, 5, 10]
        for lag in lag_periods:
            df[f'{prefix}return_lag_{lag}'] = df[f'{prefix}log_return'].shift(lag)
            df[f'{prefix}volume_change_lag_{lag}'] = np.log(volume / np.roll(volume, lag))
        
        # 滚动统计特征
        windows = [10, 20, 50]
        returns_series = pd.Series(df[f'{prefix}log_return'])
        for window in windows:
            df[f'{prefix}return_mean_{window}'] = returns_series.rolling(window).mean()
            df[f'{prefix}return_std_{window}'] = returns_series.rolling(window).std()
            df[f'{prefix}return_skew_{window}'] = returns_series.rolling(window).skew()
            df[f'{prefix}return_kurt_{window}'] = returns_series.rolling(window).kurt()
            
            # 价格统计
            close_series = pd.Series(close)
            df[f'{prefix}close_max_{window}'] = close_series.rolling(window).max()
            df[f'{prefix}close_min_{window}'] = close_series.rolling(window).min()
            df[f'{prefix}close_range_{window}'] = df[f'{prefix}close_max_{window}'] - df[f'{prefix}close_min_{window}']
        
        return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """添加跨资产交叉特征"""
        print("添加跨资产交叉特征...")
        
        # 计算所有交易对的收益率
        returns_dict = {}
        price_dict = {}
        volume_dict = {}
        
        for symbol in symbols:
            close_col = f'{symbol}_close' if f'{symbol}_close' in df.columns else 'close'
            volume_col = f'{symbol}_volume' if f'{symbol}_volume' in df.columns else 'volume'
            
            returns_dict[symbol] = np.log(df[close_col] / df[close_col].shift(1))
            price_dict[symbol] = df[close_col]
            volume_dict[symbol] = df[volume_col]
        
        # 交叉相关性特征
        windows = [10, 20, 50]
        for symbol1, symbol2 in combinations(symbols, 2):
            returns1 = returns_dict[symbol1]
            returns2 = returns_dict[symbol2]
            price1 = price_dict[symbol1]
            price2 = price_dict[symbol2]
            
            for window in windows:
                # 收益率相关性
                corr = returns1.rolling(window).corr(returns2)
                df[f'{symbol1}_{symbol2}_return_corr_{window}'] = corr
                
                # 价格比率
                price_ratio = price1 / price2
                df[f'{symbol1}_{symbol2}_price_ratio'] = price_ratio
                df[f'{symbol1}_{symbol2}_price_ratio_ma_{window}'] = price_ratio.rolling(window).mean()
                df[f'{symbol1}_{symbol2}_price_ratio_std_{window}'] = price_ratio.rolling(window).std()
                
                # 价格差值
                price_diff = price1 - price2
                df[f'{symbol1}_{symbol2}_price_diff_ma_{window}'] = price_diff.rolling(window).mean()
                
                # 成交量比率
                volume_ratio = volume_dict[symbol1] / (volume_dict[symbol2] + 1e-10)
                df[f'{symbol1}_{symbol2}_volume_ratio_{window}'] = volume_ratio.rolling(window).mean()
        
        # 多资产组合特征
        if len(symbols) >= 2:
            # 计算投资组合收益率（等权重）
            portfolio_returns = sum(returns_dict.values()) / len(returns_dict)
            df['portfolio_return'] = portfolio_returns
            
            # 投资组合与个股的相关性
            for symbol in symbols:
                for window in windows:
                    corr_with_portfolio = returns_dict[symbol].rolling(window).corr(portfolio_returns)
                    df[f'{symbol}_portfolio_corr_{window}'] = corr_with_portfolio
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        print("添加时间特征...")
        
        # 尝试从start_time列获取时间，如果不存在则从索引获取
        if 'start_time' in df.columns:
            # 基于开始时间的时间特征
            df['start_timestamp'] = pd.to_datetime(df['start_time'], unit='ms')
        else:
            # 如果没有start_time列，使用DataFrame索引
            if isinstance(df.index, pd.DatetimeIndex):
                df['start_timestamp'] = df.index
            else:
                # 如果索引也不是datetime，创建一个默认的时间序列
                print("警告：无法获取时间信息，使用默认时间序列")
                df['start_timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='5min')
        
        # 小时特征
        df['hour'] = df['start_timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 星期特征
        df['dayofweek'] = df['start_timestamp'].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # 月份特征
        df['month'] = df['start_timestamp'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 是否工作日
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # 时间段特征
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        
        return df
    
    def _add_advanced_combination_features(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """添加高级组合特征"""
        print("添加高级组合特征...")
        
        # 技术指标组合
        for symbol in symbols:
            prefix = f"{symbol}_" if f'{symbol}_close' in df.columns else ""
            
            # RSI和布林带组合
            rsi_col = f'{prefix}rsi_14'
            bb_position_col = f'{prefix}bb_position'
            if rsi_col in df.columns and bb_position_col in df.columns:
                df[f'{prefix}rsi_bb_combo'] = df[rsi_col] * df[bb_position_col]
            
            # MACD和价格位置组合
            macd_col = f'{prefix}macd_hist'
            price_vs_sma_col = f'{prefix}price_vs_sma_20'
            if macd_col in df.columns and price_vs_sma_col in df.columns:
                df[f'{prefix}macd_price_combo'] = df[macd_col] * df[price_vs_sma_col]
            
            # 波动率和成交量组合
            atr_col = f'{prefix}atr_14'
            volume_col = f'{prefix}volume' if f'{prefix}volume' in df.columns else 'volume'
            if atr_col in df.columns:
                df[f'{prefix}atr_volume_combo'] = df[atr_col] * np.log(df[volume_col] + 1)
        
        # 市场状态特征
        # 使用多个指标综合判断市场状态
        main_symbol = symbols[0]
        prefix = f"{main_symbol}_" if f'{main_symbol}_close' in df.columns else ""
        
        # 趋势强度
        sma_5_col = f'{prefix}sma_5'
        sma_20_col = f'{prefix}sma_20'
        if sma_5_col in df.columns and sma_20_col in df.columns:
            df['trend_strength'] = (df[sma_5_col] - df[sma_20_col]) / df[sma_20_col]
        
        # 市场情绪指标
        rsi_col = f'{prefix}rsi_14'
        if rsi_col in df.columns:
            df['market_sentiment'] = np.where(df[rsi_col] > 70, 1,  # 超买
                                            np.where(df[rsi_col] < 30, -1, 0))  # 超卖
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """添加统计学习特征"""
        print("添加统计学习特征...")
        
        # 主成分分析相关特征（手动实现简化版本）
        # 收集所有价格相关特征
        price_features = []
        for symbol in symbols:
            prefix = f"{symbol}_" if f'{symbol}_close' in df.columns else ""
            close_col = 'close' if prefix == "" else f'{symbol}_close'
            
            if close_col in df.columns:
                price_features.append(df[close_col])
        
        if len(price_features) > 1:
            # 计算价格的协方差矩阵特征
            price_matrix = np.column_stack(price_features)
            
            # 滚动窗口计算协方差矩阵的特征值
            window = 20
            for i in range(window, len(df)):
                window_data = price_matrix[i-window:i]
                try:
                    cov_matrix = np.cov(window_data.T)
                    eigenvals = np.linalg.eigvals(cov_matrix)
                    
                    # 特征值相关指标
                    df.loc[df.index[i], 'price_cov_max_eigenval'] = np.max(eigenvals)
                    df.loc[df.index[i], 'price_cov_eigenval_ratio'] = np.max(eigenvals) / (np.sum(eigenvals) + 1e-10)
                except:
                    df.loc[df.index[i], 'price_cov_max_eigenval'] = np.nan
                    df.loc[df.index[i], 'price_cov_eigenval_ratio'] = np.nan
        
        # 熵特征
        for symbol in symbols:
            prefix = f"{symbol}_" if f'{symbol}_close' in df.columns else ""
            return_col = f'{prefix}log_return'
            
            if return_col in df.columns:
                # 计算收益率的近似熵
                returns = df[return_col].dropna()
                for window in [20, 50]:
                    entropy_values = []
                    for i in range(window, len(returns)):
                        window_returns = returns.iloc[i-window:i]
                        # 简化的熵计算
                        hist, _ = np.histogram(window_returns, bins=10, density=True)
                        hist = hist[hist > 0]
                        entropy = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0
                        entropy_values.append(entropy)
                    
                    # 将熵值分配到对应的行，简化处理避免长度问题
                    try:
                        # 创建一个与df长度相同的空Series
                        entropy_col = pd.Series(np.nan, index=df.index)
                        
                        # 从window位置开始填入entropy_values
                        for i, val in enumerate(entropy_values):
                            if window + i < len(entropy_col):
                                entropy_col.iloc[window + i] = val
                        
                        df[f'{prefix}entropy_{window}'] = entropy_col
                    except Exception as e:
                        print(f"警告: 创建entropy特征时出错: {e}，跳过该特征")
                        continue
        
        return df
    
    def _build_target_variables(self, df: pd.DataFrame, target_symbol: str, future_periods: int) -> pd.DataFrame:
        """构建目标变量"""
        print("构建目标变量...")
        
        close_col = f'{target_symbol}_close' if f'{target_symbol}_close' in df.columns else 'close'
        
        # 未来收益率
        future_return = np.log(df[close_col].shift(-future_periods) / df[close_col])
        df['future_return'] = future_return
        
        # 分类目标
        threshold = 0.002  # 0.2%
        df['target'] = 0
        df.loc[future_return > threshold, 'target'] = 1
        df.loc[future_return < -threshold, 'target'] = -1
        
        # 多级别目标
        thresholds = [0.001, 0.003, 0.005]  # 0.1%, 0.3%, 0.5%
        for i, thresh in enumerate(thresholds):
            target_col = f'target_level_{i+1}'
            df[target_col] = 0
            df.loc[future_return > thresh, target_col] = 1
            df.loc[future_return < -thresh, target_col] = -1
        
        # 输出目标变量分布
        target_counts = df['target'].value_counts().sort_index()
        print("目标变量分布:")
        for val, count in target_counts.items():
            percentage = count / len(df) * 100
            label = {-1: "下跌", 0: "横盘", 1: "上涨"}[val]
            print(f"  {label}({val}): {count} ({percentage:.2f}%)")
        
        return df
    
    def _clean_and_convert_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数据并转换为指定格式"""
        print("清理数据并转换格式...")
        
        # 移除时间戳列（保留数值特征）
        timestamp_cols = ['start_timestamp']
        df = df.drop(columns=[col for col in timestamp_cols if col in df.columns])
        
        # 处理无穷大值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 移除全为NaN的列
        df = df.dropna(axis=1, how='all')
        
        # 移除NaN比例过高的行（>50%）
        nan_ratio = df.isnull().sum(axis=1) / len(df.columns)
        df = df[nan_ratio <= 0.5]
        
        # 识别特征列和目标列
        target_cols = [col for col in df.columns if col.startswith('target') or col == 'future_return']
        feature_cols = [col for col in df.columns if col not in target_cols and col not in ['start_time', 'end_time']]
        
        # 转换特征列为指定的浮点类型
        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].astype(self.float_dtype)
            elif df[col].dtype in ['int64', 'int32']:
                # 整数列转换为浮点（因为可能有运算产生小数）
                df[col] = df[col].astype(self.float_dtype)
        
        # 保持目标列为float32（用于训练）
        for col in target_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        
        # 最终清理：移除仍然全为NaN的行
        df = df.dropna(subset=target_cols, how='all')
        
        print(f"数据清理完成，最终数据类型分布：")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} 列")
        
        return df
    
    def get_feature_importance_analysis(self, df: pd.DataFrame) -> Dict:
        """获取特征重要性分析"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        # 分离特征和目标
        target_cols = [col for col in df.columns if col.startswith('target') or col == 'future_return']
        time_cols = [col for col in df.columns if 'time' in col or df[col].dtype.name.startswith('datetime')]
        feature_cols = [col for col in df.columns 
                       if col not in target_cols 
                       and col not in time_cols 
                       and col not in ['start_time', 'end_time']
                       and df[col].dtype.kind in ['i', 'f']]  # 只包含数值类型
        
        X = df[feature_cols].fillna(0)
        y = df['target'].fillna(0)
        
        # 处理无穷大值
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 移除目标为NaN的行
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 随机森林特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 互信息特征重要性
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # 特征分组分析
        feature_groups = {
            'basic': [col for col in feature_cols if any(x in col for x in ['body', 'shadow', 'range', 'position'])],
            'technical': [col for col in feature_cols if any(x in col for x in ['rsi', 'macd', 'bb_', 'atr', 'sma', 'ema'])],
            'volume': [col for col in feature_cols if 'volume' in col or 'obv' in col or 'mfi' in col],
            'cross_asset': [col for col in feature_cols if '_corr_' in col or '_ratio' in col or '_diff_' in col],
            'temporal': [col for col in feature_cols if any(x in col for x in ['hour', 'dow', 'month', 'weekend'])],
            'statistical': [col for col in feature_cols if any(x in col for x in ['entropy', 'eigenval', 'skew', 'kurt'])]
        }
        
        return {
            'feature_importance': dict(zip(feature_cols, rf.feature_importances_)),
            'mutual_info': dict(zip(feature_cols, mi_scores)),
            'feature_groups': feature_groups,
            'total_features': len(feature_cols),
            'total_samples': len(X)
        } 