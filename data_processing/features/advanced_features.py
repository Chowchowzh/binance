# -*- coding: utf-8 -*-
"""
高级特征工程模块
包含滚动波动率、市场微观结构特征、高级统计特征等
基于金融机器学习最佳实践构建高质量特征
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
from numba import jit
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("Warning: TA-Lib not available. Technical features will be skipped.")

warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    高级特征工程器
    
    核心功能：
    1. 滚动波动率估计器（多种方法）
    2. 市场微观结构特征
    3. 高级统计特征
    4. 分数化差分特征
    5. 趋势和周期分解特征
    """
    
    def __init__(self, 
                 volatility_windows: List[int] = [5, 10, 20, 50],
                 microstructure_windows: List[int] = [5, 15, 30],
                 statistical_windows: List[int] = [10, 20, 50, 100]):
        """
        初始化高级特征工程器
        
        Args:
            volatility_windows: 波动率计算窗口列表
            microstructure_windows: 微观结构特征窗口列表  
            statistical_windows: 统计特征窗口列表
        """
        self.volatility_windows = volatility_windows
        self.microstructure_windows = microstructure_windows
        self.statistical_windows = statistical_windows
        
        print(f"初始化高级特征工程器:")
        print(f"  - 波动率窗口: {volatility_windows}")
        print(f"  - 微观结构窗口: {microstructure_windows}")
        print(f"  - 统计特征窗口: {statistical_windows}")
    
    def build_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建多种波动率特征
        
        Args:
            df: OHLCV数据DataFrame
            
        Returns:
            包含波动率特征的DataFrame
        """
        print("构建波动率特征...")
        
        features_df = df.copy()
        
        for window in self.volatility_windows:
            # 1. 简单波动率（对数收益率标准差）
            log_returns = np.log(df['close'] / df['close'].shift(1))
            features_df[f'volatility_simple_{window}'] = log_returns.rolling(window).std()
            
            # 2. Garman-Klass波动率估计器
            features_df[f'volatility_gk_{window}'] = self._garman_klass_volatility(
                df['high'], df['low'], df['open'], df['close'], window
            )
            
            # 3. Rogers-Satchell波动率估计器
            features_df[f'volatility_rs_{window}'] = self._rogers_satchell_volatility(
                df['high'], df['low'], df['open'], df['close'], window
            )
            
            # 4. Yang-Zhang波动率估计器
            features_df[f'volatility_yz_{window}'] = self._yang_zhang_volatility(
                df['high'], df['low'], df['open'], df['close'], window
            )
            
            # 5. Parkinson波动率（高低价差）
            features_df[f'volatility_parkinson_{window}'] = self._parkinson_volatility(
                df['high'], df['low'], window
            )
            
            # 6. 实现波动率的分位数
            rv = log_returns.abs().rolling(window).sum()
            features_df[f'realized_vol_rank_{window}'] = rv.rolling(window*2).rank(pct=True)
            
            # 7. 波动率的波动率
            vol_simple = features_df[f'volatility_simple_{window}']
            features_df[f'vol_of_vol_{window}'] = vol_simple.rolling(window).std()
        
        print(f"  - 生成波动率特征: {len([c for c in features_df.columns if 'volatility' in c])} 个")
        
        return features_df
    
    def build_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建市场微观结构特征
        
        Args:
            df: OHLCV数据DataFrame
            
        Returns:
            包含微观结构特征的DataFrame
        """
        print("构建市场微观结构特征...")
        
        features_df = df.copy()
        
        for window in self.microstructure_windows:
            # 1. 成交量相关特征
            features_df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            features_df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            features_df[f'volume_rank_{window}'] = df['volume'].rolling(window*2).rank(pct=True)
            
            # 2. 价格-成交量关系
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            features_df[f'price_volume_corr_{window}'] = price_change.rolling(window).corr(volume_change)
            
            # 3. VWAP特征
            vwap = (df['close'] * df['volume']).rolling(window).sum() / df['volume'].rolling(window).sum()
            features_df[f'vwap_{window}'] = vwap
            features_df[f'price_to_vwap_{window}'] = df['close'] / vwap
            
            # 4. 买卖压力指标（基于影线）
            body_size = abs(df['close'] - df['open'])
            upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
            lower_shadow = np.minimum(df['open'], df['close']) - df['low']
            
            features_df[f'upper_shadow_ratio_{window}'] = (upper_shadow / body_size).rolling(window).mean()
            features_df[f'lower_shadow_ratio_{window}'] = (lower_shadow / body_size).rolling(window).mean()
            
            # 5. 流动性代理指标
            high_low_spread = (df['high'] - df['low']) / df['close']
            features_df[f'liquidity_proxy_{window}'] = high_low_spread.rolling(window).mean()
            
            # 6. 价格跳跃检测
            log_returns = np.log(df['close'] / df['close'].shift(1))
            jump_threshold = log_returns.rolling(window*2).std() * 3
            features_df[f'price_jump_{window}'] = (log_returns.abs() > jump_threshold).rolling(window).sum()
            
            # 7. 订单流不平衡代理（基于收盘价相对位置）
            close_position = (df['close'] - df['low']) / (df['high'] - df['low'])
            features_df[f'order_flow_imbalance_{window}'] = close_position.rolling(window).mean()
        
        print(f"  - 生成微观结构特征: {len([c for c in features_df.columns if any(x in c for x in ['volume', 'vwap', 'shadow', 'liquidity', 'jump', 'order_flow'])])} 个")
        
        return features_df
    
    def build_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建高级统计特征
        
        Args:
            df: 数据DataFrame
            
        Returns:
            包含统计特征的DataFrame
        """
        print("构建高级统计特征...")
        
        features_df = df.copy()
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        for window in self.statistical_windows:
            # 1. 高阶矩特征
            features_df[f'skewness_{window}'] = log_returns.rolling(window).skew()
            features_df[f'kurtosis_{window}'] = log_returns.rolling(window).kurt()
            
            # 2. 分位数特征
            features_df[f'quantile_25_{window}'] = log_returns.rolling(window).quantile(0.25)
            features_df[f'quantile_75_{window}'] = log_returns.rolling(window).quantile(0.75)
            features_df[f'iqr_{window}'] = (features_df[f'quantile_75_{window}'] - 
                                           features_df[f'quantile_25_{window}'])
            
            # 3. 尾部风险特征
            features_df[f'var_5_{window}'] = log_returns.rolling(window).quantile(0.05)
            features_df[f'cvar_5_{window}'] = log_returns[log_returns <= features_df[f'var_5_{window}']].rolling(window).mean()
            
            # 4. 自相关特征
            features_df[f'autocorr_1_{window}'] = log_returns.rolling(window).apply(lambda x: x.autocorr(lag=1))
            features_df[f'autocorr_5_{window}'] = log_returns.rolling(window).apply(lambda x: x.autocorr(lag=5))
            
            # 5. 趋势强度
            price_sma = df['close'].rolling(window).mean()
            features_df[f'trend_strength_{window}'] = (df['close'] - price_sma) / price_sma
            
            # 6. 动量特征
            features_df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            
            # 7. 均值回归特征
            z_score = (df['close'] - price_sma) / df['close'].rolling(window).std()
            features_df[f'mean_reversion_{window}'] = z_score
            
            # 8. 波动率聚集性
            vol_cluster = log_returns.abs().rolling(window).mean()
            features_df[f'volatility_clustering_{window}'] = vol_cluster
            
            # 9. 信息比率
            if window >= 20:
                excess_return = log_returns - log_returns.rolling(window*2).mean()
                tracking_error = excess_return.rolling(window).std()
                features_df[f'information_ratio_{window}'] = excess_return.rolling(window).mean() / tracking_error
        
        print(f"  - 生成统计特征: {len([c for c in features_df.columns if any(x in c for x in ['skewness', 'kurtosis', 'quantile', 'var_', 'autocorr', 'trend', 'momentum', 'mean_reversion', 'clustering', 'information'])])} 个")
        
        return features_df
    
    def build_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建技术分析特征
        
        Args:
            df: OHLCV数据DataFrame
            
        Returns:
            包含技术特征的DataFrame
        """
        print("构建技术分析特征...")
        
        features_df = df.copy()
        
        if not HAS_TALIB:
            print("  - 跳过技术分析特征（TA-Lib不可用）")
            return features_df
        
        # 确保数据类型正确
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values
        close = df['close'].astype(float).values
        volume = df['volume'].astype(float).values
        
        # 1. 趋势指标
        features_df['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
        features_df['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
        features_df['dx_14'] = talib.DX(high, low, close, timeperiod=14)
        
        # 2. 动量指标
        features_df['rsi_14'] = talib.RSI(close, timeperiod=14)
        features_df['roc_10'] = talib.ROC(close, timeperiod=10)
        features_df['mom_10'] = talib.MOM(close, timeperiod=10)
        
        # 3. 波动性指标
        features_df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        features_df['natr_14'] = talib.NATR(high, low, close, timeperiod=14)
        
        # 4. 成交量指标
        features_df['ad'] = talib.AD(high, low, close, volume)
        features_df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        features_df['obv'] = talib.OBV(close, volume)
        
        # 5. 价格位置指标
        features_df['willr_14'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # 6. 布林带
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        features_df['bb_upper'] = bb_upper
        features_df['bb_middle'] = bb_middle
        features_df['bb_lower'] = bb_lower
        features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features_df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # 7. MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        features_df['macd'] = macd
        features_df['macd_signal'] = macd_signal
        features_df['macd_hist'] = macd_hist
        
        print(f"  - 生成技术分析特征: {len([c for c in features_df.columns if c not in df.columns])} 个")
        
        return features_df
    
    def build_fractal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建分形和分数化差分特征
        
        Args:
            df: 数据DataFrame
            
        Returns:
            包含分形特征的DataFrame
        """
        print("构建分形特征...")
        
        features_df = df.copy()
        log_prices = np.log(df['close'])
        
        # 1. 分数化差分（不同阶数）
        for d in [0.1, 0.2, 0.3, 0.4, 0.5]:
            features_df[f'frac_diff_{int(d*10)}'] = self._fractional_diff(log_prices, d)
        
        # 2. Hurst指数（不同窗口）
        for window in [50, 100, 200]:
            if len(df) >= window:
                features_df[f'hurst_{window}'] = log_prices.rolling(window).apply(
                    lambda x: self._hurst_exponent(x.values) if len(x.dropna()) >= 20 else np.nan
                )
        
        # 3. 去趋势波动分析（DFA）
        for window in [50, 100]:
            if len(df) >= window:
                features_df[f'dfa_{window}'] = log_prices.rolling(window).apply(
                    lambda x: self._detrended_fluctuation_analysis(x.values) if len(x.dropna()) >= 20 else np.nan
                )
        
        print(f"  - 生成分形特征: {len([c for c in features_df.columns if any(x in c for x in ['frac_diff', 'hurst', 'dfa'])])} 个")
        
        return features_df
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建所有高级特征
        
        Args:
            df: OHLCV数据DataFrame
            
        Returns:
            包含所有特征的DataFrame
        """
        print("\n" + "="*80)
        print("🔧 开始构建所有高级特征")
        print("="*80)
        
        start_time = pd.Timestamp.now()
        
        # 按顺序构建各类特征
        features_df = self.build_volatility_features(df)
        features_df = self.build_microstructure_features(features_df)
        features_df = self.build_statistical_features(features_df)
        features_df = self.build_technical_features(features_df)
        features_df = self.build_fractal_features(features_df)
        
        # 特征清理
        features_df = self._clean_features(features_df)
        
        end_time = pd.Timestamp.now()
        elapsed = (end_time - start_time).total_seconds()
        
        original_features = len(df.columns)
        new_features = len(features_df.columns) - original_features
        
        print(f"\n✅ 特征构建完成:")
        print(f"  - 原始特征: {original_features}")
        print(f"  - 新增特征: {new_features}")
        print(f"  - 总特征数: {len(features_df.columns)}")
        print(f"  - 用时: {elapsed:.2f}秒")
        
        return features_df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理特征数据"""
        print("清理特征数据...")
        
        # 处理无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 计算特征质量统计
        nan_ratios = df.isnull().sum() / len(df)
        high_nan_features = nan_ratios[nan_ratios > 0.5].index.tolist()
        
        if high_nan_features:
            print(f"  - 移除高缺失率特征: {len(high_nan_features)} 个")
            df = df.drop(columns=high_nan_features)
        
        # 前向填充缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"  - 最终特征数: {len(df.columns)}")
        
        return df
    
    # === 内部辅助方法 ===
    
    def _garman_klass_volatility(self, high, low, open_price, close, window):
        """Garman-Klass波动率估计器"""
        hl = np.log(high / low)
        co = np.log(close / open_price)
        gk = 0.5 * hl**2 - (2*np.log(2)-1) * co**2
        return gk.rolling(window).mean().apply(np.sqrt)
    
    def _rogers_satchell_volatility(self, high, low, open_price, close, window):
        """Rogers-Satchell波动率估计器"""
        ho = np.log(high / open_price)
        hc = np.log(high / close)
        lo = np.log(low / open_price)
        lc = np.log(low / close)
        rs = ho * hc + lo * lc
        return rs.rolling(window).mean().apply(np.sqrt)
    
    def _yang_zhang_volatility(self, high, low, open_price, close, window):
        """Yang-Zhang波动率估计器"""
        log_ho = np.log(high / open_price)
        log_lo = np.log(low / open_price)
        log_co = np.log(close / open_price)
        log_oc = np.log(open_price / close.shift(1))
        log_cc = np.log(close / close.shift(1))
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        close_to_close = log_cc**2
        open_to_close = log_oc**2
        
        k = 0.34 / (1.34 + (window+1)/(window-1))
        yz = open_to_close + k*close_to_close + (1-k)*rs
        
        return yz.rolling(window).mean().apply(np.sqrt)
    
    def _parkinson_volatility(self, high, low, window):
        """Parkinson波动率估计器"""
        hl = np.log(high / low)
        parkinson = hl**2 / (4 * np.log(2))
        return parkinson.rolling(window).mean().apply(np.sqrt)
    
    def _fractional_diff(self, series: pd.Series, d: float) -> pd.Series:
        """计算分数化差分"""
        try:
            from statsmodels.tsa.stattools import adfuller
            from sklearn.linear_model import LinearRegression
            
            # 简化实现：使用差分近似
            if d == 0:
                return series
            elif d == 1:
                return series.diff()
            else:
                # 使用线性插值近似分数化差分
                diff1 = series.diff()
                diff0 = series
                return diff0 * (1 - d) + diff1 * d
        except:
            return series.diff() * d
    
    def _hurst_exponent(self, ts: np.ndarray) -> float:
        """计算Hurst指数"""
        try:
            lags = range(2, min(20, len(ts)//4))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5  # 随机游走的理论值
    
    def _detrended_fluctuation_analysis(self, ts: np.ndarray) -> float:
        """去趋势波动分析"""
        try:
            N = len(ts)
            if N < 20:
                return np.nan
            
            # 积分序列
            y = np.cumsum(ts - np.mean(ts))
            
            # 不同窗口大小
            scales = np.logspace(1, np.log10(N//4), 10).astype(int)
            fluctuations = []
            
            for scale in scales:
                # 分段
                segments = N // scale
                variance = []
                
                for i in range(segments):
                    start, end = i * scale, (i + 1) * scale
                    segment = y[start:end]
                    
                    # 去趋势
                    t = np.arange(len(segment))
                    poly = np.polyfit(t, segment, 1)
                    trend = np.polyval(poly, t)
                    
                    variance.append(np.mean((segment - trend) ** 2))
                
                fluctuations.append(np.sqrt(np.mean(variance)))
            
            # 计算缩放指数
            coeffs = np.polyfit(np.log(scales), np.log(fluctuations), 1)
            return coeffs[0]
        except:
            return 0.5


# 便利函数
def add_advanced_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    为DataFrame添加高级特征
    
    Args:
        df: OHLCV数据DataFrame
        **kwargs: AdvancedFeatureEngineer的参数
        
    Returns:
        包含高级特征的DataFrame
    """
    engineer = AdvancedFeatureEngineer(**kwargs)
    return engineer.build_all_features(df)


def analyze_feature_importance(df: pd.DataFrame, 
                              target_column: str,
                              feature_columns: List[str] = None) -> pd.DataFrame:
    """
    分析特征重要性
    
    Args:
        df: 包含特征和目标的DataFrame
        target_column: 目标列名
        feature_columns: 特征列名列表，如果为None则自动检测
        
    Returns:
        特征重要性分析结果
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    from scipy.stats import pearsonr
    
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # 准备数据
    X = df[feature_columns].fillna(0)
    y = df[target_column].fillna(0)
    
    # 有效样本
    valid_mask = ~(np.isnan(y) | np.isinf(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) == 0:
        return pd.DataFrame()
    
    # 随机森林重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # 互信息
    try:
        mi_scores = mutual_info_regression(X, y, random_state=42)
    except:
        mi_scores = np.zeros(len(feature_columns))
    
    # 相关系数
    correlations = []
    for col in feature_columns:
        try:
            corr, _ = pearsonr(X[col], y)
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        except:
            correlations.append(0)
    
    # 汇总结果
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'rf_importance': rf_importance,
        'mutual_info': mi_scores,
        'correlation': correlations
    })
    
    # 综合评分
    importance_df['composite_score'] = (
        0.5 * importance_df['rf_importance'] + 
        0.3 * importance_df['mutual_info'] + 
        0.2 * importance_df['correlation']
    )
    
    return importance_df.sort_values('composite_score', ascending=False) 