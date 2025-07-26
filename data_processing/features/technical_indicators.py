# -*- coding: utf-8 -*-
"""
技术指标计算模块
提供各种技术分析指标的计算功能
"""

import talib as ta
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def add_price_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加价格相关特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            symbol: 交易对符号
            
        Returns:
            添加了价格特征的DataFrame
        """
        open_col = f'{symbol}_open'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'
        
        # 基础价格特征
        df[f'{symbol}_price_range'] = df[high_col] - df[low_col]
        
        # 改为对数收益率
        df[f'{symbol}_log_return'] = np.log(df[close_col] / df[close_col].shift(1).replace(0, np.nan))
        
        # 价格位置指标
        df[f'{symbol}_price_position'] = (df[close_col] - df[low_col]) / (df[high_col] - df[low_col] + 1e-10)
        
        # VWAP相关
        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        df[f'{symbol}_vwap'] = (typical_price * df[volume_col]).cumsum() / df[volume_col].cumsum()
        df[f'{symbol}_vwap_distance'] = df[close_col] / df[f'{symbol}_vwap'] - 1
        
        return df
    
    @staticmethod  
    def add_order_flow_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加订单流和流动性特征
        
        Args:
            df: 包含完整K线数据的DataFrame
            symbol: 交易对符号
            
        Returns:
            添加了订单流特征的DataFrame
        """
        volume_col = f'{symbol}_volume'
        quote_volume_col = f'{symbol}_quote_asset_volume'
        taker_buy_vol_col = f'{symbol}_taker_buy_base_asset_volume'
        taker_buy_quote_vol_col = f'{symbol}_taker_buy_quote_asset_volume'
        trades_col = f'{symbol}_number_of_trades'
        close_col = f'{symbol}_close'
        
        # 检查必需列
        required_cols = [volume_col, quote_volume_col, taker_buy_vol_col, taker_buy_quote_vol_col, trades_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"警告：{symbol} 缺少订单流数据列: {missing_cols}")
            return df
        
        # 计算卖方成交量
        taker_sell_vol = df[volume_col] - df[taker_buy_vol_col]
        taker_sell_quote_vol = df[quote_volume_col] - df[taker_buy_quote_vol_col]
        
        # 订单流不平衡 (Order Flow Imbalance)
        df[f'{symbol}_ofi'] = df[taker_buy_vol_col] - taker_sell_vol
        df[f'{symbol}_ofi_quote'] = df[taker_buy_quote_vol_col] - taker_sell_quote_vol
        
        # 买卖压力和比率
        df[f'{symbol}_buy_pressure'] = df[taker_buy_vol_col] / df[volume_col]
        df[f'{symbol}_sell_pressure'] = taker_sell_vol / df[volume_col]
        df[f'{symbol}_buy_sell_ratio'] = df[taker_buy_vol_col] / (taker_sell_vol + 1e-10)
        
        # 平均交易规模
        df[f'{symbol}_avg_trade_size'] = df[volume_col] / (df[trades_col] + 1e-10)
        df[f'{symbol}_taker_buy_avg_price'] = df[taker_buy_quote_vol_col] / (df[taker_buy_vol_col] + 1e-10)
        df[f'{symbol}_avg_price'] = df[quote_volume_col] / (df[volume_col] + 1e-10)
        
        # 滚动订单流特征 - 大幅减少窗口数量
        for window in [10, 30]:  # 从[5, 10, 20, 60]减少到[10, 30]
            # OFI滚动统计 - 只保留关键指标
            df[f'{symbol}_ofi_sum_{window}'] = df[f'{symbol}_ofi'].rolling(window).sum()
            df[f'{symbol}_ofi_mean_{window}'] = df[f'{symbol}_ofi'].rolling(window).mean()
            
            # 买卖压力滚动统计 - 简化
            df[f'{symbol}_buy_pressure_mean_{window}'] = df[f'{symbol}_buy_pressure'].rolling(window).mean()
            
            # 成交频率 - 简化
            df[f'{symbol}_trade_frequency_{window}'] = df[trades_col].rolling(window).sum()
        
        return df
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加动量指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            symbol: 交易对符号
            
        Returns:
            添加了动量指标的DataFrame
        """
        close_col = f'{symbol}_close'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        volume_col = f'{symbol}_volume'
        
        close = df[close_col]
        high = df[high_col]
        low = df[low_col]
        volume = df[volume_col]
        
        # RSI
        df[f'{symbol}_rsi_14'] = ta.RSI(close, timeperiod=14)
        df[f'{symbol}_rsi_21'] = ta.RSI(close, timeperiod=21)
        
        # MACD
        macd, macd_signal, macd_hist = ta.MACD(close)
        df[f'{symbol}_macd'] = macd
        df[f'{symbol}_macd_signal'] = macd_signal
        df[f'{symbol}_macd_hist'] = macd_hist
        
        # 动量
        df[f'{symbol}_momentum_5'] = ta.MOM(close, timeperiod=5)
        df[f'{symbol}_momentum_10'] = ta.MOM(close, timeperiod=10)
        df[f'{symbol}_momentum_20'] = ta.MOM(close, timeperiod=20)
        
        # ROC (变化率)
        df[f'{symbol}_roc_5'] = ta.ROC(close, timeperiod=5)
        df[f'{symbol}_roc_10'] = ta.ROC(close, timeperiod=10)
        df[f'{symbol}_roc_20'] = ta.ROC(close, timeperiod=20)
        
        # 威廉指标
        df[f'{symbol}_williams_r_14'] = ta.WILLR(high, low, close, timeperiod=14)
        
        # 随机指标
        slowk, slowd = ta.STOCH(high, low, close)
        df[f'{symbol}_stoch_k'] = slowk
        df[f'{symbol}_stoch_d'] = slowd
        
        return df
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加波动率指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            symbol: 交易对符号
            
        Returns:
            添加了波动率指标的DataFrame
        """
        close_col = f'{symbol}_close'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        
        close = df[close_col]
        high = df[high_col]
        low = df[low_col]
        
        # ATR (真实波幅)
        df[f'{symbol}_atr_14'] = ta.ATR(high, low, close, timeperiod=14)
        df[f'{symbol}_atr_21'] = ta.ATR(high, low, close, timeperiod=21)
        
        # 布林带
        upper, middle, lower = ta.BBANDS(close, timeperiod=20)
        df[f'{symbol}_bb_upper'] = upper
        df[f'{symbol}_bb_middle'] = middle  
        df[f'{symbol}_bb_lower'] = lower
        df[f'{symbol}_bb_width'] = (upper - lower) / middle
        df[f'{symbol}_bb_position'] = (close - lower) / (upper - lower + 1e-10)
        
        # 历史波动率 - 基于对数收益率
        returns = np.log(close / close.shift(1).replace(0, np.nan))
        for window in [5, 10, 20, 60]:
            df[f'{symbol}_volatility_{window}'] = returns.rolling(window).std() * np.sqrt(1440)  # 年化波动率
            df[f'{symbol}_volatility_rank_{window}'] = (
                df[f'{symbol}_volatility_{window}'].rolling(252).rank() / 252  # 252期排名
            )
        
        # 平均真实波幅百分比
        df[f'{symbol}_atr_pct_14'] = df[f'{symbol}_atr_14'] / close * 100
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加成交量指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            symbol: 交易对符号
            
        Returns:
            添加了成交量指标的DataFrame
        """
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        
        close = df[close_col]
        volume = df[volume_col]
        high = df[high_col]
        low = df[low_col]
        
        # 成交量移动平均 - 减少窗口数量
        for period in [10, 30]:  # 从[5, 10, 20, 60]减少到[10, 30]
            df[f'{symbol}_volume_sma_{period}'] = ta.SMA(volume, timeperiod=period)
            df[f'{symbol}_volume_ratio_{period}'] = volume / df[f'{symbol}_volume_sma_{period}']
        
        # 成交量加权移动平均价格偏离 - 减少窗口
        typical_price = (high + low + close) / 3
        for period in [20]:  # 从[10, 20]减少到[20]
            vwma = (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
            df[f'{symbol}_vwma_deviation_{period}'] = close / vwma - 1

        
        # 替代的成交量指标
        df[f'{symbol}_mfi_14'] = ta.MFI(high, low, close, volume, timeperiod=14)  # 资金流指标
        
        # OBV (能量潮)
        df[f'{symbol}_obv'] = ta.OBV(close, volume)
        
        # A/D Line (累积/派发线)
        df[f'{symbol}_ad'] = ta.AD(high, low, close, volume)
        
        # Chaikin A/D Oscillator
        df[f'{symbol}_adosc'] = ta.ADOSC(high, low, close, volume)
        
        return df
    
    @staticmethod
    def add_market_microstructure_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加高级市场微观结构特征
        
        Args:
            df: 包含完整交易数据的DataFrame
            symbol: 交易对符号
            
        Returns:
            添加了市场微观结构特征的DataFrame
        """
        volume_col = f'{symbol}_volume'
        taker_buy_vol_col = f'{symbol}_taker_buy_base_asset_volume'
        trades_col = f'{symbol}_number_of_trades'
        close_col = f'{symbol}_close'
        
        # 检查必需列
        required_cols = [volume_col, taker_buy_vol_col, trades_col, close_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"警告：{symbol} 缺少微观结构数据列: {missing_cols}")
            return df
        
        # 市场深度代理
        returns = np.log(df[close_col] / df[close_col].shift(1).replace(0, np.nan))
        
        # 价格影响 (Price Impact) - 减少窗口
        for window in [10, 30]:  # 从[5, 10, 20]减少到[10, 30]
            # Kyle's Lambda (价格影响系数)
            volume_window = df[volume_col].rolling(window)
            returns_window = returns.rolling(window)
            
            df[f'{symbol}_price_impact_{window}'] = (
                abs(returns_window.mean()) / (volume_window.mean() + 1e-10)
            )
            
            # Amihud非流动性比率
            df[f'{symbol}_amihud_illiquidity_{window}'] = (
                abs(returns_window.mean()) / (df[volume_col] + 1e-10)
            ).rolling(window).mean()
        
        # 买卖不平衡强度 - 减少窗口
        buy_volume = df[taker_buy_vol_col]
        sell_volume = df[volume_col] - buy_volume
        
        for window in [20]:  # 从[5, 10, 20]减少到[20]
            # 订单流不平衡的标准化版本
            ofi_raw = buy_volume - sell_volume
            total_volume = df[volume_col].rolling(window).sum()
            df[f'{symbol}_normalized_ofi_{window}'] = (
                ofi_raw.rolling(window).sum() / (total_volume + 1e-10)
            )
        
        # 交易频率和规模特征 - 减少窗口
        avg_trade_size = df[volume_col] / (df[trades_col] + 1e-10)
        
        for window in [20]:  # 从[10, 20, 60]减少到[20]
            # 平均交易规模变化
            df[f'{symbol}_avg_trade_size_change_{window}'] = (
                avg_trade_size / avg_trade_size.rolling(window).mean() - 1
            )
            
            # 交易密度 (trades per volume)
            df[f'{symbol}_trade_density_{window}'] = (
                df[trades_col].rolling(window).sum() / (df[volume_col].rolling(window).sum() + 1e-10)
            )
        
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, symbol: str, windows: List[int] = None) -> pd.DataFrame:
        """
        添加移动平均线差值指标 - 不保存绝对值，只保存不同窗口的差值
        
        Args:
            df: 数据DataFrame
            symbol: 交易对符号
            windows: 移动平均窗口期列表
            
        Returns:
            添加了移动平均差值特征的DataFrame
        """
        if windows is None:
            windows = [5, 10, 20, 60]
        
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'
        
        # 临时计算各种移动平均（不保存绝对值）
        temp_mas = {}
        
        for window in windows:
            # 计算各种移动平均但不直接保存到DataFrame
            temp_mas[f'sma_{window}'] = ta.SMA(df[close_col], timeperiod=window)
            temp_mas[f'ema_{window}'] = ta.EMA(df[close_col], timeperiod=window)
            temp_mas[f'wma_{window}'] = ta.WMA(df[close_col], timeperiod=window)
            temp_mas[f'vwma_{window}'] = (df[close_col] * df[volume_col]).rolling(window).sum() / df[volume_col].rolling(window).sum()
        
        # 计算不同窗口的移动平均差值
        for i, window1 in enumerate(windows):
            for j, window2 in enumerate(windows):
                if i < j:  # 避免重复计算，只计算 window1 < window2 的情况
                    # SMA差值
                    df[f'{symbol}_sma_diff_{window1}_{window2}'] = temp_mas[f'sma_{window1}'] - temp_mas[f'sma_{window2}']
                    
                    # EMA差值
                    df[f'{symbol}_ema_diff_{window1}_{window2}'] = temp_mas[f'ema_{window1}'] - temp_mas[f'ema_{window2}']
                    
                    # WMA差值
                    df[f'{symbol}_wma_diff_{window1}_{window2}'] = temp_mas[f'wma_{window1}'] - temp_mas[f'wma_{window2}']
                    
                    # VWMA差值
                    df[f'{symbol}_vwma_diff_{window1}_{window2}'] = temp_mas[f'vwma_{window1}'] - temp_mas[f'vwma_{window2}']
        
        # 计算价格相对于最短期移动平均的位置（保留这个重要特征）
        shortest_window = min(windows)
        df[f'{symbol}_price_vs_sma_{shortest_window}'] = df[close_col] / temp_mas[f'sma_{shortest_window}'] - 1
        df[f'{symbol}_price_vs_ema_{shortest_window}'] = df[close_col] / temp_mas[f'ema_{shortest_window}'] - 1
        
        return df
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加趋势指标
        
        Args:
            df: 数据DataFrame
            symbol: 交易对符号
            
        Returns:
            添加了趋势指标的DataFrame
        """
        close_col = f'{symbol}_close'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        
        # ADX (平均趋向指标)
        df[f'{symbol}_adx_14'] = ta.ADX(df[high_col], df[low_col], df[close_col], timeperiod=14)
        
        # DI指标
        df[f'{symbol}_plus_di'] = ta.PLUS_DI(df[high_col], df[low_col], df[close_col], timeperiod=14)
        df[f'{symbol}_minus_di'] = ta.MINUS_DI(df[high_col], df[low_col], df[close_col], timeperiod=14)
        
        # 抛物线SAR
        df[f'{symbol}_sar'] = ta.SAR(df[high_col], df[low_col])
        
        # 商品通道指数
        df[f'{symbol}_cci_14'] = ta.CCI(df[high_col], df[low_col], df[close_col], timeperiod=14)
        df[f'{symbol}_cci_20'] = ta.CCI(df[high_col], df[low_col], df[close_col], timeperiod=20)
        
        return df
    
    @staticmethod
    def add_pattern_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加形态指标
        
        Args:
            df: 数据DataFrame
            symbol: 交易对符号
            
        Returns:
            添加了形态指标的DataFrame
        """
        open_col = f'{symbol}_open'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'
        
        # 蜡烛图形态
        df[f'{symbol}_doji'] = ta.CDLDOJI(df[open_col], df[high_col], df[low_col], df[close_col])
        df[f'{symbol}_hammer'] = ta.CDLHAMMER(df[open_col], df[high_col], df[low_col], df[close_col])
        df[f'{symbol}_engulfing'] = ta.CDLENGULFING(df[open_col], df[high_col], df[low_col], df[close_col])
        df[f'{symbol}_harami'] = ta.CDLHARAMI(df[open_col], df[high_col], df[low_col], df[close_col])
        
        # 价格形态
        df[f'{symbol}_gap_up'] = (df[low_col] > df[high_col].shift(1)).astype(int)
        df[f'{symbol}_gap_down'] = (df[high_col] < df[low_col].shift(1)).astype(int)
        
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        添加所有技术指标
        
        Args:
            df: 数据DataFrame
            symbol: 交易对符号
            
        Returns:
            添加了所有技术指标的DataFrame
        """
        df = TechnicalIndicators.add_price_features(df, symbol)
        df = TechnicalIndicators.add_moving_averages(df, symbol)
        df = TechnicalIndicators.add_momentum_indicators(df, symbol)
        df = TechnicalIndicators.add_volatility_indicators(df, symbol)
        df = TechnicalIndicators.add_volume_indicators(df, symbol)
        df = TechnicalIndicators.add_trend_indicators(df, symbol)
        df = TechnicalIndicators.add_pattern_indicators(df, symbol)
        
        # 添加市场微观结构特征
        df = TechnicalIndicators.add_market_microstructure_features(df, symbol)
        
        return df 