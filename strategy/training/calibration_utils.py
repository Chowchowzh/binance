# -*- coding: utf-8 -*-
"""
信号校准工具模块
负责信号校准和阈值计算
"""

import pandas as pd
import numpy as np
from utils.threshold_manager import threshold_manager


def create_signal_calibrator(signals, future_returns, n_bins=20):
    """
    创建一个信号校准器，将原始信号映射到真实的经验未来收益
    """
    print("\n--- 创建信号校准器 (基于真实收益) ---")
    
    calibration_df = pd.DataFrame({'signal': signals, 'future_return': future_returns}).dropna()
    
    try:
        calibration_df['bin'] = pd.qcut(calibration_df['signal'], q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        calibration_df['bin'] = pd.cut(calibration_df['signal'], bins=n_bins, labels=False, duplicates='drop')

    calibrator = calibration_df.groupby('bin').agg(
        mean_signal=('signal', 'mean'),
        empirical_edge=('future_return', 'mean')
    )
    print("校准器创建完成:")
    print(calibrator)
    
    return calibrator


def apply_signal_calibrator(signals, calibrator):
    """将校准器应用于新信号"""
    print("\n--- 应用信号校准器 ---")
    
    bin_edges = [-np.inf] + calibrator['mean_signal'].tolist()
    signal_series = pd.Series(signals, name="signal")
    
    binned = pd.cut(signal_series, bins=bin_edges, labels=False, right=False)
    binned_series = pd.Series(binned, dtype=float)
    
    binned_series_filled = binned_series.fillna(float(len(calibrator) - 1))
    signal_bins = binned_series_filled.astype(int)

    calibrated_signals = signal_bins.map(calibrator['empirical_edge'])
    final_signals = calibrated_signals.fillna(0.0)
    
    print("信号校准完成")
    return final_signals.values


def calculate_dynamic_threshold(train_signals, trading_config, logger=None):
    """计算动态交易阈值"""
    threshold_percentile = trading_config.prediction_threshold_percentile
    prediction_threshold = threshold_manager.get_trading_threshold(
        train_signals, threshold_percentile)
    
    if logger:
        logger.log(f"动态交易阈值 (p{threshold_percentile}): {prediction_threshold:.6f}")
    
    return prediction_threshold 