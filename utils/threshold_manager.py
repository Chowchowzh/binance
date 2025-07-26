# -*- coding: utf-8 -*-
"""
统一阈值管理系统
所有阈值都基于分位数控制，确保逻辑一致性
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union


class ThresholdManager:
    """统一的阈值管理器，基于分位数控制"""
    
    def __init__(self):
        self.cached_thresholds: Dict[str, float] = {}
        self.signal_history: Dict[str, np.ndarray] = {}
        self.strategy_configs = {
            "center_based": {"percentile": 5.0, "adaptive": False},
            "kelly": {"percentile": 10.0, "adaptive": False}, 
            "adaptive": {"percentile": 3.0, "adaptive": True},
            "volatility_adjusted": {"percentile": 8.0, "adaptive": True},
            "momentum_mean_reversion": {"percentile": 5.0, "adaptive": True},
            "ensemble": {"percentile": 6.0, "adaptive": True}
        }
    
    def calculate_percentile_threshold(self, 
                                       signals: Union[np.ndarray, pd.Series], 
                                       percentile: float,
                                       cache_key: Optional[str] = None) -> float:
        """
        基于分位数计算阈值
        
        Args:
            signals: 信号数组或Series
            percentile: 分位数 (0-100)
            cache_key: 缓存键名，如果提供则缓存结果
        
        Returns:
            阈值
        """
        if isinstance(signals, pd.Series):
            signals = signals.values
            
        if len(signals) == 0:
            return 0.0
            
        # 移除NaN和无限值
        valid_signals = signals[~(np.isnan(signals) | np.isinf(signals))]
        
        if len(valid_signals) == 0:
            return 0.0
            
        # 使用绝对值计算阈值
        abs_signals = np.abs(valid_signals)
        threshold = float(np.percentile(abs_signals, percentile))
        
        if cache_key:
            self.cached_thresholds[cache_key] = threshold
            self.signal_history[cache_key] = valid_signals.copy()
        
        return threshold
    
    def get_trading_threshold(self, signals: Union[np.ndarray, pd.Series], 
                             percentile: float = 5.0) -> float:
        """获取交易信号阈值"""
        return self.calculate_percentile_threshold(signals, percentile, "trading")
    
    def get_position_threshold(self, signals: Union[np.ndarray, pd.Series], 
                              percentile: float = 10.0) -> float:
        """获取仓位控制阈值"""
        return self.calculate_percentile_threshold(signals, percentile, "position")
    
    def get_dynamic_threshold(self, signals: Union[np.ndarray, pd.Series], 
                             window: int = 100, percentile: float = 5.0) -> np.ndarray:
        """
        计算动态阈值（滚动窗口）
        
        Args:
            signals: 信号序列
            window: 滚动窗口大小
            percentile: 分位数
            
        Returns:
            动态阈值数组
        """
        if isinstance(signals, pd.Series):
            signals_array = signals.values
            use_pandas = True
        else:
            signals_array = signals
            use_pandas = False
        
        dynamic_thresholds = np.zeros_like(signals_array)
        
        for i in range(len(signals_array)):
            start_idx = max(0, i - window + 1)
            window_signals = signals_array[start_idx:i+1]
            
            if len(window_signals) > 0:
                dynamic_thresholds[i] = self.calculate_percentile_threshold(
                    window_signals, percentile
                )
        
        if use_pandas:
            return pd.Series(dynamic_thresholds, index=signals.index)
        else:
            return dynamic_thresholds
    
    def get_adaptive_threshold(self, signals: Union[np.ndarray, pd.Series], 
                              base_percentile: float = 5.0,
                              volatility_adjustment: bool = True) -> float:
        """
        获取自适应阈值
        
        Args:
            signals: 信号数组
            base_percentile: 基础分位数
            volatility_adjustment: 是否进行波动率调整
            
        Returns:
            自适应阈值
        """
        if isinstance(signals, pd.Series):
            signals = signals.values
            
        # 基础阈值
        base_threshold = self.calculate_percentile_threshold(signals, base_percentile)
        
        if not volatility_adjustment or len(signals) < 10:
            return base_threshold
        
        # 计算信号波动率
        signal_volatility = np.std(signals)
        avg_volatility = np.mean(np.abs(signals))
        
        # 波动率调整因子
        if avg_volatility > 0:
            volatility_factor = signal_volatility / avg_volatility
            # 高波动时提高阈值，低波动时降低阈值
            adjusted_threshold = base_threshold * (0.5 + 0.5 * volatility_factor)
        else:
            adjusted_threshold = base_threshold
        
        return adjusted_threshold
    
    def get_strategy_specific_threshold(self, strategy_name: str, 
                                        signals: Union[np.ndarray, pd.Series]) -> float:
        """获取特定策略的阈值"""
        config = self.strategy_configs.get(strategy_name, {"percentile": 5.0, "adaptive": False})
        
        if config["adaptive"]:
            return self.get_adaptive_threshold(signals, config["percentile"])
        else:
            return self.calculate_percentile_threshold(signals, config["percentile"])
    
    def get_multi_level_thresholds(self, signals: Union[np.ndarray, pd.Series],
                                  levels: List[float] = None) -> Dict[str, float]:
        """
        获取多级阈值
        
        Args:
            signals: 信号数组
            levels: 阈值级别列表（分位数）
            
        Returns:
            多级阈值字典
        """
        if levels is None:
            levels = [1.0, 5.0, 10.0, 25.0, 50.0]
        
        thresholds = {}
        for level in levels:
            key = f"level_{level}p"
            thresholds[key] = self.calculate_percentile_threshold(signals, level)
        
        return thresholds
    
    def validate_threshold(self, threshold: float, signals: Union[np.ndarray, pd.Series],
                          min_activation_rate: float = 0.01, 
                          max_activation_rate: float = 0.5) -> Dict[str, Any]:
        """
        验证阈值的有效性
        
        Args:
            threshold: 要验证的阈值
            signals: 信号数组
            min_activation_rate: 最小激活率
            max_activation_rate: 最大激活率
            
        Returns:
            验证结果
        """
        if isinstance(signals, pd.Series):
            signals = signals.values
            
        # 计算激活率
        abs_signals = np.abs(signals)
        activation_count = np.sum(abs_signals > threshold)
        activation_rate = activation_count / len(signals) if len(signals) > 0 else 0
        
        # 判断是否在合理范围内
        is_valid = min_activation_rate <= activation_rate <= max_activation_rate
        
        validation_result = {
            'threshold': threshold,
            'activation_count': activation_count,
            'total_signals': len(signals),
            'activation_rate': activation_rate,
            'is_valid': is_valid,
            'min_activation_rate': min_activation_rate,
            'max_activation_rate': max_activation_rate
        }
        
        if not is_valid:
            if activation_rate < min_activation_rate:
                validation_result['issue'] = 'threshold_too_high'
                validation_result['suggestion'] = 'lower_threshold'
            else:
                validation_result['issue'] = 'threshold_too_low'
                validation_result['suggestion'] = 'raise_threshold'
        
        return validation_result
    
    def auto_adjust_threshold(self, signals: Union[np.ndarray, pd.Series],
                             target_activation_rate: float = 0.1,
                             tolerance: float = 0.02) -> float:
        """
        自动调整阈值以达到目标激活率
        
        Args:
            signals: 信号数组
            target_activation_rate: 目标激活率
            tolerance: 容忍度
            
        Returns:
            调整后的阈值
        """
        if isinstance(signals, pd.Series):
            signals = signals.values
            
        abs_signals = np.abs(signals)
        sorted_signals = np.sort(abs_signals)
        
        # 计算目标分位数
        target_percentile = (1 - target_activation_rate) * 100
        
        # 确保在有效范围内
        target_percentile = max(1.0, min(99.0, target_percentile))
        
        optimal_threshold = np.percentile(sorted_signals, target_percentile)
        
        # 验证结果
        validation = self.validate_threshold(
            optimal_threshold, signals,
            target_activation_rate - tolerance,
            target_activation_rate + tolerance
        )
        
        return optimal_threshold
    
    def get_threshold_summary(self) -> Dict[str, Any]:
        """获取阈值管理器的摘要信息"""
        return {
            'cached_thresholds': self.cached_thresholds.copy(),
            'signal_history_keys': list(self.signal_history.keys()),
            'strategy_configs': self.strategy_configs.copy(),
            'total_cached_signals': sum(len(signals) for signals in self.signal_history.values())
        }
    
    def clear_cache(self, specific_key: Optional[str] = None):
        """
        清除缓存
        
        Args:
            specific_key: 特定键名，如果不提供则清除所有缓存
        """
        if specific_key:
            self.cached_thresholds.pop(specific_key, None)
            self.signal_history.pop(specific_key, None)
        else:
            self.cached_thresholds.clear()
            self.signal_history.clear()
    
    def export_thresholds(self) -> pd.DataFrame:
        """导出阈值配置为DataFrame"""
        data = []
        for strategy, config in self.strategy_configs.items():
            data.append({
                'strategy': strategy,
                'percentile': config['percentile'],
                'adaptive': config['adaptive'],
                'cached_threshold': self.cached_thresholds.get(strategy, None)
            })
        
        return pd.DataFrame(data)


# 全局阈值管理器实例
threshold_manager = ThresholdManager()


def calculate_dynamic_threshold(signals: Union[np.ndarray, pd.Series], 
                               percentile: float = 5.0) -> float:
    """
    兼容性函数 - 计算动态阈值
    
    Args:
        signals: 信号数组
        percentile: 分位数
        
    Returns:
        阈值
    """
    return threshold_manager.calculate_percentile_threshold(signals, percentile) 