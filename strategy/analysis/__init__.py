# -*- coding: utf-8 -*-
"""
分析模块
负责策略分析、结果可视化和性能评估
"""

from .alpha_analysis import analyze_signal_alpha
from .backtest_analysis import plot_backtest_results
from .result_logger import ResultLogger

__all__ = [
    # Alpha分析
    'analyze_signal_alpha',
    
    # 回测分析
    'plot_backtest_results',
    
    # 结果记录
    'ResultLogger'
]
