# -*- coding: utf-8 -*-
"""
回测模块
负责策略回测、仓位控制和风险管理
"""

from .backtest_runner import run_backtest
# from .smart_position_control import (
#     create_smart_controller,
#     get_strategy_info,
#     STRATEGY_DESCRIPTIONS
# )
# from .threshold_manager import ThresholdManager

__all__ = [
    # 回测引擎
    'run_backtest',
    
    # 仓位控制 (临时注释)
    # 'create_smart_controller',
    # 'get_strategy_info',
    # 'STRATEGY_DESCRIPTIONS',
    
    # 阈值管理 (临时注释)
    # 'ThresholdManager'
]
