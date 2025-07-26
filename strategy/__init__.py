# -*- coding: utf-8 -*-
"""
策略模块
包含模型训练、回测、分析等完整的量化交易策略体系
"""

# 导入子模块
from . import training
from . import backtesting
from . import analysis

# 导入主要功能 (临时注释)
# from .market_making import main as run_market_making

__all__ = [
    # 子模块
    'training',
    'backtesting', 
    'analysis',
    
    # 主要功能 (临时注释)
    # 'run_market_making'
]

__version__ = '2.0.0'
__author__ = 'Trading Strategy Team'
__description__ = 'ETH量化交易策略 - 基于Transformer模型的市场做市策略'
