# -*- coding: utf-8 -*-
"""
全局配置模块
统一管理项目的所有配置信息
"""

__version__ = "1.0.0"
__author__ = "Binance Trading Strategy Team"

from .settings import (
    DatabaseConfig, 
    TradingConfig, 
    ModelConfig,
    APIConfig,
    load_config,
    save_config,
    get_default_config
)

__all__ = [
    'DatabaseConfig',
    'TradingConfig',
    'ModelConfig', 
    'APIConfig',
    'load_config',
    'save_config',
    'get_default_config'
] 