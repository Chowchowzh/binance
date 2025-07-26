# -*- coding: utf-8 -*-
"""
数据采集模块
负责从各种数据源（主要是币安API）获取市场数据
"""

__version__ = "1.0.0"
__author__ = "Binance Trading Strategy Team"

from .binance_api import BinanceMarketData
from .data_fetcher import DataFetcher
from .run_fetcher import FetcherRunner

__all__ = [
    'BinanceMarketData',
    'DataFetcher', 
    'FetcherRunner'
] 