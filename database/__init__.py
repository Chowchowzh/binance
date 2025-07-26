# -*- coding: utf-8 -*-
"""
数据库连接模块
提供统一的数据库连接和操作接口
"""

__version__ = "1.0.0"
__author__ = "Binance Trading Strategy Team"

from .mongodb_client import MongoDBClient
from .connection import DatabaseConnection

__all__ = [
    'MongoDBClient',
    'DatabaseConnection'
] 