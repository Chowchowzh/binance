# -*- coding: utf-8 -*-
"""
数据库连接管理器
提供统一的数据库连接管理接口
"""

import os
import sys
from typing import Dict, Any, Optional

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .mongodb_client import MongoDBClient


class DatabaseConnection:
    """数据库连接管理器 - 统一管理所有数据库连接"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据库连接管理器
        
        Args:
            config: 数据库配置信息
        """
        self.config = config
        self._connections = {}
    
    def get_mongodb_client(self, database_name: Optional[str] = None) -> MongoDBClient:
        """
        获取MongoDB客户端
        
        Args:
            database_name: 数据库名称，如果不指定则使用配置中的默认数据库
            
        Returns:
            MongoDB客户端实例
        """
        db_name = database_name or self.config.get('mongodb_db_name')
        if not db_name:
            raise ValueError("未指定数据库名称")
        
        connection_key = f"mongodb_{db_name}"
        
        # 如果连接已存在且有效，直接返回
        if connection_key in self._connections:
            client = self._connections[connection_key]
            try:
                # 测试连接是否有效
                client.client.admin.command('ping')
                return client
            except Exception:
                # 连接已断开，移除无效连接
                del self._connections[connection_key]
        
        # 创建新连接
        try:
            client = MongoDBClient(
                uri=self.config['mongodb_uri'],
                database_name=db_name
            )
            self._connections[connection_key] = client
            return client
        except Exception as e:
            print(f"创建MongoDB连接失败: {e}")
            raise
    
    def setup_database_for_symbol(self, symbol: str, interval: str = None) -> tuple:
        """
        为指定交易对设置数据库连接和集合
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔，如果不指定则使用配置中的默认值
            
        Returns:
            (客户端, 集合) 元组
        """
        try:
            # 获取MongoDB客户端
            client = self.get_mongodb_client()
            
            # 确定集合名称
            interval = interval or self.config.get('interval', '1m')
            collection_name = self.config['collection_name_template'].format(
                symbol=symbol,
                interval=interval
            )
            
            # 获取集合
            collection = client.get_collection(collection_name)
            
            # 确保索引存在
            client.ensure_index(collection_name, [("open_time", 1)])
            
            print(f"成功设置数据库集合: {collection_name}")
            return client, collection
            
        except Exception as e:
            print(f"设置数据库失败: {e}")
            raise
    
    def get_collection_for_symbol(self, symbol: str, interval: str = None):
        """
        获取指定交易对的集合
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            
        Returns:
            MongoDB集合对象
        """
        client, collection = self.setup_database_for_symbol(symbol, interval)
        return collection
    
    def close_all_connections(self):
        """关闭所有数据库连接"""
        for connection_key, client in self._connections.items():
            try:
                client.close()
                print(f"已关闭连接: {connection_key}")
            except Exception as e:
                print(f"关闭连接时出错: {connection_key}, {e}")
        
        self._connections.clear()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        获取连接统计信息
        
        Returns:
            连接统计信息
        """
        stats = {
            'total_connections': len(self._connections),
            'active_connections': [],
            'connection_details': {}
        }
        
        for connection_key, client in self._connections.items():
            try:
                # 测试连接是否活跃
                client.client.admin.command('ping')
                stats['active_connections'].append(connection_key)
                
                # 获取数据库统计信息
                if hasattr(client, 'get_database_stats'):
                    db_stats = client.get_database_stats()
                    stats['connection_details'][connection_key] = {
                        'database': client.database_name,
                        'collections': len(db_stats.get('collections', 0)),
                        'data_size': db_stats.get('dataSize', 0),
                        'storage_size': db_stats.get('storageSize', 0)
                    }
                    
            except Exception as e:
                stats['connection_details'][connection_key] = {
                    'status': 'inactive',
                    'error': str(e)
                }
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查
        
        Returns:
            健康检查结果
        """
        results = {
            'overall_status': 'healthy',
            'mongodb_status': 'unknown',
            'error_messages': []
        }
        
        try:
            # 测试MongoDB连接
            client = self.get_mongodb_client()
            client.client.admin.command('ping')
            results['mongodb_status'] = 'healthy'
            
        except Exception as e:
            results['mongodb_status'] = 'unhealthy'
            results['overall_status'] = 'unhealthy'
            results['error_messages'].append(f"MongoDB连接失败: {e}")
        
        return results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all_connections()


# 兼容性函数：保持与原有代码的兼容性
def setup_database(config: Dict[str, Any], symbol: str):
    """
    兼容性函数 - 模拟原有的setup_database函数
    
    Args:
        config: 配置信息
        symbol: 交易对符号
        
    Returns:
        (客户端, 集合) 元组
    """
    connection_manager = DatabaseConnection(config)
    return connection_manager.setup_database_for_symbol(symbol) 