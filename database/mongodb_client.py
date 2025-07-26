# -*- coding: utf-8 -*-
"""
MongoDB客户端模块
提供MongoDB数据库的连接和基本操作功能
"""

import pymongo
import certifi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import Dict, List, Any, Optional, Tuple


class MongoDBClient:
    """MongoDB客户端类 - 封装MongoDB连接和基本操作"""
    
    def __init__(self, uri: str, database_name: str):
        """
        初始化MongoDB客户端
        
        Args:
            uri: MongoDB连接URI
            database_name: 数据库名称
        """
        self.uri = uri
        self.database_name = database_name
        self.client = None
        self.database = None
        self._connect()
    
    def _connect(self):
        """建立MongoDB连接"""
        try:
            ca = certifi.where()  # 获取证书路径
            self.client = MongoClient(
                self.uri, 
                server_api=ServerApi('1'), 
                tlsCAFile=ca
            )
            
            # 验证连接
            self.client.admin.command('ping')
            self.database = self.client[self.database_name]
            
            print(f"成功连接到MongoDB数据库: {self.database_name}")
            
        except ConnectionFailure as e:
            print(f"MongoDB连接失败: {e}")
            raise
        except Exception as e:
            print(f"MongoDB连接出现未知错误: {e}")
            raise
    
    def get_collection(self, collection_name: str):
        """
        获取指定的集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            MongoDB集合对象
        """
        if self.database is None:
            raise RuntimeError("数据库连接未建立")
        
        return self.database[collection_name]
    
    def ensure_index(self, collection_name: str, index_fields: List[Tuple[str, int]], 
                     unique: bool = False) -> str:
        """
        确保集合上存在指定的索引
        
        Args:
            collection_name: 集合名称
            index_fields: 索引字段列表，格式为 [(字段名, 方向), ...]
            unique: 是否为唯一索引
            
        Returns:
            索引名称
        """
        try:
            collection = self.get_collection(collection_name)
            index_name = collection.create_index(index_fields, unique=unique)
            print(f"集合 {collection_name} 的索引已确保存在: {index_name}")
            return index_name
        except Exception as e:
            print(f"创建索引时出错: {e}")
            raise
    
    def get_last_record(self, collection_name: str, sort_field: str = "open_time") -> Optional[Dict[str, Any]]:
        """
        获取集合中的最后一条记录
        
        Args:
            collection_name: 集合名称
            sort_field: 排序字段
            
        Returns:
            最后一条记录，如果没有记录则返回None
        """
        collection = self.get_collection(collection_name)
        cursor = collection.find().sort(sort_field, -1).limit(1)
        records = list(cursor)
        return records[0] if records else None
    
    def count_documents(self, collection_name: str, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """
        计算集合中的文档数量
        
        Args:
            collection_name: 集合名称
            filter_dict: 过滤条件
            
        Returns:
            文档数量
        """
        collection = self.get_collection(collection_name)
        filter_dict = filter_dict or {}
        return collection.count_documents(filter_dict)
    
    def insert_many_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[Any]:
        """
        批量插入文档
        
        Args:
            collection_name: 集合名称
            documents: 要插入的文档列表
            
        Returns:
            插入的文档ID列表
        """
        if not documents:
            return []
        
        collection = self.get_collection(collection_name)
        result = collection.insert_many(documents)
        return result.inserted_ids
    
    def find_documents(self, collection_name: str, filter_dict: Optional[Dict[str, Any]] = None,
                      sort_field: Optional[str] = None, sort_direction: int = 1,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        查询文档
        
        Args:
            collection_name: 集合名称
            filter_dict: 过滤条件
            sort_field: 排序字段
            sort_direction: 排序方向 (1: 升序, -1: 降序)
            limit: 限制返回数量
            
        Returns:
            文档列表
        """
        collection = self.get_collection(collection_name)
        filter_dict = filter_dict or {}
        
        cursor = collection.find(filter_dict)
        
        if sort_field:
            cursor = cursor.sort(sort_field, sort_direction)
        
        if limit:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    def delete_many_documents(self, collection_name: str, filter_dict: Dict[str, Any]) -> int:
        """
        批量删除文档
        
        Args:
            collection_name: 集合名称
            filter_dict: 删除条件
            
        Returns:
            删除的文档数量
        """
        collection = self.get_collection(collection_name)
        result = collection.delete_many(filter_dict)
        return result.deleted_count
    
    def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行聚合查询
        
        Args:
            collection_name: 集合名称
            pipeline: 聚合管道
            
        Returns:
            聚合结果
        """
        collection = self.get_collection(collection_name)
        return list(collection.aggregate(pipeline))
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            数据库统计信息
        """
        if self.database is None:
            raise RuntimeError("数据库连接未建立")
        
        return self.database.command("dbstats")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合统计信息
        """
        if self.database is None:
            raise RuntimeError("数据库连接未建立")
        
        return self.database.command("collstats", collection_name)
    
    def list_collections(self) -> List[str]:
        """
        列出数据库中的所有集合
        
        Returns:
            集合名称列表
        """
        if self.database is None:
            raise RuntimeError("数据库连接未建立")
        
        return self.database.list_collection_names()
    
    def close(self):
        """关闭数据库连接"""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            print(f"已关闭MongoDB连接: {self.database_name}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        """析构函数 - 确保连接被关闭"""
        self.close() 