# -*- coding: utf-8 -*-
"""
数据集构建模块
从数据库读取数据并构建机器学习数据集
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database.mongodb_client import MongoDBClient
from .features import FeatureBuilder
from config.settings import DatabaseConfig, DataCollectionConfig


class DatasetBuilder:
    """数据集构建器 - 从数据库读取数据并构建特征"""
    
    def __init__(self, db_config: DatabaseConfig, data_collection_config: DataCollectionConfig):
        """
        初始化数据集构建器
        
        Args:
            db_config: 数据库配置
            data_collection_config: 数据收集配置
        """
        self.db_config = db_config
        self.data_collection_config = data_collection_config
        self.db_client = None
        self.feature_builder = FeatureBuilder()
        
    def build_dataset(self, target_symbol: str, 
                     feature_symbols: Optional[List[str]] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     interval: str = '1m') -> pd.DataFrame:
        """
        构建完整的数据集
        
        Args:
            target_symbol: 目标交易对
            feature_symbols: 特征交易对列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔
            
        Returns:
            包含所有特征的DataFrame
        """
        print(f"开始构建数据集，目标交易对: {target_symbol}")
        
        # 设置默认值
        if feature_symbols is None:
            feature_symbols = self.data_collection_config.feature_symbols
            
        all_symbols = list(set([target_symbol] + feature_symbols))
        print(f"使用交易对: {all_symbols}")
        
        # 从数据库读取数据
        combined_df = self._load_multi_symbol_data(all_symbols, start_date, end_date, interval)
        
        if combined_df.empty:
            print("警告：未能从数据库读取到任何数据")
            return pd.DataFrame()
        
        print(f"原始数据形状: {combined_df.shape}")
        
        # 构建特征
        features_df = self.feature_builder.build_features(
            combined_df, 
            target_symbol, 
            feature_symbols
        )
        
        print(f"特征构建完成，最终数据形状: {features_df.shape}")
        return features_df
    
    def _load_multi_symbol_data(self, symbols: List[str], start_date: Optional[str], 
                               end_date: Optional[str], interval: str) -> pd.DataFrame:
        """
        从数据库加载多个交易对的数据
        
        Args:
            symbols: 交易对列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔
            
        Returns:
            合并的DataFrame
        """
        try:
            # 建立数据库连接
            self.db_client = MongoDBClient(
                uri=self.db_config.mongodb_uri,
                database_name=self.db_config.mongodb_db_name
            )
            
            # 为每个交易对加载数据
            symbol_dataframes = {}
            
            for symbol in symbols:
                print(f"加载 {symbol} 数据...")
                
                # 获取集合名称
                collection_name = f"{symbol.lower()}_{interval}"
                
                # 构建查询条件
                query_filter = self._build_time_filter(start_date, end_date)
                
                # 从数据库读取数据
                symbol_df = self._load_symbol_data(collection_name, query_filter)
                
                if not symbol_df.empty:
                    # 重命名列以包含交易对前缀
                    symbol_df = self._rename_columns_with_symbol_prefix(symbol_df, symbol)
                    symbol_dataframes[symbol] = symbol_df
                    print(f"  {symbol} 数据加载完成: {symbol_df.shape}")
                else:
                    print(f"  警告：{symbol} 未找到数据")
            
            # 合并所有交易对的数据
            if symbol_dataframes:
                combined_df = self._merge_symbol_dataframes(symbol_dataframes)
                print(f"数据合并完成，总形状: {combined_df.shape}")
                return combined_df
            else:
                print("警告：没有任何交易对的数据被加载")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return pd.DataFrame()
            
        finally:
            if self.db_client:
                self.db_client.close()
    
    def _load_symbol_data(self, collection_name: str, query_filter: Dict[str, Any]) -> pd.DataFrame:
        """
        从数据库加载单个交易对的数据
        
        Args:
            collection_name: 集合名称
            query_filter: 查询过滤器
            
        Returns:
            DataFrame
        """
        try:
            # 获取集合
            collection = self.db_client.get_collection(collection_name)
            
            # 查询数据
            cursor = collection.find(query_filter).sort("open_time", 1)
            
            # 转换为DataFrame
            data_list = list(cursor)
            
            if not data_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(data_list)
            
            # 移除MongoDB的_id字段
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # 确保包含所有预期字段
            expected_fields = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
            ]
            
            # 添加缺失字段（填充为0或NaN）
            for field in expected_fields:
                if field not in df.columns:
                    print(f"警告：{collection_name} 缺少字段 {field}，使用默认值")
                    if field in ['open_time', 'close_time']:
                        df[field] = 0
                    elif field in ['number_of_trades']:
                        df[field] = 0
                    else:
                        df[field] = 0.0
            
            # 设置时间索引
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.set_index('open_time')
            
            # 确保数据类型正确
            numeric_fields = ['open', 'high', 'low', 'close', 'volume', 
                            'quote_asset_volume', 'taker_buy_base_asset_volume', 
                            'taker_buy_quote_asset_volume']
            
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
            
            # 确保整数字段的类型
            integer_fields = ['close_time', 'number_of_trades']
            for field in integer_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0).astype(int)
            
            return df
            
        except Exception as e:
            print(f"加载 {collection_name} 数据时出错: {e}")
            return pd.DataFrame()
    
    def _rename_columns_with_symbol_prefix(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        为DataFrame的列名添加交易对前缀
        
        Args:
            df: 原始DataFrame
            symbol: 交易对符号
            
        Returns:
            重命名后的DataFrame
        """
        # 需要重命名的列（不包括时间索引）
        columns_to_rename = [
            'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
        # 创建重命名映射
        rename_mapping = {}
        for col in columns_to_rename:
            if col in df.columns:
                rename_mapping[col] = f"{symbol}_{col}"
        
        # 应用重命名
        df_renamed = df.rename(columns=rename_mapping)
        
        return df_renamed
    
    def _merge_symbol_dataframes(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并多个交易对的DataFrame
        
        Args:
            symbol_dataframes: 交易对DataFrame字典
            
        Returns:
            合并后的DataFrame
        """
        if not symbol_dataframes:
            return pd.DataFrame()
        
        # 获取所有DataFrame的列表
        dfs = list(symbol_dataframes.values())
        
        # 使用外连接合并所有DataFrame
        combined_df = dfs[0]
        
        for df in dfs[1:]:
            combined_df = pd.merge(
                combined_df, 
                df, 
                left_index=True, 
                right_index=True, 
                how='outer'
            )
        
        # 按时间排序
        combined_df = combined_df.sort_index()
        
        # 向前填充缺失值（对于价格数据）
        price_columns = [col for col in combined_df.columns if any(
            price_type in col for price_type in ['open', 'high', 'low', 'close']
        )]
        
        for col in price_columns:
            combined_df[col] = combined_df[col].fillna(method='ffill')
        
        # 对于成交量相关字段，填充为0
        volume_columns = [col for col in combined_df.columns if any(
            vol_type in col for vol_type in ['volume', 'trades', 'taker_buy']
        )]
        
        for col in volume_columns:
            combined_df[col] = combined_df[col].fillna(0)
        
        return combined_df
    
    def _build_time_filter(self, start_date: Optional[str], end_date: Optional[str]) -> Dict[str, Any]:
        """
        构建时间过滤器
        
        Args:
            start_date: 开始日期字符串
            end_date: 结束日期字符串
            
        Returns:
            MongoDB查询过滤器
        """
        filter_dict = {}
        
        if start_date:
            try:
                start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                filter_dict['open_time'] = {'$gte': start_timestamp}
            except ValueError:
                print(f"警告：无效的开始日期格式: {start_date}")
        
        if end_date:
            try:
                end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
                if 'open_time' in filter_dict:
                    filter_dict['open_time']['$lte'] = end_timestamp
                else:
                    filter_dict['open_time'] = {'$lte': end_timestamp}
            except ValueError:
                print(f"警告：无效的结束日期格式: {end_date}")
        
        return filter_dict
    
    def get_data_info(self, symbols: List[str], interval: str = '1m') -> Dict[str, Any]:
        """
        获取数据库中数据的基本信息
        
        Args:
            symbols: 交易对列表
            interval: 时间间隔
            
        Returns:
            数据信息字典
        """
        info = {
            'symbols': {},
            'total_records': 0,
            'date_range': {},
            'missing_symbols': []
        }
        
        try:
            # 建立数据库连接
            self.db_client = MongoDBClient(
                uri=self.db_config.mongodb_uri,
                database_name=self.db_config.mongodb_db_name
            )
            
            for symbol in symbols:
                collection_name = f"{symbol.lower()}_{interval}"
                
                try:
                    collection = self.db_client.get_collection(collection_name)
                    
                    # 获取记录数
                    count = collection.count_documents({})
                    
                    if count > 0:
                        # 获取时间范围
                        first_record = collection.find().sort("open_time", 1).limit(1)
                        last_record = collection.find().sort("open_time", -1).limit(1)
                        
                        first_time = list(first_record)[0]['open_time']
                        last_time = list(last_record)[0]['open_time']
                        
                        info['symbols'][symbol] = {
                            'count': count,
                            'start_time': datetime.fromtimestamp(first_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                            'end_time': datetime.fromtimestamp(last_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        info['total_records'] += count
                    else:
                        info['missing_symbols'].append(symbol)
                        
                except Exception as e:
                    print(f"获取 {symbol} 信息时出错: {e}")
                    info['missing_symbols'].append(symbol)
            
            return info
            
        except Exception as e:
            print(f"获取数据信息时出错: {e}")
            return info
            
        finally:
            if self.db_client:
                self.db_client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db_client:
            self.db_client.close() 