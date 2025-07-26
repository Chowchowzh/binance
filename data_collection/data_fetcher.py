# -*- coding: utf-8 -*-
"""
数据获取器模块
负责协调API调用和数据库存储
"""

import sys
import os
import time
import datetime
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .binance_api import BinanceMarketData
from database.mongodb_client import MongoDBClient
from config.settings import DatabaseConfig, TradingConfig


class DataFetcher:
    """数据获取器 - 协调API和数据库操作"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据获取器
        
        Args:
            config: 配置字典，包含数据库连接、API设置等
        """
        self.config = config
        self.api_client = BinanceMarketData()
        self.db_client = None
        
    def setup_database(self, symbol: str) -> Any:
        """
        设置数据库连接和集合
        
        Args:
            symbol: 交易对符号
            
        Returns:
            MongoDB集合对象
        """
        try:
            self.db_client = MongoDBClient(
                uri=self.config['mongodb_uri'],
                database_name=self.config['mongodb_db_name']
            )
            
            collection_name = self.config['collection_name_template'].format(
                symbol=symbol,
                interval=self.config['interval']
            )
            
            collection = self.db_client.get_collection(collection_name)
            
            # 确保索引存在
            self.db_client.ensure_index(collection_name, [("open_time", 1)])
            
            print(f"数据库连接成功: {collection_name}")
            return collection
            
        except Exception as e:
            print(f"数据库连接失败: {e}")
            raise
    
    def fetch_symbol_data(self, symbol: str, start_date: Optional[str] = None, 
                         full_refresh: bool = False) -> bool:
        """
        获取单个交易对的数据
        
        Args:
            symbol: 交易对符号
            start_date: 开始日期 (格式: 'YYYY-MM-DD HH:MM:SS.ms')
            full_refresh: 是否全量刷新
            
        Returns:
            是否成功
        """
        try:
            print(f"\n开始获取 {symbol} 的数据...")
            
            # 设置数据库
            collection = self.setup_database(symbol)
            
            # 确定时间范围
            start_time, end_time = self._determine_time_range(
                collection, start_date, full_refresh
            )
            
            if start_time >= end_time:
                print(f"{symbol} 数据已是最新，无需更新")
                return True
            
            # 批量获取数据
            success = self._fetch_and_store_data(
                symbol, collection, start_time, end_time
            )
            
            if success:
                print(f"{symbol} 数据获取完成")
            else:
                print(f"{symbol} 数据获取失败")
                
            return success
            
        except Exception as e:
            print(f"获取 {symbol} 数据时出错: {e}")
            return False
        finally:
            if self.db_client:
                self.db_client.close()
    
    def fetch_multiple_symbols(self, symbols: List[str], 
                              full_refresh: bool = False) -> Dict[str, bool]:
        """
        获取多个交易对的数据
        
        Args:
            symbols: 交易对列表
            full_refresh: 是否全量刷新
            
        Returns:
            每个交易对的获取结果
        """
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_symbol_data(
                    symbol=symbol,
                    full_refresh=full_refresh
                )
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
                results[symbol] = False
        
        return results
    
    def _determine_time_range(self, collection: Any, start_date: Optional[str], 
                             full_refresh: bool) -> tuple:
        """确定数据获取的时间范围"""
        
        if full_refresh and start_date:
            # 全量刷新：删除现有数据，从指定日期开始
            collection.delete_many({})
            start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S.%f")
            start_time = int(start_datetime.timestamp() * 1000)
        else:
            # 增量更新：从最后一条记录开始
            last_record = collection.find().sort("open_time", -1).limit(1)
            last_record_list = list(last_record)
            
            if last_record_list:
                start_time = last_record_list[0]["open_time"] + 1
            else:
                # 如果没有数据，使用默认开始时间
                default_start = start_date or self.config.get('start_date')
                if default_start:
                    start_datetime = datetime.datetime.strptime(default_start, "%Y-%m-%d %H:%M:%S.%f")
                    start_time = int(start_datetime.timestamp() * 1000)
                else:
                    # 默认从30天前开始
                    start_time = int((datetime.datetime.now() - datetime.timedelta(days=30)).timestamp() * 1000)
        
        # 结束时间为当前时间
        end_time = int(datetime.datetime.now().timestamp() * 1000)
        
        return start_time, end_time
    
    def _fetch_and_store_data(self, symbol: str, collection: Any, 
                             start_time: int, end_time: int) -> bool:
        """获取并存储数据"""
        
        try:
            batch_count = 0
            total_records = 0
            
            print(f"开始获取 {symbol} 从 {datetime.datetime.fromtimestamp(start_time/1000)} 到 {datetime.datetime.fromtimestamp(end_time/1000)} 的数据")
            
            # 使用生成器批量获取数据
            for batch_info in self.api_client.get_klines_batch(
                symbol=symbol,
                interval=self.config['interval'],
                start_time=start_time,
                end_time=end_time,
                delay=self.config.get('api_request_delay_seconds', 0.5)
            ):
                batch_data = batch_info['data']
                
                if not batch_data:
                    continue
                
                # 转换数据格式
                documents = self._convert_klines_to_documents(batch_data)
                
                # 存储到数据库
                if documents:
                    collection.insert_many(documents)
                    total_records += len(documents)
                
                batch_count += 1
                
                # 定期报告进度
                if batch_count % self.config.get('progress_report_interval_batches', 10) == 0:
                    print(f"已处理 {batch_count} 批数据，总计 {total_records} 条记录 ({batch_info['progress']:.1f}%)")
            
            print(f"{symbol} 数据获取完成，总计 {total_records} 条记录")
            return True
            
        except Exception as e:
            print(f"获取和存储数据时出错: {e}")
            return False
    
    def _convert_klines_to_documents(self, klines_data: List[List[Any]]) -> List[Dict[str, Any]]:
        """将K线数据转换为MongoDB文档格式"""
        
        documents = []
        
        for kline in klines_data:
            try:
                doc = {
                    "open_time": int(kline[0]),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "close_time": int(kline[6]),
                    "quote_asset_volume": float(kline[7]),
                    "number_of_trades": int(kline[8]),
                    "taker_buy_base_asset_volume": float(kline[9]),
                    "taker_buy_quote_asset_volume": float(kline[10])
                }
                documents.append(doc)
            except (ValueError, IndexError) as e:
                print(f"数据格式错误，跳过该条记录: {e}")
                continue
        
        return documents
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db_client:
            self.db_client.close()
        self.api_client.close() 