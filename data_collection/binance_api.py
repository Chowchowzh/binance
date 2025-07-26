# -*- coding: utf-8 -*-
"""
币安API接口模块
提供从币安获取K线数据的核心功能
"""

import requests
import time
import json
from typing import List, Dict, Optional, Any
from datetime import datetime


class BinanceMarketData:
    """币安市场数据API客户端"""
    
    def __init__(self, base_url: str = "https://api.binance.com"):
        """
        初始化币安API客户端
        
        Args:
            base_url: API基础URL
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_klines(self, symbol: str, interval: str, start_time: int, 
                   end_time: int, limit: int = 1000) -> List[List[Any]]:
        """
        获取K线数据
        
        Args:
            symbol: 交易对符号 (例如: 'ETHUSDT')
            interval: 时间间隔 (例如: '1m', '1h', '1d')
            start_time: 开始时间戳 (毫秒)
            end_time: 结束时间戳 (毫秒)
            limit: 数据条数限制 (最大1000)
            
        Returns:
            K线数据列表，每个元素包含:
            [开盘时间, 开盘价, 最高价, 最低价, 收盘价, 成交量, 收盘时间, ...]
            
        Raises:
            requests.RequestException: 网络请求失败
            ValueError: 参数错误
        """
        if limit > 1000:
            raise ValueError("limit不能超过1000")
            
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 检查返回的数据格式
            if not isinstance(data, list):
                raise ValueError(f"API返回数据格式错误: {data}")
                
            return data
            
        except requests.exceptions.Timeout:
            raise requests.RequestException("请求超时")
        except requests.exceptions.ConnectionError:
            raise requests.RequestException("连接错误")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise requests.RequestException("API请求频率限制")
            else:
                raise requests.RequestException(f"HTTP错误: {e}")
        except json.JSONDecodeError:
            raise requests.RequestException("API返回数据不是有效的JSON格式")
    
    def get_klines_batch(self, symbol: str, interval: str, start_time: int,
                        end_time: int, batch_size: int = 1000, 
                        delay: float = 0.5) -> List[List[Any]]:
        """
        批量获取K线数据，自动处理分页
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间戳 (毫秒)
            end_time: 结束时间戳 (毫秒)
            batch_size: 每批数据量 (最大1000)
            delay: 请求间隔时间 (秒)
            
        Returns:
            完整的K线数据列表
            
        Yields:
            每批数据和进度信息
        """
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            try:
                batch_data = self.get_klines(
                    symbol=symbol,
                    interval=interval, 
                    start_time=current_start,
                    end_time=end_time,
                    limit=batch_size
                )
                
                if not batch_data:
                    break
                    
                all_data.extend(batch_data)
                
                # 更新下一批的开始时间
                last_close_time = int(batch_data[-1][6])  # 收盘时间
                current_start = last_close_time + 1
                
                # 避免请求过于频繁
                if delay > 0:
                    time.sleep(delay)
                    
                # 生成进度信息
                progress = min(100, (current_start - start_time) / (end_time - start_time) * 100)
                yield {
                    'data': batch_data,
                    'progress': progress,
                    'current_time': current_start,
                    'total_records': len(all_data)
                }
                
            except requests.RequestException as e:
                print(f"获取数据时出错: {e}, 正在重试...")
                time.sleep(2)  # 错误时等待更长时间
                continue
                
        return all_data
    
    def close(self):
        """关闭session"""
        if self.session:
            self.session.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 