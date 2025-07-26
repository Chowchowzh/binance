# -*- coding: utf-8 -*-
"""
通用工具函数模块
提供项目中常用的辅助函数
"""

import os
import sys
import time
import hashlib
import pickle
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path


class CommonUtils:
    """通用工具函数类"""
    
    @staticmethod
    def ensure_dir(path: str) -> str:
        """
        确保目录存在，如果不存在则创建
        
        Args:
            path: 目录路径
            
        Returns:
            目录路径
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_project_root() -> str:
        """
        获取项目根目录
        
        Returns:
            项目根目录路径
        """
        current_file = os.path.abspath(__file__)
        # 向上查找包含pyproject.toml或requirements.txt的目录
        current_dir = os.path.dirname(current_file)
        
        while current_dir != os.path.dirname(current_dir):  # 不是根目录
            if (os.path.exists(os.path.join(current_dir, 'pyproject.toml')) or 
                os.path.exists(os.path.join(current_dir, 'requirements.txt'))):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        
        # 如果找不到，返回当前文件所在目录的上级目录
        return os.path.dirname(os.path.dirname(current_file))
    
    @staticmethod
    def add_project_to_path():
        """将项目根目录添加到Python路径"""
        project_root = CommonUtils.get_project_root()
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        return project_root
    
    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
        """
        计算文件哈希值
        
        Args:
            file_path: 文件路径
            algorithm: 哈希算法 ('md5', 'sha256')
            
        Returns:
            文件哈希值
        """
        hash_func = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            raise ValueError(f"计算文件哈希失败: {e}")
    
    @staticmethod
    def save_pickle(obj: Any, file_path: str) -> bool:
        """
        保存对象为pickle文件
        
        Args:
            obj: 要保存的对象
            file_path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            CommonUtils.ensure_dir(os.path.dirname(file_path))
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
            return True
        except Exception as e:
            print(f"保存pickle文件失败: {e}")
            return False
    
    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """
        从pickle文件加载对象
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的对象
        """
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError(f"加载pickle文件失败: {e}")
    
    @staticmethod
    def save_json(obj: Any, file_path: str, indent: int = 4) -> bool:
        """
        保存对象为JSON文件
        
        Args:
            obj: 要保存的对象
            file_path: 保存路径
            indent: 缩进空格数
            
        Returns:
            是否保存成功
        """
        try:
            CommonUtils.ensure_dir(os.path.dirname(file_path))
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"保存JSON文件失败: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Any:
        """
        从JSON文件加载对象
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的对象
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"加载JSON文件失败: {e}")
    
    @staticmethod
    def format_number(num: float, precision: int = 2, use_comma: bool = True) -> str:
        """
        格式化数字显示
        
        Args:
            num: 数字
            precision: 小数位数
            use_comma: 是否使用千分位分隔符
            
        Returns:
            格式化后的字符串
        """
        if np.isnan(num) or np.isinf(num):
            return "N/A"
        
        if use_comma:
            return f"{num:,.{precision}f}"
        else:
            return f"{num:.{precision}f}"
    
    @staticmethod
    def format_percentage(value: float, precision: int = 2) -> str:
        """
        格式化百分比显示
        
        Args:
            value: 数值 (0.1 = 10%)
            precision: 小数位数
            
        Returns:
            百分比字符串
        """
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"{value * 100:.{precision}f}%"
    
    @staticmethod
    def format_time_duration(seconds: float) -> str:
        """
        格式化时间duration
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
        """
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{int(minutes)}分{remaining_seconds:.1f}秒"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{int(hours)}小时{int(remaining_minutes)}分"
    
    @staticmethod
    def timestamp_to_datetime(timestamp: Union[int, float], unit: str = 'ms') -> datetime:
        """
        时间戳转换为datetime对象
        
        Args:
            timestamp: 时间戳
            unit: 时间戳单位 ('s', 'ms', 'us', 'ns')
            
        Returns:
            datetime对象
        """
        if unit == 'ms':
            timestamp = timestamp / 1000
        elif unit == 'us':
            timestamp = timestamp / 1000000
        elif unit == 'ns':
            timestamp = timestamp / 1000000000
        
        return datetime.fromtimestamp(timestamp)
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime, unit: str = 'ms') -> int:
        """
        datetime对象转换为时间戳
        
        Args:
            dt: datetime对象
            unit: 输出单位 ('s', 'ms', 'us', 'ns')
            
        Returns:
            时间戳
        """
        timestamp = dt.timestamp()
        
        if unit == 'ms':
            return int(timestamp * 1000)
        elif unit == 'us':
            return int(timestamp * 1000000)
        elif unit == 'ns':
            return int(timestamp * 1000000000)
        else:
            return int(timestamp)
    
    @staticmethod
    def get_memory_usage_mb(obj: Any) -> float:
        """
        获取对象的内存使用量（MB）
        
        Args:
            obj: 要检查的对象
            
        Returns:
            内存使用量（MB）
        """
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum() / 1024 / 1024
        elif isinstance(obj, (pd.Series, np.ndarray)):
            return obj.nbytes / 1024 / 1024
        else:
            # 对于其他对象，使用sys.getsizeof的估算
            import sys
            return sys.getsizeof(obj) / 1024 / 1024
    
    @staticmethod
    def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0, 
                        backoff_factor: float = 2.0, exceptions: Tuple = (Exception,)):
        """
        重试装饰器/函数
        
        Args:
            func: 要重试的函数
            max_retries: 最大重试次数
            delay: 初始延迟时间（秒）
            backoff_factor: 延迟递增因子
            exceptions: 需要重试的异常类型
            
        Returns:
            装饰后的函数或函数结果
        """
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise e
                    
                    print(f"第{attempt + 1}次尝试失败: {e}, {current_delay}秒后重试...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
        return wrapper
    
    @staticmethod
    def chunked_processing(data: List[Any], chunk_size: int, 
                          process_func, *args, **kwargs) -> List[Any]:
        """
        分块处理数据
        
        Args:
            data: 要处理的数据列表
            chunk_size: 块大小
            process_func: 处理函数
            *args, **kwargs: 传递给处理函数的参数
            
        Returns:
            处理结果列表
        """
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_result = process_func(chunk, *args, **kwargs)
            results.extend(chunk_result if isinstance(chunk_result, list) else [chunk_result])
        
        return results
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, 
                   default: float = 0.0) -> float:
        """
        安全除法，避免除零错误
        
        Args:
            numerator: 分子
            denominator: 分母
            default: 除零时的默认值
            
        Returns:
            除法结果
        """
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        
        result = numerator / denominator
        
        if np.isnan(result) or np.isinf(result):
            return default
        
        return result
    
    @staticmethod
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化DataFrame列名
        
        Args:
            df: 输入DataFrame
            
        Returns:
            列名标准化后的DataFrame
        """
        df_copy = df.copy()
        
        # 转换为小写，替换空格和特殊字符
        df_copy.columns = (df_copy.columns
                           .str.lower()
                           .str.replace(' ', '_')
                           .str.replace('-', '_')
                           .str.replace('.', '_')
                           .str.replace('(', '')
                           .str.replace(')', ''))
        
        return df_copy
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'disk_usage_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024
        }


# 便捷函数
def ensure_dir(path: str) -> str:
    """确保目录存在"""
    return CommonUtils.ensure_dir(path)

def format_number(num: float, precision: int = 2) -> str:
    """格式化数字"""
    return CommonUtils.format_number(num, precision)

def format_percentage(value: float, precision: int = 2) -> str:
    """格式化百分比"""
    return CommonUtils.format_percentage(value, precision)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法"""
    return CommonUtils.safe_divide(numerator, denominator, default) 