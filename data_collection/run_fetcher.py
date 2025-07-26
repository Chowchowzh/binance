# -*- coding: utf-8 -*-
"""
数据获取运行器
提供命令行接口和自动重试机制
"""

import subprocess
import time
import sys
import argparse
import os
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .data_fetcher import DataFetcher
from config.settings import load_project_config, get_legacy_config_dict


class FetcherRunner:
    """数据获取运行器 - 提供命令行接口和重试机制"""
    
    def __init__(self, config_path: str = 'config/config.json'):
        """
        初始化运行器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = load_project_config(config_path)
        self.retry_delay = 300  # 5分钟重试间隔
        self.max_retries = 5
    
    def run_single_fetch(self, symbols: List[str], full_refresh: bool = False) -> bool:
        """
        执行单次数据获取
        
        Args:
            symbols: 要获取的交易对列表
            full_refresh: 是否全量刷新
            
        Returns:
            是否成功
        """
        try:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始数据获取任务...")
            print(f"目标交易对: {symbols}")
            print(f"全量刷新: {full_refresh}")
            
            # 将ProjectConfig对象转换为字典格式供DataFetcher使用
            legacy_config = get_legacy_config_dict(self.config)
            with DataFetcher(legacy_config) as fetcher:
                results = fetcher.fetch_multiple_symbols(symbols, full_refresh)
            
            # 检查结果
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            print(f"数据获取完成: {success_count}/{total_count} 个交易对成功")
            
            # 显示详细结果
            for symbol, success in results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {symbol}")
            
            return success_count == total_count
            
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 数据获取失败: {e}")
            return False
    
    def run_with_retry(self, symbols: List[str], full_refresh: bool = False) -> bool:
        """
        带重试机制的数据获取
        
        Args:
            symbols: 要获取的交易对列表
            full_refresh: 是否全量刷新
            
        Returns:
            最终是否成功
        """
        retry_count = 0
        
        while retry_count < self.max_retries:
            if self.run_single_fetch(symbols, full_refresh):
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 数据获取成功!")
                return True
            
            retry_count += 1
            if retry_count < self.max_retries:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 数据获取失败，{self.retry_delay}秒后进行第{retry_count}次重试...")
                time.sleep(self.retry_delay)
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 达到最大重试次数，数据获取任务失败")
        
        return False
    
    def run_continuous(self, symbols: List[str], interval_hours: int = 1) -> None:
        """
        持续运行数据获取（用于实时更新）
        
        Args:
            symbols: 要获取的交易对列表
            interval_hours: 获取间隔（小时）
        """
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始持续数据获取模式")
        print(f"获取间隔: {interval_hours} 小时")
        
        try:
            while True:
                # 执行增量更新
                self.run_with_retry(symbols, full_refresh=False)
                
                # 等待下一次执行
                sleep_seconds = interval_hours * 3600
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 等待{interval_hours}小时后进行下次更新...")
                time.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 收到中断信号，停止持续获取")
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 持续获取过程中出错: {e}")


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="币安数据获取工具")
    parser.add_argument("--full-refresh", action="store_true", 
                       help="对所有交易对执行全量刷新，删除现有数据")
    parser.add_argument("--symbols", nargs='+', 
                       help="指定要获取的交易对（覆盖配置文件）")
    parser.add_argument("--continuous", action="store_true",
                       help="持续运行模式（实时更新）")
    parser.add_argument("--interval", type=int, default=1,
                       help="持续模式下的获取间隔（小时，默认1小时）")
    parser.add_argument("--config", default="config/config.json",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        # 初始化运行器
        runner = FetcherRunner(args.config)
        
        # 确定要获取的交易对
        if args.symbols:
            symbols = args.symbols
        else:
            symbols = runner.config.data_collection.feature_symbols
            
        if not symbols:
            print("错误：未指定要获取的交易对")
            return 1
        
        # 选择运行模式
        if args.continuous:
            runner.run_continuous(symbols, args.interval)
        else:
            success = runner.run_with_retry(symbols, args.full_refresh)
            return 0 if success else 1
            
    except Exception as e:
        print(f"程序运行出错: {e}")
        return 1


def run_legacy_script():
    """运行原有的数据获取脚本（兼容性接口）"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 正在启动数据获取脚本...")
    
    retry_count = 0
    max_retries = 5
    retry_delay = 300
    
    while retry_count < max_retries:
        try:
            # 运行主函数
            result = main()
            if result == 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 数据获取脚本正常退出")
                return True
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 数据获取脚本退出，返回码: {result}")
                
        except FileNotFoundError:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误：未找到数据获取脚本")
            return False
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 运行数据获取脚本时出错: {e}")
        
        retry_count += 1
        if retry_count < max_retries:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {retry_delay}秒后进行第{retry_count}次重试...")
            time.sleep(retry_delay)
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 达到最大重试次数，数据获取失败")
    return False


if __name__ == "__main__":
    sys.exit(main()) 