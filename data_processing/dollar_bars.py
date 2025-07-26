# -*- coding: utf-8 -*-
"""
成交额K线（Dollar Bars）生成器

按照累计成交额达到预设阈值时生成新的K线，而不是按照固定时间间隔。
这种方法能够更好地应对资产价格的剧烈波动，生成的时间序列在统计特性上更接近理想的IID高斯分布。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

from utils.logger import ResultLogger


class DollarBarsGenerator:
    """成交额K线生成器"""
    
    def __init__(self, threshold_usd: float = 50_000_000, auto_threshold: bool = True):
        """
        初始化成交额K线生成器
        
        Args:
            threshold_usd: 成交额阈值（美元），默认5000万美元
            auto_threshold: 是否自动计算动态阈值
        """
        self.threshold_usd = threshold_usd
        self.auto_threshold = auto_threshold
        # 创建一个简单的日志记录器
        self.logger = logging.getLogger(__name__)
        
    def calculate_dynamic_threshold(self, df: pd.DataFrame, 
                                  symbol: str = 'ETHUSDT',
                                  window_days: int = 30,
                                  target_bars_per_day: int = 50) -> float:
        """
        计算动态成交额阈值
        
        Args:
            df: 原始数据DataFrame
            symbol: 交易对符号
            window_days: 滚动窗口天数
            target_bars_per_day: 目标每日K线数量
            
        Returns:
            动态阈值（美元）
        """
        self.logger.info(f"计算 {symbol} 的动态成交额阈值")
        
        # 计算每分钟的成交额（价格 * 成交量）
        quote_volume_col = f'{symbol}_quote_asset_volume'
        if quote_volume_col not in df.columns:
            raise ValueError(f"找不到列 {quote_volume_col}")
        
        # 计算滚动窗口内的日均成交额
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['open_time'], unit='ms')
        df_copy = df_copy.set_index('timestamp')
        
        # 按日汇总成交额
        daily_volume = df_copy[quote_volume_col].resample('D').sum()
        
        # 计算滚动平均日成交额
        rolling_avg_daily_volume = daily_volume.rolling(window=window_days, min_periods=1).mean()
        
        # 计算动态阈值：日均成交额 / 目标每日K线数量
        latest_avg_daily_volume = rolling_avg_daily_volume.iloc[-1]
        dynamic_threshold = latest_avg_daily_volume / target_bars_per_day
        
        self.logger.info(f"过去{window_days}天日均成交额: ${latest_avg_daily_volume:,.2f}")
        self.logger.info(f"目标每日K线数: {target_bars_per_day}")
        self.logger.info(f"计算出的动态阈值: ${dynamic_threshold:,.2f}")
        
        return dynamic_threshold
    
    def generate_dollar_bars(self, df: pd.DataFrame, 
                           symbol: str = 'ETHUSDT',
                           threshold: Optional[float] = None) -> pd.DataFrame:
        """
        生成成交额K线
        
        Args:
            df: 原始分钟级数据
            symbol: 交易对符号
            threshold: 成交额阈值，如果为None则使用实例阈值或动态计算
            
        Returns:
            成交额K线DataFrame
        """
        self.logger.info(f"开始生成 {symbol} 的成交额K线")
        
        # 确定阈值
        if threshold is None:
            if self.auto_threshold:
                threshold = self.calculate_dynamic_threshold(df, symbol)
            else:
                threshold = self.threshold_usd
        
        # 准备列名
        cols = {
            'open': f'{symbol}_open',
            'high': f'{symbol}_high', 
            'low': f'{symbol}_low',
            'close': f'{symbol}_close',
            'volume': f'{symbol}_volume',
            'quote_volume': f'{symbol}_quote_asset_volume',
            'trades': f'{symbol}_number_of_trades',
            'taker_buy_volume': f'{symbol}_taker_buy_base_asset_volume',
            'taker_buy_quote_volume': f'{symbol}_taker_buy_quote_asset_volume'
        }
        
        # 检查必要的列
        for col_name, col_key in cols.items():
            if col_key not in df.columns:
                raise ValueError(f"找不到必要的列: {col_key}")
        
        dollar_bars = []
        cumulative_volume = 0.0
        current_bar = None
        
        for idx, row in df.iterrows():
            quote_volume = row[cols['quote_volume']]
            cumulative_volume += quote_volume
            
            if current_bar is None:
                # 初始化新的K线
                current_bar = {
                    'start_time': row['open_time'],
                    'end_time': row['open_time'],
                    'open': row[cols['open']],
                    'high': row[cols['high']],
                    'low': row[cols['low']],
                    'close': row[cols['close']],
                    'volume': row[cols['volume']],
                    'quote_volume': quote_volume,
                    'trades': row[cols['trades']],
                    'taker_buy_volume': row[cols['taker_buy_volume']],
                    'taker_buy_quote_volume': row[cols['taker_buy_quote_volume']],
                    'bar_count': 1
                }
            else:
                # 更新当前K线
                current_bar['end_time'] = row['open_time']
                current_bar['high'] = max(current_bar['high'], row[cols['high']])
                current_bar['low'] = min(current_bar['low'], row[cols['low']])
                current_bar['close'] = row[cols['close']]
                current_bar['volume'] += row[cols['volume']]
                current_bar['quote_volume'] += quote_volume
                current_bar['trades'] += row[cols['trades']]
                current_bar['taker_buy_volume'] += row[cols['taker_buy_volume']]
                current_bar['taker_buy_quote_volume'] += row[cols['taker_buy_quote_volume']]
                current_bar['bar_count'] += 1
            
            # 检查是否达到阈值
            if cumulative_volume >= threshold:
                # 保存当前K线
                dollar_bars.append(current_bar.copy())
                
                # 重置累计成交额和当前K线
                cumulative_volume = 0.0
                current_bar = None
        
        # 如果还有未完成的K线，也添加进去
        if current_bar is not None:
            dollar_bars.append(current_bar)
        
        # 转换为DataFrame
        result_df = pd.DataFrame(dollar_bars)
        
        # 添加时间戳列
        result_df['start_timestamp'] = pd.to_datetime(result_df['start_time'], unit='ms')
        result_df['end_timestamp'] = pd.to_datetime(result_df['end_time'], unit='ms')
        
        # 计算K线持续时间（分钟）
        result_df['duration_minutes'] = (result_df['end_time'] - result_df['start_time']) / (1000 * 60)
        
        # 重新排列列的顺序
        column_order = [
            'start_time', 'end_time', 'start_timestamp', 'end_timestamp',
            'open', 'high', 'low', 'close', 
            'volume', 'quote_volume', 'trades',
            'taker_buy_volume', 'taker_buy_quote_volume',
            'duration_minutes', 'bar_count'
        ]
        result_df = result_df[column_order]
        
        self.logger.info(f"生成了 {len(result_df)} 根成交额K线")
        self.logger.info(f"使用阈值: ${threshold:,.2f}")
        self.logger.info(f"平均每根K线持续时间: {result_df['duration_minutes'].mean():.2f} 分钟")
        self.logger.info(f"平均每根K线包含: {result_df['bar_count'].mean():.1f} 个原始分钟K线")
        
        return result_df
    
    def generate_multi_symbol_dollar_bars(self, df: pd.DataFrame, 
                                        symbols: List[str] = ['ETHUSDT', 'BTCUSDT'],
                                        thresholds: Optional[Dict[str, float]] = None) -> Dict[str, pd.DataFrame]:
        """
        为多个交易对生成成交额K线
        
        Args:
            df: 原始数据DataFrame
            symbols: 交易对列表
            thresholds: 各交易对的阈值字典
            
        Returns:
            各交易对成交额K线的字典
        """
        results = {}
        
        for symbol in symbols:
            threshold = None
            if thresholds and symbol in thresholds:
                threshold = thresholds[symbol]
            
            try:
                results[symbol] = self.generate_dollar_bars(df, symbol, threshold)
            except Exception as e:
                self.logger.error(f"生成 {symbol} 成交额K线失败: {e}")
                continue
        
        return results
    
    def save_dollar_bars(self, dollar_bars: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                        output_dir: str = 'processed_data',
                        filename_prefix: str = 'dollar_bars') -> Dict[str, str]:
        """
        保存成交额K线数据
        
        Args:
            dollar_bars: 成交额K线数据或多交易对数据字典
            output_dir: 输出目录
            filename_prefix: 文件名前缀
            
        Returns:
            保存的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        if isinstance(dollar_bars, pd.DataFrame):
            # 单个DataFrame
            filename = f"{filename_prefix}.parquet"
            filepath = output_path / filename
            dollar_bars.to_parquet(filepath, index=False)
            saved_files['default'] = str(filepath)
            self.logger.info(f"成交额K线已保存: {filepath}")
            
        elif isinstance(dollar_bars, dict):
            # 多交易对字典
            for symbol, df in dollar_bars.items():
                filename = f"{filename_prefix}_{symbol}.parquet"
                filepath = output_path / filename
                df.to_parquet(filepath, index=False)
                saved_files[symbol] = str(filepath)
                self.logger.info(f"{symbol} 成交额K线已保存: {filepath}")
        
        return saved_files
    
    def analyze_dollar_bars_statistics(self, dollar_bars: pd.DataFrame, 
                                     symbol: str = 'ETHUSDT') -> Dict[str, float]:
        """
        分析成交额K线的统计特性
        
        Args:
            dollar_bars: 成交额K线DataFrame
            symbol: 交易对名称
            
        Returns:
            统计信息字典
        """
        self.logger.info(f"分析 {symbol} 成交额K线的统计特性")
        
        # 计算收益率
        dollar_bars['returns'] = dollar_bars['close'].pct_change().dropna()
        
        stats = {
            'total_bars': len(dollar_bars),
            'avg_duration_minutes': dollar_bars['duration_minutes'].mean(),
            'std_duration_minutes': dollar_bars['duration_minutes'].std(),
            'min_duration_minutes': dollar_bars['duration_minutes'].min(),
            'max_duration_minutes': dollar_bars['duration_minutes'].max(),
            'avg_bars_per_dollar_bar': dollar_bars['bar_count'].mean(),
            'avg_quote_volume': dollar_bars['quote_volume'].mean(),
            'returns_mean': dollar_bars['returns'].mean(),
            'returns_std': dollar_bars['returns'].std(),
            'returns_skewness': dollar_bars['returns'].skew(),
            'returns_kurtosis': dollar_bars['returns'].kurtosis()
        }
        
        # 打印统计信息
        self.logger.info("成交额K线统计信息:")
        self.logger.info(f"  总K线数: {stats['total_bars']}")
        self.logger.info(f"  平均持续时间: {stats['avg_duration_minutes']:.2f} ± {stats['std_duration_minutes']:.2f} 分钟")
        self.logger.info(f"  持续时间范围: {stats['min_duration_minutes']:.1f} - {stats['max_duration_minutes']:.1f} 分钟")
        self.logger.info(f"  平均包含原始K线数: {stats['avg_bars_per_dollar_bar']:.1f}")
        self.logger.info(f"  平均成交额: ${stats['avg_quote_volume']:,.2f}")
        self.logger.info(f"  收益率统计:")
        self.logger.info(f"    均值: {stats['returns_mean']:.6f}")
        self.logger.info(f"    标准差: {stats['returns_std']:.6f}")
        self.logger.info(f"    偏度: {stats['returns_skewness']:.3f}")
        self.logger.info(f"    峰度: {stats['returns_kurtosis']:.3f}")
        
        return stats


def main():
    """主函数：示例用法"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载原始数据
    print("加载原始数据...")
    df = pd.read_parquet('processed_data/raw_data.parquet')
    print(f"原始数据形状: {df.shape}")
    
    # 创建成交额K线生成器
    generator = DollarBarsGenerator(auto_threshold=True)
    
    # 为ETH生成成交额K线
    print("\n生成ETHUSDT成交额K线...")
    eth_dollar_bars = generator.generate_dollar_bars(df, 'ETHUSDT')
    
    # 分析统计特性
    print("\n分析统计特性...")
    stats = generator.analyze_dollar_bars_statistics(eth_dollar_bars, 'ETHUSDT')
    
    # 保存结果
    print("\n保存成交额K线...")
    saved_files = generator.save_dollar_bars(eth_dollar_bars, filename_prefix='dollar_bars_ETHUSDT')
    print(f"已保存到: {saved_files}")
    
    # 为多个交易对生成成交额K线
    print("\n生成多交易对成交额K线...")
    multi_dollar_bars = generator.generate_multi_symbol_dollar_bars(df, ['ETHUSDT', 'BTCUSDT'])
    
    # 保存多交易对结果
    saved_multi_files = generator.save_dollar_bars(multi_dollar_bars)
    print(f"多交易对文件已保存: {saved_multi_files}")


if __name__ == "__main__":
    main() 