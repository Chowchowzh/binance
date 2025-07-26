# -*- coding: utf-8 -*-
"""
三分类标签法（Triple-Barrier Method, TBM）实现
基于Marcos López de Prado的金融机器学习理论
实现动态边界和路径依赖的标签生成
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
import warnings
from numba import jit
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')


class TripleBarrierLabeler:
    """
    三分类标签法实现器
    
    核心功能：
    1. 动态边界设置（基于滚动波动率）
    2. 路径依赖的标签生成
    3. 事件触发机制
    4. 高效的向量化计算
    """
    
    def __init__(self, 
                 profit_factor: float = 2.0,
                 loss_factor: float = 1.0,
                 volatility_window: int = 20,
                 max_holding_period: int = 60,
                 min_return_threshold: float = 0.0001,
                 use_symmetric_barriers: bool = False):
        """
        初始化三分类标签器
        
        Args:
            profit_factor: 止盈因子 (c1)，相对于波动率的倍数
            loss_factor: 止损因子 (c2)，相对于波动率的倍数  
            volatility_window: 滚动波动率计算窗口
            max_holding_period: 最大持仓期（信息时钟）
            min_return_threshold: 最小收益阈值，过滤噪声
            use_symmetric_barriers: 是否使用对称边界
        """
        self.profit_factor = profit_factor
        self.loss_factor = loss_factor
        self.volatility_window = volatility_window
        self.max_holding_period = max_holding_period
        self.min_return_threshold = min_return_threshold
        self.use_symmetric_barriers = use_symmetric_barriers
        
        print(f"初始化三分类标签器:")
        print(f"  - 止盈因子: {profit_factor}")
        print(f"  - 止损因子: {loss_factor}")
        print(f"  - 波动率窗口: {volatility_window}")
        print(f"  - 最大持仓期: {max_holding_period}")
        print(f"  - 对称边界: {use_symmetric_barriers}")
    
    def compute_dynamic_barriers(self, 
                                prices: pd.Series, 
                                volatility: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算动态边界
        
        Args:
            prices: 价格序列
            volatility: 滚动波动率序列
            
        Returns:
            上边界和下边界序列
        """
        if self.use_symmetric_barriers:
            # 对称边界：使用相同的因子
            factor = max(self.profit_factor, self.loss_factor)
            upper_barrier = prices * (1 + factor * volatility)
            lower_barrier = prices * (1 - factor * volatility)
        else:
            # 非对称边界：可以设置不同的风险回报比
            upper_barrier = prices * (1 + self.profit_factor * volatility)
            lower_barrier = prices * (1 - self.loss_factor * volatility)
        
        return upper_barrier, lower_barrier
    
    def compute_realized_volatility(self, 
                                  prices: pd.Series, 
                                  method: str = 'log_return') -> pd.Series:
        """
        计算已实现波动率
        
        Args:
            prices: 价格序列
            method: 计算方法 ('log_return', 'gk_estimator', 'parkinson')
            
        Returns:
            滚动波动率序列
        """
        if method == 'log_return':
            # 对数收益率波动率
            log_returns = np.log(prices / prices.shift(1))
            volatility = log_returns.rolling(
                window=self.volatility_window, 
                min_periods=max(1, self.volatility_window // 2)
            ).std()
            
        elif method == 'gk_estimator':
            # Garman-Klass波动率估计器（需要OHLC数据）
            # 这里简化为对数收益率方法
            log_returns = np.log(prices / prices.shift(1))
            volatility = log_returns.rolling(
                window=self.volatility_window,
                min_periods=max(1, self.volatility_window // 2)
            ).std()
            
        elif method == 'parkinson':
            # Parkinson波动率估计器（需要高低价数据）
            # 这里简化为对数收益率方法
            log_returns = np.log(prices / prices.shift(1))
            volatility = log_returns.rolling(
                window=self.volatility_window,
                min_periods=max(1, self.volatility_window // 2)
            ).std()
            
        else:
            raise ValueError(f"不支持的波动率计算方法: {method}")
        
        # 填充初始值和异常值
        volatility = volatility.fillna(method='bfill').fillna(0.01)
        volatility = volatility.clip(lower=0.0001, upper=0.1)  # 限制在合理范围内
        
        return volatility
    
    @staticmethod
    @jit(nopython=True)
    def _find_barrier_touch_numba(prices: np.ndarray,
                                  upper_barriers: np.ndarray,
                                  lower_barriers: np.ndarray,
                                  start_idx: int,
                                  max_holding_period: int) -> Tuple[int, int]:
        """
        使用Numba加速的边界触碰检测
        
        Args:
            prices: 价格数组
            upper_barriers: 上边界数组
            lower_barriers: 下边界数组
            start_idx: 开始索引
            max_holding_period: 最大持仓期
            
        Returns:
            (触碰类型, 触碰时间): (1=止盈, -1=止损, 0=时间到期, 999=数据不足), 触碰索引
        """
        entry_price = prices[start_idx]
        upper_barrier = upper_barriers[start_idx]
        lower_barrier = lower_barriers[start_idx]
        
        end_idx = min(start_idx + max_holding_period, len(prices) - 1)
        
        for i in range(start_idx + 1, end_idx + 1):
            current_price = prices[i]
            
            # 检查上边界触碰（止盈）
            if current_price >= upper_barrier:
                return 1, i
            
            # 检查下边界触碰（止损）
            if current_price <= lower_barrier:
                return -1, i
        
        # 时间到期
        if end_idx < len(prices) - 1:
            return 0, end_idx
        else:
            # 数据不足
            return 999, end_idx
    
    def generate_triple_barrier_labels(self, 
                                     prices: pd.Series,
                                     event_indices: Optional[Union[List[int], pd.Index]] = None,
                                     volatility_method: str = 'log_return',
                                     n_jobs: int = 1) -> pd.DataFrame:
        """
        生成三分类标签
        
        Args:
            prices: 价格序列（通常是收盘价）
            event_indices: 事件触发点索引，如果为None则使用所有点
            volatility_method: 波动率计算方法
            n_jobs: 并行作业数
            
        Returns:
            包含标签和相关信息的DataFrame
        """
        print("开始生成三分类标签...")
        print(f"  - 价格序列长度: {len(prices)}")
        
        # 计算滚动波动率
        volatility = self.compute_realized_volatility(prices, method=volatility_method)
        print(f"  - 波动率统计: 均值={volatility.mean():.6f}, 标准差={volatility.std():.6f}")
        
        # 计算动态边界
        upper_barriers, lower_barriers = self.compute_dynamic_barriers(prices, volatility)
        
        # 确定事件触发点
        if event_indices is None:
            # 默认：使用所有有效的价格点（跳过初始的NaN值）
            valid_start = max(self.volatility_window, 1)
            event_indices = list(range(valid_start, len(prices) - self.max_holding_period))
        elif isinstance(event_indices, pd.Index):
            event_indices = event_indices.tolist()
        
        print(f"  - 事件数量: {len(event_indices)}")
        
        # 初始化结果
        results = []
        
        # 转换为numpy数组以提高性能
        prices_array = prices.values
        upper_barriers_array = upper_barriers.values
        lower_barriers_array = lower_barriers.values
        
        if n_jobs == 1:
            # 单线程处理
            for i, event_idx in enumerate(event_indices):
                if i % 1000 == 0:
                    print(f"    处理进度: {i}/{len(event_indices)} ({i/len(event_indices)*100:.1f}%)")
                
                result = self._process_single_event(
                    prices_array, upper_barriers_array, lower_barriers_array,
                    event_idx, prices.index[event_idx]
                )
                results.append(result)
        else:
            # 多线程处理
            print(f"  - 使用多线程处理，进程数: {n_jobs}")
            with mp.Pool(processes=n_jobs) as pool:
                process_func = partial(
                    self._process_single_event,
                    prices_array, upper_barriers_array, lower_barriers_array
                )
                
                tasks = [(event_idx, prices.index[event_idx]) for event_idx in event_indices]
                results = pool.starmap(process_func, tasks)
        
        # 构建结果DataFrame
        df_results = pd.DataFrame(results)
        
        # 过滤无效结果
        valid_mask = df_results['touch_type'] != 999
        df_results = df_results[valid_mask].copy()
        
        print(f"  - 有效标签数量: {len(df_results)}")
        
        # 标签分布统计
        label_counts = df_results['label'].value_counts().sort_index()
        print("  - 标签分布:")
        label_names = {-1: "止损", 0: "时间到期", 1: "止盈"}
        for label, count in label_counts.items():
            percentage = count / len(df_results) * 100
            print(f"    {label_names.get(label, label)}: {count} ({percentage:.2f}%)")
        
        return df_results
    
    def _process_single_event(self, 
                            prices_array: np.ndarray,
                            upper_barriers_array: np.ndarray,
                            lower_barriers_array: np.ndarray,
                            event_idx: int,
                            event_timestamp) -> dict:
        """
        处理单个事件的标签生成
        
        Args:
            prices_array: 价格数组
            upper_barriers_array: 上边界数组
            lower_barriers_array: 下边界数组
            event_idx: 事件索引
            event_timestamp: 事件时间戳
            
        Returns:
            事件处理结果字典
        """
        # 使用Numba加速的边界触碰检测
        touch_type, touch_idx = self._find_barrier_touch_numba(
            prices_array, upper_barriers_array, lower_barriers_array,
            event_idx, self.max_holding_period
        )
        
        # 计算相关信息
        entry_price = prices_array[event_idx]
        
        if touch_type != 999:  # 有效触碰
            exit_price = prices_array[touch_idx]
            holding_period = touch_idx - event_idx
            return_pct = (exit_price - entry_price) / entry_price
            
            # 确定最终标签
            if touch_type == 1:  # 止盈
                label = 1
            elif touch_type == -1:  # 止损
                label = -1
            else:  # 时间到期
                # 根据最终收益符号和阈值确定标签
                if abs(return_pct) < self.min_return_threshold:
                    label = 0  # 中性
                else:
                    label = 1 if return_pct > 0 else -1
        else:
            # 数据不足
            exit_price = np.nan
            holding_period = np.nan
            return_pct = np.nan
            label = np.nan
        
        return {
            'event_idx': event_idx,
            'event_timestamp': event_timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'upper_barrier': upper_barriers_array[event_idx],
            'lower_barrier': lower_barriers_array[event_idx],
            'touch_type': touch_type,
            'touch_idx': touch_idx,
            'holding_period': holding_period,
            'return_pct': return_pct,
            'label': label
        }
    
    def generate_cusum_events(self, 
                            prices: pd.Series, 
                            threshold: float = 0.01) -> List[int]:
        """
        使用对称CUSUM过滤器生成事件
        识别市场结构性变化的时刻
        
        Args:
            prices: 价格序列
            threshold: CUSUM阈值
            
        Returns:
            事件索引列表
        """
        print(f"使用CUSUM过滤器生成事件 (阈值: {threshold})")
        
        # 计算对数收益率
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # 初始化CUSUM变量
        events = []
        s_pos = 0  # 正向CUSUM
        s_neg = 0  # 负向CUSUM
        
        for i, ret in enumerate(log_returns):
            # 更新CUSUM值
            s_pos = max(0, s_pos + ret)
            s_neg = min(0, s_neg + ret)
            
            # 检查是否触发事件
            if s_pos > threshold:
                events.append(i + 1)  # +1因为log_returns从索引1开始
                s_pos = 0  # 重置
            elif s_neg < -threshold:
                events.append(i + 1)
                s_neg = 0  # 重置
        
        # 过滤事件：确保有足够的数据进行标签生成
        max_valid_idx = len(prices) - self.max_holding_period - 1
        events = [idx for idx in events if idx <= max_valid_idx]
        
        print(f"  - 生成事件数量: {len(events)}")
        print(f"  - 事件密度: {len(events)/len(prices)*100:.2f}%")
        
        return events
    
    def analyze_label_quality(self, labels_df: pd.DataFrame) -> dict:
        """
        分析标签质量
        
        Args:
            labels_df: 标签DataFrame
            
        Returns:
            质量分析结果
        """
        analysis = {}
        
        # 基本统计
        analysis['total_events'] = len(labels_df)
        analysis['label_distribution'] = labels_df['label'].value_counts().to_dict()
        
        # 持仓期分析
        valid_holding = labels_df['holding_period'].dropna()
        analysis['holding_period_stats'] = {
            'mean': valid_holding.mean(),
            'median': valid_holding.median(),
            'std': valid_holding.std(),
            'min': valid_holding.min(),
            'max': valid_holding.max()
        }
        
        # 收益率分析
        valid_returns = labels_df['return_pct'].dropna()
        analysis['return_stats'] = {
            'mean': valid_returns.mean(),
            'median': valid_returns.median(),
            'std': valid_returns.std(),
            'skewness': valid_returns.skew(),
            'kurtosis': valid_returns.kurtosis()
        }
        
        # 按标签分组的收益率分析
        for label in [-1, 0, 1]:
            label_returns = labels_df[labels_df['label'] == label]['return_pct'].dropna()
            if len(label_returns) > 0:
                analysis[f'returns_label_{label}'] = {
                    'count': len(label_returns),
                    'mean': label_returns.mean(),
                    'std': label_returns.std(),
                    'win_rate': (label_returns > 0).mean() if label != 0 else np.nan
                }
        
        # 触碰类型分析
        touch_type_dist = labels_df['touch_type'].value_counts().to_dict()
        analysis['touch_type_distribution'] = touch_type_dist
        
        return analysis


# 便利函数
def create_triple_barrier_features(df: pd.DataFrame,
                                  price_column: str = 'close',
                                  **labeler_kwargs) -> pd.DataFrame:
    """
    为DataFrame添加三分类标签特征
    
    Args:
        df: 包含价格数据的DataFrame
        price_column: 价格列名
        **labeler_kwargs: TripleBarrierLabeler的参数
        
    Returns:
        包含TBM标签的DataFrame
    """
    labeler = TripleBarrierLabeler(**labeler_kwargs)
    
    # 生成标签
    labels_df = labeler.generate_triple_barrier_labels(df[price_column])
    
    # 合并到原DataFrame
    df_with_labels = df.copy()
    
    # 将标签对齐到原始DataFrame
    for col in ['label', 'return_pct', 'holding_period', 'touch_type']:
        if col in labels_df.columns:
            # 创建与原DataFrame索引对齐的Series
            aligned_series = pd.Series(index=df.index, dtype=float)
            for _, row in labels_df.iterrows():
                event_idx = row['event_idx']
                if event_idx < len(df):
                    aligned_series.iloc[event_idx] = row[col]
            
            df_with_labels[f'tbm_{col}'] = aligned_series
    
    return df_with_labels 