#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDP交易环境 - 将交易问题形式化为马尔可夫决策过程

该模块实现了完整的MDP框架，包括：
1. 状态空间：市场特征 + 信号置信度 + 投资组合状态
2. 动作空间：目标仓位大小（离散化）
3. 奖励函数：基于风险调整后收益（夏普比率微分 + 交易成本惩罚）
4. 转移概率：通过历史数据环境交互学习

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MDPState:
    """MDP状态定义
    
    包含三个核心组件：
    1. 市场特征：技术指标、微观结构特征、动态特征
    2. 信号置信度：来自元标签模型的预测概率
    3. 投资组合状态：当前持仓、未实现盈亏等
    """
    market_features: np.ndarray        # 市场特征向量
    signal_confidence: float           # 信号置信度 [0, 1]
    signal_direction: int              # 信号方向 {-1, 0, 1}
    current_position: float            # 当前仓位 [-1, 1]
    unrealized_pnl: float             # 未实现盈亏
    portfolio_value: float             # 投资组合价值
    days_in_position: int             # 持仓天数
    recent_volatility: float          # 近期波动率
    current_price: float              # 当前价格
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """
        将状态转换为张量格式，供神经网络使用
        
        Args:
            device: 目标设备 ('cpu', 'cuda', 'mps')
            
        Returns:
            在指定设备上的状态张量
        """
        state_vector = np.concatenate([
            self.market_features,
            [self.signal_confidence, 
             self.signal_direction,
             self.current_position,
             self.unrealized_pnl / self.portfolio_value,  # 归一化
             self.days_in_position / 100.0,  # 归一化
             self.recent_volatility,
             np.log(self.current_price / 100.0)]  # 对数归一化
        ])
        tensor = torch.FloatTensor(state_vector)
        
        # 根据设备类型移动tensor
        if device in ['cuda', 'mps'] and device != 'cpu':
            tensor = tensor.to(device)
            
        return tensor


@dataclass 
class MDPAction:
    """MDP动作定义
    
    动作为目标仓位大小，而非简单的买卖信号
    """
    target_position: float  # 目标仓位 [-1.0, 1.0]
    
    @classmethod
    def from_discrete_action(cls, discrete_action: int) -> 'MDPAction':
        """从离散动作索引转换为连续目标仓位"""
        # 动作空间: {-1.0, -0.5, 0, +0.5, +1.0}
        action_map = {
            0: -1.0,   # 全仓做空
            1: -0.5,   # 半仓做空
            2:  0.0,   # 空仓
            3:  0.5,   # 半仓做多
            4:  1.0    # 全仓做多
        }
        return cls(target_position=action_map[discrete_action])
    
    def to_discrete(self) -> int:
        """将连续目标仓位转换为离散动作索引"""
        position_to_action = {
            -1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4
        }
        # 找到最接近的离散动作
        closest_pos = min(position_to_action.keys(), 
                         key=lambda x: abs(x - self.target_position))
        return position_to_action[closest_pos]


class DifferentialSharpeReward:
    """差分夏普比率奖励计算器
    
    计算从一个时间步到下一个时间步投资组合夏普比率的变化，
    激励Agent寻找风险效率高的交易策略。
    """
    
    def __init__(self, window_size: int = 252, risk_free_rate: float = 0.05):
        """
        Args:
            window_size: 计算夏普比率的滚动窗口大小
            risk_free_rate: 无风险利率（年化）
        """
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate / 252  # 转换为日收益率
        self.returns_buffer = deque(maxlen=window_size)
        self.prev_sharpe = 0.0
    
    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - self.risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def get_differential_reward(self, current_return: float) -> float:
        """获取差分夏普比率奖励"""
        self.returns_buffer.append(current_return)
        
        if len(self.returns_buffer) < 10:  # 最少需要10个观测值
            return 0.0
        
        current_sharpe = self.calculate_sharpe_ratio(list(self.returns_buffer))
        differential_sharpe = current_sharpe - self.prev_sharpe
        self.prev_sharpe = current_sharpe
        
        return differential_sharpe


class TradingMDPEnvironment:
    """交易MDP环境
    
    实现完整的马尔可夫决策过程交易环境，支持：
    1. 复合状态空间（市场+信号+投资组合）
    2. 离散动作空间（目标仓位）
    3. 差分夏普比率+交易成本的复合奖励
    4. 历史数据驱动的状态转移
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.00075,
                 max_position: float = 1.0,
                 lookback_window: int = 60,
                 volatility_penalty: float = 0.1,
                 sharpe_window: int = 252):
        """
        Args:
            data: 包含特征和标签的历史数据
            initial_capital: 初始资金
            transaction_cost: 交易成本（单边）
            max_position: 最大仓位
            lookback_window: 状态特征回看窗口
            volatility_penalty: 波动率惩罚权重
            sharpe_window: 夏普比率计算窗口
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.volatility_penalty = volatility_penalty
        
        # 初始化奖励计算器
        self.sharpe_reward = DifferentialSharpeReward(window_size=sharpe_window)
        
        # 环境状态
        self.current_step = 0
        self.current_position = 0.0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.shares_held = 0.0
        self.entry_price = 0.0
        self.days_in_position = 0
        
        # 性能追踪
        self.portfolio_history = []
        self.position_history = []
        self.action_history = []
        self.reward_history = []
        self.trade_history = []
        
        # 预处理数据
        self._prepare_data()
        
        logger.info(f"初始化交易MDP环境: {len(self.data)} 个时间步")
    
    def _prepare_data(self):
        """准备和验证输入数据"""
        # 检查必要列
        required_columns = ['close', 'volume', 'signal_confidence', 'signal_direction']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"数据缺少必要列: {missing_columns}")
        
        # 计算收益率 - 使用numpy避免pandas兼容性问题
        if 'returns' not in self.data.columns:
            close_values = self.data['close'].values
            returns = np.zeros_like(close_values)
            returns[1:] = (close_values[1:] - close_values[:-1]) / close_values[:-1]
            self.data['returns'] = returns
        
        # 计算波动率
        if 'volatility' not in self.data.columns:
            returns = self.data['returns'].values
            volatility = np.zeros_like(returns)
            window = getattr(self, 'lookback_window', 100)  # 默认100
            for i in range(window, len(returns)):
                volatility[i] = np.std(returns[i-window:i]) if i >= window else 0
            self.data['volatility'] = volatility
        
        # 填充NaN值 - 避免pandas兼容性问题
        for col in self.data.columns:
            if self.data[col].dtype.kind in 'biufc':  # 数值列
                self.data[col] = self.data[col].fillna(0)
        
        logger.info(f"数据准备完成: {self.data.shape}")
        logger.info(f"收益率统计: mean={self.data['returns'].mean():.6f}, std={self.data['returns'].std():.6f}")
        logger.info(f"波动率统计: mean={self.data['volatility'].mean():.6f}, std={self.data['volatility'].std():.6f}")
        
        # 提取特征列（除了基础价格信息外的所有特征）
        exclude_columns = [
            'open', 'high', 'low', 'close', 'volume', 
            'signal_confidence', 'signal_direction',
            'returns', 'volatility',
            'start_time', 'end_time', 'end_timestamp', 'timestamp'  # 排除时间相关列
        ]
        
        # 只保留数值型特征列
        self.feature_columns = []
        for col in self.data.columns:
            if col not in exclude_columns and self.data[col].dtype.kind in 'biufc':  # 数值类型
                self.feature_columns.append(col)
        
        logger.info(f"提取到 {len(self.feature_columns)} 个市场特征")
        
        # 验证特征列都是数值型
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                logger.warning(f"非数值特征列: {col}, 类型: {self.data[col].dtype}")
        
        logger.info(f"初始化交易MDP环境: {len(self.data)} 个时间步")
    
    def reset(self, start_step: Optional[int] = None) -> MDPState:
        """重置环境到初始状态"""
        if start_step is None:
            self.current_step = self.lookback_window
        else:
            self.current_step = max(start_step, self.lookback_window)
        
        self.current_position = 0.0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.shares_held = 0.0
        self.entry_price = 0.0
        self.days_in_position = 0
        
        # 清空历史记录
        self.portfolio_history = []
        self.position_history = []
        self.action_history = []
        self.reward_history = []
        self.trade_history = []
        
        # 重置奖励计算器
        self.sharpe_reward = DifferentialSharpeReward(window_size=252)
        
        return self._get_current_state()
    
    def _get_current_state(self) -> MDPState:
        """获取当前状态"""
        current_row = self.data.iloc[self.current_step]
        
        # 1. 市场特征（使用回看窗口内的最新特征）
        if len(self.feature_columns) > 0:
            market_features = current_row[self.feature_columns].values.astype(np.float32)
        else:
            # 如果没有额外特征，使用基础技术指标
            market_features = np.array([
                current_row['returns'],
                current_row['volatility'],
                current_row['volume'] / current_row['volume'] if current_row['volume'] > 0 else 0
            ], dtype=np.float32)
        
        # 处理NaN值
        market_features = np.nan_to_num(market_features, nan=0.0)
        
        # 2. 信号置信度和方向
        signal_confidence = float(current_row.get('signal_confidence', 0.5))
        signal_direction = int(current_row.get('signal_direction', 0))
        
        # 3. 投资组合状态
        current_price = float(current_row['close'])
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        recent_volatility = float(current_row.get('volatility', 0.02))
        
        return MDPState(
            market_features=market_features,
            signal_confidence=signal_confidence,
            signal_direction=signal_direction,
            current_position=self.current_position,
            unrealized_pnl=unrealized_pnl,
            portfolio_value=self.portfolio_value,
            days_in_position=self.days_in_position,
            recent_volatility=recent_volatility,
            current_price=current_price
        )
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        if self.shares_held == 0 or self.entry_price == 0:
            return 0.0
        
        return self.shares_held * (current_price - self.entry_price)
    
    def step(self, action: MDPAction) -> Tuple[MDPState, float, bool, Dict[str, Any]]:
        """执行一步动作
        
        Returns:
            next_state: 下一个状态
            reward: 获得的奖励
            done: 是否结束
            info: 额外信息
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_current_state(), 0.0, True, {'message': 'Episode finished'}
        
        # 获取当前价格
        current_price = float(self.data.iloc[self.current_step]['close'])
        
        # 执行交易
        trade_info = self._execute_trade(action, current_price)
        
        # 移动到下一时间步
        self.current_step += 1
        next_price = float(self.data.iloc[self.current_step]['close'])
        
        # 更新投资组合价值
        self._update_portfolio_value(next_price)
        
        # 计算奖励
        reward = self._calculate_reward(trade_info, current_price, next_price)
        
        # 更新持仓天数
        if abs(self.current_position) > 1e-6:
            self.days_in_position += 1
        else:
            self.days_in_position = 0
        
        # 记录历史
        self.portfolio_history.append(self.portfolio_value)
        self.position_history.append(self.current_position)
        self.action_history.append(action.target_position)
        self.reward_history.append(reward)
        
        # 获取下一状态
        next_state = self._get_current_state()
        
        # 检查是否结束
        done = (self.current_step >= len(self.data) - 1 or 
                self.portfolio_value <= 0.1 * self.initial_capital)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.current_position,
            'trade_info': trade_info,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _execute_trade(self, action: MDPAction, current_price: float) -> Dict[str, Any]:
        """执行交易操作"""
        target_position = np.clip(action.target_position, -self.max_position, self.max_position)
        position_change = target_position - self.current_position
        
        if abs(position_change) < 1e-6:
            return {'trade_executed': False, 'transaction_cost': 0.0}
        
        # 计算需要交易的份额
        trade_value = abs(position_change) * self.portfolio_value
        transaction_cost = trade_value * self.transaction_cost
        
        # 更新现金和持仓
        if position_change > 0:  # 买入
            required_cash = trade_value + transaction_cost
            if required_cash <= self.cash:
                shares_to_buy = trade_value / current_price
                self.shares_held += shares_to_buy
                self.cash -= required_cash
                self.current_position = target_position
                if abs(self.current_position) > 1e-6:
                    self.entry_price = current_price
            else:
                transaction_cost = 0.0  # 资金不足，无法交易
        else:  # 卖出
            shares_to_sell = abs(trade_value) / current_price
            if shares_to_sell <= self.shares_held:
                self.shares_held -= shares_to_sell
                self.cash += trade_value - transaction_cost
                self.current_position = target_position
                if abs(self.current_position) < 1e-6:
                    self.entry_price = 0.0
            else:
                transaction_cost = 0.0  # 持仓不足，无法交易
        
        # 记录交易
        if transaction_cost > 0:
            self.trade_history.append({
                'step': self.current_step,
                'action': target_position,
                'price': current_price,
                'cost': transaction_cost,
                'position_change': position_change
            })
        
        return {
            'trade_executed': transaction_cost > 0,
            'transaction_cost': transaction_cost,
            'position_change': position_change
        }
    
    def _update_portfolio_value(self, current_price: float):
        """更新投资组合价值"""
        self.portfolio_value = self.cash + self.shares_held * current_price
    
    def _calculate_reward(self, trade_info: Dict[str, Any], 
                         prev_price: float, current_price: float) -> float:
        """计算复合奖励函数
        
        奖励 = 差分夏普比率 - 交易成本惩罚 - 波动率惩罚
        """
        # 1. 计算当期收益率
        price_return = (current_price - prev_price) / prev_price
        position_return = self.current_position * price_return
        
        # 2. 差分夏普比率奖励
        sharpe_reward = self.sharpe_reward.get_differential_reward(position_return)
        
        # 3. 交易成本惩罚
        transaction_cost_penalty = trade_info.get('transaction_cost', 0.0) / self.portfolio_value
        
        # 4. 波动率惩罚（鼓励平滑的资金曲线）
        volatility_penalty = 0.0
        if len(self.portfolio_history) >= 2:
            recent_returns = np.diff(self.portfolio_history[-10:]) / self.portfolio_history[-10:-1]
            if len(recent_returns) > 0:
                volatility_penalty = self.volatility_penalty * np.std(recent_returns)
        
        # 5. 复合奖励
        total_reward = (sharpe_reward 
                       - transaction_cost_penalty 
                       - volatility_penalty)
        
        return total_reward
    
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return 5  # {-1.0, -0.5, 0, +0.5, +1.0}
    
    def get_state_size(self) -> int:
        """获取状态空间大小"""
        sample_state = self._get_current_state()
        return len(sample_state.to_tensor())  # 这里只需要长度，使用默认CPU即可
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取环境性能指标"""
        if len(self.portfolio_history) < 2:
            return {}
        
        portfolio_returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        metrics = {
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'annual_return': np.mean(portfolio_returns) * 252,
            'volatility': np.std(portfolio_returns) * np.sqrt(252),
            'sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'total_trades': len(self.trade_history),
            'avg_position': np.mean(np.abs(self.position_history)) if self.position_history else 0
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        peak = self.portfolio_history[0]
        max_dd = 0.0
        
        for value in self.portfolio_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd


if __name__ == "__main__":
    # 简单测试
    print("测试MDP交易环境...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_steps = 1000
    
    # 模拟价格序列
    returns = np.random.normal(0.0001, 0.02, n_steps)
    prices = 100 * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, n_steps),
        'signal_confidence': np.random.uniform(0.4, 0.8, n_steps),
        'signal_direction': np.random.choice([-1, 0, 1], n_steps),
        'feature_1': np.random.normal(0, 1, n_steps),
        'feature_2': np.random.normal(0, 1, n_steps),
    })
    
    # 创建环境
    env = TradingMDPEnvironment(test_data)
    
    # 测试环境
    state = env.reset()
    print(f"初始状态维度: {len(state.to_tensor())}")  # 只需要长度，使用默认CPU即可
    print(f"动作空间大小: {env.get_action_space_size()}")
    
    # 执行几步随机动作
    for i in range(10):
        action = MDPAction.from_discrete_action(np.random.randint(0, 5))
        next_state, reward, done, info = env.step(action)
        print(f"Step {i}: Reward={reward:.6f}, Portfolio={info['portfolio_value']:.2f}")
        
        if done:
            break
    
    # 获取性能指标
    metrics = env.get_performance_metrics()
    print(f"\n性能指标: {metrics}")
    
    print("MDP环境测试完成！") 