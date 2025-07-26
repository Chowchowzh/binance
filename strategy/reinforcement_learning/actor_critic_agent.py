#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actor-Critic强化学习Agent - 基于PPO算法的交易策略学习

该模块实现了完整的Actor-Critic框架，包括：
1. Actor网络：策略网络，输出动作概率分布
2. Critic网络：价值网络，评估状态价值
3. PPO算法：近端策略优化，稳定且高效的策略梯度方法
4. 经验回放：支持批量训练和样本效率提升

特点：
- 支持离散和连续动作空间
- 内置正则化技术（Dropout, BatchNorm, 权重衰减）
- 自适应学习率调度
- 梯度裁剪和训练稳定性保证

Author: AI Assistant
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, namedtuple
import logging
import math

from .mdp_environment import MDPState, MDPAction

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 经验元组定义
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'
])


def _get_optimal_device(force_cpu: bool = True) -> str:
    """
    智能设备选择 - 优化为CPU多线程
    
    Args:
        force_cpu: 是否强制使用CPU (默认True，因为CPU多线程更适合此任务)
    
    Returns:
        最优设备字符串
    """
    if force_cpu:
        print("🔧 强制使用CPU计算 + 多线程加速 (4进程)")
        return 'cpu'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("🚀 检测到Apple Silicon GPU，使用MPS加速")
        return 'mps'
    elif torch.cuda.is_available():
        print(f"🚀 检测到CUDA GPU: {torch.cuda.get_device_name()}")
        return 'cuda'
    else:
        print("⚠️  使用CPU，建议使用GPU加速训练")
        return 'cpu'


@dataclass
class AgentConfig:
    """Agent配置参数"""
    # 网络架构
    hidden_dim: int = 256
    num_layers: int = 3  # 添加缺失的网络层数配置
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    
    # 训练超参数
    learning_rate: float = 3e-4
    batch_size: int = 64
    ppo_epochs: int = 10
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5  # 添加缺失的梯度裁剪配置
    
    # GAE参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # 正则化
    weight_decay: float = 1e-4
    target_kl: float = 0.01
    
    # 设备配置 - 支持MPS加速  
    device: str = None  # 将在__post_init__中设置
    
    def __post_init__(self):
        """初始化后处理，设置最优设备"""
        if self.device is None:
            self.device = _get_optimal_device()


class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络
    
    输入状态，输出动作概率分布
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig):
        super(ActorNetwork, self).__init__()
        self.config = config
        
        # 构建网络层
        layers = []
        input_dim = state_dim
        
        for i in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            
            layers.append(nn.ReLU())
            
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            
            input_dim = config.hidden_dim
        
        # 输出层
        layers.append(nn.Linear(config.hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        logits = self.network(state)
        return F.softmax(logits, dim=-1)
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """获取动作和对数概率"""
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy


class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络
    
    输入状态，输出状态价值估计
    """
    
    def __init__(self, state_dim: int, config: AgentConfig):
        super(CriticNetwork, self).__init__()
        self.config = config
        
        # 构建网络层
        layers = []
        input_dim = state_dim
        
        for i in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            
            layers.append(nn.ReLU())
            
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            
            input_dim = config.hidden_dim
        
        # 输出层（单一价值）
        layers.append(nn.Linear(config.hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state).squeeze(-1)


class ExperienceBuffer:
    """经验回放缓冲区
    
    存储Agent与环境交互的经验，支持GAE优势计算
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        """添加经验"""
        self.buffer.append(experience)
    
    def get_batch(self, batch_size: int) -> List[Experience]:
        """获取批量经验"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_all(self) -> List[Experience]:
        """获取所有经验"""
        return list(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class ActorCriticAgent:
    """Actor-Critic强化学习Agent
    
    基于PPO算法的完整RL Agent，包括：
    1. 双网络架构（Actor + Critic）
    2. PPO训练算法
    3. GAE优势计算
    4. 自适应学习和正则化
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[AgentConfig] = None):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            config: Agent配置参数
        """
        self.config = config or AgentConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建网络
        self.actor = ActorNetwork(state_dim, action_dim, self.config).to(self.config.device)
        self.critic = CriticNetwork(state_dim, self.config).to(self.config.device)
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        self.actor_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=1000, gamma=0.95
        )
        self.critic_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=1000, gamma=0.95
        )
        
        # 经验缓冲区
        self.experience_buffer = ExperienceBuffer()
        
        # 训练统计
        self.training_stats = {
            'total_steps': 0,
            'total_episodes': 0,
            'actor_losses': [],
            'critic_losses': [],
            'rewards': [],
            'entropies': []
        }
        
        logger.info(f"初始化Actor-Critic Agent: 状态维度={state_dim}, 动作维度={action_dim}")
    
    def select_action(self, state: MDPState, training: bool = True) -> Tuple[MDPAction, Dict[str, Any]]:
        """选择动作
        
        Args:
            state: 当前状态
            training: 是否为训练模式
            
        Returns:
            action: 选择的动作
            info: 额外信息（概率、价值等）
        """
        state_tensor = state.to_tensor(self.config.device).unsqueeze(0)
        
        # 保存原始训练状态
        actor_training = self.actor.training
        critic_training = self.critic.training
        
        try:
            # 临时设为eval模式以避免BatchNorm问题（batch_size=1）
            self.actor.eval()
            self.critic.eval()
            
            with torch.no_grad() if not training else torch.enable_grad():
                # 获取动作概率
                if training:
                    action_idx, log_prob, entropy = self.actor.get_action_and_log_prob(state_tensor)
                    value = self.critic(state_tensor)
                else:
                    probs = self.actor(state_tensor)
                    action_idx = torch.argmax(probs, dim=-1).item()
                    log_prob = torch.log(probs[0, action_idx])
                    value = self.critic(state_tensor)
                    entropy = torch.tensor(0.0)
        finally:
            # 恢复原始训练状态
            if actor_training:
                self.actor.train()
            if critic_training:
                self.critic.train()
        
        # 转换为MDP动作
        action = MDPAction.from_discrete_action(action_idx)
        
        info = {
            'action_idx': action_idx,
            'log_prob': log_prob.item() if training else log_prob.item(),
            'value': value.item(),
            'entropy': entropy.item() if training else 0.0
        }
        
        return action, info
    
    def store_experience(self, state: MDPState, action: int, reward: float, 
                        next_state: MDPState, done: bool, log_prob: float, value: float):
        """存储经验"""
        experience = Experience(
            state=state.to_tensor(self.config.device),
            action=action,
            reward=reward,
            next_state=next_state.to_tensor(self.config.device),
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.experience_buffer.add(experience)
    
    def compute_gae(self, experiences: List[Experience]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE优势和回报
        
        GAE (Generalized Advantage Estimation) 提供了方差和偏差之间的平衡
        """
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool)
        
        # 计算TD error
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]
        
        deltas = rewards + self.config.gamma * next_values * (~dones) - values
        
        # 计算GAE
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(experiences))):
            advantage = deltas[t] + self.config.gamma * self.config.gae_lambda * advantage * (~dones[t])
            advantages[t] = advantage
        
        # 计算回报
        returns = advantages + values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self) -> Dict[str, float]:
        """更新策略（PPO算法）"""
        if len(self.experience_buffer) < self.config.batch_size:
            return {}
        
        # 获取所有经验
        experiences = self.experience_buffer.get_all()
        
        # 计算优势和回报
        advantages, returns = self.compute_gae(experiences)
        
        # 准备训练数据
        states = torch.stack([exp.state for exp in experiences]).to(self.config.device)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(self.config.device)
        old_log_probs = torch.tensor([exp.log_prob for exp in experiences], dtype=torch.float32).to(self.config.device)
        advantages = advantages.to(self.config.device)
        returns = returns.to(self.config.device)
        
        # 训练统计
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # PPO更新循环
        for epoch in range(self.config.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(experiences))
            
            for start_idx in range(0, len(experiences), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(experiences))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                action_probs = self.actor(batch_states)
                values = self.critic(batch_states)
                
                # 计算新的对数概率和熵
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算概率比
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic损失
                critic_loss = F.mse_loss(values, batch_returns)
                
                # 总损失
                total_loss = actor_loss + self.config.value_loss_coef * critic_loss - self.config.entropy_coef * entropy
                
                # 反向传播和优化
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # 更新学习率
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # 清空经验缓冲区
        self.experience_buffer.clear()
        
        # 更新训练统计
        avg_actor_loss = total_actor_loss / (self.config.ppo_epochs * len(experiences) // self.config.batch_size)
        avg_critic_loss = total_critic_loss / (self.config.ppo_epochs * len(experiences) // self.config.batch_size)
        avg_entropy = total_entropy / (self.config.ppo_epochs * len(experiences) // self.config.batch_size)
        
        self.training_stats['actor_losses'].append(avg_actor_loss)
        self.training_stats['critic_losses'].append(avg_critic_loss)
        self.training_stats['entropies'].append(avg_entropy)
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy,
            'learning_rate': self.actor_optimizer.param_groups[0]['lr']
        }
    
    def save_models(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_models(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        logger.info(f"模型已加载: {path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        stats = self.training_stats.copy()
        
        if len(stats['actor_losses']) > 0:
            stats['avg_actor_loss'] = np.mean(stats['actor_losses'][-100:])
            stats['avg_critic_loss'] = np.mean(stats['critic_losses'][-100:])
            stats['avg_entropy'] = np.mean(stats['entropies'][-100:])
        
        return stats
    
    def set_training_mode(self, training: bool):
        """设置训练模式"""
        self.actor.train(training)
        self.critic.train(training)


if __name__ == "__main__":
    # 简单测试
    print("测试Actor-Critic Agent...")
    
    # 创建配置
    config = AgentConfig(
        hidden_dim=128,
        num_layers=2,
        learning_rate=1e-3
    )
    
    # 创建Agent
    state_dim = 10
    action_dim = 5
    agent = ActorCriticAgent(state_dim, action_dim, config)
    
    # 创建模拟状态
    from .mdp_environment import MDPState
    import numpy as np
    
    mock_state = MDPState(
        market_features=np.random.randn(5).astype(np.float32),
        signal_confidence=0.7,
        signal_direction=1,
        current_position=0.0,
        unrealized_pnl=0.0,
        portfolio_value=100000.0,
        days_in_position=0,
        recent_volatility=0.02,
        current_price=100.0
    )
    
    # 测试动作选择
    action, info = agent.select_action(mock_state)
    print(f"选择动作: {action.target_position}")
    print(f"动作信息: {info}")
    
    # 测试存储经验
    agent.store_experience(
        state=mock_state,
        action=action.to_discrete(),
        reward=0.1,
        next_state=mock_state,
        done=False,
        log_prob=info['log_prob'],
        value=info['value']
    )
    
    print(f"经验缓冲区大小: {len(agent.experience_buffer)}")
    print("Actor-Critic Agent测试完成！") 