#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习训练管道 - 完整的RL交易策略训练系统

该模块实现了端到端的强化学习训练流程，包括：
1. 数据预处理和特征工程
2. MDP环境配置和初始化
3. Actor-Critic Agent训练
4. 稳健回测和性能评估
5. 模型保存和部署准备

核心功能：
- 分阶段训练（预训练、主训练、微调）
- 超参数优化和自动调优
- 实时监控和早停机制
- 多进程并行训练支持
- 完整的实验跟踪和日志记录

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor
import optuna
from tqdm import tqdm

from .mdp_environment import TradingMDPEnvironment, MDPState, MDPAction
from .actor_critic_agent import ActorCriticAgent, AgentConfig
from .robust_backtester import RobustBacktester, BacktestConfig

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy.training.enhanced_signal_generator import EnhancedSignalGenerator
from strategy.training.enhanced_transformer import EnhancedTimeSeriesTransformer, TransformerConfig
from config.settings import load_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RLTrainingConfig:
    """强化学习训练配置"""
    # 训练阶段
    num_pretraining_episodes: int = 50
    num_main_training_episodes: int = 200
    num_fine_tuning_episodes: int = 50
    
    # 环境参数
    initial_capital: float = 100000.0
    transaction_cost: float = 0.00075
    max_position: float = 1.0
    lookback_window: int = 60
    
    # 训练策略
    curriculum_learning: bool = True
    experience_replay: bool = True
    prioritized_replay: bool = False
    
    # 超参数优化
    use_hyperparameter_tuning: bool = False
    n_trials: int = 50
    pruning_enabled: bool = True
    
    # 并行训练
    use_multiprocessing: bool = True  # 启用多进程加速
    n_workers: int = 4
    
    # 监控和保存
    save_interval: int = 50
    eval_frequency: int = 25
    checkpoint_dir: str = "rl_checkpoints"
    log_dir: str = "rl_logs"
    
    # 早停和调度
    early_stopping_patience: int = 20
    learning_rate_schedule: bool = True
    
    # 实验跟踪
    experiment_name: str = "rl_trading_experiment"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["rl", "trading", "actor_critic"]


class ExperimentTracker:
    """实验跟踪器 - 记录训练过程和结果"""
    
    def __init__(self, config):
        self.config = config
        rl_config = config.reinforcement_learning
        self.experiment_dir = Path(rl_config.experiment_dir) / rl_config.experiment_name
        self.log_dir = Path(rl_config.log_dir)
        self.results_dir = Path(rl_config.results_dir)
        
        # 创建目录
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化记录
        self.metrics = {}
        self.hyperparameters = {}
        self.artifacts = []
        
    def save_config(self):
        """保存实验配置"""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, phase: str = "train"):
        """记录指标"""
        log_entry = {
            'step': step,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics[step] = log_entry # 使用字典存储，方便后续保存
        
        # 实时保存
        if len(self.metrics) % 10 == 0:
            self.save_metrics()
    
    def save_metrics(self):
        """保存指标到文件"""
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """记录超参数"""
        self.hyperparameters.update(hyperparameters)
        
        hyperparams_path = self.experiment_dir / "hyperparameters.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(self.hyperparameters, f, indent=2, default=str)
    
    def save_artifact(self, artifact: Any, name: str, artifact_type: str = "model"):
        """保存训练产物"""
        artifact_dir = self.experiment_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        
        if artifact_type == "model":
            artifact_path = artifact_dir / f"{name}.pth"
            torch.save(artifact, artifact_path)
        elif artifact_type == "data":
            artifact_path = artifact_dir / f"{name}.pkl"
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
        else:
            artifact_path = artifact_dir / f"{name}.json"
            with open(artifact_path, 'w') as f:
                json.dump(artifact, f, indent=2, default=str)
        
        self.artifacts.append(str(artifact_path))
        logger.info(f"保存产物: {name} -> {artifact_path}")


class CurriculumLearning:
    """课程学习策略
    
    逐步增加训练难度，提高学习效率
    """
    
    def __init__(self, stages: List[Dict[str, Any]]):
        self.current_stage = 0
        self.stages = stages
    
    def get_current_stage(self) -> Dict[str, Any]:
        """获取当前训练阶段配置"""
        return self.stages[min(self.current_stage, len(self.stages) - 1)]
    
    def advance_stage(self, performance_metric: float, threshold: float = 0.6):
        """根据性能决定是否进入下一阶段"""
        if performance_metric > threshold and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            logger.info(f"课程学习进入阶段: {self.stages[self.current_stage]['name']}")
            return True
        return False


class HyperparameterOptimizer:
    """超参数优化器
    
    使用Optuna进行自动超参数搜索
    """
    
    def __init__(self, config: RLTrainingConfig):
        self.config = config
        self.study = None
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """建议超参数组合"""
        hyperparameters = {
            # Agent参数
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'clip_epsilon': trial.suggest_float('clip_epsilon', 0.1, 0.3),
            'entropy_coef': trial.suggest_float('entropy_coef', 0.001, 0.1, log=True),
            
            # 环境参数
            'transaction_cost': trial.suggest_float('transaction_cost', 0.0001, 0.001),
            'volatility_penalty': trial.suggest_float('volatility_penalty', 0.05, 0.2),
            
            # 训练参数
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'gamma': trial.suggest_float('gamma', 0.95, 0.999)
        }
        
        return hyperparameters
    
    def optimize(self, objective_function: Callable, n_trials: int = 50) -> Dict[str, Any]:
        """执行超参数优化"""
        self.study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner() if self.config.pruning_enabled else None
        )
        
        self.study.optimize(objective_function, n_trials=n_trials)
        
        return self.study.best_params


class RLTrainingPipeline:
    """强化学习训练管道
    
    完整的端到端RL训练系统，包括：
    1. 数据预处理和环境设置
    2. Agent训练和优化
    3. 性能评估和模型保存
    4. 超参数优化和实验跟踪
    """
    
    def __init__(self, config):
        self.config = config
        self.rl_config = config.reinforcement_learning
        
        # 设置随机种子
        np.random.seed(self.rl_config.random_seed)
        torch.manual_seed(self.rl_config.random_seed)
        
        # 设置设备 - 支持MPS加速
        self.device = self._get_optimal_device()
        print(f"🔧 强化学习训练设备: {self.device}")
        
        # 初始化组件
        self.environment = None
        self.agent = None
        self.backtester = None
        self.tracker = ExperimentTracker(self.config)
        
        # 初始化训练历史记录
        self.training_history = []
        
        # 初始化早停相关属性
        self.best_performance = float('-inf')
        self.patience_counter = 0
        # 创建课程学习阶段配置
        self.curriculum_phases = self._create_curriculum_phases()
        # 初始化课程学习
        self.curriculum = CurriculumLearning(self.curriculum_phases) if self.rl_config.curriculum_learning else None
    
    def _get_optimal_device(self) -> torch.device:
        """
        智能设备选择 - 优化为CPU多线程
        
        Returns:
            最优PyTorch设备对象
        """
        print("🔧 强制使用CPU进行强化学习训练 + 多线程加速")
        return torch.device('cpu')
    
    def _create_curriculum_phases(self) -> List[Dict[str, Any]]:
        """创建课程学习阶段配置"""
        return [
            {"name": "预训练阶段", "difficulty": 0.3, "volatility_factor": 0.5, "episodes": self.rl_config.pre_training_episodes},
            {"name": "主训练阶段", "difficulty": 0.6, "volatility_factor": 0.8, "episodes": self.rl_config.num_episodes},
            {"name": "微调阶段", "difficulty": 1.0, "volatility_factor": 1.0, "episodes": self.rl_config.fine_tuning_episodes}
        ]
    
    def prepare_data(self, data_path: str) -> pd.DataFrame:
        """准备训练数据"""
        logger.info(f"加载训练数据: {data_path}")
        
        if data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
        
        # 数据验证
        required_columns = ['close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"数据缺少必要列: {missing_columns}")
        
        # 添加信号特征（如果不存在）
        if 'signal_confidence' not in data.columns:
            logger.info("生成模拟信号特征...")
            # 这里可以集成实际的信号生成器
            data['signal_confidence'] = np.random.uniform(0.4, 0.8, len(data))
            data['signal_direction'] = np.random.choice([-1, 0, 1], len(data))
        
        logger.info(f"数据准备完成: {len(data)} 行, {len(data.columns)} 列")
        return data
    
    def create_environment(self, data: pd.DataFrame, 
                          stage_config: Optional[Dict[str, Any]] = None) -> TradingMDPEnvironment:
        """创建MDP交易环境"""
        env_config = {
            'initial_capital': self.rl_config.initial_cash,
            'transaction_cost': self.rl_config.transaction_cost_bps / 10000,  # 转换bps为小数
            'max_position': self.rl_config.max_position,
            'lookback_window': self.rl_config.lookback_window
        }
        
        # 应用课程学习配置
        if stage_config:
            if 'volatility_scale' in stage_config:
                # 调整数据的波动性
                if 'returns' in data.columns:
                    data = data.copy()
                    data['returns'] *= stage_config['volatility_scale']
        
        return TradingMDPEnvironment(data, **env_config)
    
    def create_agent(self, state_dim: int, action_dim: int, 
                    hyperparameters: Optional[Dict[str, Any]] = None) -> ActorCriticAgent:
        """创建Actor-Critic Agent"""
        # 默认配置
        agent_config = AgentConfig()
        
        # 应用超参数
        if hyperparameters:
            for key, value in hyperparameters.items():
                if hasattr(agent_config, key):
                    setattr(agent_config, key, value)
        
        return ActorCriticAgent(state_dim, action_dim, agent_config)
    
    def train_single_episode(self, agent: ActorCriticAgent, 
                           env: TradingMDPEnvironment, 
                           episode_idx: int) -> Dict[str, Any]:
        """训练单个episode"""
        state = env.reset()
        total_reward = 0
        step_count = 0
        episode_metrics = {
            'rewards': [],
            'actions': [],
            'values': [],
            'entropies': []
        }
        
        while True:
            # 选择动作
            action, action_info = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_experience(
                state=state,
                action=action.to_discrete(),
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=action_info['log_prob'],
                value=action_info['value']
            )
            
            # 记录指标
            episode_metrics['rewards'].append(reward)
            episode_metrics['actions'].append(action.target_position)
            episode_metrics['values'].append(action_info['value'])
            episode_metrics['entropies'].append(action_info['entropy'])
            
            total_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        # 更新策略
        update_info = agent.update_policy()
        
        # 计算episode统计
        episode_stats = {
            'episode': episode_idx,
            'total_reward': total_reward,
            'episode_length': step_count,
            'avg_reward': total_reward / step_count if step_count > 0 else 0,
            'avg_value': np.mean(episode_metrics['values']) if episode_metrics['values'] else 0,
            'avg_entropy': np.mean(episode_metrics['entropies']) if episode_metrics['entropies'] else 0,
            'position_variance': np.var(episode_metrics['actions']) if episode_metrics['actions'] else 0,
            **update_info
        }
        
        # 环境性能指标
        env_metrics = env.get_performance_metrics()
        episode_stats.update(env_metrics)
        
        return episode_stats
    
    def evaluate_agent(self, agent: ActorCriticAgent, 
                      eval_data: pd.DataFrame) -> Dict[str, Any]:
        """评估Agent性能"""
        # 创建评估环境
        eval_env = self.create_environment(eval_data)
        
        # 设置为评估模式
        agent.set_training_mode(False)
        
        # 运行评估
        state = eval_env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            action, _ = agent.select_action(state, training=False)
            next_state, reward, done, info = eval_env.step(action)
            
            total_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        # 获取环境性能指标
        env_metrics = eval_env.get_performance_metrics()
        
        eval_metrics = {
            'eval_total_reward': total_reward,
            'eval_avg_reward': total_reward / step_count if step_count > 0 else 0,
            'eval_steps': step_count,
            **{f'eval_{k}': v for k, v in env_metrics.items()}
        }
        
        # 恢复训练模式
        agent.set_training_mode(True)
        
        return eval_metrics
    
    def save_checkpoint(self, agent: ActorCriticAgent, episode: int, 
                       metrics: Dict[str, Any]):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'agent_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'optimizer_state_dict': agent.actor_optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(self.rl_config.checkpoint_dir, f"checkpoint_episode_{episode}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if metrics.get('eval_sharpe_ratio', -np.inf) > self.best_performance:
            self.best_performance = metrics.get('eval_sharpe_ratio', -np.inf)
            best_model_path = os.path.join(self.rl_config.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            logger.info(f"保存最佳模型: 夏普比率={self.best_performance:.4f}")
    
    def check_early_stopping(self, metrics: Dict[str, Any]) -> bool:
        """检查早停条件"""
        current_performance = metrics.get('eval_sharpe_ratio', -np.inf)
        
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.rl_config.early_stopping_patience:
            logger.info(f"早停触发: 连续{self.patience_counter}次评估无改善")
            return True
        
        return False
    
    def run_training(self, data_path: str, 
                    eval_data_path: Optional[str] = None,
                    hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """运行完整训练流程"""
        logger.info("开始强化学习训练...")
        
        # 准备数据
        train_data = self.prepare_data(data_path)
        
        if eval_data_path:
            eval_data = self.prepare_data(eval_data_path)
        else:
            # 使用训练数据的最后20%作为评估数据
            split_idx = int(len(train_data) * 0.8)
            eval_data = train_data.iloc[split_idx:].reset_index(drop=True)
            train_data = train_data.iloc[:split_idx].reset_index(drop=True)
        
        # 创建环境和Agent
        initial_env = self.create_environment(train_data)
        state_dim = initial_env.get_state_size()
        action_dim = initial_env.get_action_space_size()
        
        agent = self.create_agent(state_dim, action_dim, hyperparameters)
        
        # 记录超参数
        if hyperparameters:
            self.tracker.log_hyperparameters(hyperparameters)
        
        # 训练循环
        total_episodes = (self.rl_config.pre_training_episodes + 
                                                   self.rl_config.num_episodes + 
                          self.rl_config.fine_tuning_episodes)
        
        training_phases = [
            ("pretraining", self.rl_config.pre_training_episodes),
            ("main_training", self.rl_config.num_episodes),
            ("fine_tuning", self.rl_config.fine_tuning_episodes)
        ]
        
        episode_count = 0
        
        for phase_name, num_episodes in training_phases:
            logger.info(f"开始{phase_name}阶段: {num_episodes} episodes")
            
            for episode in tqdm(range(num_episodes), desc=f"{phase_name}"):
                episode_count += 1
                
                # 课程学习
                if self.curriculum:
                    stage_config = self.curriculum.get_current_stage()
                    env = self.create_environment(train_data, stage_config)
                else:
                    env = self.create_environment(train_data)
                
                # 训练单个episode
                episode_metrics = self.train_single_episode(agent, env, episode_count)
                
                # 记录训练指标
                self.tracker.log_metrics(episode_metrics, episode_count, phase_name)
                self.training_history.append(episode_metrics)
                
                # 定期评估
                if episode_count % self.rl_config.eval_frequency == 0:
                    eval_metrics = self.evaluate_agent(agent, eval_data)
                    self.tracker.log_metrics(eval_metrics, episode_count, "eval")
                    
                    # 课程学习进展
                    if self.curriculum:
                        self.curriculum.advance_stage(
                            eval_metrics.get('eval_sharpe_ratio', 0)
                        )
                    
                    # 检查早停
                    if self.check_early_stopping(eval_metrics):
                        logger.info("早停触发，结束训练")
                        break
                    
                    # 保存检查点
                    if episode_count % self.rl_config.save_frequency == 0:
                        combined_metrics = {**episode_metrics, **eval_metrics}
                        self.save_checkpoint(agent, episode_count, combined_metrics)
                
                # 打印进度
                if episode_count % 25 == 0:
                    logger.info(f"Episode {episode_count}: "
                              f"Reward={episode_metrics['total_reward']:.4f}, "
                              f"Sharpe={episode_metrics.get('sharpe_ratio', 0):.4f}")
        
        # 最终评估
        final_eval_metrics = self.evaluate_agent(agent, eval_data)
        self.tracker.log_metrics(final_eval_metrics, episode_count, "final_eval")
        
        # 保存最终模型
        self.save_checkpoint(agent, episode_count, final_eval_metrics)
        
        # 保存训练历史
        self.tracker.save_artifact(self.training_history, "training_history", "data")
        
        logger.info("强化学习训练完成")
        
        return {
            'final_metrics': final_eval_metrics,
            'training_history': self.training_history,
            'best_performance': self.best_performance,
            'total_episodes': episode_count
        }
    
    def run_hyperparameter_optimization(self, data_path: str) -> Dict[str, Any]:
        """运行超参数优化"""
        if not self.optimizer:
            raise ValueError("超参数优化器未初始化")
        
        def objective(trial):
            hyperparameters = self.optimizer.suggest_hyperparameters(trial)
            
            # 运行简化训练
            old_config = self.config
            self.config = RLTrainingConfig(
                num_pretraining_episodes=10,
                num_main_training_episodes=20,
                num_fine_tuning_episodes=5,
                eval_frequency=10
            )
            
            try:
                results = self.run_training(data_path, hyperparameters=hyperparameters)
                performance = results['final_metrics'].get('eval_sharpe_ratio', 0)
                
                # 报告中间结果用于剪枝
                trial.report(performance, self.config.num_main_training_episodes)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return performance
            
            except Exception as e:
                logger.error(f"Trial失败: {e}")
                return -np.inf
            
            finally:
                self.config = old_config
        
        # 执行优化
        best_params = self.optimizer.optimize(objective, self.config.n_trials)
        
        # 记录最佳超参数
        self.tracker.log_hyperparameters({"best_hyperparameters": best_params})
        
        return best_params
    
    def run_robust_backtest(self, data_path: str, 
                           model_path: Optional[str] = None) -> Dict[str, Any]:
        """运行稳健回测"""
        logger.info("开始稳健回测...")
        
        # 准备数据
        data = self.prepare_data(data_path)
        
        # 加载模型（如果提供）
        if model_path:
            checkpoint = torch.load(model_path)
            initial_env = self.create_environment(data)
            agent = self.create_agent(
                initial_env.get_state_size(), 
                initial_env.get_action_space_size()
            )
            agent.actor.load_state_dict(checkpoint['agent_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        else:
            # 使用最佳模型
            best_model_path = os.path.join(self.rl_config.checkpoint_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                return self.run_robust_backtest(data_path, best_model_path)
            else:
                raise ValueError("未找到训练好的模型")
        
        # 配置回测
        backtest_config = BacktestConfig(
            train_window_size=252 * 2,
            test_window_size=63,
            step_size=21,
            embargo_period=5
        )
        
        # 创建回测器
        backtester = RobustBacktester(backtest_config)
        
        # 运行RL回测
        environment_config = {
            'initial_capital': self.rl_config.initial_cash,
            'transaction_cost': self.rl_config.transaction_cost_bps / 10000,  # 转换bps为小数
            'max_position': self.rl_config.max_position
        }
        
        backtest_results = backtester.run_rl_backtest(
            data, agent, environment_config
        )
        
        # 生成报告
        report = backtester.generate_report()
        self.tracker.save_artifact(report, "backtest_report", "data")
        self.tracker.save_artifact(backtest_results, "backtest_results", "data")
        
        logger.info("稳健回测完成")
        return backtest_results


def main():
    """主函数 - 演示完整的训练流程"""
    # 创建配置
    config = RLTrainingConfig(
        num_pretraining_episodes=20,
        num_main_training_episodes=100,
        num_fine_tuning_episodes=20,
        experiment_name="demo_rl_trading",
        use_hyperparameter_tuning=False,
        curriculum_learning=True
    )
    
    # 创建训练管道
    pipeline = RLTrainingPipeline(config)
    
    # 模拟数据路径（实际使用时替换为真实路径）
    data_path = "processed_data/featured_data_reduced.parquet"
    
    try:
        # 检查数据文件是否存在
        if not os.path.exists(data_path):
            logger.warning(f"数据文件不存在: {data_path}")
            logger.info("创建模拟数据进行演示...")
            
            # 创建模拟数据
            np.random.seed(42)
            n_samples = 2000
            
            returns = np.random.normal(0.0001, 0.02, n_samples)
            prices = 100 * np.exp(np.cumsum(returns))
            
            demo_data = pd.DataFrame({
                'close': prices,
                'volume': np.random.lognormal(10, 0.5, n_samples),
                'signal_confidence': np.random.uniform(0.4, 0.8, n_samples),
                'signal_direction': np.random.choice([-1, 0, 1], n_samples),
                'feature_1': np.random.randn(n_samples),
                'feature_2': np.random.randn(n_samples),
                'returns': returns
            })
            
            # 保存模拟数据
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            demo_data.to_parquet(data_path)
        
        # 运行训练
        if config.use_hyperparameter_tuning:
            logger.info("开始超参数优化...")
            best_params = pipeline.run_hyperparameter_optimization(data_path)
            logger.info(f"最佳超参数: {best_params}")
            
            # 使用最佳超参数重新训练
            results = pipeline.run_training(data_path, hyperparameters=best_params)
        else:
            results = pipeline.run_training(data_path)
        
        # 运行稳健回测
        backtest_results = pipeline.run_robust_backtest(data_path)
        
        # 输出结果摘要
        logger.info("训练完成!")
        logger.info(f"最终性能: 夏普比率={results['final_metrics'].get('eval_sharpe_ratio', 0):.4f}")
        logger.info(f"训练轮数: {results['total_episodes']}")
        logger.info(f"回测结果: {len(backtest_results.get('rl_fold_results', []))} 折验证")
        
        return results, backtest_results
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


if __name__ == "__main__":
    # 运行演示
    results, backtest_results = main() 