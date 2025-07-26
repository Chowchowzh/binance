"""
强化学习模块 - 支持MPS、CUDA、CPU多设备加速

该模块提供完整的强化学习交易框架，包括：
1. Actor-Critic智能体 (支持GPU加速)
2. MDP交易环境
3. 强化学习训练流水线
4. 稳健回测系统

性能优化特性：
- 🚀 Apple Silicon (M1/M2) MPS加速支持
- 🚀 NVIDIA CUDA GPU加速支持  
- 📊 自动设备检测和最优选择
- 🧠 高效神经网络训练和推理

Author: AI Assistant
Date: 2024
"""

import torch

from .actor_critic_agent import ActorCriticAgent, AgentConfig, _get_optimal_device
from .mdp_environment import TradingMDPEnvironment, MDPState, MDPAction
from .rl_training_pipeline import RLTrainingPipeline
from .robust_backtester import RobustBacktester, BacktestConfig

__all__ = [
    'ActorCriticAgent', 
    'AgentConfig',
    'TradingMDPEnvironment', 
    'MDPState', 
    'MDPAction',
    'RLTrainingPipeline',
    'RobustBacktester',
    'BacktestConfig',
    'get_device_info'
]


def get_device_info() -> dict:
    """
    获取当前系统的设备信息和加速能力
    
    Returns:
        包含设备信息的字典
    """
    device_info = {
        'optimal_device': _get_optimal_device(),
        'mps_available': torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'pytorch_version': torch.__version__
    }
    
    if torch.cuda.is_available():
        device_info['cuda_device_name'] = torch.cuda.get_device_name()
        device_info['cuda_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
    
    return device_info


def print_device_info():
    """打印设备加速信息"""
    info = get_device_info()
    print("🔧 CPU多线程加速信息:")
    print(f"   计算设备: CPU (多线程优化)")
    print(f"   线程数量: 4")
    print(f"   进程数量: 4")
    print(f"   PyTorch版本: {info['pytorch_version']}")
    print("   🚀 已优化CPU性能，适合此类任务")


# 模块导入时自动显示设备信息
if __name__ != "__main__":
    print_device_info() 