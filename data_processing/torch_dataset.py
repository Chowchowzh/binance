# -*- coding: utf-8 -*-
"""
PyTorch时间序列数据集模块
用于模型训练的数据集类
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_processing.preprocessor import DataPreprocessor


class TimeSeriesDataset(Dataset):
    """时间序列数据集类 - 使用已有的预处理器"""
    
    def __init__(self, data_path: str, sequence_length: int, 
                 preprocessor: DataPreprocessor = None,
                 scaler=None,
                 config_path: str = 'config/config.json'):
        """
        初始化时间序列数据集
        
        Args:
            data_path: 特征数据文件路径（如果为None则使用预处理器的输出）
            sequence_length: 序列长度
            preprocessor: 数据预处理器实例（可选）
            config_path: 配置文件路径
        """
        self.sequence_length = sequence_length
        self.config_path = config_path
        
        # 如果提供了预处理器，使用它；否则创建新的
        if preprocessor is None:
            self.preprocessor = DataPreprocessor(config_path)
        else:
            self.preprocessor = preprocessor
        
        # 确定数据路径
        if data_path is None:
            data_path = self.preprocessor.get_featured_data_path()
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 加载数据（假设预处理器已经处理了标准化）
        print(f"从预处理数据加载: {data_path}")
        df = pd.read_parquet(data_path)
        
        # 分离特征和目标
        if 'target' in df.columns:
            feature_columns = [col for col in df.columns 
                             if col not in ['target', 'open_time', 'close_time', 'future_return']]
            self.features = df[feature_columns].values.astype(np.float32)
            self.targets = df['target'].values.astype(np.int64)
        else:
            raise ValueError("数据中未找到'target'列")
        
        # 数据已经在预处理阶段标准化，这里只需要清理
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"数据形状: features={self.features.shape}, targets={self.targets.shape}")
        
        # 创建序列
        self._create_sequences()
    
    def _create_sequences(self):
        """创建时间序列数据"""
        sequences = []
        labels = []
        
        for i in range(len(self.features) - self.sequence_length):
            seq = self.features[i:i + self.sequence_length]
            label = self.targets[i + self.sequence_length]
            sequences.append(seq)
            labels.append(label)
        
        self.sequences = np.array(sequences, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.labels[idx]) 