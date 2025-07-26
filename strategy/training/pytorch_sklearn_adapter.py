# -*- coding: utf-8 -*-
"""
PyTorch模型的sklearn适配器
将PyTorch模型包装成sklearn风格的接口，支持fit/predict方法
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class PyTorchSklearnAdapter(BaseEstimator, ClassifierMixin):
    """
    PyTorch模型的sklearn适配器
    
    将PyTorch模型包装成sklearn接口，提供fit/predict方法
    """
    
    def __init__(self,
                 model_class,
                 model_params: dict = None,
                 sequence_length: int = 60,
                 epochs: int = 50,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 device: str = 'auto',
                 verbose: bool = False):
        """
        初始化适配器
        
        Args:
            model_class: PyTorch模型类
            model_params: 模型初始化参数
            sequence_length: 序列长度
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            device: 设备选择
            verbose: 是否详细输出
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 初始化组件
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        self.num_features = None
        self.num_classes = None
        
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> Union[DataLoader, Tuple[DataLoader, int]]:
        """准备数据加载器"""
        # 确保X是3D数组 (samples, sequence_length, features)
        if X.ndim == 2:
            # 假设输入是 (samples, features)，需要重塑为序列
            n_samples, n_features = X.shape
            if n_samples < self.sequence_length:
                # 如果样本数少于序列长度，重复填充
                X_padded = np.tile(X, (self.sequence_length // n_samples + 1, 1))[:self.sequence_length]
                X = X_padded.reshape(1, self.sequence_length, n_features)
                # 对应的标签处理
                if y is not None:
                    y = np.array([y[-1]])  # 使用最后一个标签
            else:
                # 创建滑动窗口序列
                sequences = []
                sequence_labels = []
                for i in range(n_samples - self.sequence_length + 1):
                    sequences.append(X[i:i + self.sequence_length])
                    # 对应的标签是序列的最后一个时间步的标签
                    if y is not None:
                        sequence_labels.append(y[i + self.sequence_length - 1])
                
                X = np.array(sequences)
                if y is not None:
                    y = np.array(sequence_labels)
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        if y is not None:
            # 编码标签
            y_encoded = self.label_encoder.fit_transform(y)
            y_tensor = torch.LongTensor(y_encoded).to(self.device)
            
            # 现在X和y的长度应该已经匹配了
            assert len(X_tensor) == len(y_tensor), f"特征样本数 {len(X_tensor)} 与标签样本数 {len(y_tensor)} 不匹配"
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return dataloader, len(self.label_encoder.classes_)
        else:
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            return dataloader
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
            sample_weight: 样本权重（暂不支持）
        """
        if self.verbose:
            logger.info(f"开始训练PyTorch模型，输入维度: {X.shape}")
        
        # 准备数据
        train_loader, num_classes = self._prepare_data(X, y)
        
        # 存储特征信息
        if X.ndim == 3:
            self.num_features = X.shape[2]
        else:
            self.num_features = X.shape[1]
        self.num_classes = num_classes
        
        # 初始化模型
        model_params = self.model_params.copy()
        model_params.update({
            'num_features': self.num_features,
            'num_classes': self.num_classes
        })
        
        self.model = self.model_class(**model_params).to(self.device)
        
        # 初始化优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 前向传播
                output = self.model(data)
                loss = criterion(output, target)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                acc = 100. * correct / total
                avg_loss = total_loss / len(train_loader)
                logger.info(f'Epoch {epoch+1}/{self.epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%')
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签
        
        Args:
            X: 测试特征
            
        Returns:
            预测标签
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 保存原始样本数量用于输出调整
        original_samples = len(X)
        
        # 准备数据
        test_loader = self._prepare_data(X)
        
        # 预测
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for (data,) in test_loader:
                output = self.model(data)
                _, predicted = output.max(1)
                predictions.extend(predicted.cpu().numpy())
        
        # 解码标签
        predictions = np.array(predictions)
        
        # 如果输入是2D且经过了滑动窗口处理，需要调整输出长度
        if len(predictions) != original_samples and original_samples >= self.sequence_length:
            # 对于滑动窗口处理，我们有 original_samples - sequence_length + 1 个预测
            # 需要为前面的 sequence_length - 1 个样本生成预测
            if len(predictions) > 0:
                # 使用第一个预测值填充前面的样本
                first_pred = predictions[0]
                padding = np.full(self.sequence_length - 1, first_pred)
                predictions = np.concatenate([padding, predictions])
        
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 测试特征
            
        Returns:
            类别概率
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 保存原始样本数量用于输出调整
        original_samples = len(X)
        
        # 准备数据
        test_loader = self._prepare_data(X)
        
        # 预测
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for (data,) in test_loader:
                output = self.model(data)
                proba = torch.softmax(output, dim=1)
                probabilities.extend(proba.cpu().numpy())
        
        probabilities = np.array(probabilities)
        
        # 如果输入是2D且经过了滑动窗口处理，需要调整输出长度
        if len(probabilities) != original_samples and original_samples >= self.sequence_length:
            # 对于滑动窗口处理，我们有 original_samples - sequence_length + 1 个预测
            # 需要为前面的 sequence_length - 1 个样本生成预测
            if len(probabilities) > 0:
                # 使用第一个预测概率填充前面的样本
                first_proba = probabilities[0]
                padding = np.tile(first_proba, (self.sequence_length - 1, 1))
                probabilities = np.concatenate([padding, probabilities])
        
        return probabilities
    
    def get_params(self, deep=True):
        """获取参数（sklearn接口要求）"""
        return {
            'model_class': self.model_class,
            'model_params': self.model_params,
            'sequence_length': self.sequence_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'device': self.device.type if hasattr(self.device, 'type') else str(self.device),
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """设置参数（sklearn接口要求）"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def create_transformer_adapter(sequence_length: int = 60,
                             epochs: int = 30,
                             batch_size: int = 16,
                             learning_rate: float = 0.001,
                             verbose: bool = False) -> PyTorchSklearnAdapter:
    """
    创建Transformer模型的sklearn适配器
    
    Args:
        sequence_length: 序列长度
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        verbose: 是否详细输出
        
    Returns:
        适配器实例
    """
    from .transformer_model import TimeSeriesTransformer
    
    model_params = {
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1
    }
    
    return PyTorchSklearnAdapter(
        model_class=TimeSeriesTransformer,
        model_params=model_params,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=verbose
    ) 