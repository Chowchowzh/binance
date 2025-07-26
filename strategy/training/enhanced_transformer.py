#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版Transformer模型 - 专为金融时间序列设计的高级架构

该模块实现了针对金融深度学习优化的Transformer架构，包括：
1. 激进的正则化策略：多层Dropout、DropPath、权重衰减
2. 稳定性改进：LayerNorm、梯度裁剪、残差连接
3. 注意力优化：多头注意力、相对位置编码
4. 早停和自适应学习率
5. 金融特有的归一化和特征处理

特点：
- 专为低信噪比的金融数据设计
- 内置过拟合防护机制  
- 支持与传统模型（LSTM/GRU）的性能对比
- 可解释性增强（注意力权重可视化）

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Transformer配置参数"""
    # 基础架构
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 3
    dim_feedforward: int = 512
    max_seq_length: int = 100
    
    # 正则化参数
    dropout: float = 0.3
    attention_dropout: float = 0.2
    feedforward_dropout: float = 0.3
    path_dropout: float = 0.1  # DropPath/Stochastic Depth
    weight_decay: float = 1e-4
    
    # 训练稳定性
    layer_norm_eps: float = 1e-6
    use_pre_norm: bool = True  # Pre-LayerNorm vs Post-LayerNorm
    gradient_clip_val: float = 1.0
    
    # 位置编码
    use_relative_position: bool = True
    max_relative_position: int = 32
    
    # 其他
    activation: str = 'gelu'  # 'relu', 'gelu', 'swish'
    use_batch_norm: bool = False  # 在金融数据中通常不推荐
    
    # 早停参数
    patience: int = 10
    min_delta: float = 1e-4


class DropPath(nn.Module):
    """DropPath (Stochastic Depth) 正则化
    
    在训练过程中随机丢弃整个残差路径，强迫网络学习冗余表示
    """
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class RelativePositionalEncoding(nn.Module):
    """相对位置编码
    
    相比于绝对位置编码，相对位置编码能更好地捕捉时间序列中的相对关系
    """
    def __init__(self, d_model: int, max_relative_position: int = 32):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 创建相对位置嵌入
        self.relative_position_embedding = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
    def forward(self, seq_len: int):
        """生成相对位置编码"""
        # 创建相对位置矩阵
        range_vec = torch.arange(seq_len)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        
        # 裁剪到最大相对距离
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # 转换为非负索引
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # 获取嵌入
        embeddings = self.relative_position_embedding(final_mat)
        return embeddings


class EnhancedMultiHeadAttention(nn.Module):
    """增强的多头注意力机制
    
    包含相对位置编码和注意力Dropout
    """
    def __init__(self, config: TransformerConfig):
        super(EnhancedMultiHeadAttention, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.d_k = config.d_model // config.nhead
        
        assert config.d_model % config.nhead == 0
        
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.dropout = nn.Dropout(config.dropout)
        
        # 相对位置编码
        if config.use_relative_position:
            self.relative_position_encoding = RelativePositionalEncoding(
                self.d_k, config.max_relative_position
            )
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # 线性变换和重塑
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # 注意力计算
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 添加相对位置编码
        if self.config.use_relative_position:
            rel_pos_embed = self.relative_position_encoding(seq_len)
            rel_pos_scores = torch.matmul(Q, rel_pos_embed.transpose(-2, -1)) / math.sqrt(self.d_k)
            attention_scores = attention_scores + rel_pos_scores
        
        # 应用mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(context)
        return self.dropout(output), attention_weights


class EnhancedTransformerLayer(nn.Module):
    """增强的Transformer编码器层
    
    包含Pre-LayerNorm、DropPath和改进的前馈网络
    """
    def __init__(self, config: TransformerConfig):
        super(EnhancedTransformerLayer, self).__init__()
        self.config = config
        
        # 多头注意力
        self.attention = EnhancedMultiHeadAttention(config)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            self._get_activation(config.activation),
            nn.Dropout(config.feedforward_dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # DropPath
        self.drop_path = DropPath(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
        
    def _get_activation(self, activation: str):
        """获取激活函数"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def forward(self, x, mask=None):
        # Pre-LayerNorm架构
        if self.config.use_pre_norm:
            # 注意力子层
            norm_x = self.norm1(x)
            attn_output, attn_weights = self.attention(norm_x, norm_x, norm_x, mask)
            x = x + self.drop_path(attn_output)
            
            # 前馈子层
            norm_x = self.norm2(x)
            ff_output = self.feed_forward(norm_x)
            x = x + self.drop_path(ff_output)
        else:
            # Post-LayerNorm架构（传统）
            attn_output, attn_weights = self.attention(x, x, x, mask)
            x = self.norm1(x + self.drop_path(attn_output))
            
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.drop_path(ff_output))
        
        return x, attn_weights


class FinancialFeatureEmbedding(nn.Module):
    """金融特征嵌入层
    
    专门处理金融时间序列的特征嵌入和归一化
    """
    def __init__(self, num_features: int, d_model: int, config: TransformerConfig):
        super(FinancialFeatureEmbedding, self).__init__()
        self.num_features = num_features
        self.d_model = d_model
        
        # 特征投影
        self.feature_projection = nn.Linear(num_features, d_model)
        
        # 可学习的特征缩放
        self.feature_scale = nn.Parameter(torch.ones(num_features))
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # 批归一化（可选，一般不用于Transformer）
        if config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features)
        else:
            self.batch_norm = None
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        
        # 可选的批归一化
        if self.batch_norm is not None:
            # 需要调整维度用于BatchNorm
            x = x.transpose(1, 2)  # (batch_size, num_features, seq_len)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)  # (batch_size, seq_len, num_features)
        
        # 特征缩放
        x = x * self.feature_scale
        
        # 投影到模型维度
        x = self.feature_projection(x) * math.sqrt(self.d_model)
        
        return self.dropout(x)


class EnhancedTimeSeriesTransformer(nn.Module):
    """增强版时间序列Transformer
    
    专为金融数据设计的高级Transformer架构
    """
    def __init__(self, num_features: int, num_classes: int, config: Optional[TransformerConfig] = None):
        super(EnhancedTimeSeriesTransformer, self).__init__()
        
        self.config = config or TransformerConfig()
        self.num_features = num_features
        self.num_classes = num_classes
        
        # 特征嵌入
        self.feature_embedding = FinancialFeatureEmbedding(
            num_features, self.config.d_model, self.config
        )
        
        # 位置编码
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(self.config) 
            for _ in range(self.config.num_encoder_layers)
        ])
        
        # 最终归一化
        if self.config.use_pre_norm:
            self.final_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            self._get_activation(self.config.activation),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info(f"初始化增强版Transformer: 特征={num_features}, 类别={num_classes}, 参数={self._count_parameters()}")
    
    def _create_positional_encoding(self):
        """创建位置编码"""
        pe = torch.zeros(self.config.max_seq_length, self.config.d_model)
        position = torch.arange(0, self.config.max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.config.d_model, 2).float() * 
                           (-math.log(10000.0) / self.config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)
        
        self.register_buffer('pe', pe)
        return nn.Dropout(self.config.dropout)
    
    def _get_activation(self, activation: str):
        """获取激活函数"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化用于线性层
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm参数初始化
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Embedding):
                # 嵌入层初始化
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def _count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x, mask=None, return_attention=False):
        """前向传播
        
        Args:
            x: 输入特征 (batch_size, seq_len, num_features)
            mask: 注意力mask (batch_size, seq_len, seq_len)
            return_attention: 是否返回注意力权重
        
        Returns:
            logits: 分类输出 (batch_size, num_classes)
            attention_weights: 注意力权重（可选）
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 特征嵌入
        x = self.feature_embedding(x)
        
        # 2. 位置编码
        x = x + self.pe[:, :seq_len, :]
        x = self.positional_encoding(x)
        
        # 3. Transformer编码器
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, mask)
            if return_attention:
                attention_weights.append(attn_weights)
        
        # 4. 最终归一化
        if self.config.use_pre_norm:
            x = self.final_norm(x)
        
        # 5. 全局池化（使用最后一个时间步）
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # 6. 分类
        logits = self.classifier(x)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits
    
    def get_attention_weights(self, x, mask=None):
        """获取注意力权重用于可解释性分析"""
        with torch.no_grad():
            _, attention_weights = self.forward(x, mask, return_attention=True)
        return attention_weights


class ModelComparison:
    """模型对比工具类
    
    用于比较Transformer与传统模型（LSTM/GRU）的性能
    """
    def __init__(self):
        pass
    
    @staticmethod
    def create_lstm_baseline(num_features: int, num_classes: int, hidden_dim: int = 128, 
                           num_layers: int = 2, dropout: float = 0.3):
        """创建LSTM基线模型"""
        return nn.Sequential(
            nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            ),
            nn.Linear(hidden_dim, num_classes)
        )
    
    @staticmethod
    def create_gru_baseline(num_features: int, num_classes: int, hidden_dim: int = 128, 
                          num_layers: int = 2, dropout: float = 0.3):
        """创建GRU基线模型"""
        class GRUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=num_features,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_first=True
                )
                self.classifier = nn.Linear(hidden_dim, num_classes)
                
            def forward(self, x):
                output, _ = self.gru(x)
                return self.classifier(output[:, -1, :])
        
        return GRUModel()


class EarlyStopping:
    """早停回调类"""
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def _is_better(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


if __name__ == "__main__":
    # 测试增强版Transformer
    print("测试增强版Transformer...")
    
    # 创建配置
    config = TransformerConfig(
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dropout=0.2,
        use_relative_position=True
    )
    
    # 创建模型
    num_features = 50
    num_classes = 3
    model = EnhancedTimeSeriesTransformer(num_features, num_classes, config)
    
    # 创建测试数据
    batch_size, seq_len = 32, 60
    x = torch.randn(batch_size, seq_len, num_features)
    
    # 前向传播
    with torch.no_grad():
        logits = model(x)
        print(f"输出形状: {logits.shape}")
        
        # 测试注意力权重
        attention_weights = model.get_attention_weights(x)
        print(f"注意力层数: {len(attention_weights)}")
        print(f"注意力权重形状: {attention_weights[0].shape}")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 创建基线模型对比
    lstm_model = ModelComparison.create_lstm_baseline(num_features, num_classes)
    gru_model = ModelComparison.create_gru_baseline(num_features, num_classes)
    
    print(f"\n模型对比:")
    print(f"  Transformer: {trainable_params:,} 参数")
    print(f"  LSTM: {sum(p.numel() for p in lstm_model.parameters() if p.requires_grad):,} 参数") 
    print(f"  GRU: {sum(p.numel() for p in gru_model.parameters() if p.requires_grad):,} 参数")
    
    print("增强版Transformer测试完成！") 