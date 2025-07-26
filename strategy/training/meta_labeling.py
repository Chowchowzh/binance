# -*- coding: utf-8 -*-
"""
元标签（Meta-Labeling）技术实现
基于Marcos López de Prado的两阶段学习框架
解决信号精度低的问题，提升信噪比
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import pickle
import os
from datetime import datetime

warnings.filterwarnings('ignore')


class MetaLabeler:
    """
    元标签技术实现器
    
    两阶段学习框架：
    1. 主模型：追求高召回率，识别所有潜在机会
    2. 次级模型：追求高精度，过滤主模型的误报
    """
    
    def __init__(self,
                 primary_model=None,
                 meta_model_type: str = 'random_forest',
                 meta_model_params: Dict = None,
                 probability_threshold: float = 0.5,
                 calibration_method: str = 'platt'):
        """
        初始化元标签器
        
        Args:
            primary_model: 主模型（通常是已训练的Transformer）
            meta_model_type: 次级模型类型 ('random_forest', 'gbm', 'logistic', 'neural_net')
            meta_model_params: 次级模型参数
            probability_threshold: 概率阈值
            calibration_method: 概率校准方法
        """
        self.primary_model = primary_model
        self.meta_model_type = meta_model_type
        self.meta_model_params = meta_model_params or {}
        self.probability_threshold = probability_threshold
        self.calibration_method = calibration_method
        
        # 初始化次级模型
        self.meta_model = self._create_meta_model()
        self.meta_scaler = StandardScaler()
        self.is_fitted = False
        
        print(f"初始化元标签器:")
        print(f"  - 次级模型类型: {meta_model_type}")
        print(f"  - 概率阈值: {probability_threshold}")
        print(f"  - 校准方法: {calibration_method}")
    
    def _create_meta_model(self):
        """创建次级模型"""
        if self.meta_model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.meta_model_params)
            return RandomForestClassifier(**default_params)
            
        elif self.meta_model_type == 'gbm':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
            default_params.update(self.meta_model_params)
            return GradientBoostingClassifier(**default_params)
            
        elif self.meta_model_type == 'logistic':
            default_params = {
                'C': 1.0,
                'random_state': 42,
                'max_iter': 1000
            }
            default_params.update(self.meta_model_params)
            return LogisticRegression(**default_params)
            
        elif self.meta_model_type == 'neural_net':
            # 使用PyTorch实现的小型神经网络
            return self._create_neural_meta_model()
            
        else:
            raise ValueError(f"不支持的次级模型类型: {self.meta_model_type}")
    
    def _create_neural_meta_model(self):
        """创建神经网络次级模型"""
        class MetaNet(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
                super(MetaNet, self).__init__()
                layers = []
                
                # 输入层
                layers.append(nn.Linear(input_dim, hidden_dims[0]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                
                # 隐藏层
                for i in range(len(hidden_dims) - 1):
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))
                
                # 输出层
                layers.append(nn.Linear(hidden_dims[-1], 1))
                layers.append(nn.Sigmoid())
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return MetaNet
    
    def generate_primary_predictions(self, 
                                   features: np.ndarray,
                                   device: str = 'cpu',
                                   batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用主模型生成预测
        
        Args:
            features: 特征数组 (n_samples, seq_len, n_features)
            device: 计算设备
            batch_size: 批次大小
            
        Returns:
            预测标签和概率
        """
        if self.primary_model is None:
            raise ValueError("主模型未设置")
        
        print("生成主模型预测...")
        
        self.primary_model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_features).to(device)
                
                # 获取模型输出
                outputs = self.primary_model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        print(f"  - 主模型预测完成，样本数: {len(predictions)}")
        print(f"  - 预测分布: {np.bincount(predictions)}")
        
        return predictions, probabilities
    
    def create_meta_features(self,
                           original_features: np.ndarray,
                           primary_predictions: np.ndarray,
                           primary_probabilities: np.ndarray,
                           additional_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        创建次级模型的特征
        
        Args:
            original_features: 原始特征
            primary_predictions: 主模型预测
            primary_probabilities: 主模型概率
            additional_features: 额外特征
            
        Returns:
            次级模型特征数组
        """
        meta_features_list = []
        
        # 1. 主模型的预测概率（所有类别）
        meta_features_list.append(primary_probabilities)
        
        # 2. 主模型的预测标签（one-hot编码）
        n_classes = primary_probabilities.shape[1]
        pred_onehot = np.eye(n_classes)[primary_predictions]
        meta_features_list.append(pred_onehot)
        
        # 3. 概率的统计特征
        prob_max = np.max(primary_probabilities, axis=1).reshape(-1, 1)
        prob_entropy = -np.sum(primary_probabilities * np.log(primary_probabilities + 1e-10), axis=1).reshape(-1, 1)
        prob_std = np.std(primary_probabilities, axis=1).reshape(-1, 1)
        
        meta_features_list.extend([prob_max, prob_entropy, prob_std])
        
        # 4. 原始特征的聚合统计（取最后一个时间步）
        if len(original_features.shape) == 3:  # (n_samples, seq_len, n_features)
            # 取最后时间步的特征
            last_step_features = original_features[:, -1, :]
            
            # 计算序列的统计特征
            seq_mean = np.mean(original_features, axis=1)
            seq_std = np.std(original_features, axis=1)
            seq_max = np.max(original_features, axis=1)
            seq_min = np.min(original_features, axis=1)
            
            meta_features_list.extend([last_step_features, seq_mean, seq_std, seq_max, seq_min])
        else:
            # 2D特征直接使用
            meta_features_list.append(original_features)
        
        # 5. 添加额外特征
        if additional_features is not None:
            meta_features_list.append(additional_features)
        
        # 合并所有特征
        meta_features = np.concatenate(meta_features_list, axis=1)
        
        print(f"创建次级模型特征: {meta_features.shape}")
        
        return meta_features
    
    def create_meta_labels(self,
                         primary_predictions: np.ndarray,
                         true_labels: np.ndarray,
                         prediction_type: str = 'any_direction') -> np.ndarray:
        """
        创建元标签
        
        Args:
            primary_predictions: 主模型预测
            true_labels: 真实标签
            prediction_type: 预测类型 ('any_direction', 'exact_match', 'profitable_only')
            
        Returns:
            元标签数组 (1=主模型正确, 0=主模型错误)
        """
        if prediction_type == 'exact_match':
            # 精确匹配：主模型预测必须完全正确
            meta_labels = (primary_predictions == true_labels).astype(int)
            
        elif prediction_type == 'any_direction':
            # 方向正确：只要方向对就算正确（不区分中性）
            # 将三分类转为二分类方向
            primary_direction = np.sign(primary_predictions - 1)  # -1, 0, 1 -> -1, -1, 0
            true_direction = np.sign(true_labels - 1)
            
            # 如果预测和真实都是非零且符号相同，或都是零，则正确
            meta_labels = ((primary_direction * true_direction > 0) | 
                          ((primary_direction == 0) & (true_direction == 0))).astype(int)
            
        elif prediction_type == 'profitable_only':
            # 仅盈利：主模型预测为盈利方向且真实确实盈利
            profitable_pred = (primary_predictions != 1)  # 非中性预测
            profitable_true = (true_labels != 1)  # 非中性真实
            direction_correct = (primary_predictions == true_labels)
            
            meta_labels = (profitable_pred & profitable_true & direction_correct).astype(int)
            
        else:
            raise ValueError(f"不支持的预测类型: {prediction_type}")
        
        positive_rate = np.mean(meta_labels)
        print(f"元标签创建完成:")
        print(f"  - 正确率: {positive_rate:.4f}")
        print(f"  - 正确样本数: {np.sum(meta_labels)}")
        print(f"  - 总样本数: {len(meta_labels)}")
        
        return meta_labels
    
    def fit_meta_model(self,
                      meta_features: np.ndarray,
                      meta_labels: np.ndarray,
                      validation_split: float = 0.2) -> Dict[str, Any]:
        """
        训练次级模型
        
        Args:
            meta_features: 次级模型特征
            meta_labels: 元标签
            validation_split: 验证集比例
            
        Returns:
            训练结果统计
        """
        print("开始训练次级模型...")
        
        # 数据预处理
        meta_features = np.nan_to_num(meta_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 特征标准化
        meta_features_scaled = self.meta_scaler.fit_transform(meta_features)
        
        # 数据集划分
        n_samples = len(meta_features_scaled)
        n_train = int(n_samples * (1 - validation_split))
        
        train_features = meta_features_scaled[:n_train]
        train_labels = meta_labels[:n_train]
        val_features = meta_features_scaled[n_train:]
        val_labels = meta_labels[n_train:]
        
        print(f"  - 训练样本: {len(train_features)}")
        print(f"  - 验证样本: {len(val_features)}")
        print(f"  - 特征维度: {meta_features_scaled.shape[1]}")
        
        # 训练模型
        if self.meta_model_type == 'neural_net':
            results = self._fit_neural_meta_model(train_features, train_labels, val_features, val_labels)
        else:
            results = self._fit_sklearn_meta_model(train_features, train_labels, val_features, val_labels)
        
        self.is_fitted = True
        print("次级模型训练完成")
        
        return results
    
    def _fit_sklearn_meta_model(self, train_features, train_labels, val_features, val_labels):
        """训练sklearn次级模型"""
        # 训练模型
        self.meta_model.fit(train_features, train_labels)
        
        # 验证集评估
        val_pred_proba = self.meta_model.predict_proba(val_features)[:, 1]
        val_predictions = (val_pred_proba >= self.probability_threshold).astype(int)
        
        # 计算指标
        precision = np.mean(val_labels[val_predictions == 1]) if np.sum(val_predictions) > 0 else 0
        recall = np.mean(val_predictions[val_labels == 1]) if np.sum(val_labels) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        auc = roc_auc_score(val_labels, val_pred_proba) if len(np.unique(val_labels)) > 1 else 0.5
        
        # 交叉验证
        cv_scores = cross_val_score(self.meta_model, train_features, train_labels, 
                                   cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
        
        results = {
            'validation_precision': precision,
            'validation_recall': recall,
            'validation_f1': f1,
            'validation_auc': auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'feature_importance': getattr(self.meta_model, 'feature_importances_', None)
        }
        
        print(f"  - 验证精度: {precision:.4f}")
        print(f"  - 验证召回: {recall:.4f}")
        print(f"  - 验证F1: {f1:.4f}")
        print(f"  - 验证AUC: {auc:.4f}")
        print(f"  - 交叉验证AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results
    
    def _fit_neural_meta_model(self, train_features, train_labels, val_features, val_labels):
        """训练神经网络次级模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        input_dim = train_features.shape[1]
        self.meta_model = self.meta_model(input_dim).to(device)
        
        # 转换为张量
        train_X = torch.FloatTensor(train_features).to(device)
        train_y = torch.FloatTensor(train_labels).unsqueeze(1).to(device)
        val_X = torch.FloatTensor(val_features).to(device)
        val_y = torch.FloatTensor(val_labels).unsqueeze(1).to(device)
        
        # 训练设置
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.001)
        
        best_val_auc = 0
        patience = 10
        patience_counter = 0
        
        # 训练循环
        for epoch in range(100):
            self.meta_model.train()
            optimizer.zero_grad()
            
            outputs = self.meta_model(train_X)
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()
            
            # 验证
            if epoch % 5 == 0:
                self.meta_model.eval()
                with torch.no_grad():
                    val_outputs = self.meta_model(val_X)
                    val_pred_proba = val_outputs.cpu().numpy().flatten()
                    
                    if len(np.unique(val_labels)) > 1:
                        val_auc = roc_auc_score(val_labels, val_pred_proba)
                        
                        if val_auc > best_val_auc:
                            best_val_auc = val_auc
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= patience:
                            break
        
        # 最终评估
        self.meta_model.eval()
        with torch.no_grad():
            val_outputs = self.meta_model(val_X)
            val_pred_proba = val_outputs.cpu().numpy().flatten()
            val_predictions = (val_pred_proba >= self.probability_threshold).astype(int)
        
        precision = np.mean(val_labels[val_predictions == 1]) if np.sum(val_predictions) > 0 else 0
        recall = np.mean(val_predictions[val_labels == 1]) if np.sum(val_labels) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        auc = roc_auc_score(val_labels, val_pred_proba) if len(np.unique(val_labels)) > 1 else 0.5
        
        results = {
            'validation_precision': precision,
            'validation_recall': recall,
            'validation_f1': f1,
            'validation_auc': auc,
            'training_epochs': epoch + 1
        }
        
        print(f"  - 训练轮数: {epoch + 1}")
        print(f"  - 验证精度: {precision:.4f}")
        print(f"  - 验证召回: {recall:.4f}")
        print(f"  - 验证AUC: {auc:.4f}")
        
        return results
    
    def predict_meta_confidence(self,
                              meta_features: np.ndarray,
                              return_probabilities: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        使用次级模型预测置信度
        
        Args:
            meta_features: 次级模型特征
            return_probabilities: 是否返回概率
            
        Returns:
            预测结果，如果return_probabilities=True则返回(预测, 概率)
        """
        if not self.is_fitted:
            raise ValueError("次级模型未训练")
        
        # 特征预处理
        meta_features = np.nan_to_num(meta_features, nan=0.0, posinf=1e6, neginf=-1e6)
        meta_features_scaled = self.meta_scaler.transform(meta_features)
        
        # 预测
        if self.meta_model_type == 'neural_net':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.meta_model.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(meta_features_scaled).to(device)
                probabilities = self.meta_model(X_tensor).cpu().numpy().flatten()
        else:
            probabilities = self.meta_model.predict_proba(meta_features_scaled)[:, 1]
        
        predictions = (probabilities >= self.probability_threshold).astype(int)
        
        if return_probabilities:
            return predictions, probabilities
        else:
            return predictions
    
    def generate_filtered_signals(self,
                                 original_features: np.ndarray,
                                 device: str = 'cpu') -> Dict[str, np.ndarray]:
        """
        生成经过元标签过滤的高质量信号
        
        Args:
            original_features: 原始特征
            device: 计算设备
            
        Returns:
            包含原始信号和过滤信号的字典
        """
        print("生成过滤后的高质量信号...")
        
        # 1. 主模型预测
        primary_preds, primary_probs = self.generate_primary_predictions(original_features, device)
        
        # 2. 创建次级模型特征
        meta_features = self.create_meta_features(original_features, primary_preds, primary_probs)
        
        # 3. 次级模型预测置信度
        confidence_preds, confidence_probs = self.predict_meta_confidence(meta_features)
        
        # 4. 生成过滤信号
        # 原始信号：主模型的预测概率差
        if primary_probs.shape[1] == 3:  # 三分类
            raw_signals = primary_probs[:, 2] - primary_probs[:, 0]  # P(up) - P(down)
        else:
            raw_signals = primary_probs[:, 1] - primary_probs[:, 0]
        
        # 过滤信号：原始信号 * 置信度
        filtered_signals = raw_signals * confidence_probs
        
        # 高置信度信号：只保留高置信度的预测
        high_confidence_mask = confidence_probs >= self.probability_threshold
        high_conf_signals = np.where(high_confidence_mask, raw_signals, 0)
        
        results = {
            'primary_predictions': primary_preds,
            'primary_probabilities': primary_probs,
            'confidence_predictions': confidence_preds,
            'confidence_probabilities': confidence_probs,
            'raw_signals': raw_signals,
            'filtered_signals': filtered_signals,
            'high_confidence_signals': high_conf_signals,
            'high_confidence_mask': high_confidence_mask
        }
        
        # 统计信息
        signal_stats = {
            'raw_signal_mean': np.mean(np.abs(raw_signals)),
            'filtered_signal_mean': np.mean(np.abs(filtered_signals)),
            'high_conf_signal_ratio': np.mean(high_confidence_mask),
            'avg_confidence': np.mean(confidence_probs)
        }
        
        print(f"信号生成完成:")
        print(f"  - 原始信号强度: {signal_stats['raw_signal_mean']:.4f}")
        print(f"  - 过滤信号强度: {signal_stats['filtered_signal_mean']:.4f}")
        print(f"  - 高置信度比例: {signal_stats['high_conf_signal_ratio']:.4f}")
        print(f"  - 平均置信度: {signal_stats['avg_confidence']:.4f}")
        
        results['statistics'] = signal_stats
        
        return results
    
    def save_meta_model(self, save_path: str):
        """保存元标签模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'meta_model_type': self.meta_model_type,
            'meta_model_params': self.meta_model_params,
            'probability_threshold': self.probability_threshold,
            'calibration_method': self.calibration_method,
            'meta_scaler': self.meta_scaler,
            'is_fitted': self.is_fitted
        }
        
        if self.meta_model_type == 'neural_net':
            model_data['meta_model_state_dict'] = self.meta_model.state_dict()
        else:
            model_data['meta_model'] = self.meta_model
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"元标签模型已保存: {save_path}")
    
    def load_meta_model(self, load_path: str):
        """加载元标签模型"""
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.meta_model_type = model_data['meta_model_type']
        self.meta_model_params = model_data['meta_model_params']
        self.probability_threshold = model_data['probability_threshold']
        self.calibration_method = model_data['calibration_method']
        self.meta_scaler = model_data['meta_scaler']
        self.is_fitted = model_data['is_fitted']
        
        if self.meta_model_type == 'neural_net':
            # 重建神经网络模型
            self.meta_model = self._create_neural_meta_model()
            self.meta_model.load_state_dict(model_data['meta_model_state_dict'])
        else:
            self.meta_model = model_data['meta_model']
        
        print(f"元标签模型已加载: {load_path}")


# 便利函数
def create_meta_labeling_pipeline(primary_model,
                                 features: np.ndarray,
                                 true_labels: np.ndarray,
                                 meta_model_type: str = 'random_forest',
                                 device: str = 'cpu') -> MetaLabeler:
    """
    创建完整的元标签流水线
    
    Args:
        primary_model: 主模型
        features: 训练特征
        true_labels: 真实标签
        meta_model_type: 次级模型类型
        device: 计算设备
        
    Returns:
        训练好的MetaLabeler
    """
    print("创建元标签流水线...")
    
    # 初始化元标签器
    meta_labeler = MetaLabeler(
        primary_model=primary_model,
        meta_model_type=meta_model_type
    )
    
    # 生成主模型预测
    primary_preds, primary_probs = meta_labeler.generate_primary_predictions(features, device)
    
    # 创建次级模型特征
    meta_features = meta_labeler.create_meta_features(features, primary_preds, primary_probs)
    
    # 创建元标签
    meta_labels = meta_labeler.create_meta_labels(primary_preds, true_labels)
    
    # 训练次级模型
    training_results = meta_labeler.fit_meta_model(meta_features, meta_labels)
    
    print("元标签流水线创建完成")
    
    return meta_labeler 