# -*- coding: utf-8 -*-
"""
增强信号生成器
整合TBM和元标签技术，生成高质量、低噪声的交易信号
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

from strategy.training.transformer_model import TimeSeriesTransformer
from strategy.training.meta_labeling import MetaLabeler
from config.settings import ProjectConfig, load_config


class EnhancedSignalGenerator:
    """
    增强信号生成器
    
    核心功能：
    1. 加载训练好的主模型和元标签模型
    2. 生成高质量、高置信度的交易信号
    3. 提供信号强度、置信度、风险评估
    4. 支持实时推理和批量处理
    """
    
    def __init__(self, 
                 config: ProjectConfig = None,
                 primary_model_path: str = None,
                 meta_model_path: str = None,
                 scaler_path: str = None):
        """
        初始化增强信号生成器
        
        Args:
            config: 项目配置
            primary_model_path: 主模型路径
            meta_model_path: 元标签模型路径  
            scaler_path: 标准化器路径
        """
        self.config = config or load_config()
        
        # 设备配置
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # 模型路径
        self.primary_model_path = primary_model_path or self.config.model.model_save_path
        self.meta_model_path = meta_model_path or self.config.model.meta_model_save_path
        self.scaler_path = scaler_path or self.config.model.scaler_path
        
        # 模型组件
        self.primary_model = None
        self.meta_labeler = None
        self.scaler = None
        self.is_loaded = False
        
        print(f"初始化增强信号生成器:")
        print(f"  - 设备: {self.device}")
        print(f"  - 主模型路径: {self.primary_model_path}")
        print(f"  - 元标签模型路径: {self.meta_model_path}")
        
    def load_models(self, num_features: int = None):
        """
        加载所有模型组件
        
        Args:
            num_features: 特征数量，如果为None则从数据中推断
        """
        print("\n加载模型组件...")
        
        # 1. 加载标准化器
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ 标准化器加载成功: {self.scaler_path}")
            
            # 从标准化器推断特征数量
            if num_features is None and hasattr(self.scaler, 'n_features_in_'):
                num_features = self.scaler.n_features_in_
        else:
            print(f"⚠️ 标准化器不存在: {self.scaler_path}")
        
        # 2. 加载主模型
        if os.path.exists(self.primary_model_path):
            if num_features is None:
                raise ValueError("无法确定特征数量，请手动指定")
            
            # 创建模型架构
            model_params = self.config.model.get_model_params()
            self.primary_model = TimeSeriesTransformer(
                num_features=num_features,
                **model_params
            ).to(self.device)
            
            # 加载权重
            state_dict = torch.load(self.primary_model_path, map_location=self.device)
            self.primary_model.load_state_dict(state_dict)
            self.primary_model.eval()
            
            print(f"✅ 主模型加载成功: {self.primary_model_path}")
            print(f"   - 特征数量: {num_features}")
            print(f"   - 模型参数: {sum(p.numel() for p in self.primary_model.parameters()):,}")
        else:
            print(f"❌ 主模型不存在: {self.primary_model_path}")
            return False
        
        # 3. 加载元标签模型
        if self.config.model.use_meta_labeling and os.path.exists(self.meta_model_path):
            self.meta_labeler = MetaLabeler()
            self.meta_labeler.load_meta_model(self.meta_model_path)
            self.meta_labeler.primary_model = self.primary_model
            
            print(f"✅ 元标签模型加载成功: {self.meta_model_path}")
        elif self.config.model.use_meta_labeling:
            print(f"⚠️ 元标签模型不存在: {self.meta_model_path}")
        else:
            print("📝 元标签功能未启用")
        
        self.is_loaded = True
        print("🎉 所有模型组件加载完成")
        return True
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        预处理特征数据
        
        Args:
            features: 原始特征数组
            
        Returns:
            预处理后的特征数组
        """
        # 处理NaN和无穷大值
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 标准化
        if self.scaler is not None:
            original_shape = features.shape
            if len(original_shape) == 3:  # (n_samples, seq_len, n_features)
                # 重塑为2D进行标准化
                features_2d = features.reshape(-1, features.shape[-1])
                features_2d = self.scaler.transform(features_2d)
                features = features_2d.reshape(original_shape)
            else:  # 2D features
                features = self.scaler.transform(features)
        
        return features.astype(np.float32)
    
    def generate_primary_signals(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        使用主模型生成基础信号
        
        Args:
            features: 预处理后的特征数组
            
        Returns:
            主模型信号结果
        """
        if not self.is_loaded:
            raise ValueError("模型未加载，请先调用load_models()")
        
        self.primary_model.eval()
        
        # 批量推理
        batch_size = 256
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch_features).to(self.device)
                
                outputs = self.primary_model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # 生成基础信号
        if probabilities.shape[1] == 3:  # 三分类
            # P(上涨) - P(下跌)
            raw_signals = probabilities[:, 2] - probabilities[:, 0]
        else:  # 二分类
            raw_signals = probabilities[:, 1] - probabilities[:, 0]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'raw_signals': raw_signals,
            'signal_strength': np.abs(raw_signals),
            'confidence_score': np.max(probabilities, axis=1)
        }
    
    def generate_enhanced_signals(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        生成增强的交易信号（包含元标签过滤）
        
        Args:
            features: 原始特征数组
            
        Returns:
            增强信号结果字典
        """
        if not self.is_loaded:
            raise ValueError("模型未加载，请先调用load_models()")
        
        print(f"生成增强信号，样本数: {len(features)}")
        
        # 1. 预处理特征
        processed_features = self.preprocess_features(features)
        
        # 2. 生成主模型信号
        primary_results = self.generate_primary_signals(processed_features)
        
        # 3. 如果有元标签模型，进行信号增强
        if self.meta_labeler is not None and self.meta_labeler.is_fitted:
            print("使用元标签模型增强信号...")
            
            # 生成过滤后的信号
            meta_results = self.meta_labeler.generate_filtered_signals(
                processed_features,
                device=str(self.device)
            )
            
            # 合并结果
            enhanced_results = {
                **primary_results,
                'meta_confidence': meta_results['confidence_probabilities'],
                'filtered_signals': meta_results['filtered_signals'],
                'high_confidence_signals': meta_results['high_confidence_signals'],
                'high_confidence_mask': meta_results['high_confidence_mask'],
                'statistics': meta_results['statistics']
            }
            
            print(f"信号增强完成:")
            print(f"  - 高置信度比例: {meta_results['statistics']['high_conf_signal_ratio']:.4f}")
            print(f"  - 平均置信度: {meta_results['statistics']['avg_confidence']:.4f}")
            
        else:
            print("使用主模型信号（无元标签增强）")
            enhanced_results = primary_results
        
        return enhanced_results
    
    def generate_trading_decisions(self, 
                                 signals: Dict[str, np.ndarray],
                                 signal_threshold: float = 0.1,
                                 confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        基于信号生成交易决策
        
        Args:
            signals: 信号结果字典
            signal_threshold: 信号阈值
            confidence_threshold: 置信度阈值
            
        Returns:
            交易决策DataFrame
        """
        # 使用增强信号还是原始信号
        if 'filtered_signals' in signals:
            main_signals = signals['filtered_signals']
            confidence_scores = signals['meta_confidence']
        else:
            main_signals = signals['raw_signals']
            confidence_scores = signals['confidence_score']
        
        # 生成交易决策
        decisions = []
        
        for i in range(len(main_signals)):
            signal = main_signals[i]
            confidence = confidence_scores[i]
            
            # 基于信号强度和置信度做决策
            if abs(signal) >= signal_threshold and confidence >= confidence_threshold:
                if signal > 0:
                    action = 'BUY'
                    direction = 1
                else:
                    action = 'SELL'
                    direction = -1
            else:
                action = 'HOLD'
                direction = 0
            
            # 计算仓位大小（基于信号强度和置信度）
            position_size = min(abs(signal) * confidence, 1.0) if action != 'HOLD' else 0.0
            
            decisions.append({
                'index': i,
                'signal': signal,
                'confidence': confidence,
                'action': action,
                'direction': direction,
                'position_size': position_size,
                'signal_strength': abs(signal),
                'meets_threshold': abs(signal) >= signal_threshold,
                'high_confidence': confidence >= confidence_threshold
            })
        
        df_decisions = pd.DataFrame(decisions)
        
        # 统计信息
        action_counts = df_decisions['action'].value_counts()
        high_conf_count = df_decisions['high_confidence'].sum()
        
        print(f"\n交易决策统计:")
        print(f"  - 买入信号: {action_counts.get('BUY', 0)}")
        print(f"  - 卖出信号: {action_counts.get('SELL', 0)}")
        print(f"  - 持有信号: {action_counts.get('HOLD', 0)}")
        print(f"  - 高置信度决策: {high_conf_count} ({high_conf_count/len(df_decisions)*100:.2f}%)")
        print(f"  - 平均信号强度: {df_decisions['signal_strength'].mean():.4f}")
        print(f"  - 平均置信度: {df_decisions['confidence'].mean():.4f}")
        
        return df_decisions
    
    def batch_inference(self,
                       data: Union[pd.DataFrame, np.ndarray],
                       feature_columns: List[str] = None,
                       sequence_length: int = None) -> Dict[str, Any]:
        """
        批量推理
        
        Args:
            data: 输入数据（DataFrame或numpy数组）
            feature_columns: 特征列名（如果data是DataFrame）
            sequence_length: 序列长度
            
        Returns:
            推理结果
        """
        if not self.is_loaded:
            raise ValueError("模型未加载，请先调用load_models()")
        
        sequence_length = sequence_length or self.config.model.sequence_length
        
        # 准备特征数据
        if isinstance(data, pd.DataFrame):
            if feature_columns is None:
                # 自动检测特征列（排除时间和目标列）
                exclude_cols = ['start_time', 'end_time', 'target', 'future_return']
                exclude_cols.extend([col for col in data.columns if 'target' in col or 'time' in col])
                feature_columns = [col for col in data.columns if col not in exclude_cols]
            
            # 构建序列特征
            features = []
            valid_indices = []
            
            for i in range(sequence_length, len(data)):
                try:
                    feature_sequence = data.iloc[i-sequence_length:i][feature_columns].values
                    if feature_sequence.shape[0] == sequence_length:
                        features.append(feature_sequence)
                        valid_indices.append(i)
                except:
                    continue
            
            features = np.array(features)
            
        else:  # numpy数组
            features = data
            valid_indices = list(range(len(features)))
        
        print(f"批量推理: {len(features)} 个样本")
        
        # 生成信号
        signals = self.generate_enhanced_signals(features)
        
        # 生成交易决策
        decisions = self.generate_trading_decisions(signals)
        decisions['original_index'] = valid_indices
        
        return {
            'signals': signals,
            'decisions': decisions,
            'feature_count': features.shape[-1] if len(features) > 0 else 0,
            'sample_count': len(features),
            'timestamp': datetime.now().isoformat()
        }
    
    def real_time_inference(self, 
                           latest_features: np.ndarray) -> Dict[str, Any]:
        """
        实时推理（单个样本）
        
        Args:
            latest_features: 最新的特征序列 (seq_len, n_features)
            
        Returns:
            实时推理结果
        """
        if not self.is_loaded:
            raise ValueError("模型未加载，请先调用load_models()")
        
        # 确保输入形状正确
        if len(latest_features.shape) == 2:
            latest_features = latest_features.reshape(1, *latest_features.shape)
        
        # 生成信号
        signals = self.generate_enhanced_signals(latest_features)
        
        # 提取单个样本的结果
        result = {}
        for key, value in signals.items():
            if isinstance(value, np.ndarray) and len(value) > 0:
                result[key] = value[0] if value.ndim > 0 else value
            else:
                result[key] = value
        
        # 生成交易决策
        decision = self.generate_trading_decisions(signals).iloc[0].to_dict()
        
        return {
            'signal': result,
            'decision': decision,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_inference_results(self, 
                              results: Dict[str, Any], 
                              save_path: str):
        """
        保存推理结果
        
        Args:
            results: 推理结果
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存决策DataFrame
        if 'decisions' in results:
            decisions_path = save_path.replace('.pkl', '_decisions.csv')
            results['decisions'].to_csv(decisions_path, index=False)
            print(f"交易决策已保存: {decisions_path}")
        
        # 保存完整结果
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"推理结果已保存: {save_path}")


def load_enhanced_signal_generator(config: ProjectConfig = None) -> EnhancedSignalGenerator:
    """
    便利函数：加载预训练的增强信号生成器
    
    Args:
        config: 项目配置
        
    Returns:
        加载好的EnhancedSignalGenerator
    """
    generator = EnhancedSignalGenerator(config)
    
    # 尝试自动加载模型
    try:
        generator.load_models()
        print("✅ 增强信号生成器加载成功")
        return generator
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("请确保模型文件存在并正确训练")
        raise 