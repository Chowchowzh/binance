# -*- coding: utf-8 -*-
"""
高级训练流水线
整合三分类标签法(TBM)和元标签技术的完整两阶段学习框架
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import sys
import pickle
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
import warnings

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_processing.features.triple_barrier_labeling import TripleBarrierLabeler
from strategy.training.meta_labeling import MetaLabeler
from strategy.training.transformer_model import TimeSeriesTransformer
from config.settings import ProjectConfig, load_config
from data_processing.preprocessor import DataPreprocessor

warnings.filterwarnings('ignore')


class AdvancedTrainingPipeline:
    """
    高级训练流水线
    
    完整的两阶段学习框架：
    1. TBM标签生成：使用动态边界生成高质量标签
    2. 主模型训练：Transformer模型追求高召回率
    3. 元标签训练：次级模型提升精度，过滤假阳性
    """
    
    def __init__(self, config: ProjectConfig = None):
        """
        初始化高级训练流水线
        
        Args:
            config: 项目配置
        """
        self.config = config or load_config()
        
        # 设备配置
        # 强制使用CPU进行多线程训练
        self.device = torch.device("cpu")
        print("🔧 高级训练管道使用CPU + 多线程加速")
        
        # 初始化组件
        self.tbm_labeler = None
        self.primary_model = None
        self.meta_labeler = None
        self.preprocessor = None
        self.scaler = None
        
        # 数据缓存
        self.processed_data = None
        self.tbm_labels = None
        self.training_features = None
        self.training_labels = None
        
        print(f"初始化高级训练流水线:")
        print(f"  - 设备: {self.device}")
        print(f"  - TBM启用: True")
        print(f"  - 元标签启用: {self.config.model.use_meta_labeling}")
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        加载和准备数据
        
        Returns:
            处理后的数据DataFrame
        """
        print("\n" + "=" * 60)
        print("📊 数据加载与预处理")
        print("=" * 60)
        
        # 1. 加载原始数据
        data_path = self.config.model.data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        print(f"加载数据文件: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"原始数据维度: {df.shape}")
        
        # 2. 初始化预处理器
        self.preprocessor = DataPreprocessor(config_path='config/config.json')
        
        # 3. 加载或创建scaler
        scaler_path = self.config.model.scaler_path
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"已加载标准化器: {scaler_path}")
        else:
            print("标准化器不存在，将自动创建新的标准化器")
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            print("已创建新的StandardScaler")

        self.processed_data = df
        return df
    
    def generate_tbm_labels(self, price_column: str = None) -> pd.DataFrame:
        """
        生成TBM标签
        
        Args:
            price_column: 价格列名
            
        Returns:
            TBM标签DataFrame
        """
        print("\n" + "=" * 60)
        print("🎯 三分类标签法 (TBM) 标签生成")
        print("=" * 60)
        
        # 检查缓存
        cache_path = self.config.model.tbm_labels_cache_path
        if os.path.exists(cache_path):
            print(f"从缓存加载TBM标签: {cache_path}")
            self.tbm_labels = pd.read_parquet(cache_path)
            return self.tbm_labels
        
        # 确定价格列
        if price_column is None:
            # 自动检测价格列
            possible_columns = ['close', 'ETHUSDT_close', f'{self.config.data_collection.target_symbol}_close']
            price_column = None
            for col in possible_columns:
                if col in self.processed_data.columns:
                    price_column = col
                    break
            
            if price_column is None:
                raise ValueError("无法找到价格列，请手动指定")
        
        print(f"使用价格列: {price_column}")
        
        # 初始化TBM标签器
        tbm_params = self.config.model.get_tbm_params()
        self.tbm_labeler = TripleBarrierLabeler(**tbm_params)
        
        # 提取价格序列
        prices = self.processed_data[price_column].copy()
        
        # 生成事件触发点
        if self.config.model.tbm_use_cusum_events:
            print("使用CUSUM过滤器生成事件...")
            event_indices = self.tbm_labeler.generate_cusum_events(
                prices, 
                threshold=self.config.model.tbm_cusum_threshold
            )
        else:
            event_indices = None  # 使用默认：所有有效点
        
        # 生成TBM标签
        tbm_labels = self.tbm_labeler.generate_triple_barrier_labels(
            prices=prices,
            event_indices=event_indices,
            volatility_method=self.config.model.tbm_volatility_method,
            n_jobs=1  # 避免多进程问题
        )
        
        # 分析标签质量
        quality_analysis = self.tbm_labeler.analyze_label_quality(tbm_labels)
        print(f"\n标签质量分析:")
        print(f"  - 总事件数: {quality_analysis['total_events']}")
        print(f"  - 平均持仓期: {quality_analysis['holding_period_stats']['mean']:.2f}")
        print(f"  - 平均收益率: {quality_analysis['return_stats']['mean']:.6f}")
        
        # 保存到缓存
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tbm_labels.to_parquet(cache_path)
        print(f"TBM标签已保存到缓存: {cache_path}")
        
        self.tbm_labels = tbm_labels
        return tbm_labels
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据 - 优化版本，增加样本数量
        
        Returns:
            Tuple[features, labels]: 训练特征和标签
        """
        print("\n" + "=" * 60)
        print("🔧 训练数据准备")
        print("=" * 60)
        
        if self.processed_data is None:
            raise ValueError("处理后的数据不存在，请先调用load_and_prepare_data()")
        
        if self.tbm_labels is None:
            raise ValueError("TBM标签不存在，请先调用generate_tbm_labels()")
        
        # 1. 构建特征-标签对应关系
        feature_label_pairs = []
        
        # 获取特征列（排除时间和目标列）- 更严格的过滤
        exclude_cols = ['start_time', 'end_time', 'target', 'future_return', 'tbm_label', 'tbm_return_pct', 'tbm_holding_period', 'tbm_touch_type']
        
        # 扩展排除列表，包含所有时间相关列
        for col in self.processed_data.columns:
            if ('target' in col or 'time' in col.lower() or 'timestamp' in col.lower() or
                self.processed_data[col].dtype.name.startswith('datetime')):
                exclude_cols.append(col)
        
        feature_cols = [col for col in self.processed_data.columns if col not in exclude_cols]
        
        # 强制转换特征列为float32
        for col in feature_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].astype(np.float32)
        
        print(f"特征列数量: {len(feature_cols)}")
        
        # 2. 使用滑动窗口方法提取更多训练样本 - 解决样本数量少的问题
        sequence_length = self.config.model.sequence_length
        overlap_ratio = 0.5  # 50%重叠，增加样本数量
        step_size = max(1, int(sequence_length * (1 - overlap_ratio)))
        
        print(f"使用滑动窗口方法，窗口大小: {sequence_length}, 步长: {step_size}")
        
        # 首先收集所有有效的TBM事件
        valid_events = []
        for _, tbm_row in self.tbm_labels.iterrows():
            event_idx = tbm_row['event_idx']
            label = tbm_row['label']
            
            # 检查标签是否有效
            if pd.isna(label):
                continue
                
            # 检查是否有足够的历史数据
            if event_idx < sequence_length:
                continue
                
            # 检查是否有足够的未来数据用于验证
            if event_idx >= len(self.processed_data) - 1:
                continue
                
            valid_events.append((event_idx, label))
        
        print(f"找到 {len(valid_events)} 个有效TBM事件")
        
        # 3. 为每个有效事件生成多个训练样本
        for event_idx, label in valid_events:
            # 在事件前的窗口内生成多个样本
            max_start_idx = event_idx - sequence_length
            min_start_idx = max(0, max_start_idx - sequence_length)
            
            # 生成多个起始位置
            start_positions = list(range(min_start_idx, max_start_idx + 1, step_size))
            if len(start_positions) == 0:
                start_positions = [max_start_idx]
            
            for start_idx in start_positions:
                end_idx = start_idx + sequence_length
                
                # 确保不超出边界
                if end_idx > len(self.processed_data):
                    continue
                
                try:
                    feature_sequence = self.processed_data.iloc[start_idx:end_idx][feature_cols].values
                    
                    # 检查特征序列是否完整
                    if feature_sequence.shape[0] != sequence_length:
                        continue
                    
                    # 检查是否有过多的NaN值
                    nan_ratio = np.isnan(feature_sequence).sum() / feature_sequence.size
                    if nan_ratio > 0.1:  # 超过10%的NaN值就跳过
                        continue
                    
                    # 处理NaN值
                    feature_sequence = np.nan_to_num(feature_sequence, nan=0.0)
                    
                    feature_label_pairs.append((feature_sequence, label))
                    
                except Exception as e:
                    print(f"警告: 处理序列 {start_idx}:{end_idx} 时出错: {e}")
                    continue
        
        print(f"成功创建 {len(feature_label_pairs)} 个训练样本")
        
        # 4. 转换为numpy数组
        if len(feature_label_pairs) == 0:
            raise ValueError("没有有效的训练样本，请检查数据质量和TBM标签")
        
        features = np.array([pair[0] for pair in feature_label_pairs])
        labels = np.array([pair[1] for pair in feature_label_pairs])
        
        # 5. 创建或更新标准化器
        if self.scaler is not None:
            # 重塑为2D进行标准化
            original_shape = features.shape
            features_2d = features.reshape(-1, features.shape[-1])
            
            # 如果是新创建的scaler，需要先fit
            if not hasattr(self.scaler, 'scale_'):
                print("正在拟合新的标准化器...")
                features_2d = self.scaler.fit_transform(features_2d)
                
                # 保存标准化器
                scaler_path = self.config.model.scaler_path
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                print(f"标准化器已保存到: {scaler_path}")
            else:
                print("使用已有标准化器进行转换...")
                features_2d = self.scaler.transform(features_2d)
            
            features = features_2d.reshape(original_shape)
            print("特征已标准化")
        
        # 6. 转换标签格式 (TBM: -1,0,1 -> 0,1,2)
        labels = labels + 1
        
        print(f"最终训练数据:")
        print(f"  - 特征形状: {features.shape}")
        print(f"  - 标签形状: {labels.shape}")
        print(f"  - 标签分布: {np.bincount(labels.astype(int))}")
        
        # 强制转换数据类型
        self.training_features = features.astype(np.float32)
        self.training_labels = labels.astype(np.int64)
        
        return self.training_features, self.training_labels
    
    def train_primary_model(self) -> Dict[str, Any]:
        """
        训练主模型 (Transformer)
        
        Returns:
            训练结果统计
        """
        print("\n" + "=" * 60)
        print("🧠 主模型训练 (第一阶段: 高召回率)")
        print("=" * 60)
        
        if self.training_features is None or self.training_labels is None:
            raise ValueError("训练数据未准备，请先调用prepare_training_data()")
        
        # 1. 数据集划分
        n_samples = len(self.training_features)
        split_idx = int(n_samples * self.config.model.train_test_split_ratio)
        
        train_features = self.training_features[:split_idx]
        train_labels = self.training_labels[:split_idx]
        val_features = self.training_features[split_idx:]
        val_labels = self.training_labels[split_idx:]
        
        print(f"数据集划分:")
        print(f"  - 训练样本: {len(train_features)}")
        print(f"  - 验证样本: {len(val_features)}")
        
        # 2. 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(train_features),
            torch.LongTensor(train_labels)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_features),
            torch.LongTensor(val_labels)
        )
        
        batch_size = self.config.model.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 3. 初始化模型
        num_features = train_features.shape[-1]
        model_params = self.config.model.get_model_params()
        
        self.primary_model = TimeSeriesTransformer(
            num_features=num_features,
            **model_params
        ).to(self.device)
        
        print(f"主模型参数:")
        total_params = sum(p.numel() for p in self.primary_model.parameters())
        print(f"  - 总参数数: {total_params:,}")
        print(f"  - 特征维度: {num_features}")
        
        # 4. 训练设置
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.primary_model.parameters(), 
            lr=self.config.model.learning_rate
        )
        
        # 5. 训练循环
        best_val_accuracy = 0.0
        training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.config.model.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.model.epochs}")
            
            # 训练阶段
            self.primary_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.primary_model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.primary_model.parameters(), 
                        self.config.model.clip_value
                    )
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # 验证阶段
            self.primary_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.primary_model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # 记录历史
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_accuracy'].append(val_accuracy)
            
            print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
            print(f"  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%")
            
            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_save_path = self.config.model.model_save_path
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(self.primary_model.state_dict(), model_save_path)
                print(f"  🏆 新的最佳模型已保存: {val_accuracy:.2f}%")
        
        print(f"\n主模型训练完成!")
        print(f"  - 最佳验证准确率: {best_val_accuracy:.2f}%")
        print(f"  - 模型保存路径: {self.config.model.model_save_path}")
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'training_history': training_history,
            'model_path': self.config.model.model_save_path
        }
    
    def train_meta_labeling(self) -> Dict[str, Any]:
        """
        训练元标签模型 (第二阶段)
        
        Returns:
            元标签训练结果
        """
        print("\n" + "=" * 60)
        print("🔍 元标签训练 (第二阶段: 高精度)")
        print("=" * 60)
        
        if not self.config.model.use_meta_labeling:
            print("元标签功能未启用，跳过")
            return {}
        
        if self.primary_model is None:
            raise ValueError("主模型未训练，请先调用train_primary_model()")
        
        # 1. 初始化元标签器
        meta_params = self.config.model.get_meta_labeling_params()
        self.meta_labeler = MetaLabeler(
            primary_model=self.primary_model,
            **meta_params
        )
        
        # 2. 使用训练数据训练元标签模型
        # 注意：这里使用整个数据集，但会在内部分割
        train_features = self.training_features
        
        # 将TBM标签转换回原始格式 (0,1,2 -> -1,0,1)
        true_labels = self.training_labels - 1
        
        # 3. 生成主模型预测
        primary_preds, primary_probs = self.meta_labeler.generate_primary_predictions(
            train_features, 
            device=str(self.device)
        )
        
        # 4. 创建次级模型特征
        meta_features = self.meta_labeler.create_meta_features(
            train_features, 
            primary_preds, 
            primary_probs
        )
        
        # 5. 创建元标签
        meta_labels = self.meta_labeler.create_meta_labels(
            primary_preds, 
            true_labels,
            prediction_type=self.config.model.meta_prediction_type
        )
        
        # 6. 训练次级模型
        training_results = self.meta_labeler.fit_meta_model(
            meta_features, 
            meta_labels,
            validation_split=self.config.model.meta_validation_split
        )
        
        # 7. 保存元标签模型
        meta_model_path = self.config.model.meta_model_save_path
        self.meta_labeler.save_meta_model(meta_model_path)
        
        print(f"\n元标签训练完成!")
        print(f"  - 验证AUC: {training_results.get('validation_auc', 0):.4f}")
        print(f"  - 验证精度: {training_results.get('validation_precision', 0):.4f}")
        print(f"  - 模型保存路径: {meta_model_path}")
        
        return training_results
    
    def generate_enhanced_signals(self, test_features: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        生成增强的交易信号
        
        Args:
            test_features: 测试特征，如果为None则使用验证集
            
        Returns:
            信号结果字典
        """
        print("\n" + "=" * 60)
        print("🚀 生成增强交易信号")
        print("=" * 60)
        
        if self.meta_labeler is None:
            print("元标签模型未训练，使用主模型信号")
            # TODO: 实现仅使用主模型的信号生成
            return {}
        
        # 使用验证集数据
        if test_features is None:
            split_idx = int(len(self.training_features) * self.config.model.train_test_split_ratio)
            test_features = self.training_features[split_idx:]
        
        # 生成过滤后的信号
        signal_results = self.meta_labeler.generate_filtered_signals(
            test_features, 
            device=str(self.device)
        )
        
        print(f"信号生成完成!")
        print(f"  - 原始信号平均强度: {signal_results['statistics']['raw_signal_mean']:.4f}")
        print(f"  - 过滤信号平均强度: {signal_results['statistics']['filtered_signal_mean']:.4f}")
        print(f"  - 高置信度信号比例: {signal_results['statistics']['high_conf_signal_ratio']:.4f}")
        
        return signal_results
    
    def run_complete_pipeline(self, price_column: str = None) -> Dict[str, Any]:
        """
        运行完整的训练流水线
        
        Args:
            price_column: 价格列名
            
        Returns:
            完整流程结果
        """
        print("🚀 启动高级训练流水线")
        print("=" * 80)
        
        results = {}
        
        try:
            # 1. 数据加载与预处理
            self.load_and_prepare_data()
            
            # 2. TBM标签生成
            self.generate_tbm_labels(price_column)
            
            # 3. 训练数据准备
            self.prepare_training_data()
            
            # 4. 主模型训练
            primary_results = self.train_primary_model()
            results['primary_model'] = primary_results
            
            # 5. 元标签训练
            if self.config.model.use_meta_labeling:
                meta_results = self.train_meta_labeling()
                results['meta_labeling'] = meta_results
            
            # 6. 信号生成测试
            signal_results = self.generate_enhanced_signals()
            results['signals'] = signal_results
            
            results['status'] = 'success'
            results['timestamp'] = datetime.now().isoformat()
            
            print("\n" + "=" * 80)
            print("🎉 高级训练流水线完成!")
            print("=" * 80)
            print(f"✅ 主模型最佳准确率: {primary_results['best_val_accuracy']:.2f}%")
            
            if 'meta_labeling' in results:
                meta_auc = results['meta_labeling'].get('validation_auc', 0)
                print(f"✅ 元标签模型AUC: {meta_auc:.4f}")
            
            if 'signals' in results:
                high_conf_ratio = results['signals']['statistics']['high_conf_signal_ratio']
                print(f"✅ 高置信度信号比例: {high_conf_ratio:.4f}")
            
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ 训练流水线失败: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
        
        return results


def main():
    """主函数：运行完整的高级训练流水线"""
    try:
        # 加载配置
        config = load_config()
        
        # 创建训练流水线
        pipeline = AdvancedTrainingPipeline(config)
        
        # 运行完整流程
        results = pipeline.run_complete_pipeline()
        
        print("\n训练完成！可以使用以下文件进行推理：")
        print(f"  - 主模型: {config.model.model_save_path}")
        if config.model.use_meta_labeling:
            print(f"  - 元标签模型: {config.model.meta_model_save_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断训练")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        raise


if __name__ == '__main__':
    main() 