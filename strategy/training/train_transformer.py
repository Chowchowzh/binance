import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import os
import sys
import pickle
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_processing.torch_dataset import TimeSeriesDataset
from data_processing.preprocessor import DataPreprocessor
from strategy.training.transformer_model import TimeSeriesTransformer
from config.settings import ProjectConfig, load_config

def train_model(config: ProjectConfig = None):
    """
    Main function to orchestrate the model training process.
    """
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    print("=" * 80)
    print("🚀 开始 Transformer 模型训练")
    print("=" * 80)
    print(f"⏰ 训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Setup Device (MPS for Apple Silicon, CUDA for Nvidia, else CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️  使用设备: {device}")
    
    # DataLoader num_workers should be 0 if using MPS
    num_workers = 0 if device.type == 'mps' else 4
    print(f"⚙️  数据加载器工作进程数: {num_workers}")

    # 2. 初始化数据预处理器并加载 scaler
    print("\n" + "=" * 60)
    print("📊 数据预处理与加载")
    print("=" * 60)
    
    try:
        # 初始化数据预处理器
        preprocessor = DataPreprocessor(config_path='config/config.json')
        print("✅ 数据预处理器初始化成功")
        
        # 加载 scaler
        scaler_path = config.model.scaler_path
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✅ 成功加载标准化器: {scaler_path}")
            print(f"   - 标准化器类型: {type(scaler).__name__}")
            if hasattr(scaler, 'n_features_in_'):
                print(f"   - 特征数量: {scaler.n_features_in_}")
        else:
            print(f"⚠️  标准化器文件不存在: {scaler_path}")
            print("   将使用数据集中已标准化的数据")
            scaler = None
    except Exception as e:
        print(f"❌ 预处理器初始化失败: {e}")
        return

    # 3. Load and Prepare Data
    data_path = config.model.data_path
    if not os.path.exists(data_path):
        print(f"❌ 数据文件未找到: \"{data_path}\"")
        print("   请先运行特征工程处理")
        return

    print(f"📁 加载数据文件: {data_path}")
    
    try:
        # Create the training dataset instance using preprocessed data
        full_dataset = TimeSeriesDataset(
            data_path=data_path, 
            sequence_length=config.model.sequence_length,
            preprocessor=preprocessor
        )
        
        # Get number of features from the dataset
        num_features = full_dataset.features.shape[1]
        total_samples = len(full_dataset)
        print(f"✅ 数据集创建成功")
        print(f"   - 特征数量: {num_features}")
        print(f"   - 总样本数: {total_samples}")
        print(f"   - 序列长度: {config.model.sequence_length}")
        print(f"   - 数据形状: {full_dataset.features.shape}")
        
        # 详细检查目标向量
        print(f"\n🎯 目标向量检查:")
        print(f"   - 目标数据类型: {full_dataset.targets.dtype}")
        print(f"   - 目标数据形状: {full_dataset.targets.shape}")
        print(f"   - 目标值范围: [{np.min(full_dataset.targets)}, {np.max(full_dataset.targets)}]")
        print(f"   - 目标值样例 (前10个): {full_dataset.targets[:10]}")
        
        # 检查是否存在 NaN 或无效值
        nan_count = np.isnan(full_dataset.targets).sum()
        inf_count = np.isinf(full_dataset.targets).sum()
        print(f"   - NaN 值数量: {nan_count}")
        print(f"   - 无穷大值数量: {inf_count}")
        
        # 检查目标分布
        unique_targets, counts = np.unique(full_dataset.targets, return_counts=True)
        print(f"   - 目标类别分布:")
        for target, count in zip(unique_targets, counts):
            percentage = (count / len(full_dataset.targets)) * 100
            print(f"     类别 {target}: {count} 样本 ({percentage:.2f}%)")
        
        # 检查类别数量是否合理
        num_classes = len(unique_targets)
        print(f"   - 总类别数: {num_classes}")
        
        if num_classes < 2:
            print(f"   ❌ 错误: 只有 {num_classes} 个类别，无法进行分类训练")
            print(f"   💡 建议: 检查目标生成逻辑，确保有多个类别")
            return
        elif num_classes > 10:
            print(f"   ⚠️  警告: 类别数量较多 ({num_classes})，请确认这是预期的")
        else:
            print(f"   ✅ 类别数量合理")
        
        # 检查特征和标准化器一致性
        if scaler is not None:
            print(f"\n🔧 标准化器一致性检查:")
            if hasattr(scaler, 'n_features_in_'):
                print(f"   - 标准化器特征数: {scaler.n_features_in_}")
                print(f"   - 数据特征数: {num_features}")
                if scaler.n_features_in_ == num_features:
                    print(f"   - ✅ 特征维度完全匹配")
                else:
                    print(f"   - ⚠️  特征维度不匹配，可能影响推理一致性")
            
            # 检查特征数据的统计信息
            print(f"   - 特征值范围: [{np.min(full_dataset.features):.4f}, {np.max(full_dataset.features):.4f}]")
            print(f"   - 特征值均值: {np.mean(full_dataset.features):.4f}")
            print(f"   - 特征值标准差: {np.std(full_dataset.features):.4f}")
            
            # 检查是否已经标准化
            mean_close_to_zero = abs(np.mean(full_dataset.features)) < 0.1
            std_close_to_one = abs(np.std(full_dataset.features) - 1.0) < 0.2
            if mean_close_to_zero and std_close_to_one:
                print(f"   ✅ 数据似乎已经标准化（均值≈0，标准差≈1）")
            else:
                print(f"   ⚠️  数据可能未标准化，请检查预处理流程")
        
        print(f"   🎯 数据检查完成，准备开始训练")

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return

    # 4. 数据集划分
    print(f"\n📊 数据集划分 (训练/验证比例: {config.model.train_test_split_ratio:.2f})")
    train_size = int(len(full_dataset) * config.model.train_test_split_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    training_params = config.model.get_training_params()
    model_params = config.model.get_model_params()
    
    print(f"   - 训练样本: {len(train_dataset)}")
    print(f"   - 验证样本: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False, num_workers=num_workers)
    
    print(f"   - 训练批次数: {len(train_loader)}")
    print(f"   - 验证批次数: {len(val_loader)}")
    print(f"   - 批次大小: {training_params['batch_size']}")

    # 5. Initialize Model, Loss, and Optimizer
    print(f"\n" + "=" * 60)
    print("🧠 模型初始化")
    print("=" * 60)
    
    model = TimeSeriesTransformer(
        num_features=num_features,
        **model_params
    ).to(device)

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 模型创建成功")
    print(f"   - 模型参数: {model_params}")
    print(f"   - 总参数数量: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    
    print(f"   - 损失函数: {type(criterion).__name__}")
    print(f"   - 优化器: {type(optimizer).__name__}")
    print(f"   - 学习率: {training_params['learning_rate']}")

    # 6. Training Loop
    print(f"\n" + "=" * 60)
    print("🏋️  开始模型训练")
    print("=" * 60)
    print(f"   - 训练轮数: {training_params['epochs']}")
    print(f"   - 梯度裁剪: {training_params['clip_value']}")
    
    best_val_accuracy = 0.0
    training_start_time = datetime.now()

    for epoch in range(training_params['epochs']):
        epoch_start_time = datetime.now()
        print(f"\n🔄 Epoch {epoch+1}/{training_params['epochs']} - {epoch_start_time.strftime('%H:%M:%S')}")
        print("-" * 50)
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        print("📈 训练阶段...")
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Check for nan loss before backpropagation
            if torch.isnan(loss):
                print(f"⚠️  检测到 NaN 损失 - Epoch [{epoch+1}/{training_params['epochs']}], Step [{i+1}/{len(train_loader)}]. 跳过此批次.")
                continue

            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_params['clip_value'])
            
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if (i + 1) % 50 == 0:
                current_acc = 100 * correct_train / total_train if total_train > 0 else 0
                print(f"   Step [{i+1}/{len(train_loader)}] - Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # --- Validation Phase ---
        print("📊 验证阶段...")
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for i, (sequences, labels) in enumerate(val_loader):
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"\n📋 Epoch {epoch+1} 结果总结:")
        print(f"   ⏱️  耗时: {epoch_duration}")
        print(f"   📈 训练损失: {avg_train_loss:.4f}")
        print(f"   📈 训练准确率: {train_accuracy:.2f}%")
        print(f"   📊 验证损失: {avg_val_loss:.4f}")
        print(f"   📊 验证准确率: {val_accuracy:.2f}%")

        # --- Save Best Model ---
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = config.model.model_save_path
            
            # 确保模型保存目录存在
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            
            torch.save(model.state_dict(), model_save_path)
            print(f"🏆 新的最佳模型已保存!")
            print(f"   📁 保存路径: {model_save_path}")
            print(f"   🎯 验证准确率: {val_accuracy:.2f}%")
        else:
            print(f"   当前最佳验证准确率: {best_val_accuracy:.2f}%")

    training_end_time = datetime.now()
    total_training_time = training_end_time - training_start_time
    
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print("=" * 80)
    print(f"⏰ 训练结束时间: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总训练时长: {total_training_time}")
    print(f"🏆 最佳验证准确率: {best_val_accuracy:.2f}%")
    print(f"📁 最佳模型保存位置: {config.model.model_save_path}")
    print(f"📊 使用的标准化器: {config.model.scaler_path}")
    print("=" * 80)

if __name__ == '__main__':
    train_model() 