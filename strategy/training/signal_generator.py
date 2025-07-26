# -*- coding: utf-8 -*-
"""
信号生成模块
负责使用Transformer模型生成交易信号
"""

import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import numpy as np
import os

from config.settings import ProjectConfig, load_config


def generate_signals(model, data, scaler, device, sequence_length=None, avg_up_return=None, avg_down_return=None, config: ProjectConfig = None):
    """使用Transformer模型生成交易信号"""
    print("\n生成Transformer模型交易信号...")
    
    # 如果没有提供sequence_length，从配置中获取
    if sequence_length is None:
        if config is None:
            config = load_config()
        sequence_length = config.model.sequence_length
    
    model.eval()

    use_expected_return_signal = (
        avg_up_return is not None and pd.notna(avg_up_return) and
        avg_down_return is not None and pd.notna(avg_down_return)
    )
    if not use_expected_return_signal:
        print("Warning: 平均收益无效，回退到 P(Up)-P(Down) 信号模式")
    
    features_df = data.drop(columns=['target'])
    
    # 数据清理
    features_df.replace([np.inf, -np.inf], 0, inplace=True)
    features_df.fillna(0, inplace=True)

    finfo = np.finfo(np.float32)
    features_df.clip(lower=finfo.min, upper=finfo.max, inplace=True)

    all_features = features_df.values.astype(np.float32)
    all_features = scaler.transform(all_features)
    
    predictions = []
    
    num_samples = len(all_features) - sequence_length
    if num_samples <= 0:
        print("数据不足，无法创建序列")
        return np.array([])
        
    sequences = np.array([all_features[i : i + sequence_length] for i in range(num_samples)])
    
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(sequences).float())
    batch_size = 256
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_idx, batch_sequences_tuple in enumerate(dataloader):
            batch_sequences = batch_sequences_tuple[0].to(device)
            
            output = model(batch_sequences)

            if torch.isnan(output).any():
                batch_signal = torch.zeros(output.shape[0], device=device)
            else:
                probabilities = nn.functional.softmax(output, dim=1)
                
                if use_expected_return_signal:
                    p_up = probabilities[:, 1]
                    p_down = probabilities[:, 0]
                    avg_up_float = float(avg_up_return) if avg_up_return is not None else 0.0
                    avg_down_float = float(avg_down_return) if avg_down_return is not None else 0.0
                    batch_signal = p_up * avg_up_float + p_down * avg_down_float
                else:
                    batch_signal = probabilities[:, 1] - probabilities[:, 0]
            
            predictions.extend(batch_signal.cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                print(f"  已处理 {len(predictions)} 个信号，共 {batch_idx + 1} 批次...")
    
    print("信号生成完成")
    return np.array(predictions)


def load_or_generate_signals(model, data, scaler, device, cache_path, 
                             avg_up_return=None, avg_down_return=None, logger=None, config: ProjectConfig = None):
    """加载缓存的信号或生成新信号"""
    if os.path.exists(cache_path):
        if logger:
            logger.log(f"从缓存加载信号: {cache_path}")
        return np.load(cache_path)
    else:
        signals = generate_signals(model, data, scaler, device, config=config, avg_up_return=avg_up_return, avg_down_return=avg_down_return)
        if logger:
            logger.log(f"保存信号到缓存: {cache_path}")
        np.save(cache_path, signals)
        return signals


def calculate_conditional_returns(train_df, target_symbol, sequence_length, N=5):
    """计算条件期望收益"""
    train_future_returns = np.log(
        train_df[f'{target_symbol}_close'].shift(-N) / 
        train_df[f'{target_symbol}_close'].replace(0, np.nan)
    )
    
    # 对齐数据
    aligned_train_df = train_df.iloc[sequence_length:].copy()
    aligned_train_df['future_return'] = train_future_returns.iloc[sequence_length:]
    
    # 计算条件期望收益
    avg_up_return = aligned_train_df[aligned_train_df['target'] == 1]['future_return'].mean()
    avg_down_return = aligned_train_df[aligned_train_df['target'] == 0]['future_return'].mean()
    
    return avg_up_return, avg_down_return, aligned_train_df 