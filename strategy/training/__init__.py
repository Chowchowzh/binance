# -*- coding: utf-8 -*-
"""
模型训练模块
负责Transformer模型的训练、信号生成和校准
"""

from .transformer_model import TimeSeriesTransformer
from .train_transformer import train_model
from .signal_generator import (
    generate_signals,
    load_or_generate_signals,
    calculate_conditional_returns
)
from .calibration_utils import (
    create_signal_calibrator,
    apply_signal_calibrator,
    calculate_dynamic_threshold
)

__all__ = [
    # 模型相关
    'TimeSeriesTransformer',
    'train_model',
    
    # 信号生成
    'generate_signals',
    'load_or_generate_signals',
    'calculate_conditional_returns',
    
    # 信号校准
    'create_signal_calibrator',
    'apply_signal_calibrator',
    'calculate_dynamic_threshold'
]
