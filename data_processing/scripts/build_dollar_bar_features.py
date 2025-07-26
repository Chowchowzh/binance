#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成交额K线特征工程执行脚本
对成交额K线数据进行完整的特征工程处理
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_processing.features.dollar_bar_features import DollarBarFeatures
from utils.logger import setup_logger


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description='成交额K线特征工程')
    parser.add_argument('--input', type=str, default='processed_data/dollar_bars_ETHUSDT.parquet',
                       help='输入的成交额K线数据文件')
    parser.add_argument('--output', type=str, default='processed_data/dollar_bar_features.parquet',
                       help='输出的特征数据文件')
    parser.add_argument('--target_symbol', type=str, default='ETHUSDT',
                       help='目标交易对')
    parser.add_argument('--feature_symbols', type=str, nargs='+', default=['ETHUSDT'],
                       help='用于特征工程的交易对列表')
    parser.add_argument('--future_periods', type=int, default=15,
                       help='未来收益预测期数')
    parser.add_argument('--use_fp16', action='store_true',
                       help='是否使用fp16格式')
    parser.add_argument('--no_fp16', action='store_true',
                       help='不使用fp16格式')  
    parser.add_argument('--analysis', action='store_true',
                       help='是否进行特征重要性分析')
    
    args = parser.parse_args()
    
    # 处理fp16设置
    if args.no_fp16:
        args.use_fp16 = False
    elif not hasattr(args, 'use_fp16') or not args.use_fp16:
        args.use_fp16 = True  # 默认使用fp16
    
    # 设置日志
    logger = setup_logger('dollar_bar_features')
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return 1
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始处理成交额K线特征工程")
    logger.info(f"输入文件: {input_path}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"目标交易对: {args.target_symbol}")
    logger.info(f"特征交易对: {args.feature_symbols}")
    logger.info(f"使用fp16: {args.use_fp16}")
    
    try:
        # 1. 加载成交额K线数据
        logger.info("加载成交额K线数据...")
        df = pd.read_parquet(input_path)
        logger.info(f"原始数据维度: {df.shape}")
        logger.info(f"数据列: {list(df.columns)}")
        
        # 2. 创建特征工程器
        feature_builder = DollarBarFeatures(use_fp16=args.use_fp16)
        
        # 3. 构建特征
        logger.info("开始构建特征...")
        features_df = feature_builder.build_comprehensive_features(
            df=df,
            target_symbol=args.target_symbol,
            feature_symbols=args.feature_symbols,
            future_periods=args.future_periods
        )
        
        # 4. 保存特征数据
        logger.info("保存特征数据...")
        features_df.to_parquet(output_path, index=False)
        logger.info(f"特征数据已保存到: {output_path}")
        
        # 5. 输出统计信息
        logger.info("特征工程统计信息:")
        logger.info(f"  最终数据维度: {features_df.shape}")
        logger.info(f"  特征数量: {features_df.shape[1] - len([col for col in features_df.columns if 'target' in col or 'future_return' in col])}")
        logger.info(f"  内存使用: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # 数据类型分布
        dtype_counts = features_df.dtypes.value_counts()
        logger.info("数据类型分布:")
        for dtype, count in dtype_counts.items():
            logger.info(f"  {dtype}: {count} 列")
        
        # 目标变量分布
        if 'target' in features_df.columns:
            target_counts = features_df['target'].value_counts().sort_index()
            logger.info("目标变量分布:")
            for val, count in target_counts.items():
                percentage = count / len(features_df) * 100
                label = {-1: "下跌", 0: "横盘", 1: "上涨"}[val]
                logger.info(f"  {label}({val}): {count} ({percentage:.2f}%)")
        
        # 6. 特征重要性分析（可选）
        if args.analysis:
            logger.info("开始特征重要性分析...")
            try:
                analysis_result = feature_builder.get_feature_importance_analysis(features_df)
                
                # 保存分析结果
                analysis_path = output_path.with_suffix('.analysis.json')
                with open(analysis_path, 'w', encoding='utf-8') as f:
                    # 转换numpy数组为列表以便JSON序列化
                    serializable_result = {}
                    for key, value in analysis_result.items():
                        if isinstance(value, dict):
                            serializable_result[key] = {k: float(v) if isinstance(v, np.floating) else v 
                                                      for k, v in value.items()}
                        else:
                            serializable_result[key] = value
                    
                    json.dump(serializable_result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"特征重要性分析结果已保存到: {analysis_path}")
                
                # 输出前10个最重要的特征
                feature_importance = analysis_result['feature_importance']
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                logger.info("前10个最重要的特征:")
                for i, (feature, importance) in enumerate(top_features, 1):
                    logger.info(f"  {i:2d}. {feature}: {importance:.4f}")
                
            except Exception as e:
                logger.error(f"特征重要性分析失败: {e}")
        
        # 7. 保存特征元数据
        metadata = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'target_symbol': args.target_symbol,
            'feature_symbols': args.feature_symbols,
            'future_periods': args.future_periods,
            'use_fp16': args.use_fp16,
            'original_shape': df.shape,
            'final_shape': features_df.shape,
            'feature_count': features_df.shape[1] - len([col for col in features_df.columns if 'target' in col or 'future_return' in col]),
            'memory_usage_mb': features_df.memory_usage(deep=True).sum() / 1024**2,
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"元数据已保存到: {metadata_path}")
        logger.info("成交额K线特征工程完成!")
        
        return 0
        
    except Exception as e:
        logger.error(f"特征工程处理失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


def load_dollar_bar_features(feature_file: str, 
                           metadata_file: Optional[str] = None) -> tuple:
    """
    加载成交额K线特征数据
    
    Args:
        feature_file: 特征数据文件路径
        metadata_file: 元数据文件路径（可选）
        
    Returns:
        (features_df, metadata)
    """
    # 加载特征数据
    features_df = pd.read_parquet(feature_file)
    
    # 加载元数据
    metadata = None
    if metadata_file is None:
        metadata_file = Path(feature_file).with_suffix('.metadata.json')
    
    if Path(metadata_file).exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return features_df, metadata


def validate_dollar_bar_features(features_df: pd.DataFrame) -> dict:
    """
    验证成交额K线特征数据质量
    
    Args:
        features_df: 特征数据DataFrame
        
    Returns:
        验证结果字典
    """
    # 分离特征和目标
    target_cols = [col for col in features_df.columns if 'target' in col or 'future_return' in col]
    feature_cols = [col for col in features_df.columns if col not in target_cols and col not in ['start_time', 'end_time']]
    
    validation_result = {
        'total_samples': len(features_df),
        'total_features': len(feature_cols),
        'target_columns': target_cols,
        'missing_values': {},
        'infinite_values': {},
        'data_types': {},
        'memory_usage_mb': features_df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # 检查缺失值和无限值
    for col in feature_cols:
        missing_count = features_df[col].isnull().sum()
        infinite_count = np.isinf(features_df[col]).sum()
        
        if missing_count > 0:
            validation_result['missing_values'][col] = missing_count
        if infinite_count > 0:
            validation_result['infinite_values'][col] = infinite_count
        
        validation_result['data_types'][col] = str(features_df[col].dtype)
    
    # 目标变量分布
    if 'target' in features_df.columns:
        target_distribution = features_df['target'].value_counts().to_dict()
        validation_result['target_distribution'] = target_distribution
    
    return validation_result


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 