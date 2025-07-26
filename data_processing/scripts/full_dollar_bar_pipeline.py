#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的成交额K线数据处理流程
从原始分钟数据到成交额K线，再到完整的特征工程
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import List, Optional, Dict
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_processing.dollar_bars import DollarBarsGenerator
from data_processing.features.dollar_bar_features import DollarBarFeatures
from utils.logger import setup_logger


def generate_dollar_bars_pipeline(input_file: str, 
                                 symbols: List[str] = ['ETHUSDT'],
                                 auto_threshold: bool = True,
                                 manual_threshold: Optional[float] = None,
                                 output_dir: str = 'processed_data') -> Dict[str, str]:
    """
    生成成交额K线数据
    
    Args:
        input_file: 原始分钟数据文件
        symbols: 交易对列表
        auto_threshold: 是否自动计算阈值
        manual_threshold: 手动指定的阈值
        output_dir: 输出目录
        
    Returns:
        生成的文件路径字典
    """
    logger = setup_logger('dollar_bars_generator')
    
    # 加载原始数据
    logger.info("加载原始分钟数据...")
    df = pd.read_parquet(input_file)
    logger.info(f"原始数据维度: {df.shape}")
    
    # 创建成交额K线生成器
    if manual_threshold:
        generator = DollarBarsGenerator(threshold_usd=manual_threshold, auto_threshold=False)
    else:
        generator = DollarBarsGenerator(auto_threshold=auto_threshold)
    
    # 生成成交额K线
    if len(symbols) == 1:
        # 单个交易对
        symbol = symbols[0]
        threshold = manual_threshold if manual_threshold else None
        dollar_bars = generator.generate_dollar_bars(df, symbol, threshold)
        
        # 保存
        saved_files = generator.save_dollar_bars(
            dollar_bars, 
            output_dir=output_dir, 
            filename_prefix=f'dollar_bars_{symbol}'
        )
        
        # 分析统计特性
        stats = generator.analyze_dollar_bars_statistics(dollar_bars, symbol)
        
    else:
        # 多个交易对
        thresholds = {symbol: manual_threshold for symbol in symbols} if manual_threshold else None
        multi_dollar_bars = generator.generate_multi_symbol_dollar_bars(df, symbols, thresholds)
        
        # 保存
        saved_files = generator.save_dollar_bars(multi_dollar_bars, output_dir=output_dir)
        
        # 分析每个交易对的统计特性
        for symbol, dollar_bars in multi_dollar_bars.items():
            stats = generator.analyze_dollar_bars_statistics(dollar_bars, symbol)
    
    logger.info(f"成交额K线生成完成，文件保存在: {saved_files}")
    return saved_files


def build_features_pipeline(dollar_bars_file: str,
                          target_symbol: str = 'ETHUSDT',
                          feature_symbols: List[str] = None,
                          future_periods: int = 15,
                          use_fp16: bool = True,
                          output_file: str = None) -> str:
    """
    构建成交额K线特征工程
    
    Args:
        dollar_bars_file: 成交额K线数据文件
        target_symbol: 目标交易对
        feature_symbols: 特征交易对列表
        future_periods: 未来收益预测期数
        use_fp16: 是否使用fp16格式
        output_file: 输出文件路径
        
    Returns:
        特征文件路径
    """
    logger = setup_logger('feature_builder')
    
    if feature_symbols is None:
        feature_symbols = [target_symbol]
    
    if output_file is None:
        output_file = dollar_bars_file.replace('.parquet', '_features.parquet')
    
    # 加载成交额K线数据
    logger.info("加载成交额K线数据...")
    df = pd.read_parquet(dollar_bars_file)
    logger.info(f"成交额K线数据维度: {df.shape}")
    
    # 创建特征工程器
    feature_builder = DollarBarFeatures(use_fp16=use_fp16)
    
    # 构建特征
    logger.info("开始构建特征...")
    features_df = feature_builder.build_comprehensive_features(
        df=df,
        target_symbol=target_symbol,
        feature_symbols=feature_symbols,
        future_periods=future_periods
    )
    
    # 保存特征数据
    logger.info("保存特征数据...")
    features_df.to_parquet(output_file, index=False)
    logger.info(f"特征数据已保存到: {output_file}")
    
    # 保存元数据
    metadata = {
        'dollar_bars_file': dollar_bars_file,
        'output_file': output_file,
        'target_symbol': target_symbol,
        'feature_symbols': feature_symbols,
        'future_periods': future_periods,
        'use_fp16': use_fp16,
        'original_shape': df.shape,
        'final_shape': features_df.shape,
        'feature_count': features_df.shape[1] - len([col for col in features_df.columns if 'target' in col or 'future_return' in col]),
        'memory_usage_mb': features_df.memory_usage(deep=True).sum() / 1024**2,
        'processing_timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = Path(output_file).with_suffix('.metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return output_file


def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description='完整的成交额K线数据处理流程')
    
    # 输入输出参数
    parser.add_argument('--input', type=str, default='processed_data/raw_data.parquet',
                       help='原始分钟数据文件路径')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='输出目录')
    
    # 成交额K线参数
    parser.add_argument('--symbols', type=str, nargs='+', default=['ETHUSDT'],
                       help='交易对列表')
    parser.add_argument('--auto_threshold', action='store_true', default=True,
                       help='是否自动计算成交额阈值')
    parser.add_argument('--manual_threshold', type=float,
                       help='手动指定成交额阈值（美元）')
    
    # 特征工程参数
    parser.add_argument('--target_symbol', type=str, default='ETHUSDT',
                       help='目标交易对')
    parser.add_argument('--feature_symbols', type=str, nargs='+',
                       help='用于特征工程的交易对列表（默认与symbols相同）')
    parser.add_argument('--future_periods', type=int, default=15,
                       help='未来收益预测期数')
    parser.add_argument('--use_fp16', action='store_true',
                       help='是否使用fp16格式')
    parser.add_argument('--no_fp16', action='store_true',
                       help='不使用fp16格式')
    
    # 流程控制参数
    parser.add_argument('--skip_dollar_bars', action='store_true',
                       help='跳过成交额K线生成，直接进行特征工程')
    parser.add_argument('--skip_features', action='store_true',
                       help='只生成成交额K线，跳过特征工程')
    parser.add_argument('--analysis', action='store_true',
                       help='进行特征重要性分析')
    
    args = parser.parse_args()
    
    # 处理fp16设置
    if args.no_fp16:
        args.use_fp16 = False
    elif not hasattr(args, 'use_fp16') or not args.use_fp16:
        args.use_fp16 = True  # 默认使用fp16
    
    # 设置日志
    logger = setup_logger('dollar_bar_pipeline')
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return 1
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置默认特征交易对
    if args.feature_symbols is None:
        args.feature_symbols = args.symbols
    
    logger.info("="*80)
    logger.info("开始完整的成交额K线数据处理流程")
    logger.info("="*80)
    logger.info(f"输入文件: {input_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"交易对: {args.symbols}")
    logger.info(f"目标交易对: {args.target_symbol}")
    logger.info(f"特征交易对: {args.feature_symbols}")
    logger.info(f"使用fp16: {args.use_fp16}")
    
    start_time = time.time()
    
    try:
        # 第一步：生成成交额K线
        if not args.skip_dollar_bars:
            logger.info("\n" + "="*50)
            logger.info("第一步：生成成交额K线")
            logger.info("="*50)
            
            saved_files = generate_dollar_bars_pipeline(
                input_file=str(input_path),
                symbols=args.symbols,
                auto_threshold=args.auto_threshold,
                manual_threshold=args.manual_threshold,
                output_dir=str(output_dir)
            )
            
            # 选择主要的成交额K线文件用于特征工程
            main_dollar_bars_file = saved_files.get(args.target_symbol) or saved_files.get('default')
            
        else:
            # 跳过生成，查找现有文件
            logger.info("跳过成交额K线生成，查找现有文件...")
            main_dollar_bars_file = str(output_dir / f'dollar_bars_{args.target_symbol}.parquet')
            if not Path(main_dollar_bars_file).exists():
                logger.error(f"找不到成交额K线文件: {main_dollar_bars_file}")
                return 1
        
        # 第二步：特征工程
        if not args.skip_features:
            logger.info("\n" + "="*50)
            logger.info("第二步：特征工程")
            logger.info("="*50)
            
            feature_output_file = str(output_dir / f'dollar_bar_features_{args.target_symbol}.parquet')
            
            feature_file = build_features_pipeline(
                dollar_bars_file=main_dollar_bars_file,
                target_symbol=args.target_symbol,
                feature_symbols=args.feature_symbols,
                future_periods=args.future_periods,
                use_fp16=args.use_fp16,
                output_file=feature_output_file
            )
            
            # 第三步：特征重要性分析（可选）
            if args.analysis:
                logger.info("\n" + "="*50)
                logger.info("第三步：特征重要性分析")
                logger.info("="*50)
                
                try:
                    # 加载特征数据
                    features_df = pd.read_parquet(feature_file)
                    
                    # 创建特征工程器进行分析
                    feature_builder = DollarBarFeatures(use_fp16=args.use_fp16)
                    analysis_result = feature_builder.get_feature_importance_analysis(features_df)
                    
                    # 保存分析结果
                    analysis_path = Path(feature_file).with_suffix('.analysis.json')
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
        
        # 总结
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("流程完成总结")
        logger.info("="*80)
        logger.info(f"总处理时间: {processing_time:.2f} 秒")
        
        if not args.skip_dollar_bars:
            logger.info(f"成交额K线文件: {main_dollar_bars_file}")
        
        if not args.skip_features:
            logger.info(f"特征数据文件: {feature_file}")
            
            # 输出最终统计信息
            features_df = pd.read_parquet(feature_file)
            logger.info(f"最终特征维度: {features_df.shape}")
            logger.info(f"内存使用: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # 目标变量分布
            if 'target' in features_df.columns:
                target_counts = features_df['target'].value_counts().sort_index()
                logger.info("目标变量分布:")
                for val, count in target_counts.items():
                    percentage = count / len(features_df) * 100
                    label = {-1: "下跌", 0: "横盘", 1: "上涨"}[val]
                    logger.info(f"  {label}({val}): {count} ({percentage:.2f}%)")
        
        logger.info("完整流程执行成功!")
        return 0
        
    except Exception as e:
        logger.error(f"流程执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 