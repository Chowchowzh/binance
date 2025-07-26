#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于三分类标签法的特征工程演示脚本
使用Triple Barrier Method (TBM) 构建更高质量的标签和特征
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_processing.features import build_features_with_tbm, analyze_tbm_features_quality
from utils.logger import get_logger

# 设置日志
logger = get_logger(__name__)


def demo_tbm_feature_engineering():
    """演示TBM特征工程的完整流程"""
    
    print("="*60)
    print("基于三分类标签法的特征工程演示")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载示例数据...")
    
    # 这里假设您有成交额K线数据
    # 实际使用时请替换为您的数据路径
    data_path = "processed_data/dollar_bars.parquet"
    
    if not os.path.exists(data_path):
        print(f"警告：未找到数据文件 {data_path}")
        print("请先运行数据收集和成交额K线生成脚本")
        print("或者修改此脚本中的数据路径")
        return
    
    try:
        df = pd.read_parquet(data_path)
        print(f"数据加载成功，维度: {df.shape}")
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
        print(f"列名: {list(df.columns)}")
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # 2. 配置TBM参数
    print("\n2. 配置TBM参数...")
    
    tbm_configs = [
        {
            'name': '保守策略',
            'profit_factor': 1.5,
            'loss_factor': 1.0,
            'volatility_window': 30,
            'max_holding_period': 100,
            'use_symmetric_barriers': False
        },
        {
            'name': '积极策略', 
            'profit_factor': 2.5,
            'loss_factor': 1.5,
            'volatility_window': 20,
            'max_holding_period': 60,
            'use_symmetric_barriers': False
        },
        {
            'name': '对称策略',
            'profit_factor': 2.0,
            'loss_factor': 2.0,
            'volatility_window': 25,
            'max_holding_period': 80,
            'use_symmetric_barriers': True
        }
    ]
    
    results = {}
    
    # 3. 对每种配置进行特征工程
    for config in tbm_configs:
        print(f"\n3. 执行 {config['name']} 的特征工程...")
        
        try:
            feature_df = build_features_with_tbm(
                df=df,
                target_symbol='ETHUSDT',
                data_type='dollar_bars',
                use_fp16=True,
                profit_factor=config['profit_factor'],
                loss_factor=config['loss_factor'],
                volatility_window=config['volatility_window'],
                max_holding_period=config['max_holding_period'],
                use_symmetric_barriers=config['use_symmetric_barriers'],
                min_return_threshold=0.0001,
                volatility_method='log_return',
                use_cusum_events=False,  # 可以尝试设置为True使用CUSUM过滤器
                n_jobs=1  # 可以设置为-1使用所有CPU核心
            )
            
            # 分析特征质量
            quality_analysis = analyze_tbm_features_quality(feature_df)
            
            results[config['name']] = {
                'features': feature_df,
                'analysis': quality_analysis,
                'config': config
            }
            
            print(f"{config['name']} 完成:")
            print(f"  - 特征数量: {quality_analysis['total_features']}")
            print(f"  - 有效标签: {quality_analysis['valid_labels']}")
            print(f"  - 标签覆盖率: {quality_analysis['label_coverage']:.2%}")
            print(f"  - 内存使用: {quality_analysis['memory_usage_mb']:.2f} MB")
            
        except Exception as e:
            logger.error(f"{config['name']} 执行失败: {e}")
            continue
    
    # 4. 比较不同策略的效果
    print("\n4. 比较不同策略的效果...")
    compare_strategies(results)
    
    # 5. 保存结果
    print("\n5. 保存结果...")
    save_results(results)
    
    # 6. 可视化分析
    print("\n6. 生成可视化分析...")
    visualize_results(results)
    
    print("\n特征工程演示完成！")


def compare_strategies(results: dict):
    """比较不同TBM策略的效果"""
    
    print("\n策略对比分析:")
    print("-" * 80)
    print(f"{'策略名称':<12} {'标签覆盖率':<10} {'止盈比例':<8} {'止损比例':<8} {'中性比例':<8} {'特征数':<6}")
    print("-" * 80)
    
    for name, result in results.items():
        analysis = result['analysis']
        
        if 'label_distribution' in analysis:
            dist = analysis['label_distribution']
            total = sum(dist.values())
            
            profit_pct = dist.get(1, 0) / total * 100 if total > 0 else 0
            loss_pct = dist.get(-1, 0) / total * 100 if total > 0 else 0
            neutral_pct = dist.get(0, 0) / total * 100 if total > 0 else 0
            
            print(f"{name:<12} {analysis['label_coverage']:<10.2%} {profit_pct:<8.1f}% {loss_pct:<8.1f}% {neutral_pct:<8.1f}% {analysis['total_features']:<6}")
    
    print("-" * 80)
    
    # 详细收益分析
    print("\n各策略收益分析:")
    for name, result in results.items():
        analysis = result['analysis']
        print(f"\n{name}:")
        
        for label in [-1, 0, 1]:
            key = f'returns_label_{label}'
            if key in analysis:
                stats = analysis[key]
                label_name = {-1: "止损", 0: "中性", 1: "止盈"}[label]
                print(f"  {label_name}: 数量={stats['count']}, 平均收益={stats['mean']:.4f}, 标准差={stats['std']:.4f}")


def save_results(results: dict):
    """保存特征工程结果"""
    
    output_dir = "processed_data/tbm_features"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, result in results.items():
        # 保存特征数据
        features_df = result['features']
        safe_name = name.replace(' ', '_').lower()
        
        features_path = f"{output_dir}/{safe_name}_features.parquet"
        features_df.to_parquet(features_path)
        print(f"保存 {name} 特征数据到: {features_path}")
        
        # 保存分析报告
        analysis_path = f"{output_dir}/{safe_name}_analysis.txt"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"{name} TBM特征工程分析报告\n")
            f.write("="*50 + "\n\n")
            
            analysis = result['analysis']
            config = result['config']
            
            f.write("配置参数:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\n基本统计:\n")
            f.write(f"  总样本数: {analysis['total_samples']}\n")
            f.write(f"  有效标签数: {analysis['valid_labels']}\n")
            f.write(f"  标签覆盖率: {analysis['label_coverage']:.2%}\n")
            f.write(f"  特征数量: {analysis['total_features']}\n")
            f.write(f"  内存使用: {analysis['memory_usage_mb']:.2f} MB\n")
            
            if 'label_distribution' in analysis:
                f.write(f"\n标签分布:\n")
                for label, count in analysis['label_distribution'].items():
                    pct = count / analysis['valid_labels'] * 100
                    label_name = {-1: "止损", 0: "中性", 1: "止盈"}[int(label)]
                    f.write(f"  {label_name}: {count} ({pct:.2f}%)\n")
        
        print(f"保存 {name} 分析报告到: {analysis_path}")


def visualize_results(results: dict):
    """生成可视化分析图表"""
    
    if not results:
        print("没有结果可用于可视化")
        return
    
    output_dir = "processed_data/tbm_features"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 标签分布对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TBM策略对比分析', fontsize=16)
    
    # 标签分布饼图
    for i, (name, result) in enumerate(results.items()):
        if i >= 3:  # 最多显示3个策略
            break
            
        row = i // 2
        col = i % 2
        
        analysis = result['analysis']
        if 'label_distribution' in analysis:
            dist = analysis['label_distribution']
            labels = ['止损', '中性', '止盈']
            sizes = [dist.get(-1, 0), dist.get(0, 0), dist.get(1, 0)]
            colors = ['red', 'gray', 'green']
            
            axes[row, col].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[row, col].set_title(f'{name} 标签分布')
    
    # 如果策略少于4个，隐藏多余的子图
    if len(results) < 4:
        for i in range(len(results), 4):
            row = i // 2
            col = i % 2
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 收益分布对比
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for name, result in results.items():
        features_df = result['features']
        returns = features_df['future_return'].dropna()
        
        if len(returns) > 0:
            ax.hist(returns, bins=50, alpha=0.6, label=name, density=True)
    
    ax.set_xlabel('收益率')
    ax.set_ylabel('密度')
    ax.set_title('各策略收益率分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/returns_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 持仓期分析
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for name, result in results.items():
        features_df = result['features']
        holding_periods = features_df['tbm_holding_period'].dropna()
        
        if len(holding_periods) > 0:
            ax.hist(holding_periods, bins=30, alpha=0.6, label=name, density=True)
    
    ax.set_xlabel('持仓期 (K线数)')
    ax.set_ylabel('密度')
    ax.set_title('各策略持仓期分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/holding_period_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {output_dir}/")


def demo_with_sample_data():
    """使用模拟数据进行演示（当没有真实数据时）"""
    
    print("使用模拟数据进行TBM特征工程演示...")
    
    # 生成模拟的成交额K线数据
    np.random.seed(42)
    n_samples = 5000
    
    # 模拟价格随机游走
    returns = np.random.normal(0, 0.005, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # 创建模拟的成交额K线数据
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(15, 1, n_samples),
        'trades': np.random.poisson(1000, n_samples),
        'duration_minutes': np.random.exponential(60, n_samples),
        'bar_count': np.random.poisson(50, n_samples),
    })
    
    # 添加时间索引
    start_time = pd.Timestamp('2024-01-01')
    df.index = pd.date_range(start=start_time, periods=n_samples, freq='5min')
    
    print(f"模拟数据生成完成，维度: {df.shape}")
    
    # 执行TBM特征工程
    try:
        feature_df = build_features_with_tbm(
            df=df,
            target_symbol='ETHUSDT',  # 模拟数据中没有符号前缀
            data_type='dollar_bars',
            use_fp16=True,
            profit_factor=2.0,
            loss_factor=1.0,
            volatility_window=20,
            max_holding_period=60,
            use_symmetric_barriers=False,
            min_return_threshold=0.0001,
            n_jobs=1
        )
        
        # 分析结果
        analysis = analyze_tbm_features_quality(feature_df)
        
        print("\n模拟数据TBM特征工程结果:")
        print(f"  特征数量: {analysis['total_features']}")
        print(f"  有效标签: {analysis['valid_labels']}")
        print(f"  标签覆盖率: {analysis['label_coverage']:.2%}")
        
        # 保存模拟结果
        output_dir = "processed_data/tbm_features"
        os.makedirs(output_dir, exist_ok=True)
        
        feature_df.to_parquet(f"{output_dir}/demo_features.parquet")
        print(f"\n模拟特征数据已保存到: {output_dir}/demo_features.parquet")
        
        return feature_df
        
    except Exception as e:
        logger.error(f"模拟演示失败: {e}")
        return None


if __name__ == "__main__":
    try:
        # 首先尝试使用真实数据
        demo_tbm_feature_engineering()
    except Exception as e:
        print(f"真实数据演示失败: {e}")
        print("切换到模拟数据演示...")
        
        # 如果真实数据不可用，使用模拟数据
        demo_with_sample_data() 