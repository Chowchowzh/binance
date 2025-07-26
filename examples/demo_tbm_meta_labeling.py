#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三分类标签法 (TBM) 和元标签技术演示脚本

这个脚本演示了如何使用高级机器学习技术来提升交易信号质量：
1. 三分类标签法：使用动态边界生成路径依赖的标签
2. 元标签技术：两阶段学习框架提升信号精度
3. 增强信号生成：结合两种技术生成高质量交易信号
4. 强化学习交易：基于MDP的Actor-Critic交易策略

使用方法：
    python demo_tbm_meta_labeling.py [--mode MODE] [--data_path PATH]
    
参数：
    --mode: 运行模式 ('train', 'inference', 'demo', 'rl_demo', 'rl_train')
    --data_path: 数据文件路径（可选）
"""

import sys
import os
import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.features.triple_barrier_labeling import TripleBarrierLabeler
from data_processing.features.advanced_features import AdvancedFeatureEngineer, add_advanced_features
from strategy.training.advanced_train_pipeline import AdvancedTrainingPipeline
from strategy.training.enhanced_signal_generator import EnhancedSignalGenerator
from strategy.analysis.advanced_model_evaluation import AdvancedModelEvaluator, comprehensive_model_evaluation
from config.settings import load_config

# 强化学习相关导入
try:
    from strategy.reinforcement_learning.rl_training_pipeline import RLTrainingPipeline
    from strategy.reinforcement_learning.robust_backtester import RobustBacktester
    from strategy.reinforcement_learning.mdp_environment import TradingMDPEnvironment
    from strategy.reinforcement_learning.actor_critic_agent import ActorCriticAgent
    RL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 强化学习模块导入失败: {e}")
    RL_AVAILABLE = False

warnings.filterwarnings('ignore')


def demo_tbm_labeling():
    """演示三分类标签法的使用"""
    print("=" * 80)
    print("🎯 演示 1: 三分类标签法 (Triple-Barrier Method)")
    print("=" * 80)
    
    # 创建示例价格数据
    np.random.seed(42)
    n_periods = 1000
    
    # 模拟价格随机游走
    returns = np.random.normal(0.0001, 0.02, n_periods)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)))
    
    print(f"创建示例价格数据: {len(prices)} 个数据点")
    print(f"价格范围: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # 初始化TBM标签器
    tbm_labeler = TripleBarrierLabeler(
        profit_factor=2.0,      # 止盈因子
        loss_factor=1.0,        # 止损因子
        volatility_window=20,   # 波动率窗口
        max_holding_period=50,  # 最大持仓期
        min_return_threshold=0.001  # 最小收益阈值
    )
    
    # 生成TBM标签
    tbm_labels = tbm_labeler.generate_triple_barrier_labels(prices)
    
    # 分析标签质量
    quality_analysis = tbm_labeler.analyze_label_quality(tbm_labels)
    
    print(f"\n📊 TBM标签质量分析:")
    print(f"  - 总事件数: {quality_analysis['total_events']}")
    print(f"  - 平均持仓期: {quality_analysis['holding_period_stats']['mean']:.2f} 期")
    print(f"  - 平均收益率: {quality_analysis['return_stats']['mean']:.4f}")
    print(f"  - 收益率标准差: {quality_analysis['return_stats']['std']:.4f}")
    
    # 显示各类标签的胜率
    for label in [-1, 0, 1]:
        if f'returns_label_{label}' in quality_analysis:
            stats = quality_analysis[f'returns_label_{label}']
            label_name = {-1: "止损", 0: "时间到期", 1: "止盈"}[label]
            print(f"  - {label_name}标签: {stats['count']} 个, 平均收益: {stats['mean']:.4f}")
    
    return tbm_labels, quality_analysis


def demo_cusum_events():
    """演示CUSUM事件过滤器"""
    print("\n" + "=" * 60)
    print("🔍 演示 1.1: CUSUM事件过滤器")
    print("=" * 60)
    
    # 创建带趋势的价格数据
    np.random.seed(42)
    n_periods = 500
    
    # 加入一些趋势和突发事件
    base_returns = np.random.normal(0.0001, 0.01, n_periods)
    # 在特定位置添加突发事件
    base_returns[100:110] += 0.005   # 上涨趋势
    base_returns[200:210] -= 0.008   # 下跌趋势
    base_returns[350:360] += 0.003   # 小幅上涨
    
    prices = pd.Series(100 * np.exp(np.cumsum(base_returns)))
    
    # 初始化TBM标签器
    tbm_labeler = TripleBarrierLabeler(
        profit_factor=1.5,
        loss_factor=1.0,
        max_holding_period=30
    )
    
    # 使用CUSUM过滤器生成事件
    cusum_events = tbm_labeler.generate_cusum_events(prices, threshold=0.01)
    
    print(f"CUSUM事件过滤结果:")
    print(f"  - 总价格点数: {len(prices)}")
    print(f"  - CUSUM触发事件: {len(cusum_events)}")
    print(f"  - 事件密度: {len(cusum_events)/len(prices)*100:.2f}%")
    
    if len(cusum_events) > 0:
        print(f"  - 事件位置示例: {cusum_events[:10]}")
    
    return cusum_events


def demo_advanced_features():
    """演示高级特征工程"""
    print("\n" + "=" * 80)
    print("🔧 演示 3: 高级特征工程")
    print("=" * 80)
    
    # 创建示例OHLCV数据
    np.random.seed(42)
    n_periods = 500
    
    # 生成价格数据
    base_price = 100
    returns = np.random.normal(0.0001, 0.02, n_periods)
    close = base_price * np.exp(np.cumsum(returns))
    
    # 转换为pandas Series以便使用shift方法
    close = pd.Series(close)
    
    # 生成OHLC数据
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    open_price = close.shift(1).fillna(close.iloc[0])
    
    # 生成成交量数据
    volume = np.random.lognormal(10, 0.5, n_periods)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    print(f"创建示例OHLCV数据: {len(df)} 个数据点")
    print(f"价格范围: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # 初始化高级特征工程器
    feature_engineer = AdvancedFeatureEngineer(
        volatility_windows=[5, 10, 20],
        microstructure_windows=[5, 15],
        statistical_windows=[10, 20, 50]
    )
    
    # 构建所有特征
    features_df = feature_engineer.build_all_features(df)
    
    print(f"\n📊 特征工程结果:")
    print(f"  - 原始特征: {len(df.columns)}")
    print(f"  - 总特征数: {len(features_df.columns)}")
    print(f"  - 新增特征: {len(features_df.columns) - len(df.columns)}")
    
    # 特征类别统计
    feature_categories = {
        '波动率特征': len([c for c in features_df.columns if 'volatility' in c or 'vol_' in c]),
        '微观结构特征': len([c for c in features_df.columns if any(x in c for x in ['volume', 'vwap', 'shadow', 'liquidity'])]),
        '统计特征': len([c for c in features_df.columns if any(x in c for x in ['skewness', 'kurtosis', 'quantile', 'momentum'])]),
        '技术指标': len([c for c in features_df.columns if any(x in c for x in ['rsi', 'macd', 'bb_', 'adx'])]),
        '分形特征': len([c for c in features_df.columns if any(x in c for x in ['frac_diff', 'hurst', 'dfa'])])
    }
    
    print(f"\n📈 特征类别分布:")
    for category, count in feature_categories.items():
        if count > 0:
            print(f"  - {category}: {count} 个")
    
    # 显示一些重要特征的统计
    important_features = [c for c in features_df.columns if any(x in c for x in ['volatility_simple', 'rsi', 'macd'])][:5]
    if important_features:
        print(f"\n🎯 重要特征统计:")
        for feature in important_features:
            values = features_df[feature].dropna()
            if len(values) > 0:
                print(f"  - {feature}: 均值={values.mean():.4f}, 标准差={values.std():.4f}")
    
    return features_df


def demo_model_evaluation():
    """演示高级模型评估（需要模拟数据）"""
    print("\n" + "=" * 80)
    print("📊 演示 4: 高级模型评估")
    print("=" * 80)
    
    # 生成模拟的分类结果用于演示
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟真实标签（三分类：0=下跌, 1=中性, 2=上涨）
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    # 模拟预测结果（有一定准确性）
    y_pred = y_true.copy()
    # 添加一些预测错误
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.25), replace=False)
    y_pred[error_indices] = np.random.choice([0, 1, 2], size=len(error_indices))
    
    # 模拟预测概率
    y_proba = np.random.dirichlet([1, 1, 1], size=n_samples)
    # 让概率与预测标签更一致
    for i in range(n_samples):
        y_proba[i, y_pred[i]] = max(y_proba[i, y_pred[i]], 0.6)
        y_proba[i] = y_proba[i] / y_proba[i].sum()  # 重新标准化
    
    print(f"创建模拟分类结果: {n_samples} 个样本")
    print(f"真实标签分布: {np.bincount(y_true)}")
    print(f"预测标签分布: {np.bincount(y_pred)}")
    
    # 初始化模型评估器
    evaluator = AdvancedModelEvaluator(save_plots=False)  # 不保存图表用于演示
    
    # 分类性能评估
    class_names = ['下跌', '中性', '上涨']
    classification_results = evaluator.evaluate_classification_performance(
        y_true, y_pred, y_proba, class_names
    )
    
    # 生成模拟信号和收益率用于信号质量分析
    signals = np.random.normal(0, 0.5, n_samples)
    returns = np.random.normal(0.001, 0.02, n_samples)
    # 让信号与收益有一定相关性
    signals = signals + 0.3 * returns
    
    print(f"\n生成模拟交易信号和收益率...")
    
    # 信号质量分析
    signal_quality_results = evaluator.analyze_signal_quality(signals, returns)
    
    # 回测性能分析
    backtesting_results = evaluator.analyze_backtesting_performance(
        signals, returns, initial_capital=100000
    )
    
    print(f"\n📈 评估结果摘要:")
    print(f"✅ 分类性能:")
    print(f"  - 准确率: {classification_results['accuracy']:.4f}")
    print(f"  - F1分数: {classification_results['f1_macro']:.4f}")
    print(f"  - MCC: {classification_results['matthews_corrcoef']:.4f}")
    
    print(f"\n✅ 信号质量:")
    stats = signal_quality_results['signal_stats']
    print(f"  - 信号均值: {stats['mean']:.4f}")
    print(f"  - 信号标准差: {stats['std']:.4f}")
    if 'signal_return_correlation' in signal_quality_results:
        print(f"  - 信号-收益相关性: {signal_quality_results['signal_return_correlation']:.4f}")
    
    print(f"\n✅ 回测性能:")
    returns_data = backtesting_results['returns']
    risk_data = backtesting_results['risk']
    ratios_data = backtesting_results['ratios']
    
    print(f"  - 总收益率: {returns_data['total_return']:.2%}")
    print(f"  - 年化收益率: {returns_data['annual_return']:.2%}")
    print(f"  - 年化波动率: {risk_data['volatility']:.2%}")
    print(f"  - 夏普比率: {ratios_data['sharpe_ratio']:.4f}")
    print(f"  - 最大回撤: {risk_data['max_drawdown']:.2%}")
    
    evaluation_results = {
        'classification': classification_results,
        'signal_quality': signal_quality_results,
        'backtesting': backtesting_results
    }
    
    return evaluation_results


def demo_meta_labeling():
    """演示元标签技术（需要预训练模型）"""
    print("\n" + "=" * 80)
    print("🔍 演示 2: 元标签技术 (Meta-Labeling)")
    print("=" * 80)
    
    # 检查是否有预训练模型
    config = load_config()
    
    model_path = config.model.model_save_path
    if not os.path.exists(model_path):
        print("⚠️ 未找到预训练的主模型")
        print("请先运行训练流程：python demo_tbm_meta_labeling.py --mode train")
        return None
    
    try:
        # 加载增强信号生成器
        generator = EnhancedSignalGenerator(config)
        
        # 尝试加载模型
        generator.load_models()
        
        print("✅ 模型加载成功，元标签技术可用")
        
        # 如果有测试数据，生成一些示例信号
        data_path = config.model.data_path
        if os.path.exists(data_path):
            print(f"\n使用数据文件进行演示: {data_path}")
            
            # 加载少量数据进行演示
            df = pd.read_parquet(data_path)
            demo_data = df.tail(200)  # 使用最后200行数据
            
            # 批量推理
            results = generator.batch_inference(demo_data)
            
            print(f"\n📈 元标签增强信号演示:")
            print(f"  - 处理样本数: {results['sample_count']}")
            print(f"  - 特征维度: {results['feature_count']}")
            
            # 显示信号统计
            decisions = results['decisions']
            print(f"  - 生成交易决策: {len(decisions)}")
            print(f"  - 买入信号: {(decisions['action'] == 'BUY').sum()}")
            print(f"  - 卖出信号: {(decisions['action'] == 'SELL').sum()}")
            print(f"  - 持有信号: {(decisions['action'] == 'HOLD').sum()}")
            print(f"  - 高置信度决策: {decisions['high_confidence'].sum()}")
            
            return results
        
    except Exception as e:
        print(f"❌ 元标签演示失败: {e}")
        print("可能需要先训练元标签模型")
        return None


def run_training_pipeline():
    """运行完整的训练流水线"""
    print("=" * 80)
    print("🚀 运行完整训练流水线")
    print("=" * 80)
    
    try:
        # 检查数据文件
        config = load_config()
        data_path = config.model.data_path
        
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            print("请确保有可用的特征数据文件")
            return False
        
        # 创建训练流水线
        pipeline = AdvancedTrainingPipeline(config)
        
        # 运行完整流程
        print("\n开始训练流程...")
        results = pipeline.run_complete_pipeline()
        
        if results['status'] == 'success':
            print("\n🎉 训练流水线成功完成!")
            
            # 显示结果摘要
            if 'primary_model' in results:
                accuracy = results['primary_model']['best_val_accuracy']
                print(f"✅ 主模型验证准确率: {accuracy:.2f}%")
            
            if 'meta_labeling' in results and results['meta_labeling']:
                auc = results['meta_labeling'].get('validation_auc', 0)
                print(f"✅ 元标签模型AUC: {auc:.4f}")
            
            if 'signals' in results and results['signals']:
                stats = results['signals']['statistics']
                print(f"✅ 高置信度信号比例: {stats['high_conf_signal_ratio']:.4f}")
            
            return True
        else:
            print(f"❌ 训练失败: {results.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ 训练流水线异常: {e}")
        return False


def run_inference_demo():
    """运行推理演示"""
    print("=" * 80)
    print("📈 运行增强信号推理演示")
    print("=" * 80)
    
    try:
        config = load_config()
        
        # 检查是否有训练好的模型
        if not os.path.exists(config.model.model_save_path):
            print("❌ 未找到训练好的模型")
            print("请先运行: python demo_tbm_meta_labeling.py --mode train")
            return False
        
        # 加载增强信号生成器
        generator = EnhancedSignalGenerator(config)
        generator.load_models()
        
        # 加载测试数据
        data_path = config.model.data_path
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            return False
        
        print(f"加载数据: {data_path}")
        df = pd.read_parquet(data_path)
        
        # 使用最后的数据进行推理演示
        test_data = df.tail(500)
        print(f"使用最新 {len(test_data)} 行数据进行推理")
        
        # 批量推理
        results = generator.batch_inference(test_data)
        
        # 显示结果
        print(f"\n📊 推理结果摘要:")
        print(f"  - 处理样本数: {results['sample_count']}")
        print(f"  - 特征维度: {results['feature_count']}")
        
        decisions = results['decisions']
        signals = results['signals']
        
        # 信号统计
        print(f"\n🎯 交易信号统计:")
        action_counts = decisions['action'].value_counts()
        for action, count in action_counts.items():
            percentage = count / len(decisions) * 100
            print(f"  - {action}: {count} ({percentage:.1f}%)")
        
        # 信号质量统计
        high_conf_mask = decisions['high_confidence']
        high_conf_decisions = decisions[high_conf_mask]
        
        print(f"\n⭐ 高置信度信号分析:")
        print(f"  - 高置信度信号数: {len(high_conf_decisions)}")
        print(f"  - 高置信度比例: {len(high_conf_decisions)/len(decisions)*100:.2f}%")
        
        if len(high_conf_decisions) > 0:
            high_conf_actions = high_conf_decisions['action'].value_counts()
            print(f"  - 高置信度交易分布:")
            for action, count in high_conf_actions.items():
                print(f"    {action}: {count}")
        
        # 保存结果
        save_path = 'results/inference_demo_results.pkl'
        generator.save_inference_results(results, save_path)
        
        print(f"\n💾 结果已保存到: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ 推理演示失败: {e}")
        return False


def run_rl_training_pipeline():
    """运行强化学习训练流水线"""
    if not RL_AVAILABLE:
        print("⚠️ 强化学习模块未安装或导入失败，无法运行强化学习训练。")
        return False

    print("=" * 80)
    print("🚀 运行强化学习训练流水线")
    print("=" * 80)
    
    try:
        config = load_config()
        data_path = config.model.data_path
        
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            print("请确保有可用的强化学习数据文件")
            return False
        
        # 创建强化学习训练流水线
        pipeline = RLTrainingPipeline(config)
        
        # 运行完整流程
        print("\n开始强化学习训练流程...")
        results = pipeline.run_complete_pipeline()
        
        if results['status'] == 'success':
            print("\n🎉 强化学习训练流水线成功完成!")
            
            # 显示结果摘要
            if 'actor_critic' in results:
                print(f"✅ Actor-Critic模型训练成功。")
                print(f"  - 训练轮数: {results['actor_critic']['epochs']}")
                print(f"  - 总奖励: {results['actor_critic']['total_reward']:.2f}")
                print(f"  - 平均奖励: {results['actor_critic']['mean_reward']:.4f}")
                print(f"  - 最终策略熵: {results['actor_critic']['final_policy_entropy']:.4f}")
            
            if 'mdp_env' in results and results['mdp_env']:
                print(f"✅ MDP环境训练成功。")
                print(f"  - 环境步数: {results['mdp_env']['total_steps']}")
                print(f"  - 平均步长: {results['mdp_env']['mean_step_length']:.2f}")
                print(f"  - 平均奖励: {results['mdp_env']['mean_reward']:.4f}")
            
            if 'backtester' in results and results['backtester']:
                print(f"✅ 回测器训练成功。")
                print(f"  - 回测总步数: {results['backtester']['total_steps']}")
                print(f"  - 平均步长: {results['backtester']['mean_step_length']:.2f}")
                print(f"  - 平均奖励: {results['backtester']['mean_reward']:.4f}")
            
            return True
        else:
            print(f"❌ 强化学习训练失败: {results.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ 强化学习训练流水线异常: {e}")
        return False


def run_rl_demo():
    """运行强化学习演示"""
    if not RL_AVAILABLE:
        print("⚠️ 强化学习模块未安装或导入失败，无法运行强化学习演示。")
        return False

    print("=" * 80)
    print("📈 运行强化学习演示")
    print("=" * 80)
    
    try:
        config = load_config()
        
        # 检查是否有训练好的模型
        if not os.path.exists(config.model.model_save_path):
            print("❌ 未找到强化学习模型")
            print("请先运行: python demo_tbm_meta_labeling.py --mode rl_train")
            return False
        
        # 加载强化学习模型
        agent = ActorCriticAgent(config)
        agent.load_models()
        
        # 加载测试数据
        data_path = config.model.data_path
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            return False
        
        print(f"加载数据: {data_path}")
        df = pd.read_parquet(data_path)
        
        # 创建MDP环境
        mdp_env = TradingMDPEnvironment(df, config)
        
        # 运行强化学习演示
        print("\n开始强化学习演示...")
        backtester = RobustBacktester(mdp_env, agent, config)
        results = backtester.run_episode(initial_capital=100000)
        
        print(f"\n📊 强化学习演示结果:")
        print(f"  - 总步数: {results['total_steps']}")
        print(f"  - 总奖励: {results['total_reward']:.2f}")
        print(f"  - 平均奖励: {results['mean_reward']:.4f}")
        print(f"  - 最终策略熵: {results['final_policy_entropy']:.4f}")
        
        # 回测性能分析
        backtesting_results = backtester.analyze_backtesting_performance(
            results['actions'], results['rewards'], initial_capital=100000
        )
        
        print(f"\n📈 强化学习回测性能:")
        returns_data = backtesting_results['returns']
        risk_data = backtesting_results['risk']
        ratios_data = backtesting_results['ratios']
        
        print(f"  - 总收益率: {returns_data['total_return']:.2%}")
        print(f"  - 年化收益率: {returns_data['annual_return']:.2%}")
        print(f"  - 年化波动率: {risk_data['volatility']:.2%}")
        print(f"  - 夏普比率: {ratios_data['sharpe_ratio']:.4f}")
        print(f"  - 最大回撤: {risk_data['max_drawdown']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 强化学习演示失败: {e}")
        return False


def run_complete_demo():
    """运行完整演示"""
    print("🎬 开始完整演示")
    print("=" * 80)
    
    print("\n第一部分：理论和技术演示（无需训练数据）")
    
    # 1. TBM标签演示
    tbm_results = demo_tbm_labeling()
    
    # 2. CUSUM事件过滤演示
    cusum_results = demo_cusum_events()
    
    # 3. 高级特征工程演示
    features_results = demo_advanced_features()
    
    # 4. 高级模型评估演示
    evaluation_results = demo_model_evaluation()
    
    print("\n第二部分：实际模型演示（需要训练数据）")
    
    # 5. 元标签演示（如果有预训练模型）
    meta_results = demo_meta_labeling()
    
    print("\n" + "=" * 80)
    print("🎉 完整演示结束")
    print("=" * 80)
    
    print("\n📝 演示总结:")
    print("✅ 三分类标签法：展示了动态边界和路径依赖的标签生成")
    print("✅ CUSUM过滤器：展示了结构性变化的事件检测")
    print("✅ 高级特征工程：展示了多种类型的高级金融特征构建")
    print("✅ 高级模型评估：展示了分类性能、信号质量和回测分析")
    
    if meta_results is not None:
        print("✅ 元标签技术：展示了两阶段学习框架的信号增强")
    else:
        print("⚠️ 元标签技术：需要先训练模型才能完整演示")
    
    print(f"\n📊 技术特性统计:")
    if tbm_results:
        print(f"  - TBM标签数量: {len(tbm_results[0])}")
    if cusum_results:
        print(f"  - CUSUM事件数: {len(cusum_results)}")
    if features_results is not None:
        print(f"  - 生成特征数: {len(features_results.columns)}")
    if evaluation_results:
        acc = evaluation_results['classification']['accuracy']
        print(f"  - 演示模型准确率: {acc:.2%}")
    
    print("\n📖 下一步建议:")
    print("1. 如需训练模型：python demo_tbm_meta_labeling.py --mode train")
    print("2. 如需推理演示：python demo_tbm_meta_labeling.py --mode inference")
    print("3. 查看详细实现：")
    print("   - TBM实现：data_processing/features/triple_barrier_labeling.py")
    print("   - 高级特征：data_processing/features/advanced_features.py")
    print("   - 元标签实现：strategy/training/meta_labeling.py")
    print("   - 模型评估：strategy/analysis/advanced_model_evaluation.py")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="三分类标签法和元标签技术演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
    # 运行完整演示（理论+实践）
    python demo_tbm_meta_labeling.py --mode demo
    
    # 训练模型（使用TBM标签和元标签技术）
    python demo_tbm_meta_labeling.py --mode train
    
    # 运行推理演示
    python demo_tbm_meta_labeling.py --mode inference
    
    # 运行强化学习训练
    python demo_tbm_meta_labeling.py --mode rl_train
    
    # 运行强化学习演示
    python demo_tbm_meta_labeling.py --mode rl_demo
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['demo', 'train', 'inference', 'rl_demo', 'rl_train'],
        default='demo',
        help='运行模式：demo=完整演示, train=训练模型, inference=推理演示, rl_demo=强化学习演示, rl_train=强化学习训练'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        help='数据文件路径（可选，默认使用配置文件中的路径）'
    )
    
    args = parser.parse_args()
    
    print("🚀 三分类标签法和元标签技术演示")
    print("=" * 80)
    print(f"运行模式: {args.mode}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.data_path:
        print(f"数据路径: {args.data_path}")
        # 这里可以更新配置中的数据路径
    
    print("=" * 80)
    
    try:
        if args.mode == 'demo':
            run_complete_demo()
        elif args.mode == 'train':
            success = run_training_pipeline()
            if success:
                print("\n✅ 训练完成！现在可以运行推理演示：")
                print("python demo_tbm_meta_labeling.py --mode inference")
        elif args.mode == 'inference':
            success = run_inference_demo()
            if success:
                print("\n✅ 推理演示完成！")
        elif args.mode == 'rl_train':
            success = run_rl_training_pipeline()
            if success:
                print("\n✅ 强化学习训练完成！现在可以运行强化学习演示：")
                print("python demo_tbm_meta_labeling.py --mode rl_demo")
        elif args.mode == 'rl_demo':
            success = run_rl_demo()
            if success:
                print("\n✅ 强化学习演示完成！")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 