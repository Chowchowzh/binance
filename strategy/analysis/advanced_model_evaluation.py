# -*- coding: utf-8 -*-
"""
高级模型评估模块
包含精度召回率分析、信号质量评分、回测性能分析等
基于金融机器学习最佳实践的模型评估框架
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from scipy import stats
from datetime import datetime, timedelta
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: Plotly not available. Interactive dashboards will be skipped.")

warnings.filterwarnings('ignore')


class AdvancedModelEvaluator:
    """
    高级模型评估器
    
    核心功能：
    1. 分类性能评估
    2. 信号质量分析
    3. 时间序列交叉验证
    4. 回测性能分析
    5. 特征重要性分析
    6. 模型稳定性评估
    """
    
    def __init__(self, 
                 save_plots: bool = True,
                 plot_dir: str = "evaluation_plots",
                 figsize: Tuple[int, int] = (12, 8)):
        """
        初始化高级模型评估器
        
        Args:
            save_plots: 是否保存图表
            plot_dir: 图表保存目录
            figsize: 图表尺寸
        """
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.figsize = figsize
        
        # 创建图表保存目录
        if save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
        
        print(f"初始化高级模型评估器:")
        print(f"  - 保存图表: {save_plots}")
        print(f"  - 图表目录: {plot_dir}")
    
    def evaluate_classification_performance(self,
                                          y_true: np.ndarray,
                                          y_pred: np.ndarray,
                                          y_proba: np.ndarray = None,
                                          class_names: List[str] = None) -> Dict[str, Any]:
        """
        评估分类性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            class_names: 类别名称
            
        Returns:
            分类性能评估结果
        """
        print("评估分类性能...")
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        results = {}
        
        # 基本指标
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        results['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # 分类报告
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        results['classification_report'] = report
        
        # 如果有概率预测，计算概率相关指标
        if y_proba is not None:
            if len(np.unique(y_true)) == 2:  # 二分类
                results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                results['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
            else:  # 多分类
                try:
                    results['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                    results['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
                except:
                    pass
        
        # 打印主要指标
        print(f"  - 准确率: {results['accuracy']:.4f}")
        print(f"  - 精度(宏平均): {results['precision_macro']:.4f}")
        print(f"  - 召回率(宏平均): {results['recall_macro']:.4f}")
        print(f"  - F1(宏平均): {results['f1_macro']:.4f}")
        print(f"  - MCC: {results['matthews_corrcoef']:.4f}")
        
        if y_proba is not None and 'roc_auc' in results:
            print(f"  - ROC AUC: {results['roc_auc']:.4f}")
        
        return results
    
    def analyze_signal_quality(self,
                             signals: np.ndarray,
                             returns: np.ndarray,
                             prices: np.ndarray = None,
                             signal_threshold: float = 0.1) -> Dict[str, Any]:
        """
        分析信号质量
        
        Args:
            signals: 信号序列
            returns: 收益率序列
            prices: 价格序列（可选）
            signal_threshold: 信号阈值
            
        Returns:
            信号质量分析结果
        """
        print("分析信号质量...")
        
        results = {}
        
        # 信号统计
        results['signal_stats'] = {
            'mean': np.mean(signals),
            'std': np.std(signals),
            'min': np.min(signals),
            'max': np.max(signals),
            'skewness': stats.skew(signals),
            'kurtosis': stats.kurtosis(signals)
        }
        
        # 信号分布
        strong_long = signals > signal_threshold
        strong_short = signals < -signal_threshold
        neutral = np.abs(signals) <= signal_threshold
        
        results['signal_distribution'] = {
            'strong_long_pct': np.mean(strong_long) * 100,
            'strong_short_pct': np.mean(strong_short) * 100,
            'neutral_pct': np.mean(neutral) * 100
        }
        
        # 信号有效性分析
        if len(returns) == len(signals):
            # 信号与未来收益的相关性
            results['signal_return_correlation'] = np.corrcoef(signals[:-1], returns[1:])[0, 1]
            
            # 分层分析
            signal_quantiles = np.percentile(signals, [10, 25, 50, 75, 90])
            quantile_returns = {}
            
            for i, (low, high) in enumerate(zip([np.min(signals)] + list(signal_quantiles), 
                                             list(signal_quantiles) + [np.max(signals)])):
                mask = (signals >= low) & (signals <= high)
                if np.sum(mask) > 0:
                    quantile_returns[f'Q{i+1}'] = {
                        'mean_return': np.mean(returns[mask]),
                        'sharpe_ratio': np.mean(returns[mask]) / np.std(returns[mask]) if np.std(returns[mask]) > 0 else 0,
                        'win_rate': np.mean(returns[mask] > 0),
                        'sample_count': np.sum(mask)
                    }
            
            results['quantile_analysis'] = quantile_returns
            
            # 信号衰减分析
            decay_lags = range(1, min(21, len(signals)//4))
            signal_decay = {}
            
            for lag in decay_lags:
                if lag < len(signals) and lag < len(returns):
                    corr = np.corrcoef(signals[:-lag], returns[lag:])[0, 1]
                    signal_decay[f'lag_{lag}'] = corr
            
            results['signal_decay'] = signal_decay
        
        # 信号集中度
        results['signal_concentration'] = {
            'herfindahl_index': np.sum((signals / np.sum(np.abs(signals)))**2),
            'gini_coefficient': self._calculate_gini(np.abs(signals))
        }
        
        print(f"  - 信号均值: {results['signal_stats']['mean']:.4f}")
        print(f"  - 信号标准差: {results['signal_stats']['std']:.4f}")
        print(f"  - 强多头信号: {results['signal_distribution']['strong_long_pct']:.2f}%")
        print(f"  - 强空头信号: {results['signal_distribution']['strong_short_pct']:.2f}%")
        
        if 'signal_return_correlation' in results:
            print(f"  - 信号-收益相关性: {results['signal_return_correlation']:.4f}")
        
        return results
    
    def perform_time_series_cross_validation(self,
                                           model,
                                           X: np.ndarray,
                                           y: np.ndarray,
                                           n_splits: int = 5,
                                           test_size: float = 0.2) -> Dict[str, Any]:
        """
        时间序列交叉验证
        
        Args:
            model: 模型对象
            X: 特征数据
            y: 目标数据
            n_splits: 折数
            test_size: 测试集比例
            
        Returns:
            交叉验证结果
        """
        print(f"执行时间序列交叉验证 ({n_splits} 折)...")
        
        results = {
            'fold_scores': [],
            'predictions': [],
            'true_labels': []
        }
        
        n_samples = len(X)
        test_samples = int(n_samples * test_size)
        
        for fold in range(n_splits):
            # 计算训练集和测试集的边界
            test_end = n_samples - fold * (test_samples // n_splits)
            test_start = test_end - test_samples
            train_end = test_start
            
            if train_end <= 0:
                break
            
            # 划分数据
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            
            print(f"  折 {fold+1}: 训练 [0:{train_end}], 测试 [{test_start}:{test_end}]")
            
            try:
                # 训练模型
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train)
                
                # 预测
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = None
                
                # 评估
                fold_score = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='macro'),
                    'recall': recall_score(y_test, y_pred, average='macro'),
                    'f1': f1_score(y_test, y_pred, average='macro')
                }
                
                if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                    fold_score['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                results['fold_scores'].append(fold_score)
                results['predictions'].extend(y_pred)
                results['true_labels'].extend(y_test)
                
                print(f"    准确率: {fold_score['accuracy']:.4f}, F1: {fold_score['f1']:.4f}")
                
            except Exception as e:
                print(f"    折 {fold+1} 失败: {str(e)}")
                continue
        
        # 汇总统计
        if results['fold_scores']:
            summary_stats = {}
            for metric in results['fold_scores'][0].keys():
                scores = [fold[metric] for fold in results['fold_scores']]
                summary_stats[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            
            results['summary_stats'] = summary_stats
            
            print(f"\n交叉验证汇总:")
            for metric, stats in summary_stats.items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return results
    
    def analyze_backtesting_performance(self,
                                      signals: np.ndarray,
                                      returns: np.ndarray,
                                      prices: np.ndarray = None,
                                      initial_capital: float = 100000,
                                      transaction_cost: float = 0.001) -> Dict[str, Any]:
        """
        分析回测性能
        
        Args:
            signals: 信号序列
            returns: 收益率序列
            prices: 价格序列
            initial_capital: 初始资金
            transaction_cost: 交易成本
            
        Returns:
            回测性能分析结果
        """
        print("分析回测性能...")
        
        # 计算策略收益
        strategy_returns = signals[:-1] * returns[1:]
        
        # 考虑交易成本
        position_changes = np.diff(np.concatenate([[0], signals]))
        transaction_costs = np.abs(position_changes) * transaction_cost
        strategy_returns_net = strategy_returns - transaction_costs[1:]
        
        # 累积收益
        cumulative_returns = np.cumprod(1 + strategy_returns_net)
        portfolio_value = initial_capital * cumulative_returns
        
        # 基准收益（买入持有）
        benchmark_returns = returns[1:]
        benchmark_cumulative = np.cumprod(1 + benchmark_returns)
        benchmark_value = initial_capital * benchmark_cumulative
        
        # 性能指标
        results = {}
        
        # 收益指标
        total_return = cumulative_returns[-1] - 1
        benchmark_total_return = benchmark_cumulative[-1] - 1
        excess_return = total_return - benchmark_total_return
        
        # 年化收益率
        trading_days = len(strategy_returns_net)
        years = trading_days / 252  # 假设252个交易日每年
        annual_return = (1 + total_return) ** (1/years) - 1
        benchmark_annual_return = (1 + benchmark_total_return) ** (1/years) - 1
        
        # 风险指标
        volatility = np.std(strategy_returns_net) * np.sqrt(252)
        benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252)
        
        # 夏普比率
        risk_free_rate = 0.02  # 假设2%无风险利率
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # 最大回撤
        running_max = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        benchmark_running_max = np.maximum.accumulate(benchmark_value)
        benchmark_drawdown = (benchmark_value - benchmark_running_max) / benchmark_running_max
        benchmark_max_drawdown = np.min(benchmark_drawdown)
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 信息比率
        excess_returns = strategy_returns_net - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
        
        # 胜率
        win_rate = np.mean(strategy_returns_net > 0)
        benchmark_win_rate = np.mean(benchmark_returns > 0)
        
        # 汇总结果
        results['returns'] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'excess_return': excess_return,
            'benchmark_total_return': benchmark_total_return,
            'benchmark_annual_return': benchmark_annual_return
        }
        
        results['risk'] = {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'benchmark_volatility': benchmark_volatility,
            'benchmark_max_drawdown': benchmark_max_drawdown
        }
        
        results['ratios'] = {
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'benchmark_sharpe': benchmark_sharpe
        }
        
        results['trading'] = {
            'win_rate': win_rate,
            'benchmark_win_rate': benchmark_win_rate,
            'total_trades': np.sum(np.abs(position_changes) > 0),
            'avg_trade_return': np.mean(strategy_returns_net),
            'transaction_cost_total': np.sum(transaction_costs)
        }
        
        results['time_series'] = {
            'portfolio_value': portfolio_value,
            'benchmark_value': benchmark_value,
            'drawdown': drawdown,
            'benchmark_drawdown': benchmark_drawdown,
            'strategy_returns': strategy_returns_net,
            'benchmark_returns': benchmark_returns
        }
        
        # 打印主要指标
        print(f"  - 总收益率: {total_return:.2%}")
        print(f"  - 年化收益率: {annual_return:.2%}")
        print(f"  - 年化波动率: {volatility:.2%}")
        print(f"  - 夏普比率: {sharpe_ratio:.4f}")
        print(f"  - 最大回撤: {max_drawdown:.2%}")
        print(f"  - 信息比率: {information_ratio:.4f}")
        print(f"  - 胜率: {win_rate:.2%}")
        
        return results
    
    def analyze_feature_stability(self,
                                features: pd.DataFrame,
                                window_size: int = 252,
                                overlap: float = 0.5) -> Dict[str, Any]:
        """
        分析特征稳定性
        
        Args:
            features: 特征DataFrame
            window_size: 窗口大小
            overlap: 窗口重叠比例
            
        Returns:
            特征稳定性分析结果
        """
        print("分析特征稳定性...")
        
        results = {}
        step_size = int(window_size * (1 - overlap))
        
        # 为每个特征计算时间窗口统计
        feature_stability = {}
        
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                window_stats = []
                
                for start in range(0, len(features) - window_size + 1, step_size):
                    end = start + window_size
                    window_data = features[col].iloc[start:end]
                    
                    if len(window_data.dropna()) > window_size * 0.8:  # 至少80%有效数据
                        stats_dict = {
                            'mean': window_data.mean(),
                            'std': window_data.std(),
                            'skewness': window_data.skew(),
                            'kurtosis': window_data.kurtosis(),
                            'min': window_data.min(),
                            'max': window_data.max()
                        }
                        window_stats.append(stats_dict)
                
                if window_stats:
                    # 计算各统计量的稳定性（变异系数）
                    stability_metrics = {}
                    for stat in ['mean', 'std', 'skewness', 'kurtosis']:
                        values = [ws[stat] for ws in window_stats if not np.isnan(ws[stat])]
                        if len(values) > 1:
                            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
                            stability_metrics[f'{stat}_cv'] = cv
                    
                    feature_stability[col] = stability_metrics
        
        results['feature_stability'] = feature_stability
        
        # 总体稳定性评分
        stability_scores = {}
        for feature, metrics in feature_stability.items():
            if metrics:
                # 综合稳定性评分（变异系数的倒数，越大越稳定）
                cvs = [cv for cv in metrics.values() if not np.isinf(cv)]
                if cvs:
                    avg_cv = np.mean(cvs)
                    stability_score = 1 / (1 + avg_cv)  # 0-1之间，越大越稳定
                    stability_scores[feature] = stability_score
        
        results['stability_scores'] = stability_scores
        
        # 排序并打印最稳定和最不稳定的特征
        if stability_scores:
            sorted_features = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
            
            print(f"  - 特征数量: {len(sorted_features)}")
            print(f"  - 最稳定特征前5:")
            for feature, score in sorted_features[:5]:
                print(f"    {feature}: {score:.4f}")
            
            print(f"  - 最不稳定特征后5:")
            for feature, score in sorted_features[-5:]:
                print(f"    {feature}: {score:.4f}")
        
        return results
    
    def create_evaluation_dashboard(self,
                                  classification_results: Dict,
                                  signal_quality_results: Dict,
                                  backtesting_results: Dict,
                                  feature_stability_results: Dict = None) -> None:
        """
        创建评估仪表板
        
        Args:
            classification_results: 分类性能结果
            signal_quality_results: 信号质量结果
            backtesting_results: 回测结果
            feature_stability_results: 特征稳定性结果
        """
        print("创建评估仪表板...")
        
        if not HAS_PLOTLY:
            print("  - 跳过仪表板创建（Plotly不可用）")
            return
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "混淆矩阵", "ROC曲线",
                "累积收益对比", "回撤分析",
                "信号分布", "特征稳定性"
            ],
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "bar"}]
            ]
        )
        
        # 1. 混淆矩阵
        if 'confusion_matrix' in classification_results:
            cm = classification_results['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues', showscale=False),
                row=1, col=1
            )
        
        # 2. 累积收益对比
        if 'time_series' in backtesting_results:
            ts_data = backtesting_results['time_series']
            x_axis = list(range(len(ts_data['portfolio_value'])))
            
            fig.add_trace(
                go.Scatter(
                    x=x_axis, 
                    y=ts_data['portfolio_value'],
                    name="策略",
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=ts_data['benchmark_value'],
                    name="基准",
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        # 3. 回撤分析
        if 'time_series' in backtesting_results:
            ts_data = backtesting_results['time_series']
            x_axis = list(range(len(ts_data['drawdown'])))
            
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=ts_data['drawdown'] * 100,
                    name="策略回撤",
                    fill='tonexty',
                    line=dict(color='red', width=0)
                ),
                row=2, col=2
            )
        
        # 4. 信号分布
        if 'signal_distribution' in signal_quality_results:
            dist = signal_quality_results['signal_distribution']
            categories = ['强多头', '中性', '强空头']
            values = [dist['strong_long_pct'], dist['neutral_pct'], dist['strong_short_pct']]
            
            fig.add_trace(
                go.Bar(x=categories, y=values, name="信号分布"),
                row=3, col=1
            )
        
        # 5. 特征稳定性（如果提供）
        if feature_stability_results and 'stability_scores' in feature_stability_results:
            scores = feature_stability_results['stability_scores']
            top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features, stability = zip(*top_features)
            fig.add_trace(
                go.Bar(x=list(stability), y=list(features), orientation='h', name="稳定性评分"),
                row=3, col=2
            )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title_text="模型评估仪表板",
            showlegend=True
        )
        
        # 保存图表
        if self.save_plots:
            import plotly.io as pio
            pio.write_html(fig, f"{self.plot_dir}/evaluation_dashboard.html")
            print(f"  - 仪表板已保存: {self.plot_dir}/evaluation_dashboard.html")
        
        # 显示图表
        fig.show()
    
    # === 内部辅助方法 ===
    
    def _calculate_gini(self, values: np.ndarray) -> float:
        """计算基尼系数"""
        values = np.sort(values)
        n = len(values)
        if n == 0 or np.sum(values) == 0:
            return 0
        
        cumsum = np.cumsum(values)
        return 1 - 2 * np.sum(cumsum) / (n * cumsum[-1])


# 便利函数
def comprehensive_model_evaluation(model,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 signals: np.ndarray = None,
                                 returns: np.ndarray = None,
                                 feature_data: pd.DataFrame = None) -> Dict[str, Any]:
    """
    综合模型评估
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        signals: 信号序列
        returns: 收益率序列
        feature_data: 特征数据
        
    Returns:
        综合评估结果
    """
    evaluator = AdvancedModelEvaluator()
    
    # 模型预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # 分类性能评估
    classification_results = evaluator.evaluate_classification_performance(
        y_test, y_pred, y_proba
    )
    
    results = {'classification': classification_results}
    
    # 信号质量分析
    if signals is not None and returns is not None:
        signal_quality_results = evaluator.analyze_signal_quality(signals, returns)
        results['signal_quality'] = signal_quality_results
        
        # 回测性能分析
        backtesting_results = evaluator.analyze_backtesting_performance(signals, returns)
        results['backtesting'] = backtesting_results
    
    # 特征稳定性分析
    if feature_data is not None:
        stability_results = evaluator.analyze_feature_stability(feature_data)
        results['feature_stability'] = stability_results
    
    return results


def compare_models(models: Dict[str, Any],
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  signals_dict: Dict[str, np.ndarray] = None,
                  returns: np.ndarray = None) -> pd.DataFrame:
    """
    模型对比分析
    
    Args:
        models: 模型字典 {name: model}
        X_test: 测试特征
        y_test: 测试标签
        signals_dict: 信号字典 {name: signals}
        returns: 收益率序列
        
    Returns:
        模型对比结果DataFrame
    """
    comparison_results = []
    
    for name, model in models.items():
        print(f"\n评估模型: {name}")
        
        # 基本性能
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        result = {
            'model': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # 如果有信号和收益率，计算交易性能
        if signals_dict and name in signals_dict and returns is not None:
            signals = signals_dict[name]
            strategy_returns = signals[:-1] * returns[1:]
            
            total_return = np.prod(1 + strategy_returns) - 1
            volatility = np.std(strategy_returns) * np.sqrt(252)
            sharpe_ratio = np.mean(strategy_returns) * 252 / volatility if volatility > 0 else 0
            max_dd = np.min(np.minimum.accumulate(np.cumprod(1 + strategy_returns))) - 1
            
            result.update({
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd
            })
        
        comparison_results.append(result)
    
    return pd.DataFrame(comparison_results).set_index('model') 