import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_quantile_analysis(quantile_analysis_results, results_dir='.', filename='quantile_analysis.png'):
    """
    将信号十分位分析的结果可视化。

    :param quantile_analysis_results: 从 analyze_signal_alpha 返回的DataFrame。
    :param results_dir: 保存图表的目录。
    :param filename: 保存图表的文件名。
    """
    if quantile_analysis_results.empty:
        print("警告: 十分位分析结果为空，跳过绘图。")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 绘制平均收益率的条形图
    sns.barplot(x=quantile_analysis_results.index, y='mean_return', data=quantile_analysis_results, ax=ax1, color='b', alpha=0.6, label='Mean Return')
    ax1.set_xlabel('Signal Quantile', fontsize=12)
    ax1.set_ylabel('Mean Future Return', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 创建第二个y轴以显示每个分位的样本数
    ax2 = ax1.twinx()
    sns.lineplot(x=quantile_analysis_results.index - 1, y='count', data=quantile_analysis_results, ax=ax2, color='r', marker='o', label='Count')
    ax2.set_ylabel('Number of Samples', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Signal Quantile Analysis', fontsize=16)
    fig.tight_layout()
    save_path = os.path.join(results_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"信号十分位分析图表已保存到: {save_path}")


def plot_backtest_results(data, results_dir='.'):
    """
    绘制详细的回测结果图表，包括PNL、持仓价值和预测回报。
    """
    print("\n--- 开始绘制详细回测图表 ---")
    
    pnl_series = data.get('pnl_series')
    hold_pnl_series = data.get('hold_pnl_series')
    eth_value_history = data.get('eth_value_history')
    predicted_return_history = data.get('predicted_return_history')

    if pnl_series is None or pnl_series.empty:
        print("无法绘制PNL曲线，因为没有PNL数据。")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot 1: PnL Curve
    pnl_series.plot(ax=axes[0], label='Strategy PnL', color='blue')
    if hold_pnl_series is not None:
        hold_pnl_series.plot(ax=axes[0], label='Buy & Hold PnL', color='gray', linestyle='--')
    axes[0].set_ylabel('Profit & Loss (USD)')
    axes[0].set_title('Strategy Performance vs. Buy & Hold')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: ETH Holdings Value
    if eth_value_history is not None:
        eth_value_history.plot(ax=axes[1], label='ETH Holdings Value (USD)', color='green')
        axes[1].set_ylabel('ETH Value (USD)')
        axes[1].grid(True)
        axes[1].legend()

    # Plot 3: Predicted Return
    if predicted_return_history is not None:
        predicted_return_history.plot(ax=axes[2], label='Calibrated Signal (Predicted Return)', color='purple', alpha=0.7)
        axes[2].set_ylabel('Predicted Return')
        axes[2].grid(True)
        axes[2].legend()

    plt.xlabel('Time')
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'backtest_full_analysis.png')
    plt.savefig(save_path)
    plt.close()
    print(f"详细回测分析图表已保存到 {save_path}") 