import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def analyze_signal_alpha(signals, future_returns, n_quantiles=10):
    """
    对模型信号进行详细的Alpha能力分析。

    计算并显示：
    1. 十分位分析 (Quantile Analysis): 按信号强度分组后的平均真实收益。
    2. 信息系数 (Information Coefficient, IC): 信号与未来收益的等级相关性。

    :param signals: 模型的原始预测信号 (一维Numpy数组)。
    :param future_returns: 与信号对齐的未来真实收益率 (一维Numpy数组)。
    :param n_quantiles: 用于分析的组数 (默认为10)。
    :return: 一个包含十分位分析结果的DataFrame。
    """
    print("\n--- 开始信号Alpha能力分析 ---")
    
    # 1. 准备数据
    analysis_df = pd.DataFrame({
        'signal': signals,
        'future_return': future_returns
    }).dropna()

    if analysis_df.empty:
        print("警告: 信号或未来收益为空，无法进行Alpha分析。")
        return pd.DataFrame()

    # 2. 计算信息系数 (IC)
    # 使用Spearman等级相关性，因为它对异常值不敏感
    ic, p_value = spearmanr(analysis_df['signal'], analysis_df['future_return'])
    print(f"整体信息系数 (Spearman IC): {ic:.4f} (p-value: {p_value:.4f})")

    # 3. 进行十分位分析
    try:
        analysis_df['quantile'] = pd.qcut(analysis_df['signal'], n_quantiles, labels=False, duplicates='drop')
    except ValueError:
        print(f"警告: 无法创建 {n_quantiles} 个唯一的量化箱。减少箱数后重试。")
        try:
            analysis_df['quantile'] = pd.qcut(analysis_df['signal'], 5, labels=False, duplicates='drop')
        except ValueError:
            print("错误: 信号值过于集中，无法进行十分位分析。")
            return pd.DataFrame()
            
    # 计算每个分位的统计数据
    quantile_analysis = analysis_df.groupby('quantile').agg(
        mean_signal=('signal', 'mean'),
        mean_return=('future_return', 'mean'),
        std_return=('future_return', 'std'),
        count=('signal', 'count')
    ).reset_index()
    
    # 将分位标签从0-9调整为1-10
    quantile_analysis['quantile'] = quantile_analysis['quantile'] + 1
    quantile_analysis.set_index('quantile', inplace=True)
    
    # 计算每个分位的夏普比率 (假设为日内，无风险利率为0)
    quantile_analysis['sharpe_ratio'] = quantile_analysis['mean_return'] / quantile_analysis['std_return']
    
    print("\n信号十分位分析:")
    print(quantile_analysis)
    
    print("\n--- Alpha能力分析完成 ---")
    
    return quantile_analysis 