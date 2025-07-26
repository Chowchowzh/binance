# -*- coding: utf-8 -*-
"""
回测结果日志管理器
统一管理所有分析结果的输出和保存
"""

import os
import sys
import datetime
from typing import List, Dict, Any
from contextlib import contextmanager

class ResultLogger:
    """结果日志管理器"""
    
    def __init__(self, results_dir: str = 'backtest_results'):
        self.results_dir = results_dir
        self.log_lines: List[str] = []
        self.start_time = datetime.datetime.now()
        
        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)
        
        # 设置日志文件路径
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(results_dir, f'backtest_analysis_{timestamp}.txt')
        
    def log(self, message: str, also_print: bool = True):
        """记录消息到日志"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.log_lines.append(formatted_message)
        
        if also_print:
            print(message)
    
    def log_section(self, title: str, content: str = "", also_print: bool = True):
        """记录一个完整的分析章节"""
        separator = "=" * 60
        section_lines = [
            "",
            separator,
            f" {title}",
            separator
        ]
        
        if content:
            section_lines.extend(content.split('\n'))
        
        for line in section_lines:
            self.log(line, also_print)
    
    def log_metrics(self, metrics: Dict[str, Any], title: str = "性能指标", also_print: bool = True):
        """记录性能指标"""
        self.log_section(title, also_print=also_print)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) < 0.001:
                    formatted_value = f"{value:.6f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            self.log(f"{key}: {formatted_value}", also_print)
    
    def log_config(self, config: Dict[str, Any], title: str = "策略配置", also_print: bool = True):
        """记录配置信息"""
        self.log_section(title, also_print=also_print)
        
        for key, value in config.items():
            if isinstance(value, float):
                if key.endswith('_ratio') or key.endswith('_rate'):
                    formatted_value = f"{value:.2%}" if abs(value) <= 2 else f"{value:.4f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            self.log(f"{key}: {formatted_value}", also_print)
    
    def log_position_stats(self, trades_df, also_print: bool = True):
        """记录仓位统计信息"""
        if trades_df is None or trades_df.empty:
            self.log("没有交易数据可供分析", also_print)
            return
        
        self.log_section("仓位统计分析", also_print=also_print)
        
        stats = {
            "总交易次数": f"{len(trades_df):,}",
            "买入交易": f"{len(trades_df[trades_df['type'] == 'buy']):,}",
            "卖出交易": f"{len(trades_df[trades_df['type'] == 'sell']):,}",
            "总手续费": f"{trades_df['fee'].sum():.2f} USDT",
            "平均单笔交易金额": f"{(trades_df['price'] * trades_df['amount']).mean():.2f} USDT",
            "最大单笔交易金额": f"{(trades_df['price'] * trades_df['amount']).max():.2f} USDT",
            "最小单笔交易金额": f"{(trades_df['price'] * trades_df['amount']).min():.2f} USDT"
        }
        
        for key, value in stats.items():
            self.log(f"{key}: {value}", also_print)
        
        if 'target_position_ratio' in trades_df.columns:
            self.log("", also_print)
            position_stats = {
                "平均目标仓位": f"{trades_df['target_position_ratio'].mean():.2%}",
                "平均实际仓位": f"{trades_df['actual_position_ratio'].mean():.2%}",
                "最大仓位": f"{trades_df['actual_position_ratio'].max():.2%}",
                "最小仓位": f"{trades_df['actual_position_ratio'].min():.2%}",
                "仓位标准差": f"{trades_df['actual_position_ratio'].std():.2%}"
            }
            
            for key, value in position_stats.items():
                self.log(f"{key}: {value}", also_print)
    
    def log_alpha_analysis(self, ic: float, p_value: float, quantile_results, also_print: bool = True):
        """记录Alpha分析结果"""
        self.log_section("信号Alpha能力分析", also_print=also_print)
        
        self.log(f"信息系数 (IC): {ic:.4f} (p-value: {p_value:.6f})", also_print)
        
        if p_value < 0.001:
            significance = "***（高度显著）"
        elif p_value < 0.01:
            significance = "**（显著）"
        elif p_value < 0.05:
            significance = "*（边际显著）"
        else:
            significance = "（不显著）"
        
        self.log(f"显著性水平: {significance}", also_print)
        
        self.log("", also_print)
        self.log("十分位分析:", also_print)
        self.log("分位  平均信号    平均收益    收益标准差  样本数    夏普比率", also_print)
        self.log("-" * 65, also_print)
        
        for i, row in quantile_results.iterrows():
            quantile_num = i + 1
            self.log(
                f"{quantile_num:2.0f}    {row['mean_signal']:9.6f}  {row['mean_return']:10.6f}  "
                f"{row['std_return']:10.6f}  {row['count']:6.0f}  {row['sharpe_ratio']:8.4f}",
                also_print
            )
    
    def log_comparison(self, strategy_pnl: float, hold_pnl: float, also_print: bool = True):
        """记录策略与买入持有的对比"""
        self.log_section("策略表现对比", also_print=also_print)
        
        self.log(f"策略最终收益: {strategy_pnl:.2f} USDT", also_print)
        self.log(f"买入持有收益: {hold_pnl:.2f} USDT", also_print)
        
        if hold_pnl != 0:
            outperformance = (strategy_pnl - hold_pnl) / abs(hold_pnl) * 100
            self.log(f"超额表现: {outperformance:+.2f}%", also_print)
        
        if strategy_pnl > hold_pnl:
            self.log("✅ 策略跑赢买入持有", also_print)
        else:
            self.log("❌ 策略跑输买入持有", also_print)
    
    def save_to_file(self):
        """保存所有日志到文件"""
        end_time = datetime.datetime.now()
        duration = end_time - self.start_time
        
        # 添加文件头和尾
        header = [
            "=" * 80,
            "                    量化交易策略回测分析报告",
            "=" * 80,
            f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"运行时长: {duration}",
            ""
        ]
        
        footer = [
            "",
            "=" * 80,
            f"报告生成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"报告保存路径: {self.log_file}",
            "=" * 80
        ]
        
        # 写入文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(header))
            f.write('\n'.join(self.log_lines))
            f.write('\n'.join(footer))
        
        print(f"\n📄 完整分析报告已保存到: {self.log_file}")
        return self.log_file

@contextmanager
def capture_output(logger: ResultLogger):
    """上下文管理器，捕获print输出到logger"""
    class LoggerWriter:
        def __init__(self, logger, original_stdout):
            self.logger = logger
            self.original_stdout = original_stdout
            
        def write(self, text):
            if text.strip():  # 只记录非空行
                self.logger.log(text.rstrip(), also_print=False)
            self.original_stdout.write(text)
            
        def flush(self):
            self.original_stdout.flush()
    
    original_stdout = sys.stdout
    sys.stdout = LoggerWriter(logger, original_stdout)
    
    try:
        yield logger
    finally:
        sys.stdout = original_stdout 