# -*- coding: utf-8 -*-
"""
成交额K线生成脚本

使用示例:
python data_processing/scripts/generate_dollar_bars.py --symbol ETHUSDT --threshold 10000000
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_processing.dollar_bars import DollarBarsGenerator
from utils.logger import setup_logger


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成成交额K线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 为ETHUSDT生成成交额K线（自动计算阈值）
  python generate_dollar_bars.py --symbol ETHUSDT --auto-threshold
  
  # 指定固定阈值生成成交额K线
  python generate_dollar_bars.py --symbol ETHUSDT --threshold 50000000
  
  # 为多个交易对生成成交额K线
  python generate_dollar_bars.py --symbols ETHUSDT BTCUSDT --auto-threshold
  
  # 自定义参数
  python generate_dollar_bars.py --symbol ETHUSDT --auto-threshold --target-bars 100 --window-days 14
        """
    )
    
    # 输入参数
    parser.add_argument('--input', default='processed_data/raw_data.parquet',
                       help='输入数据文件路径')
    parser.add_argument('--output-dir', default='processed_data',
                       help='输出目录')
    
    # 交易对参数
    parser.add_argument('--symbol', type=str,
                       help='单个交易对符号（如：ETHUSDT）')
    parser.add_argument('--symbols', nargs='+', 
                       help='多个交易对符号列表')
    
    # 阈值参数
    parser.add_argument('--threshold', type=float,
                       help='固定成交额阈值（美元）')
    parser.add_argument('--auto-threshold', action='store_true',
                       help='使用自动计算的动态阈值')
    parser.add_argument('--target-bars', type=int, default=50,
                       help='自动阈值时的目标每日K线数（默认：50）')
    parser.add_argument('--window-days', type=int, default=30,
                       help='自动阈值时的滚动窗口天数（默认：30）')
    
    # 输出参数
    parser.add_argument('--prefix', default='dollar_bars',
                       help='输出文件名前缀')
    parser.add_argument('--analyze', action='store_true',
                       help='分析生成的成交额K线统计特性')
    
    args = parser.parse_args()
    
    # 参数验证
    if not args.symbol and not args.symbols:
        parser.error("必须指定 --symbol 或 --symbols")
    
    if not args.threshold and not args.auto_threshold:
        parser.error("必须指定 --threshold 或 --auto-threshold")
    
    if args.threshold and args.auto_threshold:
        parser.error("不能同时指定 --threshold 和 --auto-threshold")
    
    # 设置日志
    logger = setup_logger(__name__)
    
    try:
        # 加载数据
        logger.info(f"加载数据: {args.input}")
        import pandas as pd
        df = pd.read_parquet(args.input)
        logger.info(f"数据形状: {df.shape}")
        
        # 创建生成器
        generator = DollarBarsGenerator(
            threshold_usd=args.threshold or 50_000_000,
            auto_threshold=args.auto_threshold
        )
        
        # 确定要处理的交易对
        if args.symbol:
            symbols = [args.symbol]
        else:
            symbols = args.symbols
        
        logger.info(f"处理交易对: {symbols}")
        
        # 生成成交额K线
        if len(symbols) == 1:
            # 单个交易对
            symbol = symbols[0]
            logger.info(f"为 {symbol} 生成成交额K线...")
            
            # 设置动态阈值参数
            if args.auto_threshold:
                # 临时修改生成器的动态阈值计算参数
                original_method = generator.calculate_dynamic_threshold
                def custom_threshold_calc(df, sym):
                    return original_method(df, sym, args.window_days, args.target_bars)
                generator.calculate_dynamic_threshold = custom_threshold_calc
            
            dollar_bars = generator.generate_dollar_bars(df, symbol, args.threshold)
            
            # 分析统计特性
            if args.analyze:
                logger.info("分析统计特性...")
                stats = generator.analyze_dollar_bars_statistics(dollar_bars, symbol)
            
            # 保存结果
            filename_prefix = f"{args.prefix}_{symbol}"
            saved_files = generator.save_dollar_bars(
                dollar_bars, 
                args.output_dir, 
                filename_prefix
            )
            logger.info(f"成交额K线已保存: {saved_files}")
            
        else:
            # 多个交易对
            logger.info("为多个交易对生成成交额K线...")
            
            # 准备阈值字典
            thresholds = None
            if args.threshold:
                thresholds = {symbol: args.threshold for symbol in symbols}
            
            # 设置动态阈值参数
            if args.auto_threshold:
                original_method = generator.calculate_dynamic_threshold
                def custom_threshold_calc(df, sym, window=30, target=50):
                    return original_method(df, sym, args.window_days, args.target_bars)
                generator.calculate_dynamic_threshold = custom_threshold_calc
            
            multi_dollar_bars = generator.generate_multi_symbol_dollar_bars(
                df, symbols, thresholds
            )
            
            # 分析统计特性
            if args.analyze:
                for symbol, dollar_bars_df in multi_dollar_bars.items():
                    logger.info(f"分析 {symbol} 统计特性...")
                    stats = generator.analyze_dollar_bars_statistics(dollar_bars_df, symbol)
            
            # 保存结果
            saved_files = generator.save_dollar_bars(
                multi_dollar_bars, 
                args.output_dir, 
                args.prefix
            )
            logger.info(f"多交易对成交额K线已保存: {saved_files}")
        
        logger.info("成交额K线生成完成！")
        
    except Exception as e:
        logger.error(f"生成成交额K线时出错: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 