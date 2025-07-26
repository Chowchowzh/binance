# -*- coding: utf-8 -*-
"""
Binance交易策略项目主入口文件
展示重构后的模块化架构
"""

import argparse
import sys
import os
from typing import Dict, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入重构后的模块
from config.settings import load_project_config, migrate_config_file
from data_collection import FetcherRunner
from data_processing import DataPreprocessor
from utils import ResultLogger


def show_project_structure():
    """显示项目重构后的结构"""
    structure = """
    🏗️ Binance交易策略项目 - 重构后的模块化架构

    📁 项目结构:
    ├── 📁 data_collection/          # 数据下载模块
    │   ├── binance_api.py           # 币安API接口
    │   ├── data_fetcher.py          # 数据获取器
    │   └── run_fetcher.py           # 运行器
    │
    ├── 📁 database/                 # 数据库连接模块  
    │   ├── mongodb_client.py        # MongoDB客户端
    │   └── connection.py            # 连接管理器
    │
    ├── 📁 config/                   # 全局配置模块
    │   ├── settings.py              # 配置管理
    │   └── config.json              # 配置文件
    │
    ├── 📁 data_processing/          # 数据处理模块
    │   ├── features/                # 特征工程子模块
    │   │   ├── technical_indicators.py    # 技术指标
    │   │   ├── feature_builder.py         # 特征构建器  
    │   │   └── feature_utils.py           # 特征工具
    │   ├── preprocessor.py          # 数据预处理器
    │   └── dataset_builder.py       # 数据集构建器
    │
    ├── 📁 utils/                    # 通用工具模块
    │   ├── logger.py                # 日志管理
    │   ├── threshold_manager.py     # 阈值管理
    │   └── common.py                # 通用函数
    │
    └── 📁 strategy/                 # 策略模块 (保留原有)
        ├── market_making.py         # 主策略文件
        ├── smart_position_control.py # 仓位控制
        └── ...                      # 其他策略文件

    ✨ 主要改进:
    - 📦 模块化设计，职责清晰分离
    - 🔧 统一的配置管理系统
    - 📊 重构的特征工程流水线
    - 🗄️ 优化的数据库连接管理
    - 📋 完善的日志和工具系统
    - ⚙️ 兼容性接口保持向后兼容
    """
    print(structure)


def fetch_data(args):
    """数据获取命令"""
    with ResultLogger('logs') as logger:
        logger.info("开始数据获取任务")
        
        try:
            # 先加载配置
            config = load_project_config(args.config)
            runner = FetcherRunner(args.config)
            
            # 确定要获取的交易对
            if args.symbols:
                symbols = args.symbols
            else:
                symbols = config.data_collection.feature_symbols
            
            logger.info(f"目标交易对: {symbols}")
            logger.info(f"全量刷新: {args.full_refresh}")
            
            # 执行数据获取
            success = runner.run_with_retry(symbols, args.full_refresh)
            
            if success:
                logger.success("数据获取完成!")
            else:
                logger.error("数据获取失败!")
                return 1
                
        except Exception as e:
            logger.error(f"数据获取过程中出错: {e}")
            return 1
    
    return 0


def process_data(args):
    """数据处理命令"""
    with ResultLogger('logs') as logger:
        logger.info("开始数据处理任务")
        
        try:
            with DataPreprocessor(args.config) as preprocessor:
                # 执行两阶段数据处理
                output_path = preprocessor.process_data_two_stage(
                    chunk_size=args.chunk_size,
                    max_workers=args.max_workers
                )
                
                # 验证输出数据
                validation_result = preprocessor.validate_output_data(output_path)
                
                if validation_result['status'] == 'success':
                    logger.success(f"数据处理完成: {output_path}")
                    logger.info(f"数据形状: {validation_result['data_shape']}")
                    logger.info(f"内存使用: {validation_result['memory_usage']:.2f} MB")
                else:
                    logger.error(f"数据验证失败: {validation_result['message']}")
                    return 1
                    
        except Exception as e:
            logger.error(f"数据处理过程中出错: {e}")
            return 1
    
    return 0


def migrate_config(args):
    """配置迁移命令"""
    with ResultLogger('logs') as logger:
        logger.info("开始配置文件迁移")
        
        try:
            success = migrate_config_file(args.old_config, args.new_config)
            
            if success:
                logger.success(f"配置已迁移: {args.old_config} -> {args.new_config}")
            else:
                logger.error("配置迁移失败")
                return 1
                
        except Exception as e:
            logger.error(f"配置迁移过程中出错: {e}")
            return 1
    
    return 0


def generate_dollar_bars(args):
    """生成成交额K线命令"""
    from data_processing.dollar_bars import DollarBarsGenerator
    import pandas as pd
    
    with ResultLogger('logs') as logger:
        logger.info("开始生成成交额K线")
        
        try:
            # 参数验证
            if not args.symbol and not args.symbols:
                logger.error("必须指定 --symbol 或 --symbols")
                return 1
            
            if not args.threshold and not args.auto_threshold:
                logger.error("必须指定 --threshold 或 --auto-threshold")
                return 1
            
            if args.threshold and args.auto_threshold:
                logger.error("不能同时指定 --threshold 和 --auto-threshold")
                return 1
            
            # 加载数据
            logger.info(f"加载数据: {args.input}")
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
                filename_prefix = f"dollar_bars_{symbol}"
                saved_files = generator.save_dollar_bars(
                    dollar_bars, 
                    args.output_dir, 
                    filename_prefix
                )
                logger.success(f"成交额K线已保存: {saved_files}")
                
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
                    "dollar_bars"
                )
                logger.success(f"多交易对成交额K线已保存: {saved_files}")
            
            logger.success("成交额K线生成完成！")
            
        except Exception as e:
            logger.error(f"生成成交额K线时出错: {e}")
            return 1
    
    return 0


def show_config(args):
    """显示配置信息"""
    try:
        project_config = load_project_config(args.config)
        
        print("📋 当前项目配置:")
        print("=" * 50)
        
        print(f"\n🗄️ 数据库配置:")
        print(f"  MongoDB URI: {project_config.database.mongodb_uri[:50]}...")
        print(f"  数据库名: {project_config.database.mongodb_db_name}")
        
        print(f"\n📊 数据收集配置:")
        print(f"  目标交易对: {project_config.data_collection.target_symbol}")
        print(f"  特征交易对: {project_config.data_collection.feature_symbols}")
        print(f"  时间间隔: {project_config.data_collection.interval}")
        print(f"  开始日期: {project_config.data_collection.start_date}")
        
        print(f"\n📈 交易配置:")
        print(f"  初始资金: {project_config.trading.initial_cash:,.2f}")
        print(f"  最小交易量: {project_config.trading.min_trade_amount_eth}")
        print(f"  手续费率: {project_config.trading.fee_rate*100:.3f}%")
        
        print(f"\n🤖 模型配置:")
        print(f"  序列长度: {project_config.model.sequence_length}")
        print(f"  训练测试比: {project_config.model.train_test_split_ratio}")
        print(f"  模型维度: {project_config.model.d_model}")
        print(f"  注意力头数: {project_config.model.nhead}")
        
        print(f"\n📁 文件路径:")
        print(f"  数据文件: {project_config.model.data_path}")
        print(f"  模型文件: {project_config.model.model_save_path}")
        print(f"  标准化器: {project_config.model.scaler_path}")
        
    except Exception as e:
        print(f"❌ 读取配置失败: {e}")
        return 1
    
    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Binance交易策略项目 - 重构版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py structure                    # 显示项目结构
  python main.py config                       # 显示配置信息
  python main.py fetch --symbols ETHUSDT BTCUSDT  # 获取数据
  python main.py process                      # 处理数据
  python main.py migrate                      # 迁移配置
  python main.py dollar-bars --symbol ETHUSDT --auto-threshold --analyze  # 生成成交额K线

更多帮助请查看README_UV_SETUP.md文件。
        """
    )
    
    # 全局参数
    parser.add_argument('--config', default='config/config.json', 
                       help='配置文件路径 (默认: config/config.json)')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 显示项目结构
    structure_parser = subparsers.add_parser('structure', help='显示项目结构')
    
    # 显示配置
    config_parser = subparsers.add_parser('config', help='显示配置信息')
    
    # 数据获取命令
    fetch_parser = subparsers.add_parser('fetch', help='获取市场数据')
    fetch_parser.add_argument('--symbols', nargs='+', 
                             help='要获取的交易对列表')
    fetch_parser.add_argument('--full-refresh', action='store_true',
                             help='执行全量刷新')
    
    # 数据处理命令
    process_parser = subparsers.add_parser('process', help='处理数据')
    process_parser.add_argument('--chunk-size', type=int, default=100000,
                               help='数据块大小 (默认: 100000)')
    process_parser.add_argument('--max-workers', type=int, 
                               help='最大工作进程数')
    
    # 配置迁移命令
    migrate_parser = subparsers.add_parser('migrate', help='迁移配置文件')
    migrate_parser.add_argument('--old-config', default='dataset/config.json',
                               help='旧配置文件路径')
    migrate_parser.add_argument('--new-config', default='config/config.json',
                               help='新配置文件路径')
    
    # 成交额K线生成命令
    dollar_bars_parser = subparsers.add_parser('dollar-bars', help='生成成交额K线')
    dollar_bars_parser.add_argument('--symbol', type=str,
                                   help='单个交易对符号（如：ETHUSDT）')
    dollar_bars_parser.add_argument('--symbols', nargs='+', 
                                   help='多个交易对符号列表')
    dollar_bars_parser.add_argument('--threshold', type=float,
                                   help='固定成交额阈值（美元）')
    dollar_bars_parser.add_argument('--auto-threshold', action='store_true',
                                   help='使用自动计算的动态阈值')
    dollar_bars_parser.add_argument('--target-bars', type=int, default=50,
                                   help='目标每日K线数（默认：50）')
    dollar_bars_parser.add_argument('--window-days', type=int, default=30,
                                   help='滚动窗口天数（默认：30）')
    dollar_bars_parser.add_argument('--input', default='processed_data/raw_data.parquet',
                                   help='输入数据文件路径')
    dollar_bars_parser.add_argument('--output-dir', default='processed_data',
                                   help='输出目录')
    dollar_bars_parser.add_argument('--analyze', action='store_true',
                                   help='分析统计特性')
    
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        show_project_structure()
        parser.print_help()
        return 0
    
    # 执行对应命令
    if args.command == 'structure':
        show_project_structure()
        return 0
    elif args.command == 'config':
        return show_config(args)
    elif args.command == 'fetch':
        return fetch_data(args)
    elif args.command == 'process':
        return process_data(args)
    elif args.command == 'migrate':
        return migrate_config(args)
    elif args.command == 'dollar-bars':
        return generate_dollar_bars(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
