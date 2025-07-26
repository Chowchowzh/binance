#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段1：原始数据收集脚本
从MongoDB数据库收集原始K线数据并保存为parquet文件
"""

import argparse
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_processing import DataPreprocessor


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='收集原始K线数据')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='数据块大小 (默认: 100000)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='最大工作进程数 (默认: None，自动检测)')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='配置文件路径 (默认: config/config.json)')
    
    args = parser.parse_args()
    
    print("🗄️  阶段1：原始数据收集")
    print("=" * 50)
    print(f"⚙️  配置文件: {args.config}")
    print(f"📦 数据块大小: {args.chunk_size:,}")
    print(f"👥 最大工作进程: {args.max_workers or '自动检测'}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 初始化数据预处理器
        processor = DataPreprocessor(args.config)
        
        # 执行第一阶段数据收集
        raw_data_file = processor._stage1_collect_raw_data(
            chunk_size=args.chunk_size,
            max_workers=args.max_workers
        )
        
        print()
        print("✅ 阶段1完成！")
        print(f"📁 输出文件: {raw_data_file}")
        print(f"⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示文件信息
        if os.path.exists(raw_data_file):
            file_size = os.path.getsize(raw_data_file) / (1024 * 1024)  # MB
            print(f"📏 文件大小: {file_size:.2f} MB")
            
            # 检查数据时间范围
            try:
                import pandas as pd
                df = pd.read_parquet(raw_data_file)
                print(f"📊 数据形状: {df.shape}")
                
                if 'open_time' in df.columns:
                    start_time = datetime.fromtimestamp(df['open_time'].min() / 1000)
                    end_time = datetime.fromtimestamp(df['open_time'].max() / 1000)
                    print(f"📅 时间范围: {start_time} ~ {end_time}")
                    
            except Exception as e:
                print(f"⚠️  无法读取文件详情: {e}")
        
        print("\n🎯 下一步: 运行阶段2进行特征工程")
        print(f"   uv run python3 data_processing/scripts/stage2_feature_engineering.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中止操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 