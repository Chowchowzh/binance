#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段2：特征工程脚本
从原始数据文件进行特征工程，生成机器学习就绪的数据集
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
    parser = argparse.ArgumentParser(description='特征工程处理')
    parser.add_argument('--input', type=str, default='processed_data/raw_data.parquet',
                       help='输入的原始数据文件 (默认: processed_data/raw_data.parquet)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='不进行特征标准化')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='配置文件路径 (默认: config/config.json)')
    
    args = parser.parse_args()
    
    print("🔧 阶段2：特征工程")
    print("=" * 50)
    print(f"⚙️  配置文件: {args.config}")
    print(f"📁 输入文件: {args.input}")
    print(f"📏 特征标准化: {'否' if args.no_normalize else '是'}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件不存在: {args.input}")
        print("\n💡 请先运行阶段1收集原始数据:")
        print("   uv run python3 data_processing/scripts/stage1_collect_data.py")
        sys.exit(1)
    
    try:
        # 显示输入文件信息
        file_size = os.path.getsize(args.input) / (1024 * 1024)  # MB
        print(f"📏 输入文件大小: {file_size:.2f} MB")
        
        try:
            import pandas as pd
            input_df = pd.read_parquet(args.input)
            print(f"📊 输入数据形状: {input_df.shape}")
            
            if 'open_time' in input_df.columns:
                start_time = datetime.fromtimestamp(input_df['open_time'].min() / 1000)
                end_time = datetime.fromtimestamp(input_df['open_time'].max() / 1000)
                print(f"📅 数据时间范围: {start_time} ~ {end_time}")
                
        except Exception as e:
            print(f"⚠️  无法读取输入文件详情: {e}")
        
        print()
        
        # 初始化数据预处理器
        processor = DataPreprocessor(args.config)
        
        # 执行第二阶段特征工程
        featured_data_file = processor._stage2_feature_engineering(
            raw_data_file=args.input,
            normalize_features=not args.no_normalize
        )
        
        print()
        print("✅ 阶段2完成！")
        print(f"📁 输出文件: {featured_data_file}")
        print(f"⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示输出文件信息
        if os.path.exists(featured_data_file):
            file_size = os.path.getsize(featured_data_file) / (1024 * 1024)  # MB
            print(f"📏 输出文件大小: {file_size:.2f} MB")
            
            try:
                output_df = pd.read_parquet(featured_data_file)
                print(f"📊 输出数据形状: {output_df.shape}")
                
                # 显示目标变量分布
                if 'target' in output_df.columns:
                    target_counts = output_df['target'].value_counts().sort_index()
                    print(f"🎯 目标变量分布:")
                    for val, count in target_counts.items():
                        percentage = count / len(output_df) * 100
                        label = {-1: "下跌", 0: "横盘", 1: "上涨"}[val]
                        print(f"   {label}({val}): {count:,} ({percentage:.2f}%)")
                
                # 显示特征统计
                feature_cols = [col for col in output_df.columns 
                              if col not in ['target', 'future_return', 'open_time']]
                print(f"📈 特征数量: {len(feature_cols)}")
                
            except Exception as e:
                print(f"⚠️  无法读取输出文件详情: {e}")
        
        print("\n🎯 下一步: 运行模型训练")
        print(f"   uv run python3 -c \"from strategy.training import train_model; train_model()\"")
        
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