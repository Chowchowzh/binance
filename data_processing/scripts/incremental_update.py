#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量更新脚本
检查现有数据是否最新，如果不是则进行增量更新
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_processing import DataPreprocessor
from database.connection import DatabaseConnection
from config.settings import load_project_config, get_legacy_config_dict


def check_data_freshness(raw_data_path: str, db_connection: DatabaseConnection, 
                         target_symbol: str) -> dict:
    """
    检查数据新鲜度
    
    Args:
        raw_data_path: 原始数据文件路径
        db_connection: 数据库连接
        target_symbol: 目标交易对
        
    Returns:
        包含检查结果的字典
    """
    result = {
        'file_exists': False,
        'file_last_time': None,
        'db_last_time': None,
        'need_update': True,
        'new_records_count': 0
    }
    
    try:
        # 检查文件是否存在
        if not os.path.exists(raw_data_path):
            result['need_update'] = True
            result['new_records_count'] = 'unknown'
            return result
        
        result['file_exists'] = True
        
        # 获取文件中的最后时间
        import pandas as pd
        df = pd.read_parquet(raw_data_path)
        
        if 'open_time' in df.columns and len(df) > 0:
            result['file_last_time'] = int(df['open_time'].max())
            
            # 获取数据库中的最后时间
            collection = db_connection.get_collection_for_symbol(target_symbol)
            latest_doc = collection.find().sort('open_time', -1).limit(1)
            latest_doc = list(latest_doc)
            
            if latest_doc:
                result['db_last_time'] = int(latest_doc[0]['open_time'])
                
                # 判断是否需要更新
                if result['db_last_time'] > result['file_last_time']:
                    result['need_update'] = True
                    
                    # 计算新记录数
                    new_count = collection.count_documents({
                        'open_time': {'$gt': result['file_last_time']}
                    })
                    result['new_records_count'] = new_count
                else:
                    result['need_update'] = False
                    result['new_records_count'] = 0
            else:
                result['need_update'] = False
                
    except Exception as e:
        print(f"检查数据新鲜度时出错: {e}")
        result['need_update'] = True
        result['error'] = str(e)
    
    return result


def format_timestamp(timestamp: Optional[int]) -> str:
    """格式化时间戳为可读字符串"""
    if timestamp is None:
        return "未知"
    return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增量更新原始数据')
    parser.add_argument('--from-timestamp', type=int, 
                       help='指定起始时间戳（毫秒）')
    parser.add_argument('--from-date', type=str,
                       help='指定起始日期（格式：YYYY-MM-DD）')
    parser.add_argument('--force-update', action='store_true',
                       help='强制更新，即使数据已经是最新的')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='配置文件路径 (默认: config/config.json)')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查数据新鲜度，不进行更新')
    
    args = parser.parse_args()
    
    print("🔄 数据增量更新工具")
    print("=" * 50)
    print(f"⚙️  配置文件: {args.config}")
    print(f"⏰ 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 加载配置
        project_config = load_project_config(args.config)
        legacy_config = get_legacy_config_dict(project_config)
        target_symbol = project_config.data_collection.target_symbol
        
        print(f"🎯 目标交易对: {target_symbol}")
        print()
        
        # 建立数据库连接
        db_connection = DatabaseConnection(legacy_config)
        
        # 初始化预处理器
        preprocessor = DataPreprocessor(args.config)
        raw_data_path = preprocessor.get_raw_data_path()
        
        print(f"📁 原始数据文件: {raw_data_path}")
        
        # 检查数据新鲜度
        print("\n🔍 检查数据新鲜度...")
        freshness = check_data_freshness(raw_data_path, db_connection, target_symbol)
        
        # 显示检查结果
        print(f"   文件存在: {'是' if freshness['file_exists'] else '否'}")
        print(f"   文件最后时间: {format_timestamp(freshness['file_last_time'])}")
        print(f"   数据库最后时间: {format_timestamp(freshness['db_last_time'])}")
        print(f"   需要更新: {'是' if freshness['need_update'] else '否'}")
        print(f"   新记录数: {freshness['new_records_count']}")
        
        # 如果只是检查，直接返回
        if args.check_only:
            print("\n✅ 数据新鲜度检查完成")
            return
        
        # 确定起始时间
        last_processed_time = None
        if args.from_timestamp:
            last_processed_time = args.from_timestamp
            print(f"\n📅 使用指定起始时间戳: {format_timestamp(last_processed_time)}")
        elif args.from_date:
            try:
                date_obj = datetime.strptime(args.from_date, '%Y-%m-%d')
                last_processed_time = int(date_obj.timestamp() * 1000)
                print(f"\n📅 使用指定起始日期: {format_timestamp(last_processed_time)}")
            except ValueError:
                print(f"❌ 无效的日期格式: {args.from_date} (应为 YYYY-MM-DD)")
                sys.exit(1)
        else:
            last_processed_time = freshness.get('file_last_time')
            if last_processed_time:
                print(f"\n📅 从文件最后时间开始: {format_timestamp(last_processed_time)}")
        
        # 判断是否需要更新
        if not args.force_update and not freshness['need_update']:
            print("\n✅ 数据已经是最新的，无需更新")
            print("💡 如需强制更新，请使用 --force-update 参数")
            return
        
        if freshness['new_records_count'] == 0 and not args.force_update:
            print("\n✅ 没有新数据需要更新")
            return
        
        # 执行增量更新
        print(f"\n🚀 开始增量更新...")
        print(f"   预期新增记录: {freshness['new_records_count']}")
        
        # 更新原始数据
        start_time = datetime.now()
        new_data = preprocessor._load_incremental_data(
            target_symbol, 
            project_config.data_collection.feature_symbols,
            last_processed_time
        )
        
        if new_data.empty:
            print("✅ 没有新数据需要处理")
            return
        
        # 更新raw_data.parquet
        if os.path.exists(raw_data_path):
            print("📄 合并到现有文件...")
            existing_df = pd.read_parquet(raw_data_path)
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            
            # 去重并排序
            if 'open_time' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['open_time'], keep='last')
                combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
        else:
            print("📄 创建新文件...")
            combined_df = new_data
        
        # 保存更新后的数据
        combined_df.to_parquet(raw_data_path, index=False)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✅ 增量更新完成！")
        print(f"   新增数据: {len(new_data)} 条")
        print(f"   总数据量: {len(combined_df)} 条")
        print(f"   处理时间: {duration.total_seconds():.2f} 秒")
        print(f"   更新文件: {raw_data_path}")
        
        # 检查是否需要重新进行特征工程
        featured_data_path = preprocessor.get_featured_data_path()
        if os.path.exists(featured_data_path):
            print(f"\n💡 提示: 原始数据已更新，建议重新进行特征工程:")
            print(f"   uv run python3 data_processing/scripts/stage2_feature_engineering.py")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中止操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 确保关闭数据库连接
        if 'db_connection' in locals():
            db_connection.close_all_connections()


if __name__ == "__main__":
    main() 