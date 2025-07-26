#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化数据处理和模型训练流水线
阶段2（特征工程）→ 模型训练
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
import time

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_command(cmd, description, cwd=None):
    """
    执行系统命令并处理结果
    
    Args:
        cmd: 要执行的命令列表
        description: 命令描述
        cwd: 工作目录，默认为项目根目录
    
    Returns:
        bool: 命令是否成功执行
    """
    if cwd is None:
        cwd = project_root
        
    print(f"\n🚀 {description}")
    print("=" * 60)
    print(f"💻 执行命令: {' '.join(cmd)}")
    print(f"📁 工作目录: {cwd}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        # 使用 subprocess.run 执行命令
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,  # 实时显示输出
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print()
        print("=" * 60)
        print(f"✅ {description} - 成功完成!")
        print(f"⏱️  耗时: {duration:.2f} 秒")
        print(f"🔄 返回码: {result.returncode}")
        print("=" * 60)
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print()
        print("=" * 60)
        print(f"❌ {description} - 执行失败!")
        print(f"⏱️  耗时: {duration:.2f} 秒")
        print(f"🔄 返回码: {e.returncode}")
        print("=" * 60)
        
        return False
        
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print(f"⏹️  {description} - 用户中止操作")
        print("=" * 60)
        return False
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print()
        print("=" * 60)
        print(f"❌ {description} - 意外错误!")
        print(f"⏱️  耗时: {duration:.2f} 秒")
        print(f"🐛 错误信息: {e}")
        print("=" * 60)
        
        return False


def check_prerequisites():
    """检查运行前置条件"""
    print("🔍 检查运行前置条件...")
    
    # 检查关键文件是否存在
    required_files = [
        'data_processing/scripts/stage2_feature_engineering.py',
        'strategy/training/train_transformer.py',
        'config/config.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 以下必需文件缺失:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ 所有必需文件检查通过")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动化数据处理和模型训练流水线')
    parser.add_argument('--input', type=str, default='processed_data/raw_data.parquet',
                       help='输入的原始数据文件 (默认: processed_data/raw_data.parquet)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='不进行特征标准化')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='配置文件路径 (默认: config/config.json)')
    parser.add_argument('--skip-stage2', action='store_true',
                       help='跳过特征工程，直接运行模型训练')
    parser.add_argument('--stage2-only', action='store_true',
                       help='只运行特征工程，不运行模型训练')
    parser.add_argument('--skip-reduce', action='store_true',
                       help='跳过特征降维步骤')
    parser.add_argument('--max-features', type=int, default=50,
                       help='降维后保留的最大特征数 (默认: 50)')
    parser.add_argument('--reduce-method', type=str, default='random_forest',
                       choices=['random_forest', 'mutual_info', 'f_test', 'xgboost'],
                       help='特征选择方法 (默认: random_forest)')
    
    args = parser.parse_args()
    
    print("🎯 自动化数据处理和模型训练流水线")
    print("=" * 80)
    print(f"⚙️  配置文件: {args.config}")
    print(f"📁 输入文件: {args.input}")
    print(f"📏 特征标准化: {'否' if args.no_normalize else '是'}")
    print(f"🔄 跳过特征工程: {'是' if args.skip_stage2 else '否'}")
    print(f"🔄 仅运行特征工程: {'是' if args.stage2_only else '否'}")
    print(f"🔄 跳过特征降维: {'是' if args.skip_reduce else '否'}")
    print(f"📊 最大特征数: {args.max_features}")
    print(f"🎯 降维方法: {args.reduce_method}")
    print(f"⏰ 流水线开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 检查前置条件
    if not check_prerequisites():
        print("\n❌ 前置条件检查失败，流水线中止")
        sys.exit(1)
    
    pipeline_start_time = time.time()
    success_stages = []
    failed_stages = []
    
    try:
        # 阶段1：特征工程
        if not args.skip_stage2:
            print(f"\n🔧 阶段2：特征工程")
            print("=" * 80)
            
            # 构建特征工程命令
            stage2_cmd = [
                'uv', 'run', 'python3', 
                'data_processing/scripts/stage2_feature_engineering.py',
                '--input', args.input,
                '--config', args.config
            ]
            
            if args.no_normalize:
                stage2_cmd.append('--no-normalize')
            
            # 执行特征工程
            stage2_success = run_command(
                stage2_cmd,
                "特征工程处理",
                cwd=project_root
            )
            
            if stage2_success:
                success_stages.append("特征工程")
                print(f"✅ 特征工程阶段完成")
            else:
                failed_stages.append("特征工程")
                print(f"❌ 特征工程阶段失败")
                
                if not args.stage2_only:
                    print(f"⏹️  由于特征工程失败，跳过后续阶段")
                    
                print(f"\n📋 流水线执行总结:")
                print(f"   ✅ 成功阶段: {success_stages if success_stages else '无'}")
                print(f"   ❌ 失败阶段: {failed_stages}")
                sys.exit(1)
        else:
            print(f"⏭️  跳过特征工程阶段")
        
        # 阶段3：特征降维
        if not args.stage2_only and not args.skip_reduce:
            print(f"\n📊 特征降维阶段")
            print("=" * 80)
            
            # 等待一小段时间确保文件写入完成
            if not args.skip_stage2:
                print("⏳ 等待文件系统同步...")
                time.sleep(1)
            
            # 构建降维命令
            reduce_cmd = [
                'uv', 'run', 'python3', 
                'data_processing/scripts/reduce_features.py',
                '--input', 'processed_data/featured_data.parquet',
                '--output', 'processed_data/featured_data_reduced.parquet',
                '--max-features', str(args.max_features),
                '--method', args.reduce_method
            ]
            
            # 执行特征降维
            reduce_success = run_command(
                reduce_cmd,
                "特征降维处理",
                cwd=project_root
            )
            
            if reduce_success:
                success_stages.append("特征降维")
                print(f"✅ 特征降维阶段完成")
            else:
                failed_stages.append("特征降维")
                print(f"❌ 特征降维阶段失败")
                
                print(f"⏹️  由于特征降维失败，跳过模型训练阶段")
                print(f"\n📋 流水线执行总结:")
                print(f"   ✅ 成功阶段: {success_stages if success_stages else '无'}")
                print(f"   ❌ 失败阶段: {failed_stages}")
                sys.exit(1)
        else:
            if args.skip_reduce:
                print(f"⏭️  跳过特征降维阶段")
            else:
                print(f"⏭️  跳过特征降维阶段（仅运行特征工程）")
        
        # 阶段4：模型训练
        if not args.stage2_only:
            print(f"\n🧠 模型训练阶段")
            print("=" * 80)
            
            # 等待一小段时间确保文件写入完成
            if not args.skip_stage2 or not args.skip_reduce:
                print("⏳ 等待文件系统同步...")
                time.sleep(2)
            
            # 构建模型训练命令
            training_cmd = [
                'uv', 'run', 'python3', 
                'strategy/training/train_transformer.py'
            ]
            
            # 执行模型训练
            training_success = run_command(
                training_cmd,
                "Transformer模型训练",
                cwd=project_root
            )
            
            if training_success:
                success_stages.append("模型训练")
                print(f"✅ 模型训练阶段完成")
            else:
                failed_stages.append("模型训练")
                print(f"❌ 模型训练阶段失败")
        else:
            print(f"⏭️  跳过模型训练阶段（仅运行特征工程）")
        
        # 流水线总结
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        
        print(f"\n" + "=" * 80)
        print(f"🎉 自动化流水线执行完成!")
        print("=" * 80)
        print(f"⏰ 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
        print(f"✅ 成功阶段: {success_stages if success_stages else '无'}")
        print(f"❌ 失败阶段: {failed_stages if failed_stages else '无'}")
        
        if failed_stages:
            print(f"\n⚠️  流水线部分失败，请检查失败阶段的错误信息")
            sys.exit(1)
        else:
            print(f"\n🎯 所有阶段成功完成！")
            
            # 提供下一步建议
            if not args.stage2_only:
                print(f"\n💡 接下来可以:")
                print(f"   - 运行回测: uv run python3 -c \"from strategy.market_making import run_backtest; run_backtest()\"")
                print(f"   - 查看模型文件: ls -la models/")
                print(f"   - 检查降维结果: ls -la processed_data/featured_data_reduced.parquet")
                print(f"   - 检查日志: ls -la logs/")
            else:
                print(f"\n💡 接下来可以:")
                print(f"   - 运行特征降维: uv run python3 data_processing/scripts/reduce_features.py")
                print(f"   - 或运行完整流水线: uv run python3 data_processing/scripts/auto_pipeline.py --skip-stage2")
        
        print("=" * 80)
        
    except KeyboardInterrupt:
        print(f"\n⏹️  用户中止流水线操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 流水线执行过程中发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 