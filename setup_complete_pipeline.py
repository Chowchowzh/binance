#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整交易系统流水线设置脚本
检查环境、安装依赖、运行流水线
"""

import os
import sys
import subprocess

def check_and_install_dependencies():
    """检查并安装必要的依赖"""
    print("🔧 检查Python依赖...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'joblib', 'numba', 'pyarrow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 缺失")
    
    if missing_packages:
        print(f"\n📦 安装缺失的包: {missing_packages}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ 依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            return False
    
    return True

def check_data_availability():
    """检查数据可用性"""
    print("\n📊 检查数据文件...")
    
    data_files = [
        'processed_data/dollar_bars_ETHUSDT.parquet',
        'processed_data/dollar_bars_BTCUSDT.parquet'
    ]
    
    available_data = []
    
    for data_file in data_files:
        if os.path.exists(data_file):
            size = os.path.getsize(data_file) / (1024*1024)  # MB
            print(f"✅ {data_file} ({size:.1f}MB)")
            available_data.append(data_file)
        else:
            print(f"❌ {data_file} 不存在")
    
    return available_data

def create_output_directories():
    """创建输出目录"""
    print("\n📁 创建输出目录...")
    
    directories = [
        'pipeline_results',
        'pipeline_results/ETHUSDT',
        'pipeline_results/BTCUSDT',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}")

def check_modules():
    """检查项目模块"""
    print("\n🔍 检查项目模块...")
    
    modules_to_check = [
        'data_processing.features',
        'utils.logger',
        'strategy.training',
        'strategy.reinforcement_learning',
        'strategy.backtesting'
    ]
    
    available_modules = []
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module}")
            available_modules.append(module)
        except ImportError as e:
            print(f"⚠️  {module} - {e}")
    
    return available_modules

def run_pipeline():
    """运行完整流水线"""
    print("\n🚀 开始运行完整交易系统流水线...")
    
    try:
        # 导入主流水线
        from run_complete_trading_pipeline import CompleteTradingPipeline
        
        # 创建ETHUSDT流水线
        print("\n" + "="*60)
        print("开始处理 ETHUSDT")
        print("="*60)
        
        pipeline = CompleteTradingPipeline(
            symbol='ETHUSDT',
            test_size=0.2,
            output_dir='pipeline_results/ETHUSDT'
        )
        
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("✅ ETHUSDT 处理完成")
            
            # 显示结果摘要
            print("\n📈 结果摘要:")
            if pipeline.backtest_results:
                for key, value in pipeline.backtest_results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            
            print(f"\n📁 结果保存在: pipeline_results/ETHUSDT/")
            
        else:
            print("❌ ETHUSDT 处理失败")
            
    except Exception as e:
        print(f"❌ 流水线运行失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🎯 完整交易系统流水线设置")
    print("="*50)
    
    # 1. 检查依赖
    if not check_and_install_dependencies():
        print("❌ 依赖检查失败，无法继续")
        return
    
    # 2. 检查数据
    available_data = check_data_availability()
    if not available_data:
        print("❌ 没有可用的数据文件")
        return
    
    # 3. 创建目录
    create_output_directories()
    
    # 4. 检查模块
    available_modules = check_modules()
    
    # 5. 运行流水线
    if len(available_modules) >= 2:  # 至少需要基本模块
        run_pipeline()
    else:
        print("❌ 缺少必要的项目模块")

if __name__ == "__main__":
    main() 