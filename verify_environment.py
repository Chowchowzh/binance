#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境验证脚本
验证Binance交易策略项目的所有依赖是否正确安装
"""

import sys
import importlib
from typing import List, Tuple, Dict

def check_python_version() -> bool:
    """检查Python版本"""
    version = sys.version_info
    print(f"🐍 Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✅ Python版本符合要求 (>=3.9)")
        return True
    else:
        print("❌ Python版本过低，需要3.9或更高版本")
        return False

def check_packages() -> Dict[str, bool]:
    """检查所有必需的包"""
    required_packages = [
        # 核心计算库
        ("pandas", "数据处理和分析"),
        ("numpy", "数值计算"),
        ("scipy", "科学计算"),
        
        # 机器学习
        ("sklearn", "机器学习工具"),
        ("lightgbm", "梯度提升树"),
        ("torch", "深度学习框架"),
        ("joblib", "并行计算"),
        
        # 金融分析
        ("talib", "技术指标"),
        ("arch", "时间序列分析"),
        ("statsmodels", "统计建模"),
        
        # 数据存储
        ("pymongo", "MongoDB操作"),
        ("pyarrow", "数据存储"),
        
        # 信号处理
        ("pywt", "小波变换"),
        ("filterpy", "滤波器"),
        
        # 可视化
        ("matplotlib", "基础绘图"),
        ("seaborn", "统计可视化"),
        ("plotly", "交互式图表"),
        
        # 网络请求
        ("requests", "HTTP请求"),
    ]
    
    results = {}
    print("\n📦 检查依赖包:")
    print("-" * 50)
    
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package:<15} - {description}")
            results[package] = True
        except ImportError:
            print(f"❌ {package:<15} - {description} (未安装)")
            results[package] = False
    
    return results

def check_project_modules() -> Dict[str, bool]:
    """检查项目模块"""
    project_modules = [
        ("strategy.transformer_model", "Transformer模型"),
        ("strategy.smart_position_control", "智能仓位控制"),
        ("strategy.signal_generator", "信号生成器"),
        ("dataset.config", "数据配置"),
        ("dataset.dataset", "数据库操作"),
    ]
    
    results = {}
    print("\n🏗️ 检查项目模块:")
    print("-" * 50)
    
    for module, description in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module:<30} - {description}")
            results[module] = True
        except ImportError as e:
            print(f"❌ {module:<30} - {description} (导入失败: {e})")
            results[module] = False
    
    return results

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 测试基本功能:")
    print("-" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # 测试数据处理
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        print("✅ pandas数据处理正常")
        
        # 测试数值计算
        arr = np.array([1, 2, 3])
        assert np.sum(arr) == 6
        print("✅ numpy数值计算正常")
        
        # 测试torch
        import torch
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.sum().item() == 6.0
        print("✅ torch张量操作正常")
        
        # 测试ta-lib
        import talib
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = talib.SMA(data, timeperiod=3)
        print("✅ ta-lib技术指标计算正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 Binance交易策略环境验证")
    print("=" * 60)
    
    # 检查Python版本
    python_ok = check_python_version()
    
    # 检查依赖包
    package_results = check_packages()
    
    # 检查项目模块
    module_results = check_project_modules()
    
    # 测试基本功能
    functionality_ok = test_basic_functionality()
    
    # 总结
    print("\n📊 验证总结:")
    print("=" * 60)
    
    total_packages = len(package_results)
    successful_packages = sum(package_results.values())
    
    total_modules = len(module_results)
    successful_modules = sum(module_results.values())
    
    print(f"🐍 Python版本: {'✅ 通过' if python_ok else '❌ 失败'}")
    print(f"📦 依赖包: {successful_packages}/{total_packages} 成功安装")
    print(f"🏗️ 项目模块: {successful_modules}/{total_modules} 导入成功")
    print(f"🧪 基本功能: {'✅ 正常' if functionality_ok else '❌ 异常'}")
    
    if python_ok and successful_packages == total_packages and functionality_ok:
        print("\n🎉 环境配置完美！可以开始使用项目了。")
        return True
    else:
        print("\n⚠️ 环境存在问题，请参考README_UV_SETUP.md进行故障排除。")
        
        if successful_packages < total_packages:
            failed_packages = [pkg for pkg, status in package_results.items() if not status]
            print(f"\n缺失的包: {', '.join(failed_packages)}")
            print("建议运行: uv pip install -e .")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 