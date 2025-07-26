# Binance交易策略项目 - UV环境配置指南

这是一个基于Transformer模型的加密货币交易策略项目，使用现代化的Python包管理工具UV来管理虚拟环境和依赖。

## 🚀 快速开始

### 方法一：自动安装脚本（推荐）

```bash
# 给脚本执行权限
chmod +x setup_env.sh

# 运行自动安装脚本
./setup_env.sh
```

### 方法二：手动安装

```bash
# 1. 安装UV（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 添加UV到PATH
export PATH="$HOME/.local/bin:$PATH"

# 3. 创建虚拟环境
uv venv

# 4. 激活虚拟环境
source .venv/bin/activate

# 5. 安装项目依赖
uv pip install -e .
```

## 📦 项目结构

```
binance/
├── strategy/                    # 交易策略核心模块
│   ├── transformer_model.py    # Transformer模型定义
│   ├── train_transformer.py    # 模型训练脚本
│   ├── market_making.py         # 主策略执行脚本
│   ├── smart_position_control.py # 智能仓位控制
│   ├── signal_generator.py      # 信号生成器
│   ├── backtest_runner.py       # 回测运行器
│   └── ...
├── dataset/                     # 数据处理模块
│   ├── dataset.py              # 数据库操作
│   ├── config.py               # 数据配置
│   └── torch_dataset.py        # PyTorch数据集
├── market_api/                  # 市场数据API
│   ├── main.py                 # 数据获取主程序
│   └── marketdata.py           # 市场数据接口
├── backtest_results/           # 回测结果存储
├── signal_cache/               # 信号缓存目录
├── pyproject.toml              # UV项目配置文件
├── requirements.txt            # 传统依赖文件（备用）
└── setup_env.sh               # 环境设置脚本
```

## 🔧 主要依赖包

### 核心计算库
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **scipy**: 科学计算
- **torch**: 深度学习框架（Transformer模型）

### 机器学习
- **scikit-learn**: 机器学习工具
- **lightgbm**: 梯度提升树模型
- **joblib**: 并行计算和模型序列化

### 金融分析
- **ta-lib**: 技术指标计算
- **arch**: 时间序列分析
- **statsmodels**: 统计建模

### 数据存储
- **pymongo**: MongoDB数据库操作
- **pyarrow**: 高效数据存储格式

### 可视化
- **matplotlib**: 基础绘图
- **seaborn**: 统计可视化
- **plotly**: 交互式图表

## 🎯 主要功能模块

### 1. 数据获取和处理
```bash
# 激活环境
source .venv/bin/activate

# 获取市场数据
python market_api/main.py

# 数据预处理
python strategy/preprocess_data.py
```

### 2. 模型训练
```bash
# 训练Transformer模型
python strategy/train_transformer.py
```

### 3. 策略回测
```bash
# 运行完整的策略回测
python strategy/market_making.py
```

### 4. 智能仓位控制
项目包含多种仓位控制策略：
- **中枢策略**: 基于设定中枢的线性映射
- **凯利公式**: 基于期望收益的经典策略
- **自适应策略**: 非线性信号变换
- **波动率调整**: 根据市场波动动态调整
- **动量均值回归**: 多信号融合策略
- **集成策略**: 多策略加权组合

## 🛠️ 开发工具

### 安装开发依赖
```bash
# 安装开发工具（已在pyproject.toml中配置）
uv pip install --dev
```

### 运行测试
```bash
# 运行单元测试
pytest
```

### Jupyter环境
```bash
# 启动Jupyter
jupyter lab
```

## 🔍 环境管理

### 查看已安装包
```bash
uv pip list
```

### 更新依赖
```bash
uv pip install --upgrade -e .
```

### 添加新依赖
```bash
# 添加新包到pyproject.toml，然后运行：
uv pip install -e .
```

### 导出依赖清单
```bash
uv pip freeze > requirements-frozen.txt
```

## 🌐 网络配置

如果网络连接有问题，可以使用国内镜像：

```bash
# 使用清华大学镜像
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

# 或使用阿里云镜像
uv pip install -i https://mirrors.aliyun.com/pypi/simple/ -e .
```

## 📝 配置文件说明

### pyproject.toml
- 项目元数据和依赖管理
- 包发现配置
- 开发工具配置

### requirements.txt
- 传统pip格式的依赖文件
- 作为备用方案保留

## ⚠️ 注意事项

1. **Python版本**: 需要Python 3.9或更高版本
2. **ta-lib安装**: 如果ta-lib安装失败，可能需要先安装系统依赖
3. **MongoDB**: 数据存储需要MongoDB服务
4. **内存要求**: Transformer模型训练需要足够的内存

## 🔧 故障排除

### ta-lib安装问题
```bash
# macOS
brew install ta-lib

# Ubuntu/Debian
sudo apt-get install libta-lib-dev

# 然后重新安装Python包
uv pip install ta-lib
```

### 权限问题
```bash
chmod +x setup_env.sh
```

### 虚拟环境问题
```bash
# 删除并重新创建虚拟环境
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e .
```

## 📞 支持

如果遇到问题，请检查：
1. Python版本是否符合要求
2. 网络连接是否正常
3. 系统依赖是否已安装
4. 权限设置是否正确

---

**享受现代化的Python包管理体验！** 🎉 