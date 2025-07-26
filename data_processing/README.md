# Data Processing 数据处理模块

## 📋 概述

`data_processing` 模块是币安交易策略项目的核心数据处理引擎，负责从MongoDB数据库中获取原始K线数据，进行特征工程，生成机器学习就绪的数据集。

## 🏗️ 模块架构

```
data_processing/
├── preprocessor.py          # 主要数据预处理器
├── dataset_builder.py       # 数据集构建器
├── torch_dataset.py         # PyTorch数据集包装器
├── features/               # 特征工程子模块
│   ├── feature_builder.py     # 特征构建器
│   ├── technical_indicators.py # 技术指标计算
│   ├── feature_utils.py       # 特征工具函数
│   └── __init__.py
├── scripts/                # 执行脚本（新增）
│   ├── stage1_collect_data.py
│   ├── stage2_feature_engineering.py
│   └── incremental_update.py
└── README.md
```

## ⚡ 核心功能

### 1. 两阶段数据处理

- **阶段1**: 从MongoDB收集原始K线数据 → `processed_data/raw_data.parquet`
- **阶段2**: 特征工程和标准化 → `processed_data/featured_data.parquet`

### 2. 特征工程

- 技术指标：RSI, MACD, 布林带, ATR 等
- 订单流特征：OFI, 买卖压力, 价格影响
- 市场微观结构：流动性, 交易频率, VWAP
- 跨资产特征：相关性, 价格比率
- 滞后特征：价格和成交量的历史信息

### 3. 数据质量保证

- 智能NaN值处理
- 特征标准化（保护目标变量和价格数据）
- 数据完整性检查

## 🚀 快速开始

### 💡 快速演示

```bash
# 查看所有功能演示
uv run python3 data_processing/scripts/quick_demo.py

# 检查当前数据状态
uv run python3 data_processing/scripts/incremental_update.py --check-only
```

### 完整流程（推荐）

```bash
# 完整的两阶段处理
uv run python3 -c "
from data_processing import DataPreprocessor
processor = DataPreprocessor()
output_file = processor.process_data_two_stage(
    chunk_size=100000,
    normalize_features=True
)
print(f'处理完成：{output_file}')
"
```

### 分阶段执行

#### 🗄️ 阶段1：收集原始数据

```bash
# 方法1：使用脚本
uv run python3 data_processing/scripts/stage1_collect_data.py

# 方法2：直接调用
uv run python3 -c "
from data_processing import DataPreprocessor
processor = DataPreprocessor()
raw_file = processor._stage1_collect_raw_data(
    chunk_size=100000,
    max_workers=None
)
print(f'原始数据收集完成：{raw_file}')
"
```

#### 🔧 阶段2：特征工程

```bash
# 方法1：使用脚本
uv run python3 data_processing/scripts/stage2_feature_engineering.py

# 方法2：直接调用
uv run python3 -c "
from data_processing import DataPreprocessor
processor = DataPreprocessor()
featured_file = processor._stage2_feature_engineering(
    'processed_data/raw_data.parquet',
    normalize_features=True
)
print(f'特征工程完成：{featured_file}')
"
```

#### 🔄 增量更新

```bash
# 检查并增量更新原始数据
uv run python3 data_processing/scripts/incremental_update.py

# 或者指定起始时间
uv run python3 data_processing/scripts/incremental_update.py --from-timestamp 1640995200000
```

## 📊 输出文件说明

### 原始数据文件：`processed_data/raw_data.parquet`

包含多个交易对的OHLCV数据：

```python
# 列结构示例
[
    'open_time',           # 开盘时间戳
    'ETHUSDT_open',        # ETH开盘价
    'ETHUSDT_high',        # ETH最高价
    'ETHUSDT_low',         # ETH最低价
    'ETHUSDT_close',       # ETH收盘价
    'ETHUSDT_volume',      # ETH成交量
    'BTCUSDT_open',        # BTC开盘价
    # ... 其他交易对数据
]
```

### 特征数据文件：`processed_data/featured_data.parquet`

包含201个特征 + 目标变量：

```python
# 主要特征分类
{
    'price_features': ['*_close', '*_high', '*_low', '*_open'],      # 价格特征（未标准化）
    'volume_features': ['*_volume', '*_quote_asset_volume'],         # 成交量特征（未标准化）
    'technical_indicators': ['*_rsi_*', '*_macd_*', '*_bb_*'],       # 技术指标（已标准化）
    'order_flow': ['*_ofi', '*_buy_pressure', '*_sell_pressure'],    # 订单流（已标准化）
    'lag_features': ['*_lag_*', '*_return_*'],                       # 滞后特征（未标准化）
    'cross_asset': ['*_corr_*', '*_ratio_*'],                        # 跨资产（已标准化）
    'target': 'target',           # 目标变量：-1(下跌), 0(横盘), 1(上涨)
    'future_return': 'future_return'  # 未来收益率（未标准化）
}
```

## 🔧 配置选项

### 数据处理参数

```python
# 在 config/config.json 中配置
{
    "data_collection": {
        "target_symbol": "ETHUSDT",                    # 主要交易对
        "feature_symbols": ["ETHUSDT", "BTCUSDT"],     # 特征交易对
        "chunk_size": 100000,                          # 数据块大小
        "max_workers": null                            # 并发工作进程数
    },
    "feature_engineering": {
        "normalize_features": true,                    # 是否标准化特征
        "target_threshold": 0.0005,                    # 目标变量阈值
        "clean_initial_nans": true                     # 是否清理初始NaN值
    }
}
```

### 环境变量

```bash
# MongoDB连接配置
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB_NAME="binance_data"
```

## 📈 监控和调试

### 检查数据质量

```bash
# 验证目标变量分布
uv run python3 -c "
import pandas as pd
df = pd.read_parquet('processed_data/featured_data.parquet')
print('目标变量分布:')
print(df['target'].value_counts().sort_index())
print(f'未来收益率统计: 均值={df[\"future_return\"].mean():.6f}, 标准差={df[\"future_return\"].std():.6f}')
"
```

### 检查数据新鲜度

```bash
# 检查最新数据时间
uv run python3 -c "
import pandas as pd
from datetime import datetime
df = pd.read_parquet('processed_data/raw_data.parquet')
last_time = df['open_time'].max()
last_date = datetime.fromtimestamp(last_time/1000)
print(f'最新数据时间: {last_date}')
"
```

## 🚨 常见问题

### 1. 内存不足

```bash
# 减少chunk_size
uv run python3 -c "
from data_processing import DataPreprocessor
processor = DataPreprocessor()
processor.process_data_two_stage(chunk_size=50000)  # 减少到5万
"
```

### 2. 数据库连接失败

```bash
# 检查MongoDB连接
uv run python3 -c "
from database.connection import DatabaseConnection
from config.settings import load_project_config, get_legacy_config_dict
config = load_project_config()
legacy_config = get_legacy_config_dict(config)
db = DatabaseConnection(legacy_config)
print('数据库连接成功')
"
```

### 3. 目标变量分布异常

- 检查阈值设置是否合理（推荐0.0005）
- 确认价格数据未被错误标准化
- 验证未来收益率的计算逻辑

## 🎯 实际使用示例

### 日常数据维护

```bash
# 1. 检查数据新鲜度
uv run python3 data_processing/scripts/incremental_update.py --check-only

# 2. 如果有新数据，进行增量更新
uv run python3 data_processing/scripts/incremental_update.py

# 3. 重新生成特征数据
uv run python3 data_processing/scripts/stage2_feature_engineering.py
```

### 从零开始建立数据集

```bash
# 1. 收集所有原始数据
uv run python3 data_processing/scripts/stage1_collect_data.py --chunk-size 100000

# 2. 进行特征工程
uv run python3 data_processing/scripts/stage2_feature_engineering.py

# 3. 验证数据质量
uv run python3 -c "
import pandas as pd
df = pd.read_parquet('processed_data/featured_data.parquet')
print(f'数据形状: {df.shape}')
print('目标变量分布:')
print(df['target'].value_counts().sort_index())
"
```

### 内存优化的大数据处理

```bash
# 对于超大数据集，减少内存使用
uv run python3 data_processing/scripts/stage1_collect_data.py \
  --chunk-size 50000 \
  --max-workers 2

# 跳过特征标准化以节省内存
uv run python3 data_processing/scripts/stage2_feature_engineering.py \
  --no-normalize
```

## 🔗 相关文档

- [策略指南](../strategy/STRATEGY_GUIDE.md)
- [模型训练](../strategy/training/)
- [回测分析](../strategy/backtesting/)

## 📞 技术支持

如有问题，请检查：
1. 日志输出中的错误信息
2. 数据库连接状态
3. 磁盘空间是否充足
4. Python环境和依赖包版本

### 常用调试命令

```bash
# 查看帮助信息
uv run python3 data_processing/scripts/incremental_update.py --help

# 快速功能演示
uv run python3 data_processing/scripts/quick_demo.py

# 检查数据完整性
uv run python3 -c "
from data_processing import DataPreprocessor
processor = DataPreprocessor()
info = processor.validate_output_data()
print(info)
"
``` 