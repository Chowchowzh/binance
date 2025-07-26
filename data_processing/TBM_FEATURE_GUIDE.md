# 三分类标签法特征工程使用指南

## 概述

本系统实现了基于三分类标签法（Triple Barrier Method, TBM）的特征工程，这是一种先进的金融机器学习标签生成方法，可以生成更高质量的目标变量。

## 核心优势

### 相比传统标签方法的优势

1. **路径依赖性**：考虑价格到达边界的完整路径，而不仅仅是固定时间点的收益
2. **动态边界**：基于滚动波动率动态调整止盈止损边界
3. **风险管理**：内置止损机制，更符合实际交易情况
4. **标签质量**：减少噪声，提高信号质量

### 生成的标签类型

- **止盈 (1)**：价格在最大持仓期内触碰上边界
- **止损 (-1)**：价格在最大持仓期内触碰下边界  
- **时间到期/中性 (0)**：达到最大持仓期但未触碰边界

## 快速开始

### 1. 基本用法

```python
from data_processing.features import build_features_with_tbm

# 加载您的K线数据
df = pd.read_parquet("your_data.parquet")

# 使用TBM构建特征
feature_df = build_features_with_tbm(
    df=df,
    target_symbol='ETHUSDT',
    data_type='dollar_bars',  # 或 'minute_bars'
    profit_factor=2.0,        # 止盈因子
    loss_factor=1.0,          # 止损因子
    volatility_window=20,     # 波动率计算窗口
    max_holding_period=60     # 最大持仓期
)
```

### 2. 运行演示脚本

```bash
cd data_processing/scripts
python build_tbm_features.py
```

## 参数配置

### 核心TBM参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `profit_factor` | 2.0 | 止盈因子，相对于波动率的倍数 |
| `loss_factor` | 1.0 | 止损因子，相对于波动率的倍数 |
| `volatility_window` | 20 | 滚动波动率计算窗口 |
| `max_holding_period` | 60 | 最大持仓期（K线数） |
| `min_return_threshold` | 0.0001 | 最小收益阈值，过滤噪声 |
| `use_symmetric_barriers` | False | 是否使用对称边界 |

### 建议的参数组合

#### 保守策略
```python
profit_factor=1.5,
loss_factor=1.0,
volatility_window=30,
max_holding_period=100
```

#### 积极策略
```python
profit_factor=2.5,
loss_factor=1.5,
volatility_window=20,
max_holding_period=60
```

#### 对称策略
```python
profit_factor=2.0,
loss_factor=2.0,
use_symmetric_barriers=True
```

## 高级功能

### 1. CUSUM事件过滤

使用CUSUM过滤器只在结构性变化时生成事件：

```python
feature_df = build_features_with_tbm(
    df=df,
    use_cusum_events=True,
    cusum_threshold=0.01
)
```

### 2. 并行处理

加速TBM标签生成：

```python
feature_df = build_features_with_tbm(
    df=df,
    n_jobs=-1  # 使用所有CPU核心
)
```

### 3. 特征质量分析

```python
from data_processing.features import analyze_tbm_features_quality

analysis = analyze_tbm_features_quality(feature_df)
print(f"标签覆盖率: {analysis['label_coverage']:.2%}")
```

## 输出特征

### TBM相关特征

- `target`：主要目标变量（TBM标签）
- `future_return`：实际收益率
- `tbm_holding_period`：持仓期
- `tbm_touch_type`：触碰类型（1=止盈，-1=止损，0=时间到期）

### 其他特征

包含完整的技术指标、跨资产特征、时间特征等，具体取决于选择的`data_type`。

## 使用场景

### 1. 分类模型训练

```python
# 准备训练数据
X = feature_df.drop(['target', 'future_return', 'tbm_holding_period', 'tbm_touch_type'], axis=1)
y = feature_df['target'].dropna()

# 训练分类器
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)
```

### 2. 强化学习环境

TBM标签可以直接用作强化学习的奖励信号。

### 3. 风险管理

分析不同持仓期和触碰类型的分布，优化交易策略。

## 性能优化

### 内存优化

1. 使用`use_fp16=True`减少内存使用
2. 适当设置`volatility_window`和`max_holding_period`
3. 使用CUSUM过滤器减少事件数量

### 计算优化

1. 使用并行处理`n_jobs=-1`
2. 对于大数据集，考虑分批处理
3. 缓存中间结果

## 注意事项

### 数据要求

1. 数据必须包含`close`价格列（或带前缀的收盘价）
2. 数据应按时间顺序排列
3. 建议至少有1000个样本点

### 参数选择

1. `profit_factor`和`loss_factor`应根据资产的波动性调整
2. `volatility_window`不宜过小（建议>10）
3. `max_holding_period`应考虑数据的时间频率

### 标签不平衡

TBM可能产生不平衡的标签分布，建议：

1. 调整边界参数
2. 使用类别权重或重采样
3. 考虑使用集成方法

## 故障排除

### 常见问题

1. **标签覆盖率过低**：减小`max_holding_period`或调整边界参数
2. **内存不足**：开启`use_fp16`或分批处理
3. **计算速度慢**：使用并行处理或CUSUM过滤器

### 调试建议

1. 从小数据集开始测试
2. 检查原始数据的质量
3. 分析TBM参数对结果的影响

## 示例结果

典型的TBM标签分布：

```
止盈: 1,234 (35.2%)
止损: 987 (28.1%) 
中性: 1,289 (36.7%)
```

这种分布表明较好的风险回报平衡。

## 进一步阅读

- López de Prado, M. (2018). Advances in Financial Machine Learning
- 三分类标签法理论基础
- 金融时间序列标签方法比较研究 