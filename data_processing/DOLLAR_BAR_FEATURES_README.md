# 成交额K线特征工程

基于成交额K线的高级特征工程系统，专门设计用于加密货币高频交易策略。

## 主要特点

### 1. 成交额K线数据结构
- **自适应时间间隔**: 根据市场流动性自动调整K线间隔
- **统计特性优化**: 更接近IID高斯分布的时间序列
- **高频交易友好**: 避免传统固定时间间隔的噪声问题

### 2. 完整的交叉特征体系
- **技术指标交叉**: RSI、MACD、布林带等指标的组合特征
- **跨资产相关性**: 多交易对间的动态相关性特征
- **时间序列交叉**: 不同时间窗口的统计特征组合
- **市场微观结构**: 订单流、买卖压力等高频特征

### 3. fp16内存优化
- **内存使用减半**: 相比传统float32格式
- **计算效率提升**: 现代GPU优化支持
- **精度保持**: 特征工程场景下精度损失可忽略

## 快速开始

### 1. 完整流程执行

```bash
# 从原始分钟数据到完整特征工程
python data_processing/scripts/full_dollar_bar_pipeline.py \
    --input processed_data/raw_data.parquet \
    --symbols ETHUSDT BTCUSDT \
    --target_symbol ETHUSDT \
    --use_fp16 \
    --analysis
```

### 2. 分步执行

#### 步骤1: 生成成交额K线
```bash
python -c "
from data_processing.dollar_bars import DollarBarsGenerator
import pandas as pd

# 加载原始数据
df = pd.read_parquet('processed_data/raw_data.parquet')

# 生成成交额K线
generator = DollarBarsGenerator(auto_threshold=True)
dollar_bars = generator.generate_dollar_bars(df, 'ETHUSDT')

# 保存
generator.save_dollar_bars(dollar_bars, filename_prefix='dollar_bars_ETHUSDT')
"
```

#### 步骤2: 特征工程
```bash
python data_processing/scripts/build_dollar_bar_features.py \
    --input processed_data/dollar_bars_ETHUSDT.parquet \
    --target_symbol ETHUSDT \
    --use_fp16 \
    --analysis
```

### 3. Python API调用

```python
from data_processing.features import build_dollar_bar_features
import pandas as pd

# 加载成交额K线数据
dollar_bars = pd.read_parquet('processed_data/dollar_bars_ETHUSDT.parquet')

# 构建特征工程
features = build_dollar_bar_features(
    df=dollar_bars,
    target_symbol='ETHUSDT',
    feature_symbols=['ETHUSDT'],
    future_periods=15,
    use_fp16=True
)

print(f"特征维度: {features.shape}")
print(f"内存使用: {features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

## 特征体系详解

### 1. 基础特征 (Basic Features)
- **K线形态**: 实体大小、上下影线、价格位置
- **成交量特征**: 对数成交量、买卖压力、订单流不平衡
- **时间特征**: 对数持续时间、K线包含数量

### 2. 技术特征 (Technical Features)
- **移动平均**: SMA、EMA差值特征 (不保存绝对值)
- **动量指标**: RSI、MACD、ROC及其组合
- **波动率指标**: ATR、布林带、历史波动率
- **成交量指标**: OBV、MFI、成交量比率

### 3. 交叉特征 (Cross Features)
- **移动平均交叉**: 不同周期MA的差值特征
- **指标组合**: RSI×布林带位置、MACD×价格位置
- **跨时间窗口**: 不同窗口的统计特征差值

### 4. 跨资产特征 (Cross-Asset Features)
- **价格相关性**: 滚动收益率相关性
- **价格比率**: 价格比率及其统计特征
- **成交量比率**: 跨资产成交量关系
- **投资组合特征**: 等权重组合收益率特征

### 5. 时间特征 (Temporal Features)
- **周期性编码**: 小时、星期、月份的sin/cos编码
- **市场时段**: 早盘、午盘、晚盘、夜盘标识
- **工作日特征**: 周末、工作日标识

### 6. 高级统计特征 (Statistical Features)
- **协方差矩阵**: 价格协方差矩阵特征值
- **信息熵**: 收益率分布的熵特征
- **高阶矩**: 偏度、峰度等统计特征

### 7. 市场微观结构特征 (Microstructure Features)
- **价格影响**: Kyle's Lambda、Amihud非流动性
- **订单流**: 标准化订单流不平衡
- **交易密度**: 单位成交量交易频率

## 输出数据结构

### 特征文件
- **主文件**: `dollar_bar_features_ETHUSDT.parquet`
- **元数据**: `dollar_bar_features_ETHUSDT.metadata.json`
- **分析结果**: `dollar_bar_features_ETHUSDT.analysis.json`

### 目标变量
- **主目标**: `target` (-1: 下跌, 0: 横盘, 1: 上涨)
- **多级目标**: `target_level_1`, `target_level_2`, `target_level_3`
- **连续目标**: `future_return` (对数收益率)

### 数据类型优化
```python
# 特征列: float16 (内存优化)
feature_cols: dtype = 'float16'

# 目标列: float32 (训练精度)
target_cols: dtype = 'float32'

# 时间列: int64 (保持精度)
time_cols: dtype = 'int64'
```

## 性能特点

### 内存使用
- **传统float32**: ~8GB (10万样本 × 500特征)
- **优化float16**: ~4GB (50%内存节省)
- **额外开销**: 元数据 < 1MB

### 计算性能
- **特征计算**: 并行化talib计算
- **交叉特征**: 向量化NumPy操作
- **滚动统计**: 优化的pandas滚动窗口

### 质量保证
- **缺失值处理**: 智能NaN清理策略
- **异常值检测**: 无限值自动替换
- **数据验证**: 完整的特征质量检查

## 配置选项

### 成交额K线参数
```python
DollarBarsGenerator(
    threshold_usd=50_000_000,    # 成交额阈值
    auto_threshold=True,         # 自动计算阈值
    window_days=30,              # 动态阈值窗口
    target_bars_per_day=50       # 目标日K线数量
)
```

### 特征工程参数
```python
DollarBarFeatures(
    use_fp16=True,              # 使用fp16格式
    future_periods=15,          # 预测周期
    price_threshold=0.002,      # 价格变化阈值
    windows=[10, 20, 50],       # 滚动窗口
    lag_periods=[1, 3, 5, 10]   # 滞后周期
)
```

## 特征重要性分析

系统自动进行特征重要性分析：

### 随机森林重要性
- 基于树分裂时的不纯度减少
- 适合非线性特征关系

### 互信息重要性
- 基于信息论的特征选择
- 捕捉非线性依赖关系

### 特征分组分析
- **基础特征**: K线形态、成交量
- **技术特征**: 传统技术指标
- **交叉特征**: 特征间的交互项
- **时间特征**: 时间相关特征
- **统计特征**: 高级统计学特征

## 最佳实践

### 1. 数据质量
- 确保原始数据完整性
- 检查交易对数据一致性
- 验证时间戳连续性

### 2. 内存管理
- 优先使用fp16格式
- 分批处理大数据集
- 及时释放中间变量

### 3. 特征选择
- 使用特征重要性分析
- 去除高相关性特征
- 考虑计算成本vs收益

### 4. 模型训练
- 注意前瞻偏差
- 正确划分训练/验证集
- 考虑类别不平衡问题

## 故障排除

### 常见问题

1. **内存不足**: 使用 `--use_fp16` 或分批处理
2. **特征维度过高**: 增加特征选择步骤
3. **目标不平衡**: 调整价格变化阈值
4. **计算缓慢**: 检查并行化设置

### 日志分析
所有操作都有详细日志记录，包括：
- 数据维度变化
- 特征生成进度
- 内存使用统计
- 质量检查结果

---

## 下一步开发

- [ ] 增加更多跨资产特征
- [ ] 支持自定义特征组合
- [ ] 增加实时特征更新
- [ ] GPU加速计算支持
- [ ] 分布式特征计算 