# 🚀 快速开始指南

## 📋 前置要求

1. Python 3.8+
2. 已安装依赖：`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `numba`, `joblib`
3. 真实的成交额K线数据文件

## ⚡ 一键运行完整系统

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行完整流水线
python run_simplified_pipeline.py
```

## 📊 分步骤使用

### Step 1: TBM特征工程

```python
from data_processing.features import build_features_with_tbm
import pandas as pd

# 加载数据
df = pd.read_parquet('processed_data/dollar_bars_ETHUSDT.parquet')

# TBM特征工程
features_df = build_features_with_tbm(
    df=df.tail(2000),           # 使用最近2000个数据点
    target_symbol='ETHUSDT',    # 目标交易对
    profit_factor=1.8,          # 止盈因子
    loss_factor=1.2,            # 止损因子
    volatility_window=15,       # 波动率窗口
    max_holding_period=20       # 最大持仓期
)

print(f"特征数据: {features_df.shape}")
print(f"标签分布: {features_df['target'].value_counts()}")
```

### Step 2: 模型训练

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# 准备数据
target_cols = ['target', 'future_return', 'tbm_label', 'tbm_return_pct']
feature_cols = [col for col in features_df.columns if col not in target_cols]

X = features_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = features_df['target'].dropna()

# 时间序列分割
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
model = GradientBoostingClassifier(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"测试准确率: {model.score(X_test_scaled, y_test):.4f}")
```

### Step 3: 回测分析

```python
# 生成预测
y_pred = model.predict(X_test_scaled)

# 获取收益数据
returns = features_df['future_return'].iloc[split_idx:].values

# 计算策略表现
strategy_returns = y_pred[:-1] * returns[1:]  # 滞后一期
total_return = np.prod(1 + strategy_returns) - 1
win_rate = (strategy_returns > 0).mean()

print(f"策略总收益: {total_return:.4f}")
print(f"胜率: {win_rate:.4f}")
```

### Step 4: 保存和加载模型

```python
import joblib

# 保存模型
joblib.dump(model, 'my_trading_model.pkl')
joblib.dump(scaler, 'my_scaler.pkl')

# 加载模型 (用于生产)
loaded_model = joblib.load('my_trading_model.pkl')
loaded_scaler = joblib.load('my_scaler.pkl')

# 预测新数据
new_predictions = loaded_model.predict(loaded_scaler.transform(new_features))
```

## 🎛️ 参数调优建议

### TBM参数
- `profit_factor`: 1.5-3.0 (止盈倍数)
- `loss_factor`: 0.8-2.0 (止损倍数)  
- `volatility_window`: 10-50 (波动率窗口)
- `max_holding_period`: 10-100 (最大持仓期)

### 模型参数
- `n_estimators`: 50-200 (树的数量)
- `max_depth`: 4-10 (树的深度)
- `learning_rate`: 0.05-0.2 (学习率)

## 📈 预期结果

### 正常性能范围
- **测试准确率**: 52-58%
- **策略收益**: 2-8% (短期)
- **胜率**: 50-60%
- **夏普比率**: 0.1-0.5

### 良好信号
- 测试准确率 > 55%
- 胜率 > 55%
- 正的夏普比率
- 最大回撤 < -30%

## 🔧 故障排除

### 常见问题

1. **导入错误**
   ```python
   # 确保在项目根目录运行
   import sys
   sys.path.append('.')
   ```

2. **数据问题**
   ```python
   # 检查数据质量
   print(df.isnull().sum())
   print(df.describe())
   ```

3. **内存不足**
   ```python
   # 减少样本数量
   df_sample = df.tail(1000)  # 使用更少数据
   ```

## 📱 生产环境部署

### 实时预测示例

```python
class TradingBot:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def predict_signal(self, latest_features):
        """预测交易信号"""
        features_scaled = self.scaler.transform([latest_features])
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()
        
        return {
            'signal': prediction,          # -1: 看跌, 1: 看涨
            'confidence': confidence       # 置信度
        }

# 使用示例
bot = TradingBot(
    'pipeline_results/ETHUSDT_simplified/gradient_boosting_model.pkl',
    'pipeline_results/ETHUSDT_simplified/scaler.pkl'
)

# 实时预测
result = bot.predict_signal(current_features)
print(f"交易信号: {result['signal']}, 置信度: {result['confidence']:.3f}")
```

## 📊 监控和优化

### 性能监控
- 定期评估模型准确率
- 监控实际交易表现
- 跟踪市场环境变化

### 模型更新
- 每月重新训练模型
- 使用最新数据更新特征
- 根据表现调整参数

---

🎯 **恭喜！你现在拥有了一个完整的量化交易系统！**

📧 如有问题，请查看 `COMPLETE_TRADING_SYSTEM_SUMMARY.md` 获取详细技术文档。 