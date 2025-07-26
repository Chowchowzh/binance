#!/bin/bash
# Binance Trading Strategy Environment Setup Script
# 币安交易策略环境设置脚本

echo "🚀 正在设置Binance交易策略环境..."

# 检查uv是否已安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv安装完成"
fi

# 设置uv路径
export PATH="$HOME/.local/bin:$PATH"

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo "📦 创建虚拟环境..."
    uv venv
    echo "✅ 虚拟环境创建完成"
fi

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source .venv/bin/activate

# 安装/更新依赖
echo "📚 安装项目依赖..."
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

echo "✅ 环境设置完成！"
echo ""
echo "📋 使用说明："
echo "1. 运行策略回测: python strategy/market_making.py"
echo "2. 训练Transformer模型: python strategy/train_transformer.py"
echo "3. 数据预处理: python strategy/preprocess_data.py"
echo "4. 获取市场数据: python market_api/main.py"
echo ""
echo "💡 要激活环境，请运行: source .venv/bin/activate"
echo "💡 要退出环境，请运行: deactivate" 