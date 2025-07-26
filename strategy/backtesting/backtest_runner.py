# -*- coding: utf-8 -*-
"""
回测运行器模块
负责执行回测并生成结果
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional

from config.settings import TradingConfig


def run_backtest(predictions, X_test, test_df_full, config: TradingConfig, 
                 target_symbol, position_controller=None, logger=None, results_dir='.', 
                 train_signals=None):
    """运行回测并评估策略表现"""
    if logger:
        logger.log("开始回测...")
    
    # 转换配置为字典格式
    config_dict = config.to_dict()
    
    initial_cash = config.initial_cash
    prediction_threshold = config_dict.get('prediction_threshold', 0.0)
    min_trade_amount_eth = config.min_trade_amount_eth
    fee_rate = config.fee_rate
    position_adjustment_factor = config.position_adjustment_factor

    cash, inventory, short_collateral = initial_cash, 0.0, 0.0
    pnl_history, trades_history, eth_value_history, predicted_return_history = [], [], [], []

    close_price_col = f'{target_symbol}_close'
    open_price_col = f'{target_symbol}_open'

    test_data = test_df_full.loc[X_test.index]
    
    num_predictions = len(predictions)
    pred_series = pd.Series(predictions, index=X_test.index[:num_predictions])
    
    for i in range(len(test_data)):
        current_time = test_data.index[i]
        
        if current_time not in pred_series.index:
            continue

        current_price = test_data.iloc[i][close_price_col]
        open_price = test_data.iloc[i][open_price_col]

        total_equity = cash + short_collateral + inventory * current_price
        if total_equity <= 0:
            if logger:
                logger.log(f"--- 爆仓警告 at {current_time} ---")
            break

        predicted_signal = pred_series[current_time]

        # 使用智能仓位控制器
        if position_controller is not None:
            kwargs = {'train_signals': train_signals} if train_signals is not None else {}
            target_position_ratio = position_controller.calculate_target_position(
                signal=predicted_signal,
                current_price=current_price,
                total_equity=total_equity,
                current_position=inventory,
                **kwargs
            )
        else:
            # 回退到简单线性映射
            base_position_ratio = config_dict.get('base_position_ratio', 0.5)
            signal_amplifier = config_dict.get('signal_amplifier', 10.0)
            max_position_ratio = config_dict.get('max_position_ratio', 1.0)
            min_position_ratio = config_dict.get('min_position_ratio', -0.5)
            
            target_position_ratio = base_position_ratio + (predicted_signal * signal_amplifier)
            target_position_ratio = max(min_position_ratio, min(max_position_ratio, target_position_ratio))
        
        target_inventory_usdt = total_equity * target_position_ratio
        target_inventory_eth = target_inventory_usdt / open_price if open_price > 0 else 0
        
        trade_eth_needed = (target_inventory_eth - inventory) * position_adjustment_factor

        if abs(predicted_signal) < prediction_threshold or abs(trade_eth_needed) < min_trade_amount_eth:
            trade_eth_needed = 0

        # 风险控制
        max_long_inventory_eth = (total_equity * config_dict.get('max_position_ratio', 1.0)) / open_price if open_price > 0 else 0
        max_short_inventory_eth = (total_equity * abs(config_dict.get('min_position_ratio', -0.5))) / open_price if open_price > 0 else 0
        
        if trade_eth_needed > 0:
            max_buy = max_long_inventory_eth - inventory
            trade_eth_needed = min(trade_eth_needed, max_buy)
        else:
            max_sell = inventory + max_short_inventory_eth
            trade_eth_needed = max(trade_eth_needed, -max_sell)
        
        # 执行交易
        if abs(trade_eth_needed) > 0:
            fee = abs(trade_eth_needed) * open_price * fee_rate
            trade_type = 'sell' if trade_eth_needed < 0 else 'buy'
            
            old_inventory = inventory
            inventory += trade_eth_needed

            if trade_type == 'sell':
                closing_long = min(abs(trade_eth_needed), old_inventory) if old_inventory > 0 else 0
                opening_short = abs(trade_eth_needed) - closing_long
                if closing_long > 0: 
                    cash += closing_long * open_price
                if opening_short > 0: 
                    short_collateral += opening_short * open_price
            else:
                closing_short = min(trade_eth_needed, abs(old_inventory)) if old_inventory < 0 else 0
                opening_long = trade_eth_needed - closing_short
                if closing_short > 0:
                    avg_short_entry = short_collateral / abs(old_inventory) if old_inventory < 0 else open_price
                    collateral_released = closing_short * avg_short_entry
                    cost_to_close = closing_short * open_price
                    cash += (collateral_released - cost_to_close)
                    short_collateral -= collateral_released
                if opening_long > 0: 
                    cash -= opening_long * open_price
            
            cash -= fee
            
            current_pnl = cash + short_collateral + inventory * current_price - initial_cash
            trades_history.append({
                'time': current_time, 
                'type': trade_type, 
                'price': open_price, 
                'amount': abs(trade_eth_needed), 
                'fee': fee, 
                'pnl_at_trade': current_pnl,
                'eth_holdings_value_at_trade': inventory * current_price,
                'predicted_signal_at_trade': predicted_signal,
                'target_position_ratio': target_position_ratio,
                'actual_position_ratio': (inventory * current_price) / total_equity if total_equity > 0 else 0
            })

        pnl_history.append(cash + short_collateral + inventory * current_price - initial_cash)
        eth_value_history.append(inventory * current_price)
        predicted_return_history.append(predicted_signal)

    # 处理结果
    if not pnl_history:
        empty_series = pd.Series(dtype=float)
        return {
            'metrics': {'final_pnl': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_trades': 0, 'return_rate': 0},
            'pnl_series': empty_series,
            'hold_pnl_series': empty_series,
            'eth_value_history': empty_series,
            'predicted_return_history': empty_series,
            'trades_df': pd.DataFrame()
        }

    pnl_series = pd.Series(pnl_history, index=test_data.index[:len(pnl_history)])
    eth_value_series = pd.Series(eth_value_history, index=test_data.index[:len(pnl_history)])
    predicted_return_series = pd.Series(predicted_return_history, index=test_data.index[:len(pnl_history)])
    
    # 计算指标
    daily_returns = pnl_series.resample('D').last().ffill().pct_change()
    sharpe_ratio = np.sqrt(365) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
    
    cumulative_pnl = initial_cash + pnl_series
    peak = cumulative_pnl.cummax()
    max_drawdown = (cumulative_pnl - peak).min() / peak.max() if not peak.empty and peak.max() > 0 else 0

    initial_eth_price = test_data[open_price_col].iloc[0]
    hold_pnl_series = (test_data[close_price_col] * (initial_cash / initial_eth_price)) - initial_cash
    hold_pnl_series = hold_pnl_series.loc[pnl_series.index]
    
    metrics = {
        'final_pnl': pnl_series.iloc[-1], 
        'sharpe_ratio': sharpe_ratio, 
        'max_drawdown': max_drawdown, 
        'total_trades': len(trades_history), 
        'return_rate': pnl_series.iloc[-1] / initial_cash
    }
    
    # 保存交易记录
    trades_df = pd.DataFrame(trades_history)
    if not trades_df.empty:
        trades_csv_path = os.path.join(results_dir, 'backtest_trades.csv')
        trades_df.to_csv(trades_csv_path, index=False)
        if logger:
            logger.log(f"交易日志已保存到: {trades_csv_path}")
    
    return {
        'metrics': metrics, 
        'pnl_series': pnl_series, 
        'hold_pnl_series': hold_pnl_series, 
        'trades_df': trades_df, 
        'eth_value_history': eth_value_series, 
        'predicted_return_history': predicted_return_series
    } 