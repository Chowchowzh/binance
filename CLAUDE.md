# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

This codebase uses UV for package management. Key commands:

```bash
# Environment setup
./setup_env.sh                                    # Automated environment setup
source .venv/bin/activate                         # Activate virtual environment  
uv pip install -e .                              # Install dependencies

# Model training and strategy execution
uv run python3 strategy/training/train_transformer.py    # Train Transformer model
uv run python3 -m strategy.market_making                 # Run trading strategy (default: adaptive)
uv run python3 -m strategy.market_making --strategy center_based --skip-alpha  # Run specific strategy

# Data operations
uv run python3 main.py fetch --symbols ETHUSDT BTCUSDT   # Fetch market data
uv run python3 main.py process                           # Process data with feature engineering
uv run python3 main.py config                            # Show current configuration
uv run python3 main.py structure                         # Display project structure

# Advanced data processing
uv run python3 main.py dollar-bars --symbol ETHUSDT --auto-threshold --analyze  # Generate dollar bars
uv run python3 run_pipeline.py                           # Run complete automated pipeline

# Environment verification
uv run python3 verify_environment.py                     # Verify all dependencies

# Strategy testing commands (available strategies)
--strategy {center_based,kelly,adaptive,volatility_adjusted,momentum_mean_reversion,ensemble}

# Configuration levels
--config {conservative,standard,aggressive}

# Testing and analysis
pytest                                             # Run tests (if available)
```

## High-Level Architecture

This is a cryptocurrency trading strategy project focused on ETH market making using Transformer models:

### Core Components

1. **Strategy Module (`strategy/`)** - Modular trading system with three main sub-modules:
   - **Training (`strategy/training/`)**: 
     - `train_transformer.py`: Main model training pipeline
     - `transformer_model.py`: Enhanced Transformer architecture with calibration
     - `signal_generator.py`: Signal generation and caching system
     - `enhanced_signal_generator.py`: Advanced signal processing with meta-labeling
   - **Backtesting (`strategy/backtesting/`)**: 
     - `backtest_runner.py`: Comprehensive backtesting engine
     - `smart_position_control.py`: 6 different position control strategies
     - `threshold_manager.py`: Dynamic threshold optimization
   - **Analysis (`strategy/analysis/`)**: 
     - `alpha_analysis.py`: Alpha performance evaluation
     - `backtest_analysis.py`: Results visualization and reporting
     - `advanced_model_evaluation.py`: Model performance metrics
   - **Reinforcement Learning (`strategy/reinforcement_learning/`)**: 
     - `actor_critic_agent.py`: Deep RL trading agent
     - `mdp_environment.py`: Trading environment for RL
     - `rl_training_pipeline.py`: RL training infrastructure

2. **Data Pipeline** - Complete end-to-end data processing:
   - `data_collection/`: Binance API integration with retry logic and MongoDB storage
   - `data_processing/`: Advanced feature engineering with multiple pipelines
     - `features/`: Technical indicators, dollar bars, advanced features
     - `scripts/`: Automated pipeline scripts for streamlined processing
   - `database/`: MongoDB connection management with connection pooling
   - Signal caching in `signal_cache/` for performance optimization

3. **Configuration System** - Hierarchical configuration management:
   - `config/settings.py`: Dataclass-based configuration with validation
   - Three configuration levels: conservative, standard, aggressive
   - Automatic config migration from legacy format

### Key Technical Details

- **Transformer Model**: 3-class classification (Up/Down/Flat) with configurable architecture
- **Position Control**: 6 different strategies from conservative center_based to complex ensemble
- **Feature Engineering**: Technical indicators via ta-lib, statistical features, volatility measures
- **Backtesting**: Comprehensive analysis with Alpha metrics, drawdown analysis, and visualization
- **Signal Caching**: Automatic caching of train/test signals to avoid recomputation

### Data Flow

1. Raw market data fetched from Binance API → MongoDB
2. Feature engineering creates technical indicators → `dataset/featured_data.parquet`
3. Transformer model trained on features → `transformer_model.pth`
4. Strategy generates signals → cached in `signal_cache/`
5. Position control algorithms convert signals to trades → backtest results

### Model Configuration

Key files for model behavior:
- `strategy/model_config.py`: Transformer architecture, training params, file paths
- `config/settings.py`: Database, trading, and data collection settings
- Position control configs in strategy code with get_aggressive_config/get_adaptive_config

The system is designed for modularity with clear separation between data collection, feature engineering, model training, and strategy execution.

## Development Workflows

### Complete Pipeline Workflow
```bash
# 1. Environment setup
uv run python3 verify_environment.py
./setup_env.sh

# 2. Data collection and processing
uv run python3 main.py fetch --symbols ETHUSDT BTCUSDT
uv run python3 run_pipeline.py  # Automated feature engineering + training

# 3. Strategy execution
uv run python3 -m strategy.market_making --strategy adaptive --config standard
```

### Manual Training Workflow  
```bash
# 1. Feature engineering
uv run python3 main.py process

# 2. Model training
uv run python3 strategy/training/train_transformer.py

# 3. Strategy backtesting
uv run python3 -m strategy.market_making
```

### Advanced Data Processing
```bash
# Dollar bars for alternative time sampling
uv run python3 main.py dollar-bars --symbol ETHUSDT --auto-threshold --analyze

# Incremental data updates
uv run python3 data_processing/scripts/incremental_update.py
```

## Important Development Patterns

### Configuration Management
- Always use `uv run python3` prefix for command execution
- Configuration lives in `config/settings.py` with dataclass structure
- Three config levels: conservative/standard/aggressive for different risk profiles
- Use `load_config()` function from `config.settings` for programmatic access

### Signal Caching System
- Training/test signals cached in `signal_cache/` to avoid recomputation
- Cache automatically invalidated when model changes
- Significantly speeds up backtesting iterations

### MongoDB Integration  
- Connection strings in `config/settings.py`
- Database name: "binance" 
- Collection naming: `{symbol}_{interval}_klines`
- Connection pooling handled automatically

### Model Training Best Practices
- Models saved to `transformer_model.pth` in project root
- Scalers saved to `scaler.pkl` 
- Use calibration for improved signal quality
- Feature importance analysis available in training pipeline

### Error Handling & Logging
- Comprehensive logging system in `utils/logger.py`
- Results automatically saved to `logs/` directory with timestamps
- MongoDB connection errors handled gracefully with retries