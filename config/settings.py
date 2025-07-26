# -*- coding: utf-8 -*-
"""
项目设置和配置管理模块
统一管理所有配置信息，包括数据库、交易、模型等
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


@dataclass
class DatabaseConfig:
    """数据库配置类"""
    mongodb_uri: str = "mongodb+srv://luthercheu1129:Wy1EhFETs5QvJtT1@binance.mjdmrhf.mongodb.net/?retryWrites=true&w=majority"
    mongodb_db_name: str = "binance"
    collection_name_template: str = "{symbol}_{interval}_klines"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class APIConfig:
    """API配置类"""
    base_url: str = "https://api.binance.com"
    api_request_delay_seconds: float = 0.5
    progress_report_interval_batches: int = 10
    max_retries: int = 5
    retry_delay_seconds: int = 300
    timeout_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataCollectionConfig:
    """数据收集配置类"""
    # 数据收集相关参数
    feature_symbols: List[str] = field(default_factory=lambda: ["ETHUSDT", "BTCUSDT"])
    target_symbol: str = "ETHUSDT"
    interval: str = "1m" 
    start_date: str = "2020-07-04 13:44:59.999000"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradingConfig:
    """交易配置类"""
    # 交易参数
    initial_cash: float = 100000.0
    min_trade_amount_eth: float = 0.001
    fee_rate: float = 0.00075
    position_adjustment_factor: float = 0.2
    
    # 信号处理参数
    prediction_threshold_percentile: float = 5.0
    calibration_bins: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """模型配置类"""
    # 数据参数
    sequence_length: int = 60
    train_test_split_ratio: float = 0.8
    
    # 模型超参数
    num_classes: int = 3
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # 训练参数
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-5
    clip_value: float = 1.0
    
    # TBM (Triple-Barrier Method) 配置
    tbm_profit_factor: float = 2.0
    tbm_loss_factor: float = 1.0
    tbm_volatility_window: int = 20
    tbm_max_holding_period: int = 60
    tbm_min_return_threshold: float = 0.0001
    tbm_use_symmetric_barriers: bool = False
    tbm_volatility_method: str = 'log_return'
    tbm_use_cusum_events: bool = False
    tbm_cusum_threshold: float = 0.01
    
    # 元标签 (Meta-Labeling) 配置
    use_meta_labeling: bool = True
    meta_model_type: str = 'random_forest'  # 'random_forest', 'gbm', 'logistic', 'neural_net'
    meta_probability_threshold: float = 0.5
    meta_prediction_type: str = 'any_direction'  # 'exact_match', 'any_direction', 'profitable_only'
    meta_validation_split: float = 0.2
    
    # 元标签模型参数
    meta_rf_n_estimators: int = 100
    meta_rf_max_depth: int = 10
    meta_gbm_learning_rate: float = 0.1
    meta_neural_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    
    # 文件路径
    data_path: str = 'processed_data/featured_data_reduced.parquet'
    model_save_path: str = 'model/transformer_model.pth'
    meta_model_save_path: str = 'model/meta_model.pkl'
    scaler_path: str = 'model/scaler.pkl'
    train_signals_cache_path: str = 'signal_cache/train_signals.npy'
    test_signals_cache_path: str = 'signal_cache/test_signals.npy'
    tbm_labels_cache_path: str = 'processed_data/tbm_labels.parquet'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            'num_classes': self.num_classes,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """获取训练参数"""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'clip_value': self.clip_value
        }
    
    def get_tbm_params(self) -> Dict[str, Any]:
        """获取TBM参数"""
        return {
            'profit_factor': self.tbm_profit_factor,
            'loss_factor': self.tbm_loss_factor,
            'volatility_window': self.tbm_volatility_window,
            'max_holding_period': self.tbm_max_holding_period,
            'min_return_threshold': self.tbm_min_return_threshold,
            'use_symmetric_barriers': self.tbm_use_symmetric_barriers
        }
    
    def get_meta_labeling_params(self) -> Dict[str, Any]:
        """获取元标签参数"""
        meta_model_params = {}
        
        if self.meta_model_type == 'random_forest':
            meta_model_params = {
                'n_estimators': self.meta_rf_n_estimators,
                'max_depth': self.meta_rf_max_depth
            }
        elif self.meta_model_type == 'gbm':
            meta_model_params = {
                'learning_rate': self.meta_gbm_learning_rate
            }
        elif self.meta_model_type == 'neural_net':
            meta_model_params = {
                'hidden_dims': self.meta_neural_hidden_dims
            }
        
        return {
            'meta_model_type': self.meta_model_type,
            'meta_model_params': meta_model_params,
            'probability_threshold': self.meta_probability_threshold
        }
    
    def get_paths(self) -> Dict[str, str]:
        """获取文件路径"""
        return {
            'data_path': self.data_path,
            'model_save_path': self.model_save_path,
            'meta_model_save_path': self.meta_model_save_path,
            'scaler_path': self.scaler_path,
            'train_signals_cache_path': self.train_signals_cache_path,
            'test_signals_cache_path': self.test_signals_cache_path,
            'tbm_labels_cache_path': self.tbm_labels_cache_path
        }


@dataclass
class RLConfig:
    """强化学习配置类"""
    # 环境配置
    initial_cash: float = 100000.0
    transaction_cost_bps: float = 7.5
    market_impact_factor: float = 0.1
    max_position: float = 1.0
    min_position: float = -1.0
    position_levels: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0])
    lookback_window: int = 252
    price_column: str = "close"
    volume_column: str = "volume"
    
    # 奖励配置
    reward_type: str = "differential_sharpe"
    sharpe_window: int = 252
    transaction_cost_penalty: float = 1.0
    volatility_penalty: float = 0.1
    risk_free_rate: float = 0.02
    normalization_factor: float = 16.0
    
    # Agent配置
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    weight_decay: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 10000
    update_epochs: int = 10
    target_kl: float = 0.01
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2
    use_batch_norm: bool = True
    
    # 训练配置
    num_episodes: int = 1000
    max_steps_per_episode: int = 5000
    eval_frequency: int = 50
    save_frequency: int = 100
    early_stopping_patience: int = 100
    early_stopping_threshold: float = 0.001
    pre_training_episodes: int = 100
    fine_tuning_episodes: int = 200
    curriculum_stages: int = 3
    curriculum_learning: bool = True  # 添加缺失的课程学习配置
    random_seed: int = 42
    num_parallel_envs: int = 4  # 启用4个并行环境
    log_frequency: int = 10
    
    # 回测配置
    walk_forward_window: int = 2520
    min_train_size: int = 1260
    test_size: int = 252
    step_size: int = 63
    purge_length: int = 5
    embargo_length: int = 3
    enable_purging: bool = True
    enable_embargoing: bool = True
    cv_folds: int = 5
    sample_weight_decay: float = 0.95
    min_sample_weight: float = 0.1
    use_sample_weights: bool = True
    
    # 超参数优化配置
    hp_enabled: bool = False
    hp_n_trials: int = 100
    hp_timeout: int = 7200
    hp_sampler: str = "tpe"
    hp_pruner: str = "median"
    hp_optimization_direction: str = "maximize"
    hp_metric: str = "sharpe_ratio"
    
    # 路径配置
    model_save_dir: str = "model/rl_models"
    checkpoint_dir: str = "model/rl_checkpoints"
    log_dir: str = "logs/rl_training"
    results_dir: str = "backtest_results/rl"
    experiment_dir: str = "experiments/rl"
    experiment_name: str = "rl_experiment"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnhancedTransformerConfig:
    """增强Transformer配置类"""
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.3
    drop_path_rate: float = 0.1
    use_relative_position: bool = True
    max_relative_position: int = 128
    use_financial_embedding: bool = True
    embedding_dropout: float = 0.2
    layer_norm_eps: float = 1e-6
    activation: str = "gelu"
    pre_norm: bool = True
    gradient_clip_val: float = 1.0
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.01
    
    # 正则化配置
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.0
    stochastic_depth_rate: float = 0.1
    
    # 早停配置
    patience: int = 20
    min_delta: float = 0.001
    monitor: str = "val_loss"
    mode: str = "min"
    restore_best_weights: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectConfig:
    """项目总配置类"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reinforcement_learning: RLConfig = field(default_factory=RLConfig)
    enhanced_transformer: EnhancedTransformerConfig = field(default_factory=EnhancedTransformerConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'database': self.database.to_dict(),
            'api': self.api.to_dict(),
            'data_collection': self.data_collection.to_dict(), 
            'trading': self.trading.to_dict(),
            'model': self.model.to_dict(),
            'reinforcement_learning': self.reinforcement_learning.to_dict(),
            'enhanced_transformer': self.enhanced_transformer.to_dict()
        }
    
    def merge_legacy_config(self, legacy_config: Dict[str, Any]) -> None:
        """合并传统配置格式"""
        # 更新数据收集配置
        if 'data_collection' in legacy_config:
            dc = legacy_config['data_collection']
            if 'target_symbol' in dc:
                self.data_collection.target_symbol = dc['target_symbol']
            if 'feature_symbols' in dc:
                self.data_collection.feature_symbols = dc['feature_symbols']
            if 'interval' in dc:
                self.data_collection.interval = dc['interval']
            if 'start_date' in dc:
                self.data_collection.start_date = dc['start_date']
        
        # 向下兼容：直接从根级别读取数据收集配置
        if 'target_symbol' in legacy_config:
            self.data_collection.target_symbol = legacy_config['target_symbol']
        if 'feature_symbols' in legacy_config:
            self.data_collection.feature_symbols = legacy_config['feature_symbols']
        if 'interval' in legacy_config:
            self.data_collection.interval = legacy_config['interval']
        if 'start_date' in legacy_config:
            self.data_collection.start_date = legacy_config['start_date']
            
        # 更新数据库配置
        if 'database' in legacy_config:
            db = legacy_config['database']
            if 'mongodb_uri' in db:
                self.database.mongodb_uri = db['mongodb_uri']
            if 'mongodb_db_name' in db:
                self.database.mongodb_db_name = db['mongodb_db_name']
            if 'collection_name_template' in db:
                self.database.collection_name_template = db['collection_name_template']
        
        # 向下兼容：直接从根级别读取数据库配置
        if 'mongodb_uri' in legacy_config:
            self.database.mongodb_uri = legacy_config['mongodb_uri']
        if 'mongodb_db_name' in legacy_config:
            self.database.mongodb_db_name = legacy_config['mongodb_db_name']
        if 'collection_name_template' in legacy_config:
            self.database.collection_name_template = legacy_config['collection_name_template']
            
        # 更新API配置
        if 'api' in legacy_config:
            api = legacy_config['api']
            if 'base_url' in api:
                self.api.base_url = api['base_url']
            if 'api_request_delay_seconds' in api:
                self.api.api_request_delay_seconds = api['api_request_delay_seconds']
            if 'progress_report_interval_batches' in api:
                self.api.progress_report_interval_batches = api['progress_report_interval_batches']
            if 'max_retries' in api:
                self.api.max_retries = api['max_retries']
            if 'retry_delay_seconds' in api:
                self.api.retry_delay_seconds = api['retry_delay_seconds']
            if 'timeout_seconds' in api:
                self.api.timeout_seconds = api['timeout_seconds']
        
        # 向下兼容：直接从根级别读取API配置
        if 'api_request_delay_seconds' in legacy_config:
            self.api.api_request_delay_seconds = legacy_config['api_request_delay_seconds']
        if 'progress_report_interval_batches' in legacy_config:
            self.api.progress_report_interval_batches = legacy_config['progress_report_interval_batches']
            
        # 更新交易配置
        if 'trading' in legacy_config:
            tr = legacy_config['trading']
            for field_name in ['initial_cash', 'min_trade_amount_eth', 'fee_rate', 
                             'position_adjustment_factor', 'prediction_threshold_percentile', 'calibration_bins']:
                if field_name in tr:
                    setattr(self.trading, field_name, tr[field_name])
            
        # 🔥 关键修复：更新模型配置
        if 'model' in legacy_config:
            model_config = legacy_config['model']
            for field_name in ['sequence_length', 'train_test_split_ratio', 'num_classes', 
                             'd_model', 'nhead', 'num_encoder_layers', 'dim_feedforward', 
                             'dropout', 'batch_size', 'epochs', 'learning_rate', 'clip_value',
                             'data_path', 'model_save_path', 'scaler_path', 
                             'train_signals_cache_path', 'test_signals_cache_path']:
                if field_name in model_config:
                    setattr(self.model, field_name, model_config[field_name])


def load_config(config_path: str = 'config/config.json') -> ProjectConfig:
    """
    加载配置文件并返回ProjectConfig对象
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ProjectConfig对象
    """
    # 尝试多个可能的配置文件路径
    possible_paths = [
        config_path,
        'config.json',
        os.path.join(os.path.dirname(__file__), 'config.json'),
        os.path.join(os.path.dirname(__file__), '..', 'dataset', 'config.json')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                print(f"成功加载配置文件: {path}")
                
                # 创建ProjectConfig对象并合并旧配置
                project_config = ProjectConfig()
                project_config.merge_legacy_config(config_dict)
                return project_config
                
            except Exception as e:
                print(f"加载配置文件失败 {path}: {e}")
                continue
    
    # 如果没有找到配置文件，返回默认配置
    print("未找到配置文件，使用默认配置")
    return ProjectConfig()


def save_config(config: Union[Dict[str, Any], ProjectConfig], 
                config_path: str = 'config/config.json') -> bool:
    """
    保存配置到文件
    
    Args:
        config: 配置对象或字典
        config_path: 保存路径
        
    Returns:
        是否保存成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 转换为字典格式
        if isinstance(config, ProjectConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config
        
        # 保存到文件
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
        print(f"配置已保存到: {config_path}")
        return True
        
    except Exception as e:
        print(f"保存配置失败: {e}")
        return False


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    default_config = ProjectConfig()
    return default_config.to_dict()


def load_project_config(config_path: str = 'config/config.json') -> ProjectConfig:
    """
    加载项目配置对象
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        项目配置对象
    """
    # load_config已经返回正确的ProjectConfig对象
    return load_config(config_path)


def get_legacy_config_dict(project_config: ProjectConfig = None) -> Dict[str, Any]:
    """
    获取兼容旧格式的配置字典
    
    Args:
        project_config: 项目配置对象
        
    Returns:
        兼容旧格式的配置字典
    """
    if project_config is None:
        project_config = ProjectConfig()
    
    # 创建兼容旧格式的配置字典
    legacy_config = {}
    
    # 从data_collection配置中复制
    legacy_config.update({
        'target_symbol': project_config.data_collection.target_symbol,
        'feature_symbols': project_config.data_collection.feature_symbols,
        'interval': project_config.data_collection.interval,
        'start_date': project_config.data_collection.start_date
    })
    
    # 从database配置中复制
    legacy_config.update({
        'mongodb_uri': project_config.database.mongodb_uri,
        'mongodb_db_name': project_config.database.mongodb_db_name,
        'collection_name_template': project_config.database.collection_name_template
    })
    
    # 从api配置中复制
    legacy_config.update({
        'api_request_delay_seconds': project_config.api.api_request_delay_seconds,
        'progress_report_interval_batches': project_config.api.progress_report_interval_batches
    })
    
    return legacy_config


def migrate_config_file(old_path: str = 'dataset/config.json', 
                       new_path: str = 'config/config.json') -> bool:
    """
    迁移旧配置文件到新格式
    
    Args:
        old_path: 旧配置文件路径
        new_path: 新配置文件路径
        
    Returns:
        是否迁移成功
    """
    try:
        if not os.path.exists(old_path):
            print(f"旧配置文件不存在: {old_path}")
            return False
        
        # 加载旧配置
        with open(old_path, 'r', encoding='utf-8') as f:
            old_config = json.load(f)
        
        # 创建新配置对象
        project_config = ProjectConfig()
        project_config.merge_legacy_config(old_config)
        
        # 保存新配置
        success = save_config(project_config, new_path)
        
        if success:
            print(f"配置文件已成功迁移: {old_path} -> {new_path}")
        
        return success
        
    except Exception as e:
        print(f"配置文件迁移失败: {e}")
        return False


# 兼容性函数 - 保持与原有代码的兼容性
def load_config_legacy(config_path: str = 'dataset/config.json') -> Dict[str, Any]:
    """兼容性函数 - 加载配置并返回旧格式字典"""
    project_config = load_config(config_path)
    return get_legacy_config_dict(project_config) 