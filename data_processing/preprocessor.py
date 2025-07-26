# -*- coding: utf-8 -*-
"""
数据预处理器模块
负责数据的加载、清洗、特征工程和存储
"""

import pandas as pd
import numpy as np
import os
import sys
import math
import pickle
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database.connection import DatabaseConnection
from config.settings import load_config, get_legacy_config_dict, load_project_config
from .features.feature_builder import FeatureBuilder


class DataPreprocessor:
    """数据预处理器 - 统一处理数据获取、特征工程、存储"""
    
    def __init__(self, config_path: str = 'config/config.json'):
        """
        初始化数据预处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.project_config = load_project_config(config_path)
        self.legacy_config = get_legacy_config_dict(self.project_config)
        self.feature_builder = FeatureBuilder()
        self.db_connection = DatabaseConnection(self.legacy_config)
        self.scaler = None
        
        # 设置数据文件夹路径
        self.data_dir = "processed_data"
        self.raw_data_path = os.path.join(self.data_dir, "raw_data.parquet")
        self.featured_data_path = os.path.join(self.data_dir, "featured_data.parquet")
        
        # 确保数据文件夹存在
        os.makedirs(self.data_dir, exist_ok=True)
    
    def process_data_two_stage(self, chunk_size: int = 100000, 
                              max_workers: Optional[int] = None,
                              normalize_features: bool = True) -> str:
        """
        两阶段数据处理：先获取原始数据，再进行特征工程
        
        Args:
            chunk_size: 数据块大小
            max_workers: 最大工作进程数
            normalize_features: 是否对特征进行标准化
            
        Returns:
            最终输出文件路径
        """
        print("开始两阶段数据处理...")
        
        # 阶段1：获取原始数据
        print("\n=== 阶段1：获取原始数据 ===")
        raw_data_file = self._stage1_collect_raw_data(chunk_size, max_workers)
        
        # 阶段2：特征工程
        print("\n=== 阶段2：特征工程 ===")
        featured_data_file = self._stage2_feature_engineering(raw_data_file, normalize_features)
        
        print(f"\n数据处理完成！最终输出文件：{featured_data_file}")
        return featured_data_file
    
    def _stage1_collect_raw_data(self, chunk_size: int, max_workers: Optional[int]) -> str:
        """
        阶段1：收集原始数据并保存
        
        Args:
            chunk_size: 数据块大小
            max_workers: 最大工作进程数
            
        Returns:
            原始数据文件路径
        """
        target_symbol = self.project_config.data_collection.target_symbol
        feature_symbols = self.project_config.data_collection.feature_symbols
        
        # 获取目标数据库集合
        try:
            target_collection = self.db_connection.get_collection_for_symbol(target_symbol)
            total_docs = self.db_connection.get_mongodb_client().count_documents(
                target_collection.name
            )
        except Exception as e:
            print(f"获取数据库信息失败: {e}")
            raise
        
        if total_docs == 0:
            print(f"目标交易对 {target_symbol} 的数据库中无数据，退出。")
            raise ValueError("数据库中无数据")
        
        print(f"总计 {total_docs} 条数据 ({target_symbol})，将以 {chunk_size} 为单位进行分块获取。")
        print(f"使用的特征交易对: {feature_symbols}")
        
        num_chunks = math.ceil(total_docs / chunk_size)
        
        # 准备任务列表 - 简单的分块，不重叠
        tasks = []
        for i in range(num_chunks):
            skip = i * chunk_size
            limit = min(chunk_size, total_docs - skip)
            
            tasks.append({
                'chunk_id': i,
                'skip': skip,
                'limit': limit,
                'config_path': self.config_path,
                'target_symbol': target_symbol,
                'feature_symbols': feature_symbols
            })
        
        # 关闭当前连接避免冲突
        self.db_connection.close_all_connections()
        
        # 并行获取原始数据
        raw_chunks = []
        max_workers = max_workers or min(4, os.cpu_count())
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(collect_raw_data_chunk, tasks)
            
            for i, chunk in enumerate(results):
                if chunk is not None and not chunk.empty:
                    raw_chunks.append(chunk)
                    print(f"✓ 原始数据块 {i} 已获取: {chunk.shape}")
                else:
                    print(f"✗ 原始数据块 {i} 获取失败")
        
        if not raw_chunks:
            print("没有获取到任何原始数据，退出。")
            raise ValueError("原始数据获取失败")
        
        # 合并并保存原始数据
        print("合并原始数据...")
        full_raw_df = pd.concat(raw_chunks, ignore_index=True)
        
        # 按时间排序并去重
        if 'open_time' in full_raw_df.columns:
            full_raw_df = full_raw_df.sort_values('open_time').drop_duplicates(subset=['open_time'], keep='last')
            full_raw_df = full_raw_df.reset_index(drop=True)
        
        # 保存原始数据
        full_raw_df.to_parquet(self.raw_data_path, index=False)
        print(f"原始数据已保存到: {self.raw_data_path}")
        print(f"原始数据形状: {full_raw_df.shape}")
        
        return self.raw_data_path
    
    def _stage2_feature_engineering(self, raw_data_file: str, normalize_features: bool) -> str:
        """
        阶段2：对完整的原始数据进行特征工程
        
        Args:
            raw_data_file: 原始数据文件路径
            normalize_features: 是否进行特征标准化
            
        Returns:
            特征工程后的数据文件路径
        """
        print("加载原始数据...")
        raw_df = pd.read_parquet(raw_data_file)
        print(f"原始数据形状: {raw_df.shape}")
        
        target_symbol = self.project_config.data_collection.target_symbol
        feature_symbols = self.project_config.data_collection.feature_symbols
        
        # 构建特征 - 现在可以使用完整的数据进行特征工程
        print("开始特征工程...")
        featured_df = self.feature_builder.build_features(
            raw_df, target_symbol, feature_symbols, 
            is_first_chunk=True  # 整个数据集作为一个完整块处理
        )
        
        # 清理NaN值
        print("清理NaN值...")
        initial_shape = featured_df.shape
        featured_df = featured_df.dropna()
        final_shape = featured_df.shape
        print(f"清理前: {initial_shape}, 清理后: {final_shape}")
        
        # 添加特征标准化
        if normalize_features:
            print("正在进行特征标准化...")
            featured_df = self._normalize_features(featured_df)
        
        # 保存特征工程后的数据
        featured_df.to_parquet(self.featured_data_path, index=False)
        print(f"特征数据已保存到: {self.featured_data_path}")
        print(f"最终数据形状: {featured_df.shape}")
        
        return self.featured_data_path
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对特征进行标准化
        
        Args:
            df: 包含特征的DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        # 分离特征和目标变量
        feature_cols = [col for col in df.columns if col not in ['target', 'future_return', 'open_time']]
        target_cols = [col for col in df.columns if col in ['target', 'future_return']]
        other_cols = [col for col in df.columns if col not in feature_cols + target_cols]
        
        if not feature_cols:
            print("⚠️  警告: 没有找到需要标准化的特征列")
            return df
        
        print(f"   标准化 {len(feature_cols)} 个特征列...")
        
        # 初始化标准化器
        if self.scaler is None:
            self.scaler = StandardScaler()
            
            # 拟合并转换特征
            scaled_features = self.scaler.fit_transform(df[feature_cols])
            
            # 保存标准化器
            scaler_path = os.path.join(self.data_dir, "scaler.pkl")
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"   标准化器已保存到: {scaler_path}")
            
        else:
            # 使用已存在的标准化器进行转换
            scaled_features = self.scaler.transform(df[feature_cols])
            print(f"   使用已存在的标准化器进行转换")
        
        # 重构DataFrame
        result_df = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)
        
        # 添加目标变量和其他列
        for col in target_cols + other_cols:
            if col in df.columns:
                result_df[col] = df[col]
        
        # 确保列的顺序一致
        final_cols = feature_cols + target_cols + other_cols
        final_cols = [col for col in final_cols if col in result_df.columns]
        result_df = result_df[final_cols]
        
        print(f"   标准化完成，特征统计:")
        print(f"     特征均值: {result_df[feature_cols].mean().mean():.6f}")
        print(f"     特征标准差: {result_df[feature_cols].std().mean():.6f}")
        
        return result_df

    def load_scaler(self) -> StandardScaler:
        """
        加载已保存的标准化器
        
        Returns:
            标准化器实例
        """
        scaler_path = os.path.join(self.data_dir, "scaler.pkl")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"已加载标准化器: {scaler_path}")
        return self.scaler
    
    def get_featured_data_path(self) -> str:
        """获取特征数据文件路径"""
        return self.featured_data_path
    
    def get_raw_data_path(self) -> str:
        """获取原始数据文件路径"""
        return self.raw_data_path

    def process_incremental_data(self, last_processed_time: Optional[int] = None) -> str:
        """
        增量处理新数据
        
        Args:
            last_processed_time: 上次处理的最后时间戳
            
        Returns:
            输出文件路径
        """
        print("开始增量数据处理...")
        
        target_symbol = self.project_config.data_collection.target_symbol
        feature_symbols = self.project_config.data_collection.feature_symbols
        
        # 如果没有指定时间，尝试从现有文件中获取
        if last_processed_time is None:
            if os.path.exists(self.featured_data_path):
                try:
                    existing_df = pd.read_parquet(self.featured_data_path)
                    if 'open_time' in existing_df.columns:
                        last_processed_time = existing_df['open_time'].max()
                        print(f"从现有文件获取最后处理时间: {last_processed_time}")
                except Exception as e:
                    print(f"读取现有文件失败: {e}")
        
        # 获取新数据
        try:
            new_data = self._load_incremental_data(
                target_symbol, feature_symbols, last_processed_time
            )
            
            if new_data.empty:
                print("没有新数据需要处理")
                return self.featured_data_path
            
            # 构建特征
            featured_df = self.feature_builder.build_features(
                new_data, target_symbol, feature_symbols
            )
            
            # 合并到现有数据或创建新文件
            output_path = self._merge_or_create_data(featured_df)
            
            print(f"增量数据处理完成: {featured_df.shape}")
            return output_path
            
        except Exception as e:
            print(f"增量数据处理失败: {e}")
            raise
    
    def _load_incremental_data(self, target_symbol: str, feature_symbols: List[str],
                              last_processed_time: Optional[int]) -> pd.DataFrame:
        """加载增量数据"""
        print(f"加载增量数据，起始时间戳: {last_processed_time}")
        
        try:
            # 获取数据库中所有符合条件的新数据
            all_symbols = list(set([target_symbol] + feature_symbols))
            combined_data = {}
            
            for symbol in all_symbols:
                collection = self.db_connection.get_collection_for_symbol(symbol)
                
                # 构建时间过滤器
                time_filter = {}
                if last_processed_time is not None:
                    time_filter = {'open_time': {'$gt': last_processed_time}}
                
                # 查询新数据
                cursor = collection.find(time_filter).sort('open_time', 1)
                symbol_data = []
                
                for doc in cursor:
                    doc.pop('_id', None)  # 移除MongoDB的_id字段
                    symbol_data.append(doc)
                
                if symbol_data:
                    symbol_df = pd.DataFrame(symbol_data)
                    
                    # 确保时间戳为整数
                    symbol_df['open_time'] = pd.to_numeric(symbol_df['open_time'], errors='coerce').astype('Int64')
                    
                    # 设置时间索引
                    symbol_df.set_index('open_time', inplace=True)
                    
                    # 添加符号前缀
                    symbol_df = self._rename_columns_with_symbol_prefix(symbol_df, symbol)
                    
                    combined_data[symbol] = symbol_df
                    print(f"  {symbol}: {len(symbol_df)} 条新数据")
            
            if not combined_data:
                print("没有找到新数据")
                return pd.DataFrame()
            
            # 合并所有交易对的数据
            combined_df = self._merge_symbol_dataframes(combined_data)
            
            # 重置索引，将open_time恢复为列
            combined_df = combined_df.reset_index()
            
            print(f"增量数据加载完成: {combined_df.shape}")
            return combined_df
            
        except Exception as e:
            print(f"增量数据加载失败: {e}")
            raise
    
    def _merge_or_create_data(self, new_data: pd.DataFrame) -> str:
        """合并新数据到现有文件或创建新文件"""
        if os.path.exists(self.featured_data_path):
            try:
                existing_df = pd.read_parquet(self.featured_data_path)
                combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                
                # 去重（基于时间）
                if 'open_time' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['open_time'], keep='last')
                    combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
                
                combined_df.to_parquet(self.featured_data_path, index=False)
                print(f"数据已合并到现有文件: {self.featured_data_path}")
                
            except Exception as e:
                print(f"合并数据失败，将创建新文件: {e}")
                new_data.to_parquet(self.featured_data_path, index=False)
        else:
            new_data.to_parquet(self.featured_data_path, index=False)
            print(f"创建新数据文件: {self.featured_data_path}")
        
        return self.featured_data_path
    
    def validate_output_data(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        验证输出数据的质量
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            验证结果
        """
        if file_path is None:
            file_path = self.featured_data_path
        
        if not os.path.exists(file_path):
            return {'status': 'error', 'message': '数据文件不存在'}
        
        try:
            df = pd.read_parquet(file_path)
            
            validation_result = {
                'status': 'success',
                'file_path': file_path,
                'data_shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                'columns': list(df.columns),
                'target_distribution': {},
                'missing_values': df.isnull().sum().sum(),
                'data_types': df.dtypes.value_counts().to_dict()
            }
            
            # 目标变量分布
            if 'target' in df.columns:
                validation_result['target_distribution'] = df['target'].value_counts().to_dict()
            
            # 时间范围
            if 'open_time' in df.columns:
                validation_result['time_range'] = {
                    'start': df['open_time'].min(),
                    'end': df['open_time'].max(),
                    'duration_days': (df['open_time'].max() - df['open_time'].min()) / (24 * 60 * 60 * 1000)
                }
            
            # 特征统计信息
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                validation_result['feature_stats'] = {
                    'mean': df[numeric_columns].mean().to_dict(),
                    'std': df[numeric_columns].std().to_dict(),
                    'min': df[numeric_columns].min().to_dict(),
                    'max': df[numeric_columns].max().to_dict()
                }
            
            print(f"数据验证完成:")
            print(f"- 数据形状: {validation_result['data_shape']}")
            print(f"- 内存使用: {validation_result['memory_usage']:.2f} MB")
            print(f"- 缺失值: {validation_result['missing_values']}")
            
            return validation_result
            
        except Exception as e:
            return {'status': 'error', 'message': f'验证失败: {e}'}
    
    def _rename_columns_with_symbol_prefix(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        为DataFrame的列名添加交易对前缀
        
        Args:
            df: 原始DataFrame
            symbol: 交易对符号
            
        Returns:
            重命名后的DataFrame
        """
        # 需要重命名的列（不包括时间索引）
        columns_to_rename = [
            'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
        # 创建重命名映射
        rename_mapping = {}
        for col in columns_to_rename:
            if col in df.columns:
                rename_mapping[col] = f"{symbol}_{col}"
        
        # 应用重命名
        df_renamed = df.rename(columns=rename_mapping)
        
        return df_renamed
    
    def _merge_symbol_dataframes(self, symbol_dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        合并多个交易对的DataFrame
        
        Args:
            symbol_dataframes: 交易对DataFrame字典
            
        Returns:
            合并后的DataFrame
        """
        if not symbol_dataframes:
            return pd.DataFrame()
        
        # 获取所有DataFrame的列表
        dfs = list(symbol_dataframes.values())
        
        # 使用外连接合并所有DataFrame
        combined_df = dfs[0]
        
        for df in dfs[1:]:
            combined_df = pd.merge(
                combined_df, 
                df, 
                left_index=True, 
                right_index=True, 
                how='outer'
            )
        
        # 按时间排序
        combined_df = combined_df.sort_index()
        
        # 向前填充缺失值（对于价格数据）
        price_columns = [col for col in combined_df.columns if any(
            price_type in col for price_type in ['open', 'high', 'low', 'close']
        )]
        
        for col in price_columns:
            combined_df[col] = combined_df[col].fillna(method='ffill')
        
        # 对于成交量相关字段，填充为0
        volume_columns = [col for col in combined_df.columns if any(
            vol_type in col for vol_type in ['volume', 'trades', 'taker_buy']
        )]
        
        for col in volume_columns:
            combined_df[col] = combined_df[col].fillna(0)
        
        return combined_df

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'db_connection'):
            self.db_connection.close_all_connections()


def collect_raw_data_chunk(task_params: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    收集原始数据块（用于多进程）
    
    Args:
        task_params: 任务参数字典
        
    Returns:
        原始数据DataFrame或None
    """
    try:
        # 重新导入必要的模块（多进程环境）
        import sys
        import os
        import pandas as pd
        
        # 添加项目根目录到路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from database.connection import DatabaseConnection
        from config.settings import load_project_config, get_legacy_config_dict
        
        # 从配置路径重新加载配置
        config_path = task_params['config_path']
        project_config = load_project_config(config_path)
        legacy_config = get_legacy_config_dict(project_config)
        
        # 在子进程中创建新的数据库连接
        db_connection = DatabaseConnection(legacy_config)
        
        try:
            chunk_id = task_params['chunk_id']
            skip = task_params['skip']
            limit = task_params['limit']
            target_symbol = task_params['target_symbol']
            feature_symbols = task_params['feature_symbols']
            
            # 加载多资产原始数据
            raw_df = _load_multi_asset_raw_data(
                db_connection, target_symbol, feature_symbols, skip, limit
            )
            
            if raw_df.empty:
                print(f"原始数据块 {chunk_id} 为空，跳过")
                return pd.DataFrame()
            
            # 只在必要时输出完成信息
            if chunk_id % 10 == 0:
                print(f"原始数据块 {chunk_id} 收集完成: {raw_df.shape}")
            
            return raw_df
            
        finally:
            db_connection.close_all_connections()
        
    except Exception as e:
        print(f"收集原始数据块失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_multi_asset_raw_data(db_connection, target_symbol: str, feature_symbols: List[str],
                               skip: int = 0, limit: Optional[int] = None) -> pd.DataFrame:
    """
    加载多资产原始数据（不进行特征工程）
    """
    all_data = {}
    
    # 获取目标数据的时间范围
    target_collection = db_connection.get_collection_for_symbol(target_symbol)
    
    # 构建查询条件
    if skip > 0 or limit is not None:
        target_docs = list(target_collection.find().sort("open_time", 1).skip(skip).limit(limit or 0))
    else:
        target_docs = list(target_collection.find().sort("open_time", 1))
    
    if not target_docs:
        return pd.DataFrame()
    
    # 获取时间范围
    start_time = target_docs[0]['open_time']
    end_time = target_docs[-1]['open_time']
    
    # 为每个交易对加载数据
    for symbol in feature_symbols:
        try:
            collection = db_connection.get_collection_for_symbol(symbol)
            
            # 查询该时间范围内的数据
            docs = list(collection.find({
                'open_time': {'$gte': start_time, '$lte': end_time}
            }).sort("open_time", 1))
            
            if docs:
                symbol_df = pd.DataFrame(docs)
                
                # 确保数据类型正确
                numeric_fields = ['open', 'high', 'low', 'close', 'volume', 
                                'quote_asset_volume', 'number_of_trades',
                                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
                
                for field in numeric_fields:
                    if field in symbol_df.columns:
                        symbol_df[field] = pd.to_numeric(symbol_df[field], errors='coerce')
                
                # 确保整数字段的类型
                integer_fields = ['close_time', 'number_of_trades']
                for field in integer_fields:
                    if field in symbol_df.columns:
                        symbol_df[field] = pd.to_numeric(symbol_df[field], errors='coerce').fillna(0).astype(int)
                
                symbol_df.set_index('open_time', inplace=True)
                
                # 重命名列以包含交易对前缀
                rename_cols = {}
                columns_to_rename = [
                    'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
                ]
                
                for col in columns_to_rename:
                    if col in symbol_df.columns:
                        rename_cols[col] = f'{symbol}_{col}'
                
                symbol_df = symbol_df.rename(columns=rename_cols)
                all_data[symbol] = symbol_df[list(rename_cols.values())]
            
        except Exception as e:
            print(f"加载 {symbol} 原始数据失败: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    # 合并所有数据
    combined_df = pd.concat(all_data.values(), axis=1, join='outer')
    combined_df = combined_df.dropna()  # 只保留所有交易对都有数据的时间点
    
    # 重置索引，将时间作为列
    combined_df.reset_index(inplace=True)
    
    return combined_df 