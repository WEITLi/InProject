#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态异常检测数据处理流水线
整合现有的CERT数据集处理能力和多模态模型架构
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import warnings
from joblib import Parallel, delayed
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd
import logging # 确保导入

warnings.filterwarnings('ignore')

# 导入现有模块
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # 尝试相对导入
    from .dataset_pipeline import CERTDatasetPipeline
    from .encoder import EventEncoder
    from .config import Config, ModelConfig, TrainingConfig, DataConfig
except ImportError:
    # 如果相对导入失败，添加当前目录到路径并使用绝对导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from dataset_pipeline import CERTDatasetPipeline
    from encoder import EventEncoder
    from config import Config, ModelConfig, TrainingConfig, DataConfig

# 注意：MultiModalAnomalyDetector在train_pipeline中定义，这里不需要导入

class MultiModalDataPipeline:
    """
    多模态异常检测数据处理流水线
    
    整合功能：
    1. 现有的CERT数据集特征提取
    2. 多模态数据准备和编码
    3. 用户关系图构建
    4. 文本内容提取和预处理
    5. 结构化特征工程
    6. 训练数据生成
    """
    
    def __init__(self, 
                 config: Config = None,
                 data_version: str = 'r4.2',
                 feature_dim: int = 256,
                 num_cores: int = 8,
                 source_dir_override: Optional[str] = None,
                 work_dir_override: Optional[str] = None):
        """
        初始化多模态数据流水线
        
        Args:
            config: 完整配置对象
            data_version: 数据集版本 (会被 config.data.data_version 覆盖)
            feature_dim: 特征向量维度 (会被 config.data.feature_dim 覆盖)
            num_cores: CPU核心数 (会被 config.data.num_cores 覆盖)
            source_dir_override: 源数据目录覆盖 (会被 config.data.source_dir 覆盖)
            work_dir_override: 工作目录覆盖 (会被 CERTDatasetPipeline 的 work_dir_override 使用)
        """
        self.config = config or Config()
        
        # Values from MultiModalDataPipeline's own constructor arguments are defaults
        mmp_constructor_data_version = data_version
        mmp_constructor_feature_dim = feature_dim
        mmp_constructor_num_cores = num_cores
        mmp_constructor_source_dir_override = source_dir_override
        mmp_constructor_work_dir_override = work_dir_override

        # Determine effective values for CERTDatasetPipeline initialization
        # by prioritizing values from the config object if they are set.
        
        effective_data_version_for_base = mmp_constructor_data_version
        if self.config and hasattr(self.config.data, 'data_version') and self.config.data.data_version is not None:
            effective_data_version_for_base = self.config.data.data_version
        
        effective_feature_dim_for_base = mmp_constructor_feature_dim
        if self.config and hasattr(self.config.data, 'feature_dim') and self.config.data.feature_dim is not None:
            effective_feature_dim_for_base = self.config.data.feature_dim

        effective_num_cores_for_base = mmp_constructor_num_cores
        if self.config and hasattr(self.config.data, 'num_cores') and self.config.data.num_cores is not None:
            effective_num_cores_for_base = self.config.data.num_cores

        effective_source_dir_override_for_base = mmp_constructor_source_dir_override
        if self.config and hasattr(self.config.data, 'source_dir') and self.config.data.source_dir is not None:
            effective_source_dir_override_for_base = self.config.data.source_dir
        
        # For CERTDatasetPipeline's work_dir_override, use the one passed to MMP's constructor.
        effective_work_dir_override_for_cert = mmp_constructor_work_dir_override

        # Get seed for CERTDatasetPipeline from the main Config object
        effective_seed_for_base = getattr(self.config, 'seed', 42) # Default to 42 if not in config
        
        # 初始化基础数据流水线 (CERTDatasetPipeline)
        self.base_pipeline = CERTDatasetPipeline(
            data_version=effective_data_version_for_base,
            feature_dim=effective_feature_dim_for_base,
            num_cores=effective_num_cores_for_base,
            source_dir_override=effective_source_dir_override_for_base,
            work_dir_override=effective_work_dir_override_for_cert,
            seed=effective_seed_for_base # 传递种子
        )
        
        # Set MultiModalDataPipeline's own attributes to reflect effective values.
        self.data_version = effective_data_version_for_base
        self.feature_dim = effective_feature_dim_for_base
        self.num_cores = effective_num_cores_for_base # Corrected assignment
        
        # MMP's own work_dir and source_data_dir
        self.work_dir = self.base_pipeline.work_dir 
        self.source_data_dir = self.base_pipeline.source_data_dir
        
        # 多模态数据目录 (remains based on self.work_dir which is now CERT's work_dir)
        self.multimodal_dir = os.path.join(self.work_dir, "MultiModalData")
        self._create_multimodal_directories()
        
        # 数据缓存
        self._data_cache = {}
        self._user_graph_cache = None
        self._text_data_cache = None
        
        print(f"初始化多模态数据流水线")
        print(f"  数据版本 (effective for base): {self.data_version}") 
        print(f"  特征维度 (effective for base): {self.feature_dim}") 
        print(f"  多模态数据目录: {os.path.abspath(self.multimodal_dir)}")
    
    def _create_multimodal_directories(self):
        """创建多模态数据目录结构"""
        directories = [
            "MultiModalData",
            "MultiModalData/UserGraphs",
            "MultiModalData/TextData", 
            "MultiModalData/StructuredFeatures",
            "MultiModalData/BehaviorSequences",
            "MultiModalData/TrainingData",
            "MultiModalData/Models"
        ]
        for directory in directories:
            work_path = os.path.join(self.work_dir, directory)
            os.makedirs(work_path, exist_ok=True)
    
    def run_base_feature_extraction(self, 
                                  start_week: int = 0, 
                                  end_week: int = None,
                                  max_users: int = None,
                                  sample_ratio: float = None):
        """
        运行基础特征提取流水线
        
        Args:
            start_week: 开始周数
            end_week: 结束周数  
            max_users: 最大用户数
            sample_ratio: 数据采样比例，用于快速测试
        """
        print(f"\n{'='*60}")
        print(f"Step 1: 运行基础特征提取流水线")
        print(f"{'='*60}")

        force_regen_combined_weeks = getattr(self.config.data, 'force_regenerate_combined_weeks', False)
        force_regen_analysis_levels = getattr(self.config.data, 'force_regenerate_analysis_levels', False)
        print(f"   基础流水线是否强制重新合并周数据: {force_regen_combined_weeks}")
        print(f"   基础流水线是否强制重新生成分析级别CSV: {force_regen_analysis_levels}")
        
        # 运行完整的基础流水线
        self.base_pipeline.run_full_pipeline(
            start_week=start_week,
            end_week=end_week,
            max_users=max_users,
            modes=['week', 'day', 'session'],
            sample_ratio=sample_ratio,
            force_regenerate_combined_weeks=force_regen_combined_weeks,
            force_regenerate_analysis_levels=force_regen_analysis_levels
        )
        
        print("✅ 基础特征提取完成")
    
    def _apply_max_users_filter(self, users_df: pd.DataFrame, max_users: Optional[int]) -> pd.DataFrame:
        """
        辅助函数：对用户DataFrame应用max_users限制。
        优先保留恶意用户。
        """
        if max_users and len(users_df) > max_users:
            print(f"   统一应用用户数量限制: {len(users_df)} -> {max_users}")
            if 'malscene' in users_df.columns:
                malicious_users_df = users_df[users_df['malscene'] > 0]
                normal_users_df = users_df[users_df['malscene'] == 0]

                if len(malicious_users_df) >= max_users:
                    final_users_df = malicious_users_df.sample(n=max_users, random_state=self.config.seed) # 使用 self.config.seed
                else:
                    remaining_slots = max_users - len(malicious_users_df)
                    if remaining_slots > 0 and not normal_users_df.empty:
                        selected_normal_df = normal_users_df.sample(
                            n=min(remaining_slots, len(normal_users_df)), random_state=self.config.seed # 使用 self.config.seed
                        )
                        final_users_df = pd.concat([malicious_users_df, selected_normal_df])
                    else:
                        final_users_df = malicious_users_df
                
                if final_users_df.empty and not users_df.empty:
                     print("    ⚠️ 优先选择恶意用户后为空，但原始用户列表不为空。回退到随机采样。")
                     final_users_df = users_df.sample(n=min(max_users, len(users_df)), random_state=self.config.seed) # 使用 self.config.seed

            else: # 没有恶意场景信息，随机采样
                print("    ⚠️ 'malscene' 列不存在于users_df，执行随机用户采样。")
                final_users_df = users_df.sample(n=min(max_users, len(users_df)), random_state=self.config.seed) # 使用 self.config.seed
            
            print(f"   最终筛选用户数: {len(final_users_df)}")
            return final_users_df
        return users_df

    def prepare_training_data(self, 
                            start_week: int = 0,
                            end_week: int = None,
                            max_users: int = None, # 这个max_users将用于统一筛选
                            sequence_length: int = 128) -> Dict[str, Any]:
        """
        准备多模态训练数据
        
        Args:
            start_week: 开始周数
            end_week: 结束周数
            max_users: 最大用户数限制 (将在这里统一应用)
            sequence_length: 序列长度
            
        Returns:
            训练数据字典
        """
        logger = logging.getLogger(__name__) # 获取 logger 实例
        logger.info(f"[MultiModalDataPipeline.prepare_training_data] 接收到的参数: start_week={start_week}, end_week={end_week}, max_users={max_users}, sequence_length={sequence_length}")

        if end_week is None:
            end_week = self.base_pipeline.max_weeks
            
        print(f"\n{'='*60}")
        print(f"Step 3: 准备多模态训练数据 (统一用户筛选)")
        print(f"{'='*60}")
        
        # --- 关键改动：在这里统一加载和筛选用户 ---
        print("   加载初始用户数据...")
        initial_users_df = self.base_pipeline.step2_load_user_data() # 加载所有可能的用户
        print(f"   初始用户数: {len(initial_users_df)}")
        
        logger.info(f"[MultiModalDataPipeline.prepare_training_data] 调用 _apply_max_users_filter 之前, 传入的 max_users={max_users}")
        final_selected_users_df = self._apply_max_users_filter(initial_users_df, max_users)
        logger.info(f"[MultiModalDataPipeline.prepare_training_data] 调用 _apply_max_users_filter 之后, final_selected_users_df 长度: {len(final_selected_users_df)}")
        
        final_user_list = final_selected_users_df.index.tolist() # 假设索引是user_id
        logger.info(f"[MultiModalDataPipeline.prepare_training_data] final_user_list 长度: {len(final_user_list)}")
        
        if not final_user_list:
            print("⚠️ 经过筛选后没有用户可用于训练，将返回空数据。")
            return { # 返回一个符合后续期望结构但为空的字典
                'behavior_sequences': np.array([]),
                'node_features': np.array([]),
                'adjacency_matrix': np.array([]),
                'text_content': [],
                'structured_features': np.array([]),
                'labels': np.array([]),
                'users': [],
                'user_to_index': {}
            }

        training_data_file = os.path.join(self.multimodal_dir, "TrainingData",
                                        f"training_data_w{start_week}_{end_week}_u{len(final_user_list)}.pickle")
        logger.info(f"[MultiModalDataPipeline.prepare_training_data] 构造的 training_data_file 路径: {training_data_file}")
        
        if os.path.exists(training_data_file) and not self.config.data.force_regenerate_training_data:
            print(f"   训练数据已存在且未强制重新生成，直接加载: {training_data_file}")
            try:
                with open(training_data_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"   加载缓存训练数据失败: {e}。将重新生成。")

        # 加载各模态数据，确保它们基于或兼容 final_selected_users_df
        # _load_user_graph_data 等方法现在需要知道最终用户列表或DataFrame
        user_graph_data = self._load_user_graph_data(start_week, end_week, final_selected_users_df)
        text_data = self._load_text_data(start_week, end_week) # 文本数据通常较难按用户精确预筛选
        behavior_sequences = self._load_behavior_sequences(start_week, end_week, final_user_list) # 传递用户列表
        structured_features = self._load_structured_features(start_week, end_week, final_user_list) # 传递用户列表
        
        # labels_data 现在应该基于 final_selected_users_df 生成
        labels_data = self._load_labels_data_from_df(final_selected_users_df) 
        
        # 整合训练数据
        training_data = self._integrate_multimodal_data(
            user_graph_data, text_data, behavior_sequences, 
            structured_features, labels_data, sequence_length,
            final_user_list # 传递权威的用户列表给整合函数
        )

        # --- START: Add Normalization ---
        # Normalize behavior_sequences
        # Reshape to 2D for scaler: (num_samples * sequence_length, feature_dim)
        # Then reshape back to 3D
        if 'behavior_sequences' in training_data and training_data['behavior_sequences'].size > 0: # Check if key exists and not empty
            bs_shape = training_data['behavior_sequences'].shape
            # Only scale if there's more than one sample to avoid issues with single sample std dev
            if bs_shape[0] > 1: 
                behavior_sequences_2d = training_data['behavior_sequences'].reshape(-1, bs_shape[-1])
                scaler_bs = StandardScaler()
                normalized_bs_2d = scaler_bs.fit_transform(behavior_sequences_2d)
                training_data['behavior_sequences'] = normalized_bs_2d.reshape(bs_shape)
                print("   行为序列数据已进行标准化处理。")
            else:
                print("   行为序列数据样本过少(<2)，跳过标准化。")
        else:
            print("   未找到或行为序列数据为空，跳过标准化。")

        # Normalize structured_features
        # This is already 2D: (num_samples, feature_dim)
        if 'structured_features' in training_data and training_data['structured_features'].size > 0:
            # Only scale if there's more than one sample
            if training_data['structured_features'].shape[0] > 1:
                scaler_sf = StandardScaler()
                training_data['structured_features'] = scaler_sf.fit_transform(training_data['structured_features'])
                print("   结构化特征数据已进行标准化处理。")
            else:
                print("   结构化特征数据样本过少(<2)，跳过标准化。")
        else:
            print("   未找到或结构化特征数据为空，跳过标准化。")
        # --- END: Add Normalization ---
        
        # 保存训练数据
        with open(training_data_file, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"   训练数据保存到: {training_data_file}")
        print(f"   数据样本数: {len(training_data.get('labels', []))}")
        
        return training_data
    
    def _load_user_graph_data(self, start_week: int, end_week: int, users_df_for_graph: pd.DataFrame) -> Dict[str, Any]:
        """
        加载或构建用户图数据，基于指定的users_df_for_graph。
        Args:
            users_df_for_graph: 经过筛选的、用于构建图的用户DataFrame。
        """
        # 文件名可以包含用户数量信息以区分不同筛选下的图
        graph_file_suffix = f"_u{len(users_df_for_graph)}.pickle" if users_df_for_graph is not None else ".pickle"
        graph_file = os.path.join(self.multimodal_dir, "UserGraphs",
                                f"user_graph_w{start_week}_{end_week}{graph_file_suffix}")

        if os.path.exists(graph_file) and not self.config.data.force_regenerate_graphs:
            print(f"   用户关系图已存在，直接加载: {graph_file}")
            with open(graph_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"   构建用户关系图 (基于 {len(users_df_for_graph)} 个已筛选用户)...")
        # _construct_user_relationships 需要确保使用这个传入的 users_df_for_graph
        # 并且不再进行内部的 max_users 筛选
        user_graph_data = self._construct_user_relationships(
            users_df_for_graph, start_week, end_week 
        )
        
        with open(graph_file, 'wb') as f:
            pickle.dump(user_graph_data, f)
        print(f"   用户关系图保存到: {graph_file}")
        return user_graph_data
    
    def _load_text_data(self, start_week: int, end_week: int) -> Dict[str, Any]:
        """加载文本数据 (此方法暂不按用户筛选，后续可在整合时处理)"""
        # ... (保留原样，文本的精细筛选通常在整合或模型层面处理)
        text_file = os.path.join(self.multimodal_dir, "TextData",
                               f"text_data_w{start_week}_{end_week}.pickle")
        if os.path.exists(text_file) and not self.config.data.force_regenerate_text_data:
            print(f"   文本数据已存在，直接加载: {text_file}")
            with open(text_file, 'rb') as f:
                return pickle.load(f)

        # 如果不存在或需要重新生成 (调用原有的_extract_text_data)
        print(f"   提取原始文本数据...") # 修改日志消息
        self._extract_text_data(start_week, end_week) # 这会保存文件
        
        # 再次尝试加载
        if os.path.exists(text_file):
            with open(text_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"   ⚠️ 未能加载或生成文本数据文件: {text_file}")
            return {} # 返回空字典

    def _extract_text_data(self, start_week: int, end_week: int):
        """从基础流水线处理的 Parquet 数据中提取文本内容并保存。"""
        print(f"   深入提取文本数据 (周 {start_week}-{end_week-1})...")
        all_user_texts = {}

        # 使用 self.base_pipeline.current_parquet_dir_name 来获取正确的 Parquet 目录
        base_parquet_dir = self.base_pipeline._get_work_file_path(self.base_pipeline.current_parquet_dir_name)

        if not os.path.exists(base_parquet_dir):
            print(f"    ⚠️ 基础 Parquet 目录 {base_parquet_dir} 不存在，无法提取文本。")
            return

        for week in range(start_week, end_week):
            print(f"    处理周 {week} 的文本...")
            try:
                week_data_ddf = dd.read_parquet(
                    base_parquet_dir,
                    filters=[('week', '==', week)],
                    columns=['user', 'type', 'activity', 'url', 'filename', 'content'], # 根据实际可能包含文本的列
                    engine='pyarrow'
                )

                if week_data_ddf.npartitions == 0 or week_data_ddf.map_partitions(len).compute().sum() == 0:
                    print(f"     周 {week}: 无数据，跳过文本提取。")
                    continue
                
                week_data_df = week_data_ddf.compute()

                for _, row in week_data_df.iterrows():
                    user = row.get('user')
                    event_type = row.get('type')
                    texts_to_add = []

                    if pd.isna(user):
                        continue

                    if user not in all_user_texts:
                        all_user_texts[user] = {
                            'email_texts': [],
                            'file_texts': [],
                            'http_texts': [],
                            'other_texts': []
                        }
                    
                    content = str(row.get('content', '')) if pd.notna(row.get('content')) else ''
                    activity = str(row.get('activity', '')) if pd.notna(row.get('activity')) else ''
                    url = str(row.get('url', '')) if pd.notna(row.get('url')) else ''
                    filename = str(row.get('filename', '')) if pd.notna(row.get('filename')) else ''

                    if event_type == 'email':
                        if content: texts_to_add.append(content)
                        if activity: texts_to_add.append(activity) # 例如邮件主题
                        all_user_texts[user]['email_texts'].extend(texts_to_add)
                    elif event_type == 'file':
                        if filename: texts_to_add.append(filename)
                        if content: texts_to_add.append(content) # 例如文件内容片段
                        all_user_texts[user]['file_texts'].extend(texts_to_add)
                    elif event_type == 'http':
                        if url: texts_to_add.append(url)
                        if content: texts_to_add.append(content) # 例如网页标题或片段
                        all_user_texts[user]['http_texts'].extend(texts_to_add)
                    else: # 其他事件类型，如logon, device
                        if activity: texts_to_add.append(activity)
                        if content: texts_to_add.append(content)
                        all_user_texts[user]['other_texts'].extend(texts_to_add)
            
            except FileNotFoundError:
                print(f"     周 {week}: Parquet数据分区未找到，跳过文本提取。")
            except Exception as e:
                print(f"     周 {week}: 提取文本时发生错误: {e}")
        
        # 组织成期望的格式：{week_num: {user_id: {text_type: [texts]}}}
        # 当前 all_user_texts 是 {user_id: {text_type: [texts]}}
        # 为了与 _load_text_data 的原始期望（每个 week 一个 key）对齐，这里可以简单地将所有周的文本聚合
        # 或者，如果严格要求按周分开，则需要在循环内部构建更复杂的结构并合并。
        # 简化处理：所有周的文本聚合到一个虚拟的 "all_weeks" 键下，或者直接返回 all_user_texts
        # _load_text_data 的读取逻辑也需要相应调整。 
        # 现在的 _integrate_multimodal_data 会遍历 text_data.values(), 所以直接返回 all_user_texts 结构是OK的。

        output_text_data = {"all_collected_weeks": all_user_texts} # 保持 text_data[week_key] 的结构

        text_file_path = os.path.join(self.multimodal_dir, "TextData", f"text_data_w{start_week}_{end_week}.pickle")
        try:
            with open(text_file_path, 'wb') as f:
                pickle.dump(output_text_data, f)
            print(f"   ✅ 提取的文本数据已保存到: {text_file_path}")
        except Exception as e:
            print(f"   ❌ 保存提取的文本数据失败: {e}")

    def _load_behavior_sequences(self, start_week: int, end_week: int, final_user_list: List[str]) -> Dict[str, Any]:
        """
        加载或构建行为序列数据, 基于 final_user_list。
        Args:
            final_user_list: 权威的用户ID列表。
        """
        # 文件名可以包含用户数量信息
        sequence_file_suffix = f"_u{len(final_user_list)}.pickle"
        sequence_file = os.path.join(self.multimodal_dir, "BehaviorSequences",
                                   f"behavior_sequences_w{start_week}_{end_week}{sequence_file_suffix}")

        loaded_data = None
        if os.path.exists(sequence_file) and not self.config.data.force_regenerate_sequences:
            print(f"   行为序列已存在，直接加载: {sequence_file}")
            try:
                with open(sequence_file, 'rb') as f:
                    loaded_data = pickle.load(f)
            except Exception as e:
                print(f"   ⚠️ 加载缓存的行为序列失败: {e}. 将尝试重新构建。")
                loaded_data = None # 确保在加载失败时为None，以便后续重新构建
            
            if loaded_data is not None: # 确保不是None才返回
                # 如果加载的数据不是字典 (例如，意外保存为None或其他类型)，则返回空字典以避免下游错误
                return loaded_data if isinstance(loaded_data, dict) else {}
        
        # 如果 loaded_data is None (因为文件不存在，或强制重新生成，或加载失败/内容无效)
        # 则继续向下执行以构建新的

        print(f"   构建行为序列 (基于 {len(final_user_list)} 个已筛选用户)...")
        # _build_behavior_sequences_for_users 应该总是返回dict
        new_behavior_sequences = self._build_behavior_sequences_for_users(
            start_week, end_week, final_user_list
        )
        
        # 双重保险：确保 new_behavior_sequences 是一个字典
        if new_behavior_sequences is None:
            print("   ⚠️ _build_behavior_sequences_for_users 返回了 None，将使用空字典代替。")
            new_behavior_sequences = {}

        try:
            with open(sequence_file, 'wb') as f:
                pickle.dump(new_behavior_sequences, f)
            print(f"   行为序列保存到: {sequence_file}")
        except Exception as e:
            print(f"   ⚠️ 保存行为序列到pickle文件失败: {e}")
            # 即使保存失败，也返回构建的数据，避免影响当前运行

        return new_behavior_sequences

    def _load_structured_features(self, start_week: int, end_week: int, final_user_list: List[str]) -> Dict[str, Any]:
        """
        加载或准备结构化特征, 基于 final_user_list。
        Args:
            final_user_list: 权威的用户ID列表。
        """
        # 文件名可以包含用户数量信息
        features_file_suffix = f"_u{len(final_user_list)}.pickle"
        features_file = os.path.join(self.multimodal_dir, "StructuredFeatures",
                                   f"structured_features_w{start_week}_{end_week}{features_file_suffix}")

        if os.path.exists(features_file) and not self.config.data.force_regenerate_structured_features:
            print(f"   结构化特征已存在，直接加载: {features_file}")
            with open(features_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"   准备结构化特征 (基于 {len(final_user_list)} 个已筛选用户)...")
        # _prepare_structured_features_for_users 需要接收用户列表
        structured_features = self._prepare_structured_features_for_users(
            start_week, end_week, final_user_list
        )
        
        with open(features_file, 'wb') as f:
            pickle.dump(structured_features, f)
        print(f"   结构化特征保存到: {features_file}")
        return structured_features

    def _load_labels_data_from_df(self, final_users_df: pd.DataFrame) -> Dict[str, int]:
        """
        从传入的已筛选用户DataFrame加载标签数据。
        Args:
            final_users_df: 包含最终用户及其特征（包括标签信息）的DataFrame。
                           索引应为 user_id。
        Returns:
            一个字典 {user_id: label}
        """
        print(f"   从已筛选的DataFrame加载 {len(final_users_df)} 个用户的标签...")
        labels = {}
        for user_id, user_row_data in final_users_df.iterrows(): 
            user = user_id 
            if 'malscene' in user_row_data and user_row_data['malscene'] > 0:
                is_malicious = 1
            else:
                is_malicious = int(user_row_data.get('is_malicious', 0) == 1) 
            labels[user] = int(is_malicious)
        return labels
    
    def _integrate_multimodal_data(self, 
                                 user_graph_data: Dict[str, Any],
                                 text_data: Dict[str, Any], # 文本数据仍是全局加载的
                                 behavior_sequences: Dict[str, Any], # 已按final_user_list筛选
                                 structured_features: Dict[str, Any], # 已按final_user_list筛选
                                 labels_data: Dict[str, int], # 已按final_user_list筛选
                                 sequence_length: int,
                                 final_user_list: List[str]) -> Dict[str, Any]: # 权威用户列表
        """整合多模态数据，确保基于final_user_list"""
        logger = logging.getLogger(__name__) # <--- 添加这一行
        
        print(f"   整合多模态数据 (基于 {len(final_user_list)} 个权威用户)...")
        # 获取用户列表 - 现在直接使用 final_user_list
        # users = list(labels_data.keys()) # 旧方法，现在用final_user_list
        users = final_user_list
        
        # 确保 user_graph_data['user_to_index'] 只包含 final_user_list 中的用户
        # 如果图是为这些精确用户构建的，那应该是匹配的。
        # 但为了安全，我们基于 final_user_list 和图中实际存在的用户来调整
        
        graph_node_features = user_graph_data.get('node_features')
        graph_adj_matrix = user_graph_data.get('adjacency_matrix')
        graph_user_to_index = user_graph_data.get('user_to_index', {})

        # 如果图数据是为超集用户构建的，需要根据final_user_list进行子图提取
        # 但我们修改了_load_user_graph_data使其基于users_df_for_graph构建，
        # 所以这里的 graph_user_to_index 应该已经对应 final_user_list
        # 因此，我们信任传递过来的 graph_user_to_index
        
        # 准备各模态数据
        behavior_seq_data = []
        text_content_data = []
        structured_feat_data = []
        labels = []
        user_indices_in_graph_list = [] # 新增：用于收集每个用户的图索引
        
        for user in users:
            # 1. 行为序列数据
            if user in behavior_sequences:
                user_sequences = behavior_sequences[user]
                if len(user_sequences) > 0:
                    # 取最后一个序列或合并多个序列
                    if len(user_sequences) == 1:
                        seq = user_sequences[0]
                    else:
                        # 合并多个序列
                        seq = np.concatenate(user_sequences, axis=0)
                    
                    # 截断或填充到指定长度
                    if len(seq) > sequence_length:
                        seq = seq[-sequence_length:]
                    elif len(seq) < sequence_length:
                        padding = np.zeros((sequence_length - len(seq), seq.shape[1]))
                        seq = np.concatenate([padding, seq], axis=0)
                    
                    behavior_seq_data.append(seq)
                else:
                    behavior_seq_data.append(np.zeros((sequence_length, self.feature_dim)))
            else:
                behavior_seq_data.append(np.zeros((sequence_length, self.feature_dim)))
            
            # 2. 文本内容数据
            user_texts = []
            for week_data in text_data.values():
                if user in week_data.get('email_texts', {}):
                    user_texts.extend(week_data['email_texts'][user])
                if user in week_data.get('file_texts', {}):
                    user_texts.extend(week_data['file_texts'][user])
                if user in week_data.get('http_texts', {}):
                    user_texts.extend(week_data['http_texts'][user])
            
            # 合并文本内容
            combined_text = " ".join(user_texts[:10])  # 限制文本长度
            if not combined_text.strip():
                combined_text = "No text content"
            text_content_data.append(combined_text)
            
            # 3. 结构化特征数据
            if user in structured_features:
                user_features = structured_features[user]
                if len(user_features) > 0:
                    # 取平均值或最后一个特征向量
                    avg_features = np.mean(user_features, axis=0)
                    structured_feat_data.append(avg_features)
                else:
                    structured_feat_data.append(np.zeros(50))  # 默认特征维度
            else:
                structured_feat_data.append(np.zeros(50))
            
            # 4. 标签
            labels.append(labels_data[user])

            # 5. 用户在图中的索引 (新增)
            if user in graph_user_to_index:
                user_indices_in_graph_list.append(graph_user_to_index[user])
            else:
                # 如果用户不在图的映射中（理论上不应该发生，如果final_user_list是权威的且图是为这些用户构建的）
                # 添加一个无效索引，例如 -1
                user_indices_in_graph_list.append(-1) 
                logger.warning(f"用户 {user} 未在 graph_user_to_index 中找到，图索引设为-1")
        
        return {
            'behavior_sequences': np.array(behavior_seq_data),
            'node_features': graph_node_features,
            'adjacency_matrix': graph_adj_matrix,
            'text_content': text_content_data,
            'structured_features': np.array(structured_feat_data),
            'labels': np.array(labels),
            'users': users,
            'user_to_index': graph_user_to_index,
            'user_indices_in_graph': np.array(user_indices_in_graph_list) # 新增：添加图索引列表
        }
    
    def _construct_user_relationships(self, users_df: pd.DataFrame, # 现在这个users_df是已经筛选过的
                                    start_week: int, end_week: int) -> Dict[str, Any]:
        """构建用户关系图数据，不再进行内部max_users筛选"""
        
        # users_df 已经是筛选过的，所以不需要这里的 max_users 逻辑了
        # if max_users:
        #     # ... (旧的max_users筛选逻辑) ...

        # users = users_df['user'].tolist() # 旧的，且可能出错
        users = users_df.index.tolist() # 正确的方式，假设索引是 user_id
        
        num_users = len(users)
        user_to_index_map = {user: i for i, user in enumerate(users)}
        
        # 初始化邻接矩阵
        adjacency_matrix = np.zeros((num_users, num_users))
        
        # 构建用户特征矩阵
        user_features = []
        for _, user_row in users_df.iterrows():
            # 基础特征：角色、部门等
            features = []
            
            # 角色特征 (one-hot编码)
            role = user_row.get('role', 'Unknown')
            role_features = self._encode_categorical_feature(role, ['ITAdmin', 'Manager', 'Employee', 'Unknown'])
            features.extend(role_features)
            
            # 部门特征
            dept = user_row.get('functional_unit', 'Unknown')
            dept_features = self._encode_categorical_feature(dept, ['IT', 'Finance', 'HR', 'Sales', 'Unknown'])
            features.extend(dept_features)
            
            # OCEAN心理特征
            ocean_features = [
                user_row.get('O', 0.5), user_row.get('C', 0.5), 
                user_row.get('E', 0.5), user_row.get('A', 0.5), user_row.get('N', 0.5)
            ]
            features.extend(ocean_features)
            
            user_features.append(features)
        
        # 构建用户关系（基于共同活动、部门等）
        for i, user1 in enumerate(users):
            for j, user2 in enumerate(users):
                if i != j:
                    # 计算用户关系强度
                    relationship_strength = self._calculate_user_relationship(
                        users_df.iloc[i], users_df.iloc[j], start_week, end_week
                    )
                    adjacency_matrix[i, j] = relationship_strength
        
        return {
            'users': users,
            'node_features': np.array(user_features, dtype=np.float32),
            'adjacency_matrix': adjacency_matrix.astype(np.float32),
            'user_to_index': user_to_index_map
        }
    
    def _encode_categorical_feature(self, value: str, categories: List[str]) -> List[float]:
        """对分类特征进行one-hot编码"""
        encoding = [0.0] * len(categories)
        if value in categories:
            encoding[categories.index(value)] = 1.0
        else:
            encoding[-1] = 1.0  # Unknown类别
        return encoding
    
    def _calculate_user_relationship(self, user1: pd.Series, user2: pd.Series,
                                   start_week: int, end_week: int) -> float:
        """计算两个用户之间的关系强度"""
        relationship_score = 0.0
        
        # 1. 部门关系
        if user1.get('functional_unit') == user2.get('functional_unit'):
            relationship_score += 0.3
        
        # 2. 角色关系
        if user1.get('role') == user2.get('role'):
            relationship_score += 0.2
        
        # 3. OCEAN相似性
        ocean_features = ['O', 'C', 'E', 'A', 'N']
        ocean_similarity = 0.0
        for feature in ocean_features:
            val1 = user1.get(feature, 0.5)
            val2 = user2.get(feature, 0.5)
            ocean_similarity += 1.0 - abs(val1 - val2)
        ocean_similarity /= len(ocean_features)
        relationship_score += 0.3 * ocean_similarity
        
        # 4. 活动交互（基于共同的文件访问、邮件通信等）
        # TODO: 基于实际活动数据计算交互强度
        interaction_score = self._calculate_activity_interaction(
            user1.name, user2.name, start_week, end_week  # 使用 .name 获取Series的索引名，即user_id
        )
        relationship_score += 0.2 * interaction_score
        
        return min(relationship_score, 1.0)
    
    def _calculate_activity_interaction(self, user1: str, user2: str,
                                      start_week: int, end_week: int) -> float:
        """计算用户间的活动交互强度"""
        # 简化实现：基于用户名相似性
        # 实际应用中应该基于真实的活动交互数据
        if user1 == user2:
            return 0.0
        
        # 基于用户名的简单相似性计算
        common_chars = set(user1.lower()) & set(user2.lower())
        similarity = len(common_chars) / max(len(set(user1.lower())), len(set(user2.lower())))
        
        return min(similarity, 0.5)  # 限制最大交互强度

    def _build_behavior_sequences_for_users(self, start_week: int, end_week: int, user_list: List[str]) -> Dict[str, list]:
        """为指定用户列表构建行为序列数据 (优化版，减少I/O)"""
        
        # 1. 编码器拟合 (如果需要) - 这部分逻辑保持，但读取方式需要注意
        if not self.base_pipeline.encoder.is_fitted:
            print("   ⚠️ 行为序列编码器未拟合，尝试从基础流水线获取训练数据进行拟合...")
            temp_users_df_for_encoder = self.base_pipeline.step2_load_user_data()
            if not temp_users_df_for_encoder.empty:
                sample_events_for_encoder = []
                # 从 CERTDatasetPipeline 的 step1_combine_raw_data 输出的 Parquet 目录中读取原始事件数据
                # 该目录按周分区，例如 DataByWeek_parquet/week=0/, DataByWeek_parquet/week=1/
                base_event_parquet_dir = self.base_pipeline._get_work_file_path(self.base_pipeline.current_parquet_dir_name)
                
                if not os.path.exists(base_event_parquet_dir):
                    print(f"    ❌ 基础事件 Parquet 目录 {base_event_parquet_dir} 不存在，无法拟合编码器。")
                else:
                    for week_idx in range(start_week, min(end_week, start_week + 2)): # 用前几周数据拟合
                        print(f"      为编码器拟合读取周 {week_idx} 从 {base_event_parquet_dir}")
                        try:
                            week_event_data_ddf = dd.read_parquet(
                                base_event_parquet_dir,
                                filters=[('week', '==', week_idx)], # 假设 'week' 列存在于原始合并数据中
                                engine='pyarrow'
                            )
                            if not week_event_data_ddf.map_partitions(len).compute().sum() == 0:
                                week_event_data = week_event_data_ddf.compute()
                                if not week_event_data.empty:
                                    sample_events_for_encoder.append(week_event_data.sample(min(10000, len(week_event_data)), random_state=42)) # 增加采样数量
                                    print(f"        周 {week_idx}: 采样 {len(sample_events_for_encoder[-1])} 条事件用于编码器。")
                            else:
                                print(f"        周 {week_idx}: 无事件数据。")
                        except Exception as e_parquet_fit:
                            print(f"    读取周 {week_idx} Parquet (用于编码器拟合) 失败: {e_parquet_fit}")
                
                if sample_events_for_encoder:
                    all_sample_events = pd.concat(sample_events_for_encoder, ignore_index=True)
                    print(f"      总共采样 {len(all_sample_events)} 条事件用于编码器拟合。")
                    self.base_pipeline.encoder.fit(all_sample_events, temp_users_df_for_encoder)
                    print("   ✅ 行为序列编码器已拟合。")
                else:
                    print("   ❌ 无法为行为序列编码器找到训练数据。序列可能为空或使用默认编码。")
            else:
                 print("   ❌ 无法加载用户数据以拟合行为序列编码器。序列可能为空或使用默认编码。")
        
        # 2. 收集所有用户的周事件数据
        user_all_events_map = {user: [] for user in user_list}
        print(f"   🔄 开始为 {len(user_list)} 用户收集所有周 ({start_week}-{end_week-1}) 的事件特征数据...")
        num_data_by_week_dir = self.base_pipeline._get_work_file_path("NumDataByWeek")

        for week in range(start_week, end_week):
            # print(f"     处理周 {week} 的特征文件...") # 可以取消注释以获取更详细的日志
            feature_file_parquet = os.path.join(num_data_by_week_dir, f"{week}_features.parquet")
            feature_file_pickle = os.path.join(num_data_by_week_dir, f"{week}_features.pickle")
            
            week_data_df = None
            if os.path.exists(feature_file_parquet):
                try:
                    week_data_df = pd.read_parquet(feature_file_parquet, engine='pyarrow')
                except Exception as e_read_p:
                    # print(f"      周 {week}: 读取 Parquet 特征文件 {feature_file_parquet} 失败 ({e_read_p}). 尝试 Pickle...")
                    if os.path.exists(feature_file_pickle):
                        try: week_data_df = pd.read_pickle(feature_file_pickle)
                        except Exception as e_read_pk: 
                            # print(f"      周 {week}: 读取 Pickle 文件 {feature_file_pickle} 也失败 ({e_read_pk}).")
                            pass # 忽略单个文件读取错误，继续处理下一周
            elif os.path.exists(feature_file_pickle):
                # print(f"      周 {week}: Parquet 不存在，尝试读取 Pickle {feature_file_pickle}...")
                try: week_data_df = pd.read_pickle(feature_file_pickle)
                except Exception as e_read_pk2: 
                    # print(f"      周 {week}: 读取 Pickle 文件 {feature_file_pickle} 失败 ({e_read_pk2}).")
                    pass
            
            if week_data_df is not None and not week_data_df.empty and 'user' in week_data_df.columns:
                relevant_week_data = week_data_df[week_data_df['user'].isin(user_list)]
                if not relevant_week_data.empty:
                    for user_id_in_file, user_events_this_week in relevant_week_data.groupby('user'):
                        if user_id_in_file in user_all_events_map: # 确保key存在
                             user_all_events_map[user_id_in_file].append(user_events_this_week)
            # else:
                # if week_data_df is None: print(f"     周 {week}: 无数据文件。")
                # elif week_data_df.empty: print(f"     周 {week}: 数据文件为空。")
                # elif 'user' not in week_data_df.columns: print(f"     周 {week}: 数据文件缺少 'user' 列。")

        # 3. 为每个用户构建最终序列
        behavior_sequences = {}
        print(f"   🔄 开始为 {len(user_list)} 个用户整理和构建最终的行为序列...")
        
        for user_idx, user in enumerate(user_list):
            if (user_idx + 1) % 50 == 0:
                 print(f"     构建序列: 用户 {user_idx+1}/{len(user_list)} ({user})")

            user_final_sequences = [] # 一个用户可能只有一个主序列，或按某种逻辑切分
            user_collected_event_dfs = user_all_events_map.get(user, []) # 使用 .get 防止KeyError
            
            if user_collected_event_dfs:
                # 首先检查是否有空的DataFrame在列表中，这可能由读取错误或空文件导致
                non_empty_dfs = [df for df in user_collected_event_dfs if not df.empty and 'features' in df.columns]
                if not non_empty_dfs:
                    # print(f"      用户 {user}: 没有有效的事件DataFrame可合并。")
                    user_final_sequences.append(np.zeros((1, self.feature_dim))) # 添加默认序列
                    behavior_sequences[user] = user_final_sequences
                    continue

                try:
                    user_combined_events_df = pd.concat(non_empty_dfs, ignore_index=True)
                except ValueError as e_concat: # 例如，没有DataFrame可以合并
                    # print(f"      用户 {user}: 合并事件DataFrame时出错: {e_concat}")
                    user_final_sequences.append(np.zeros((1, self.feature_dim)))
                    behavior_sequences[user] = user_final_sequences
                    continue

                if 'date' in user_combined_events_df.columns:
                    user_combined_events_df = user_combined_events_df.sort_values('date')
                
                # 提取特征列并构建序列
                feature_vectors_list = []
                # 直接迭代Series的values，这比iterrows更快，而且我们只关心'features'列
                for features_item in user_combined_events_df['features'].values:
                    if isinstance(features_item, list):
                        feature_vectors_list.append(np.array(features_item, dtype=np.float32))
                    elif isinstance(features_item, np.ndarray):
                        feature_vectors_list.append(features_item.astype(np.float32))
                    else: # Fallback for unexpected types or missing data for an event
                        feature_vectors_list.append(np.zeros(self.feature_dim, dtype=np.float32))
                
                if feature_vectors_list:
                    try:
                        # 确保所有内部数组的维度一致，特别是特征维度 self.feature_dim
                        # np.stack 要求所有数组具有相同的形状（除了要堆叠的轴）
                        # 如果特征是 (dim,)，堆叠后是 (num_events, dim)
                        valid_feature_vectors = []
                        for f_vec in feature_vectors_list:
                            if f_vec.ndim == 1 and f_vec.shape[0] == self.feature_dim:
                                valid_feature_vectors.append(f_vec)
                            elif f_vec.ndim == 2 and f_vec.shape[1] == self.feature_dim and f_vec.shape[0] > 0: # 如果已经是 (N, dim)
                                valid_feature_vectors.extend(list(f_vec)) # 展平并添加
                            # else:
                                # print(f"    用户 {user}: 发现维度不匹配的特征向量，形状 {f_vec.shape}, 预期维度 {self.feature_dim}。将跳过此向量。")
                        
                        if valid_feature_vectors:
                            sequence_array = np.stack(valid_feature_vectors)
                            user_final_sequences.append(sequence_array)
                        # else:
                            # print(f"    用户 {user}: 没有有效的特征向量可堆叠。")

                    except ValueError as e_stack:
                        # print(f"      ⚠️ 用户 {user}: 堆叠特征向量时出错: {e_stack}. (检查特征维度是否一致)")
                        pass # 保持 user_final_sequences 为空，后续会补零

                # 如果在处理后 user_final_sequences 仍然为空 (例如，没有事件，或特征提取/堆叠失败)
                if not user_final_sequences or (len(user_final_sequences) > 0 and user_final_sequences[0].shape[0] == 0):
                    user_final_sequences = [np.zeros((1, self.feature_dim), dtype=np.float32)]
            else: # 如果用户没有任何收集到的事件DataFrame
                user_final_sequences.append(np.zeros((1, self.feature_dim), dtype=np.float32))
            
            behavior_sequences[user] = user_final_sequences
            
        print(f"   ✅ 所有用户的行为序列构建完成。")
        return behavior_sequences

    def _prepare_structured_features_for_users(self, start_week: int, end_week: int, user_list: List[str]) -> Dict[str, list]:
        """为指定用户列表准备结构化特征 (替换旧的 _prepare_structured_features)"""
        # print("📋 准备结构化特征...") # 日志移到调用处
        
        week_features_dir = os.path.join(self.work_dir, "WeekLevelFeatures")
        structured_features_all_users = {} # 先加载所有用户的周特征

        for week in range(start_week, end_week):
            # 注意：CERTDatasetPipeline 生成的周级别特征文件名是 weeks_START_END.csv
            # 而不是 week_{week}_features.csv。这里需要匹配。
            # 假设文件名是 pipeline.run_full_pipeline 中 _week_level_analysis 生成的
            # WeekLevelFeatures/weeks_{start_week}_{end_week-1}.csv
            # 为了简化，我们假设这里的文件名是按周的，或者需要调整文件名逻辑
            # 或者，更稳妥的是，我们从 NumDataByWeek/{week}_features.pickle 重新聚合我们需要的用户
            
            # 方案1: 尝试读取已聚合的周级别CSV (可能需要修改文件名逻辑)
            # week_file_path = os.path.join(week_features_dir, f"weeks_{week}_{week}.csv") # 假设有这样的文件
            # if not os.path.exists(week_file_path):
            #     week_file_path = os.path.join(week_features_dir, f"weeks_{start_week}_{end_week-1}.csv")

            # 方案2: 从 NumDataByWeek 重新聚合 (更可靠，但可能慢一点)
            # --- 修改开始: 优先读取 Parquet, 回退到 Pickle ---
            feature_file_parquet = self.base_pipeline._get_work_file_path(f"NumDataByWeek/{week}_features.parquet")
            feature_file_pickle = self.base_pipeline._get_work_file_path(f"NumDataByWeek/{week}_features.pickle")
            
            week_user_event_features_df = None
            if os.path.exists(feature_file_parquet):
                try:
                    week_user_event_features_df = pd.read_parquet(feature_file_parquet, engine='pyarrow')
                except Exception as e_parquet_read:
                    print(f"    ⚠️ 周 {week}: 读取 Parquet 特征文件 {feature_file_parquet} 失败 ({e_parquet_read}). 尝试 Pickle 回退...")
                    if os.path.exists(feature_file_pickle):
                        try:
                            week_user_event_features_df = pd.read_pickle(feature_file_pickle)
                            print(f"      ✅ 周 {week}: 成功从 Pickle 文件 {feature_file_pickle} 回退读取。")
                        except Exception as e_pickle_read:
                            print(f"      ❌ 周 {week}: 读取 Pickle 回退文件 {feature_file_pickle} 也失败 ({e_pickle_read}).")
                    else:
                         print(f"      周 {week}: Pickle 回退文件 {feature_file_pickle} 不存在。")
            elif os.path.exists(feature_file_pickle):
                print(f"    周 {week}: Parquet 特征文件不存在，尝试读取 Pickle 文件 {feature_file_pickle}...")
                try:
                    week_user_event_features_df = pd.read_pickle(feature_file_pickle)
                except Exception as e_pickle_read:
                    print(f"      ❌ 周 {week}: 读取 Pickle 文件 {feature_file_pickle} 失败 ({e_pickle_read}).")
            # --- 修改结束 ---

            if week_user_event_features_df is not None and not week_user_event_features_df.empty:
                # week_user_event_features_df = pd.read_pickle(feature_pickle_file)
                # if week_user_event_features_df.empty:
                #     continue
                for user_id_in_file in week_user_event_features_df['user'].unique():
                    if user_id_in_file not in structured_features_all_users:
                        structured_features_all_users[user_id_in_file] = []
                    
                    user_events_this_week = week_user_event_features_df[week_user_event_features_df['user'] == user_id_in_file]
                    
                    # 调用 CERTDatasetPipeline 中的聚合逻辑 (简化版)
                    # 注意：_aggregate_user_features 返回的是一个字典，包含多个平展的特征
                    aggregated_feats_dict = self.base_pipeline._aggregate_user_features(user_events_this_week, 'week')
                    
                    # 将字典转换为特征向量 (需要确定顺序和哪些特征)
                    # 为了简单，我们先假设它返回一个固定顺序和数量的数值特征列表
                    # 或者，我们直接使用 CERTDatasetPipeline 中保存到CSV的那些列
                    # 这里用一个简化方式：取字典中所有数值类型的值
                    feature_vector = [v for k, v in aggregated_feats_dict.items() if isinstance(v, (int, float, np.number))]
                    
                    # 确保特征向量长度一致 (如果需要，否则后续模型输入会出问题)
                    # 这里的默认50维是在 train_pipeline.py 中 MultiModalAnomalyDetector 假设的
                    # 需要与模型定义匹配，或者模型能够处理可变长度的结构化特征
                    # 暂时我们不强制长度，由模型部分处理或报错
                    if feature_vector: # 只有提取到特征才添加
                         structured_features_all_users[user_id_in_file].append(feature_vector)
            else:
                # print(f"   ⚠️ 未找到周 {week} 的数值特征文件 (Parquet/Pickle): {feature_file_parquet} / {feature_file_pickle}")
                # 即使文件不存在或读取失败，也继续处理下一周，而不是打印每条消息
                pass # 已经在上面处理了打印


        # 从 structured_features_all_users 中筛选出 final_user_list 所需的
        final_structured_features = {}
        for user in user_list:
            if user in structured_features_all_users and structured_features_all_users[user]:
                # 对于每个用户，可能有多个周的特征向量，可以取平均或最后一个
                # MultiModalDataPipeline._integrate_multimodal_data 中是取的平均
                # 我们这里保持列表，由 _integrate_multimodal_data 处理
                final_structured_features[user] = structured_features_all_users[user]
            else:
                final_structured_features[user] = [] # 如果某用户没有结构化特征
        
        return final_structured_features

    def run_full_multimodal_pipeline(self,
                                   start_week: int = 0,
                                   end_week: int = None,
                                   max_users: int = None, # 这个max_users将用于统一筛选
                                   sequence_length: int = 128) -> Dict[str, Any]:
        """
        运行完整的多模态数据处理流水线
        """
        print(f"\n{'='*80}")
        print(f"运行完整多模态数据处理流水线 (max_users={max_users} 将在prepare_training_data中统一应用)")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Step 1: 运行基础特征提取 (这一步理论上应该提供所有用户的数据给后续筛选)
        # max_users 在这里设置为None，让它处理所有用户，然后在prepare_training_data中筛选
        self.run_base_feature_extraction(start_week, end_week, max_users=None, 
                                         sample_ratio=getattr(self.config.data, 'sample_ratio', 1.0)) # 使用属性访问，若无则默认为1.0
        
        # Step 2: 提取多模态数据 - 此步骤已被整合到 prepare_training_data 中
        # initial_users_df_for_step2 = self.base_pipeline.step2_load_user_data()
        # final_selected_users_df_for_step2 = self._apply_max_users_filter(initial_users_df_for_step2, max_users)
        # # self.extract_multimodal_data(start_week, end_week, final_selected_users_df_for_step2) # 已移除
        
        # Step 3: 准备训练数据 - 这一步现在是核心，它会驱动各模态数据的加载/生成
        # 并确保用户一致性
        training_data = self.prepare_training_data(start_week, end_week, max_users, sequence_length)
        
        # 调试打印：检查 users 列表和 user_to_index 字典的一致性
        # 这部分调试逻辑现在应该在 prepare_training_data 返回之前，或者在 _integrate_multimodal_data 内部进行
        # 但由于 _integrate_multimodal_data 现在接收 final_user_list, 一致性应得到保证
        print("\nDEBUG: Checking user consistency in run_full_multimodal_pipeline (after prepare_training_data)")
        users_in_list = set(training_data.get('users', []))
        users_in_map = set(training_data.get('user_to_index', {}).keys())
        
        if users_in_list == users_in_map:
            print("DEBUG: Users in list and user_to_index map are CONSISTENT.")
        else:
            print("DEBUG: Users in list and user_to_index map are INCONSISTENT.")
            print(f"  Users ONLY in list: {users_in_list - users_in_map}")
            print(f"  Users ONLY in map: {users_in_map - users_in_list}")
            print(f"  Number of users in list: {len(users_in_list)}")
            print(f"  Number of users in map: {len(users_in_map)}")
            # 这可能指示了上游数据处理中用户集合不匹配的问题

        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"多模态数据处理流水线完成！")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"数据样本数: {len(training_data['labels'])}")
        print(f"正常样本: {np.sum(training_data['labels'] == 0)}")
        print(f"异常样本: {np.sum(training_data['labels'] == 1)}")
        print(f"{'='*80}")
        
        return training_data

def main():
    """主函数示例"""
    # 创建多模态数据流水线
    pipeline = MultiModalDataPipeline(
        data_version='r4.2',
        feature_dim=256,
        num_cores=8
    )
    
    # 运行完整流水线
    training_data = pipeline.run_full_multimodal_pipeline(
        start_week=0,
        end_week=5,  # 测试前5周
        max_users=100,  # 测试100个用户
        sequence_length=128
    )
    
    print("多模态数据处理完成！")

if __name__ == "__main__":
    main() 