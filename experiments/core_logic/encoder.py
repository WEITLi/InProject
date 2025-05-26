#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心编码器模块
提供统一的事件编码接口，支持各种类型的用户活动编码
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from utils import FeatureEncoder, create_time_features, normalize_features
from temporal import encode_temporal_features
from user_context import encode_user_context
from email_features import encode_email_features
from file_features import encode_file_features
from http_features import encode_http_features
from device_features import encode_device_features

class EventEncoder:
    """
    统一事件编码器
    
    负责将原始事件数据转换为固定长度的向量表示，
    支持多种事件类型和缺失值处理策略
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 data_version: str = 'r4.2',
                 device: str = 'cpu'):
        """
        初始化编码器
        
        Args:
            feature_dim: 输出特征维度
            data_version: 数据集版本 (r4.2, r5.2, r6.2等)
            device: 计算设备
        """
        self.feature_dim = feature_dim
        self.data_version = data_version
        self.device = device
        
        # 初始化各个子编码器
        self.feature_encoder = FeatureEncoder()
        
        # 事件类型映射
        self.event_type_mapping = {
            'logon': 1,
            'device': 2, 
            'email': 3,
            'file': 4,
            'http': 5
        }
        
        # 特征维度配置（根据数据版本调整）
        self.feature_dims = self._get_feature_dimensions()
        
        # 是否已拟合
        self.is_fitted = False
        
    def _get_feature_dimensions(self) -> Dict[str, int]:
        """获取各模块的特征维度配置"""
        if self.data_version in ['r4.1', 'r4.2']:
            return {
                'temporal': 12,     # 时间特征
                'user_context': 10, # 用户上下文
                'email': 9,         # 邮件特征
                'file': 5,          # 文件特征
                'http': 5,          # HTTP特征
                'device': 2,        # 设备特征
                'event_type': 6,    # 事件类型one-hot
                'activity': 20      # 活动特征
            }
        elif self.data_version in ['r5.1', 'r5.2']:
            return {
                'temporal': 12,
                'user_context': 15,
                'email': 23,        # 包含附件特征
                'file': 8,          # 包含USB传输
                'http': 5,
                'device': 3,
                'event_type': 6,
                'activity': 25
            }
        elif self.data_version in ['r6.1', 'r6.2']:
            return {
                'temporal': 12,
                'user_context': 15,
                'email': 23,
                'file': 9,
                'http': 6,          # 包含活动类型
                'device': 4,
                'event_type': 6,
                'activity': 30
            }
        else:
            # 默认配置
            return {
                'temporal': 12,
                'user_context': 10,
                'email': 15,
                'file': 8,
                'http': 6,
                'device': 3,
                'event_type': 6,
                'activity': 25
            }
    
    def fit(self, events_data: pd.DataFrame, user_data: pd.DataFrame = None):
        """
        拟合编码器参数
        
        Args:
            events_data: 事件数据
            user_data: 用户数据（包含角色、部门、OCEAN等）
        """
        print("正在拟合事件编码器...")
        
        # 拟合时间特征编码器
        if 'date' in events_data.columns:
            timestamps = pd.to_datetime(events_data['date'])
            time_features = create_time_features(timestamps)
            for name, values in time_features.items():
                if values.dtype in [np.float64, np.float32]:
                    self.feature_encoder.fit_numerical_scaler(values, f"time_{name}")
        
        # 拟合用户相关特征
        if 'user' in events_data.columns:
            users = events_data['user'].unique()
            self.feature_encoder.fit_categorical_encoder(users.tolist(), 'user_id')
        
        if 'pc' in events_data.columns:
            pcs = events_data['pc'].unique()
            self.feature_encoder.fit_categorical_encoder(pcs.tolist(), 'pc_id')
        
        # 拟合各类事件特征
        self._fit_email_features(events_data)
        self._fit_file_features(events_data) 
        self._fit_http_features(events_data)
        self._fit_device_features(events_data)
        
        # 拟合用户上下文特征
        if user_data is not None:
            self._fit_user_context_features(user_data)
        
        self.is_fitted = True
        print("编码器拟合完成")
    
    def _fit_email_features(self, data: pd.DataFrame):
        """拟合邮件特征编码器"""
        email_data = data[data['type'] == 'email']
        if len(email_data) > 0:
            # 邮件大小
            if 'size' in email_data.columns:
                sizes = pd.to_numeric(email_data['size'], errors='coerce')
                self.feature_encoder.fit_numerical_scaler(sizes.values, 'email_size')
            
            # 文本长度
            if 'content' in email_data.columns:
                text_lens = email_data['content'].str.len().values
                self.feature_encoder.fit_numerical_scaler(text_lens, 'email_text_len')
    
    def _fit_file_features(self, data: pd.DataFrame):
        """拟合文件特征编码器"""
        file_data = data[data['type'] == 'file']
        if len(file_data) > 0:
            # 文件大小
            if 'content' in file_data.columns:
                file_sizes = file_data['content'].str.len().values
                self.feature_encoder.fit_numerical_scaler(file_sizes, 'file_size')
    
    def _fit_http_features(self, data: pd.DataFrame):
        """拟合HTTP特征编码器"""
        http_data = data[data['type'] == 'http']
        if len(http_data) > 0:
            # URL长度
            if 'url/fname' in http_data.columns:
                url_lens = http_data['url/fname'].str.len().values
                self.feature_encoder.fit_numerical_scaler(url_lens, 'url_length')
    
    def _fit_device_features(self, data: pd.DataFrame):
        """拟合设备特征编码器"""
        device_data = data[data['type'] == 'device']
        if len(device_data) > 0:
            # 设备活动类型
            if 'activity' in device_data.columns:
                activities = device_data['activity'].unique().tolist()
                self.feature_encoder.fit_categorical_encoder(activities, 'device_activity')
    
    def _fit_user_context_features(self, user_data: pd.DataFrame):
        """拟合用户上下文特征编码器"""
        # 角色特征
        if 'role' in user_data.columns:
            roles = user_data['role'].unique().tolist()
            self.feature_encoder.fit_categorical_encoder(roles, 'user_role')
        
        # 部门特征
        if 'dept' in user_data.columns:
            depts = user_data['dept'].unique().tolist()
            self.feature_encoder.fit_categorical_encoder(depts, 'user_dept')
        
        # OCEAN特征
        ocean_traits = ['O', 'C', 'E', 'A', 'N']
        for trait in ocean_traits:
            if trait in user_data.columns:
                values = pd.to_numeric(user_data[trait], errors='coerce').values
                self.feature_encoder.fit_numerical_scaler(values, f'ocean_{trait}')
    
    def encode_event(self, event_dict: Dict[str, Any], 
                    user_context: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        编码单个事件
        
        Args:
            event_dict: 事件字典，包含事件的所有字段
            user_context: 用户上下文信息
            
        Returns:
            Tuple[feature_vector, mask]: 特征向量和mask
        """
        if not self.is_fitted:
            raise ValueError("编码器未拟合，请先调用fit()方法")
        
        # 初始化特征列表
        features = []
        masks = []
        
        # 1. 事件类型编码
        event_type = event_dict.get('type', 'unknown')
        event_type_vec = self._encode_event_type(event_type)
        features.append(event_type_vec)
        masks.append(np.ones(len(event_type_vec), dtype=bool))
        
        # 2. 时间特征编码
        temporal_features, temporal_mask = encode_temporal_features(
            event_dict, self.feature_encoder
        )
        features.append(temporal_features)
        masks.append(temporal_mask)
        
        # 3. 用户上下文编码
        if user_context:
            user_features, user_mask = encode_user_context(
                user_context, self.feature_encoder
            )
            features.append(user_features)
            masks.append(user_mask)
        else:
            # 用零向量填充
            user_dim = self.feature_dims['user_context']
            features.append(np.zeros(user_dim))
            masks.append(np.zeros(user_dim, dtype=bool))
        
        # 4. 特定事件类型编码
        if event_type == 'email':
            event_features, event_mask = encode_email_features(
                event_dict, self.feature_encoder, self.data_version
            )
        elif event_type == 'file':
            event_features, event_mask = encode_file_features(
                event_dict, self.feature_encoder, self.data_version
            )
        elif event_type == 'http':
            event_features, event_mask = encode_http_features(
                event_dict, self.feature_encoder, self.data_version
            )
        elif event_type == 'device':
            event_features, event_mask = encode_device_features(
                event_dict, self.feature_encoder, self.data_version
            )
        else:
            # 未知事件类型，用零向量填充
            max_dim = max(self.feature_dims[k] for k in ['email', 'file', 'http', 'device'])
            event_features = np.zeros(max_dim)
            event_mask = np.zeros(max_dim, dtype=bool)
        
        features.append(event_features)
        masks.append(event_mask)
        
        # 5. 拼接所有特征
        final_features = np.concatenate(features)
        final_mask = np.concatenate(masks)
        
        # 6. 调整到目标维度
        if len(final_features) < self.feature_dim:
            # 填充零
            padding = self.feature_dim - len(final_features)
            final_features = np.concatenate([final_features, np.zeros(padding)])
            final_mask = np.concatenate([final_mask, np.zeros(padding, dtype=bool)])
        elif len(final_features) > self.feature_dim:
            # 截断
            final_features = final_features[:self.feature_dim]
            final_mask = final_mask[:self.feature_dim]
        
        return final_features.astype(np.float32), final_mask
    
    def _encode_event_type(self, event_type: str) -> np.ndarray:
        """事件类型one-hot编码"""
        n_types = len(self.event_type_mapping)
        one_hot = np.zeros(n_types)
        if event_type in self.event_type_mapping:
            idx = self.event_type_mapping[event_type] - 1  # 0-based indexing
            one_hot[idx] = 1.0
        return one_hot
    
    def encode_event_sequence(self, events: List[Dict[str, Any]], 
                            user_context: Dict[str, Any] = None,
                            max_sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        编码事件序列
        
        Args:
            events: 事件列表
            user_context: 用户上下文
            max_sequence_length: 最大序列长度
            
        Returns:
            Tuple[sequence_features, sequence_mask]: 序列特征和mask
        """
        sequence_features = []
        sequence_masks = []
        
        # 编码每个事件
        for event in events[:max_sequence_length]:
            features, mask = self.encode_event(event, user_context)
            sequence_features.append(features)
            sequence_masks.append(mask)
        
        # 填充到最大长度
        while len(sequence_features) < max_sequence_length:
            sequence_features.append(np.zeros(self.feature_dim, dtype=np.float32))
            sequence_masks.append(np.zeros(self.feature_dim, dtype=bool))
        
        return np.array(sequence_features), np.array(sequence_masks)
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        names = []
        
        # 事件类型特征名
        for event_type in self.event_type_mapping.keys():
            names.append(f"event_type_{event_type}")
        
        # 时间特征名
        time_features = ['hour', 'day_of_week', 'is_weekend', 'is_work_hour', 
                        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                        'month_sin', 'month_cos', 'session_duration', 'time_since_last']
        for name in time_features:
            names.append(f"temporal_{name}")
        
        # 用户上下文特征名
        user_features = ['role', 'dept', 'is_it_admin', 'ocean_O', 'ocean_C', 
                        'ocean_E', 'ocean_A', 'ocean_N', 'pc_type', 'shared_pc']
        for name in user_features:
            names.append(f"user_{name}")
        
        # 活动特征名（动态生成）
        activity_features = ['n_recipients', 'email_size', 'text_length', 'file_size', 
                           'url_length', 'url_depth', 'device_duration', 'external_contact']
        for name in activity_features:
            names.append(f"activity_{name}")
        
        # 填充到目标维度
        while len(names) < self.feature_dim:
            names.append(f"padding_{len(names)}")
        
        return names[:self.feature_dim]
    
    def save_encoder(self, filepath: str):
        """保存编码器状态"""
        import pickle
        
        encoder_state = {
            'feature_dim': self.feature_dim,
            'data_version': self.data_version,
            'feature_encoder': self.feature_encoder,
            'event_type_mapping': self.event_type_mapping,
            'feature_dims': self.feature_dims,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoder_state, f)
        
        print(f"编码器已保存到: {filepath}")
    
    def load_encoder(self, filepath: str):
        """加载编码器状态"""
        import pickle
        
        with open(filepath, 'rb') as f:
            encoder_state = pickle.load(f)
        
        self.feature_dim = encoder_state['feature_dim']
        self.data_version = encoder_state['data_version']
        self.feature_encoder = encoder_state['feature_encoder']
        self.event_type_mapping = encoder_state['event_type_mapping']
        self.feature_dims = encoder_state['feature_dims']
        self.is_fitted = encoder_state['is_fitted']
        
        print(f"编码器已从 {filepath} 加载")

def create_default_encoder(data_version: str = 'r4.2', 
                          feature_dim: int = 256) -> EventEncoder:
    """
    创建默认配置的事件编码器
    
    Args:
        data_version: 数据版本
        feature_dim: 特征维度
        
    Returns:
        配置好的EventEncoder实例
    """
    encoder = EventEncoder(
        feature_dim=feature_dim,
        data_version=data_version
    )
    
    return encoder 