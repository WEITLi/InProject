#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取工具函数模块
包含标准化、分箱、embedding等工具函数，支持mask和缺省填充策略
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import warnings

class FeatureEncoder:
    """统一特征编码器，支持多种编码方式和缺失值处理"""
    
    def __init__(self, embedding_dim: int = 64, max_vocab_size: int = 10000):
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.scalers = {}
        self.encoders = {}
        self.embeddings = {}
        self.text_vectorizers = {}
        
    def fit_numerical_scaler(self, data: np.ndarray, feature_name: str, method: str = 'standard'):
        """
        拟合数值特征的标准化器
        
        Args:
            data: 数值数据
            feature_name: 特征名称
            method: 标准化方法 ('standard', 'minmax', 'robust')
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
            
        # 处理缺失值
        if hasattr(data, 'values'):
            # 如果是pandas Series，转换为numpy数组
            data_array = data.values
        else:
            data_array = np.array(data)
            
        valid_data = data_array[~np.isnan(data_array)]
        if len(valid_data) > 0:
            scaler.fit(valid_data.reshape(-1, 1))
            self.scalers[feature_name] = scaler
        
    def transform_numerical(self, data: np.ndarray, feature_name: str, 
                          fill_missing: bool = True, fill_value: float = 0.0) -> np.ndarray:
        """
        转换数值特征
        
        Args:
            data: 输入数据
            feature_name: 特征名称
            fill_missing: 是否填充缺失值
            fill_value: 填充值
            
        Returns:
            标准化后的数据和mask
        """
        if feature_name not in self.scalers:
            warnings.warn(f"No scaler found for {feature_name}, returning original data")
            return data, ~np.isnan(data)
            
        # 创建mask标记缺失值
        mask = ~np.isnan(data)
        
        # 填充缺失值
        if fill_missing:
            filled_data = np.where(mask, data, fill_value)
        else:
            filled_data = data
            
        # 标准化
        try:
            transformed = self.scalers[feature_name].transform(filled_data.reshape(-1, 1)).flatten()
            return transformed, mask
        except:
            return filled_data, mask
    
    def fit_categorical_encoder(self, data: List[str], feature_name: str):
        """拟合分类特征编码器"""
        encoder = LabelEncoder()
        valid_data = [x for x in data if x is not None and str(x) != 'nan']
        if len(valid_data) > 0:
            encoder.fit(valid_data)
            self.encoders[feature_name] = encoder
    
    def transform_categorical(self, data: List[str], feature_name: str, 
                            fill_missing: bool = True, fill_value: str = 'unknown') -> Tuple[np.ndarray, np.ndarray]:
        """
        转换分类特征
        
        Returns:
            编码后的数据和mask
        """
        if feature_name not in self.encoders:
            warnings.warn(f"No encoder found for {feature_name}")
            return np.zeros(len(data)), np.zeros(len(data), dtype=bool)
            
        # 创建mask
        mask = np.array([x is not None and str(x) != 'nan' for x in data])
        
        # 填充缺失值
        if fill_missing:
            filled_data = [x if mask[i] else fill_value for i, x in enumerate(data)]
        else:
            filled_data = data
            
        # 编码
        try:
            # 处理未见过的类别
            encoder = self.encoders[feature_name]
            encoded = []
            for item in filled_data:
                if item in encoder.classes_:
                    encoded.append(encoder.transform([item])[0])
                else:
                    encoded.append(0)  # 未知类别编码为0
            return np.array(encoded), mask
        except:
            return np.zeros(len(data)), mask
    
    def create_embedding_layer(self, feature_name: str, vocab_size: int) -> nn.Embedding:
        """创建embedding层"""
        embedding = nn.Embedding(vocab_size + 1, self.embedding_dim, padding_idx=0)
        self.embeddings[feature_name] = embedding
        return embedding
    
    def binning_transform(self, data: np.ndarray, n_bins: int = 10, 
                         strategy: str = 'quantile') -> Tuple[np.ndarray, np.ndarray]:
        """
        数值分箱处理
        
        Args:
            data: 数值数据
            n_bins: 分箱数量
            strategy: 分箱策略 ('quantile', 'uniform', 'kmeans')
            
        Returns:
            分箱后的索引和mask
        """
        mask = ~np.isnan(data)
        valid_data = data[mask]
        
        if len(valid_data) == 0:
            return np.zeros(len(data), dtype=int), mask
            
        if strategy == 'quantile':
            # 分位数分箱
            bins = np.quantile(valid_data, np.linspace(0, 1, n_bins + 1))
            bins = np.unique(bins)  # 去除重复的分位点
        elif strategy == 'uniform':
            # 等宽分箱
            bins = np.linspace(valid_data.min(), valid_data.max(), n_bins + 1)
        else:
            raise ValueError(f"Unsupported binning strategy: {strategy}")
            
        # 分箱
        binned = np.digitize(data, bins) - 1
        binned = np.clip(binned, 0, len(bins) - 2)  # 确保在有效范围内
        
        return binned, mask
    
    def text_to_features(self, texts: List[str], feature_name: str, 
                        max_features: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        文本特征提取 (TF-IDF)
        
        Args:
            texts: 文本列表
            feature_name: 特征名称
            max_features: 最大特征数
            
        Returns:
            TF-IDF特征矩阵和mask
        """
        # 创建mask
        mask = np.array([text is not None and str(text).strip() != '' for text in texts])
        
        # 处理缺失文本
        processed_texts = [str(text) if mask[i] else '' for i, text in enumerate(texts)]
        
        # 如果没有训练过的向量化器，创建新的
        if feature_name not in self.text_vectorizers:
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            valid_texts = [text for i, text in enumerate(processed_texts) if mask[i]]
            if len(valid_texts) > 0:
                vectorizer.fit(valid_texts)
                self.text_vectorizers[feature_name] = vectorizer
            else:
                # 没有有效文本，返回零矩阵
                return np.zeros((len(texts), max_features)), mask
        
        # 转换文本
        vectorizer = self.text_vectorizers[feature_name]
        try:
            tfidf_matrix = vectorizer.transform(processed_texts).toarray()
            return tfidf_matrix, mask
        except:
            return np.zeros((len(texts), max_features)), mask

def create_time_features(timestamps: pd.Series) -> Dict[str, np.ndarray]:
    """
    从时间戳创建时间特征
    
    Args:
        timestamps: 时间戳序列
        
    Returns:
        时间特征字典
    """
    dt = pd.to_datetime(timestamps)
    
    features = {
        'hour': dt.dt.hour.values,
        'day_of_week': dt.dt.dayofweek.values,
        'day_of_month': dt.dt.day.values,
        'month': dt.dt.month.values,
        'is_weekend': (dt.dt.dayofweek >= 5).astype(int).values,
        'is_work_hour': ((dt.dt.hour >= 8) & (dt.dt.hour <= 17)).astype(int).values,
        'hour_sin': np.sin(2 * np.pi * dt.dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.dt.hour / 24),
        'day_sin': np.sin(2 * np.pi * dt.dt.dayofweek / 7),
        'day_cos': np.cos(2 * np.pi * dt.dt.dayofweek / 7),
        'month_sin': np.sin(2 * np.pi * dt.dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dt.dt.month / 12),
    }
    
    return features

def normalize_features(features: Dict[str, np.ndarray], 
                      method: str = 'standard') -> Dict[str, np.ndarray]:
    """
    批量标准化特征
    
    Args:
        features: 特征字典
        method: 标准化方法
        
    Returns:
        标准化后的特征字典
    """
    normalized = {}
    
    for name, values in features.items():
        if values.dtype in [np.float64, np.float32, np.int32, np.int64]:
            if method == 'standard':
                mean = np.nanmean(values)
                std = np.nanstd(values)
                if std > 0:
                    normalized[name] = (values - mean) / std
                else:
                    normalized[name] = values - mean
            elif method == 'minmax':
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                if max_val > min_val:
                    normalized[name] = (values - min_val) / (max_val - min_val)
                else:
                    normalized[name] = np.zeros_like(values)
            else:
                normalized[name] = values
        else:
            normalized[name] = values
            
    return normalized

def pad_sequences(sequences: List[np.ndarray], max_length: int = None, 
                 padding_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    序列填充到相同长度
    
    Args:
        sequences: 序列列表
        max_length: 最大长度，None时自动计算
        padding_value: 填充值
        
    Returns:
        填充后的序列矩阵和长度mask
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    n_sequences = len(sequences)
    if len(sequences[0].shape) == 1:
        feature_dim = 1
        padded = np.full((n_sequences, max_length), padding_value)
    else:
        feature_dim = sequences[0].shape[1]
        padded = np.full((n_sequences, max_length, feature_dim), padding_value)
    
    masks = np.zeros((n_sequences, max_length), dtype=bool)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        if len(seq.shape) == 1:
            padded[i, :length] = seq[:length]
        else:
            padded[i, :length, :] = seq[:length, :]
        masks[i, :length] = True
    
    return padded, masks

def aggregate_features(features: Dict[str, np.ndarray], 
                      aggregation_methods: List[str] = ['mean', 'std', 'min', 'max', 'count']) -> Dict[str, float]:
    """
    聚合特征统计
    
    Args:
        features: 特征字典
        aggregation_methods: 聚合方法列表
        
    Returns:
        聚合后的特征字典
    """
    aggregated = {}
    
    for name, values in features.items():
        if values.dtype in [np.float64, np.float32, np.int32, np.int64]:
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                for method in aggregation_methods:
                    if method == 'mean':
                        aggregated[f"{name}_{method}"] = np.mean(valid_values)
                    elif method == 'std':
                        aggregated[f"{name}_{method}"] = np.std(valid_values)
                    elif method == 'min':
                        aggregated[f"{name}_{method}"] = np.min(valid_values)
                    elif method == 'max':
                        aggregated[f"{name}_{method}"] = np.max(valid_values)
                    elif method == 'count':
                        aggregated[f"{name}_{method}"] = len(valid_values)
                    elif method == 'median':
                        aggregated[f"{name}_{method}"] = np.median(valid_values)
            else:
                for method in aggregation_methods:
                    aggregated[f"{name}_{method}"] = 0.0
    
    return aggregated 