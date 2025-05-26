#!/usr/bin/env python3
"""LightGBM Branch for Structured Features Processing"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

class LightGBMBranch(nn.Module):
    """
    简化的LightGBM分支
    处理结构化特征，如用户属性、设备信息、时间特征等
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        output_dim: int = 128,
        dropout: float = 0.1,
        # LightGBM specific parameters
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        n_estimators: int = 100, # Added n_estimators for lgb.train
        feature_fraction: float = 0.9,
        bagging_fraction: float = 0.8, # Added bagging_fraction
        bagging_freq: int = 5 # Added bagging_freq
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        
        # Store LightGBM parameters
        self.lgbm_params = {
            'objective': 'multiclass', # Or 'binary' if only 2 classes in a different context
            'metric': 'multi_logloss', # Or 'binary_logloss'
            'num_class': output_dim, # Treating output_dim as num_classes for feature transformation
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42
        }
        self.n_estimators = n_estimators
        
        self.model = None  # LightGBM model will be trained in forward pass or separately
        self.fc = nn.Linear(input_dim, output_dim) # Fallback or initial projection
        self.dropout = nn.Dropout(dropout)
        self.is_fitted = False
        
        # 特征处理网络
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, features: Union[torch.Tensor, pd.DataFrame]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 结构化特征
            
        Returns:
            structure_embeddings: [batch_size, output_dim] 结构化特征嵌入
        """
        # 转换为张量
        if isinstance(features, pd.DataFrame):
            # 简单的数值化处理
            X = features.copy()
            for col in X.columns:
                if X[col].dtype == 'object':
                    # 简单的标签编码
                    unique_vals = X[col].unique()
                    mapping = {val: i for i, val in enumerate(unique_vals)}
                    X[col] = X[col].map(mapping)
            
            X = X.fillna(0).values.astype(np.float32)
            features_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            features_tensor = features.float()
        
        # 调整维度以匹配input_dim
        batch_size = features_tensor.shape[0]
        current_dim = features_tensor.shape[1]
        
        if current_dim != self.input_dim:
            if current_dim > self.input_dim:
                # 截断
                features_tensor = features_tensor[:, :self.input_dim]
            else:
                # 填充
                padding = torch.zeros(batch_size, self.input_dim - current_dim)
                features_tensor = torch.cat([features_tensor, padding], dim=1)
        
        # 特征处理
        output = self.feature_processor(features_tensor)
        
        return output

class StructuredFeatureExtractor:
    """结构化特征提取器"""
    
    def __init__(self):
        self.user_features = [
            'dept', 'role', 'seniority', 'ITAdmin', 'login_frequency',
            'avg_session_length', 'off_hours_access', 'weekend_access'
        ]
        
        self.temporal_features = [
            'hour', 'day_of_week', 'is_weekend', 'is_holiday',
            'time_since_last_login', 'session_duration'
        ]
        
        self.behavioral_features = [
            'email_count', 'file_access_count', 'network_access_count',
            'failed_login_attempts', 'suspicious_activity_score'
        ]
    
    def extract_features(self, data: Dict) -> pd.DataFrame:
        """提取结构化特征"""
        features = {}
        
        # 用户特征
        for feature in self.user_features:
            features[feature] = data.get(feature, 0)
        
        # 时间特征
        for feature in self.temporal_features:
            features[feature] = data.get(feature, 0)
        
        # 行为特征
        for feature in self.behavioral_features:
            features[feature] = data.get(feature, 0)
        
        return pd.DataFrame([features])

def test_lightgbm_branch():
    """测试LightGBM分支"""
    print("🧪 Testing LightGBM Branch...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 100
    n_features = 15
    
    # 创建数值特征
    numerical_data = np.random.randn(n_samples, n_features - 3)
    
    # 创建分类特征
    categorical_data = {
        'dept': np.random.choice(['IT', 'Finance', 'HR', 'Marketing'], n_samples),
        'role': np.random.choice(['Employee', 'Manager', 'Director'], n_samples),
        'is_admin': np.random.choice([0, 1], n_samples)
    }
    
    # 合并为DataFrame
    df = pd.DataFrame(numerical_data, columns=[f'feature_{i}' for i in range(numerical_data.shape[1])])
    for col, values in categorical_data.items():
        df[col] = values
    
    print(f"  输入数据形状: {df.shape}")
    print(f"  特征列: {list(df.columns)}")
    
    # 创建模型
    model = LightGBMBranch(
        input_dim=20,  # 设置较大的输入维度
        output_dim=64
    )
    
    # 前向传播
    with torch.no_grad():
        embeddings = model(df)
    
    print(f"  输出嵌入形状: {embeddings.shape}")
    print(f"  嵌入均值: {embeddings.mean().item():.4f}")
    print(f"  嵌入标准差: {embeddings.std().item():.4f}")
    
    # 测试梯度
    model.train()
    embeddings = model(df)
    loss = embeddings.sum()
    loss.backward()
    
    print("  ✅ LightGBM Branch 测试通过")
    
    # 测试特征提取器
    print("\n🧪 Testing Structured Feature Extractor...")
    
    extractor = StructuredFeatureExtractor()
    sample_data = {
        'dept': 'IT',
        'role': 'Manager',
        'seniority': 5,
        'ITAdmin': 1,
        'login_frequency': 10.5,
        'hour': 14,
        'day_of_week': 2,
        'email_count': 25
    }
    
    extracted_features = extractor.extract_features(sample_data)
    print(f"  提取特征形状: {extracted_features.shape}")
    print(f"  特征列: {list(extracted_features.columns)}")
    
    # 测试提取的特征
    with torch.no_grad():
        embeddings = model(extracted_features)
    
    print(f"  单样本嵌入形状: {embeddings.shape}")
    print("  ✅ Structured Feature Extractor 测试通过")

if __name__ == "__main__":
    test_lightgbm_branch() 