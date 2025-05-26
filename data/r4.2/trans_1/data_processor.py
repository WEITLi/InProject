#!/usr/bin/env python
# coding: utf-8

"""
数据预处理模块
负责将 CERT r4.2 数据集转换为适合 Transformer 模型的序列格式
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
import pickle
from typing import Dict, List, Tuple, Optional

class CERTDataProcessor:
    """CERT 数据集预处理器"""
    
    def __init__(self, config):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.context_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 特征和上下文信息列名
        self.removed_cols = ['user', 'day', 'week', 'starttime', 'endtime', 
                           'sessionid', 'insider', 'role', 'b_unit', 'f_unit', 
                           'dept', 'team', 'ITAdmin', 'project']
        
        self.context_cols = ['role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin']
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """加载数据文件"""
        print(f"加载数据: {data_path}")
        
        if data_path.endswith('.pkl'):
            data = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            raise ValueError("不支持的文件格式，请使用 .pkl 或 .csv 文件")
        
        print(f"数据形状: {data.shape}")
        print(f"列名: {list(data.columns)[:10]}...")
        
        return data
    
    def extract_user_contexts(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """提取用户上下文信息"""
        print("提取用户上下文信息...")
        
        user_contexts = {}
        
        # 获取每个用户的上下文信息（取第一次出现的值）
        for user in data['user'].unique():
            user_data = data[data['user'] == user].iloc[0]
            
            context_features = []
            
            # 处理分类特征
            for col in self.context_cols:
                if col in data.columns:
                    value = user_data[col]
                    if pd.isna(value):
                        context_features.append(0)  # 缺失值填充为0
                    else:
                        # 简单编码：将字符串转换为哈希值然后归一化
                        if isinstance(value, str):
                            context_features.append(hash(value) % 1000 / 1000.0)
                        else:
                            context_features.append(float(value))
                else:
                    context_features.append(0)
            
            user_contexts[user] = np.array(context_features, dtype=np.float32)
        
        print(f"提取了 {len(user_contexts)} 个用户的上下文信息")
        print(f"上下文特征维度: {len(context_features)}")
        
        return user_contexts
    
    def build_sequences(self, data: pd.DataFrame, user_contexts: Dict[str, np.ndarray]) -> Tuple[List, List, List, List]:
        """构建用户行为序列"""
        print("构建用户行为序列...")
        
        # 提取特征列
        actual_removed_cols = [col for col in self.removed_cols if col in data.columns]
        feature_cols = [col for col in data.columns if col not in actual_removed_cols]
        
        print(f"特征列数量: {len(feature_cols)}")
        
        sequences = []
        contexts = []
        labels = []
        users_list = []
        
        # 确定时间列
        time_col = 'day' if 'day' in data.columns else 'week' if 'week' in data.columns else None
        
        if time_col is None:
            raise ValueError("数据中没有找到时间列（day 或 week）")
        
        # 按用户分组构建序列
        for user in data['user'].unique():
            user_data = data[data['user'] == user].sort_values(time_col)
            
            # 提取特征和标签
            user_features = user_data[feature_cols].values
            user_labels = user_data['insider'].values
            
            # 构建滑动窗口序列
            for i in range(len(user_data) - self.config.sequence_length + 1):
                # 特征序列
                feature_seq = user_features[i:i + self.config.sequence_length]
                
                # 标签（序列中是否包含异常）
                label_seq = user_labels[i:i + self.config.sequence_length]
                has_anomaly = np.any(label_seq > 0)
                
                # 上下文信息
                context = user_contexts.get(user, np.zeros(len(self.context_cols)))
                
                sequences.append(feature_seq)
                contexts.append(context)
                labels.append(1 if has_anomaly else 0)
                users_list.append(user)
        
        print(f"生成序列数量: {len(sequences)}")
        print(f"异常序列数量: {np.sum(labels)} ({100 * np.mean(labels):.2f}%)")
        
        return sequences, contexts, labels, users_list
    
    def normalize_features(self, sequences: List[np.ndarray], contexts: List[np.ndarray], 
                          fit_scaler: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """标准化特征"""
        print("标准化特征...")
        
        # 标准化序列特征
        if fit_scaler:
            # 将所有序列展平用于拟合scaler
            all_features = np.vstack([seq.reshape(-1, seq.shape[-1]) for seq in sequences])
            self.feature_scaler.fit(all_features)
            
            # 标准化上下文特征
            context_matrix = np.array(contexts)
            self.context_scaler.fit(context_matrix)
        
        # 应用标准化
        normalized_sequences = []
        for seq in sequences:
            # 保存原始形状
            original_shape = seq.shape
            # 展平、标准化、恢复形状
            flat_seq = seq.reshape(-1, seq.shape[-1])
            normalized_flat = self.feature_scaler.transform(flat_seq)
            normalized_seq = normalized_flat.reshape(original_shape)
            normalized_sequences.append(normalized_seq)
        
        # 标准化上下文特征
        context_matrix = np.array(contexts)
        normalized_contexts = self.context_scaler.transform(context_matrix)
        normalized_contexts = [ctx for ctx in normalized_contexts]
        
        return normalized_sequences, normalized_contexts
    
    def create_few_shot_dataset(self, sequences: List, contexts: List, 
                               labels: List, users: List, num_samples: int) -> Tuple[List, List, List, List]:
        """创建小样本数据集"""
        print(f"创建小样本数据集，样本数: {num_samples}")
        
        # 按标签分层采样
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        anomaly_indices = [i for i, label in enumerate(labels) if label == 1]
        
        # 计算每类样本数
        num_anomaly = min(len(anomaly_indices), num_samples // 10)  # 10%异常样本
        num_normal = num_samples - num_anomaly
        
        # 随机采样
        np.random.shuffle(normal_indices)
        np.random.shuffle(anomaly_indices)
        
        selected_indices = normal_indices[:num_normal] + anomaly_indices[:num_anomaly]
        np.random.shuffle(selected_indices)
        
        # 提取选中的样本
        few_shot_sequences = [sequences[i] for i in selected_indices]
        few_shot_contexts = [contexts[i] for i in selected_indices]
        few_shot_labels = [labels[i] for i in selected_indices]
        few_shot_users = [users[i] for i in selected_indices]
        
        print(f"小样本数据集: {len(few_shot_sequences)} 个样本")
        print(f"异常样本: {np.sum(few_shot_labels)} ({100 * np.mean(few_shot_labels):.2f}%)")
        
        return few_shot_sequences, few_shot_contexts, few_shot_labels, few_shot_users
    
    def process_data(self, data_path: str, few_shot_samples: Optional[int] = None) -> Dict:
        """完整的数据处理流程"""
        print("开始数据处理流程...")
        
        # 1. 加载数据
        data = self.load_data(data_path)
        
        # 2. 提取用户上下文
        user_contexts = self.extract_user_contexts(data)
        
        # 3. 构建序列
        sequences, contexts, labels, users = self.build_sequences(data, user_contexts)
        
        # 4. 小样本采样（如果需要）
        if few_shot_samples is not None:
            sequences, contexts, labels, users = self.create_few_shot_dataset(
                sequences, contexts, labels, users, few_shot_samples
            )
        
        # 5. 标准化特征
        sequences, contexts = self.normalize_features(sequences, contexts, fit_scaler=True)
        
        return {
            'sequences': sequences,
            'contexts': contexts,
            'labels': labels,
            'users': users,
            'feature_scaler': self.feature_scaler,
            'context_scaler': self.context_scaler
        }

class ThreatDetectionDataset(Dataset):
    """威胁检测数据集类"""
    
    def __init__(self, sequences: List[np.ndarray], contexts: List[np.ndarray], 
                 labels: List[int], mask_prob: float = 0.15):
        self.sequences = [torch.FloatTensor(seq) for seq in sequences]
        self.contexts = [torch.FloatTensor(ctx) for ctx in contexts]
        self.labels = torch.LongTensor(labels)
        self.mask_prob = mask_prob
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        context = self.contexts[idx]
        label = self.labels[idx]
        
        # 创建掩蔽序列用于自监督学习
        masked_sequence = sequence.clone()
        mask_positions = torch.rand(sequence.shape[0]) < self.mask_prob
        
        if mask_positions.any():
            # 用零向量替换被掩蔽的位置
            masked_sequence[mask_positions] = 0
        
        return {
            'sequence': sequence,
            'masked_sequence': masked_sequence,
            'context': context,
            'label': label,
            'mask_positions': mask_positions
        }

def collate_fn(batch):
    """自定义批次整理函数，处理变长序列"""
    sequences = [item['sequence'] for item in batch]
    masked_sequences = [item['masked_sequence'] for item in batch]
    contexts = torch.stack([item['context'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    mask_positions = [item['mask_positions'] for item in batch]
    
    # 填充序列到相同长度
    sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    masked_sequences = torch.nn.utils.rnn.pad_sequence(masked_sequences, batch_first=True, padding_value=0)
    
    # 创建注意力掩码
    attention_mask = (sequences.sum(dim=-1) != 0).float()
    
    return {
        'sequences': sequences,
        'masked_sequences': masked_sequences,
        'contexts': contexts,
        'labels': labels,
        'attention_mask': attention_mask,
        'mask_positions': mask_positions
    } 