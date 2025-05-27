#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态异常检测训练器
整合多模态数据处理和模型训练
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report, confusion_matrix
)
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging # 导入 logging
from collections import Counter # 导入 Counter

warnings.filterwarnings('ignore')

# 导入相关模块
try:
    # 尝试相对导入
    from ..multimodal_pipeline import MultiModalDataPipeline
    from .multimodal_model import MultiModalAnomalyDetector
    from ..config import Config, ModelConfig, TrainingConfig, DataConfig
except ImportError:
    # 如果相对导入失败，添加路径并使用绝对导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from multimodal_pipeline import MultiModalDataPipeline
    from multimodal_model import MultiModalAnomalyDetector
    from config import Config, ModelConfig, TrainingConfig, DataConfig

class MultiModalDataset(Dataset):
    """多模态数据集类"""
    
    def __init__(self, data: Dict[str, Any], model_config: ModelConfig, data_config: DataConfig):
        """
        初始化多模态数据集
        
        Args:
            data: 训练数据字典
            model_config: 模型配置对象
            data_config: 数据配置对象
        """
        self.model_config = model_config
        self.data_config = data_config

        self.labels = data.get('labels', np.array([]))
        self.users = data.get('users', [])

        # Behavior sequences
        self.behavior_sequences = data.get('behavior_sequences', np.array([]))
        # Text content
        self.text_content = data.get('text_content', [])
        # Structured features
        self.structured_features = data.get('structured_features', np.array([]))
        
        # Graph-related features (some global, some per-sample)
        self.node_features = data.get('node_features', np.array([]))
        self.adjacency_matrix = data.get('adjacency_matrix', np.array([]))
        self.user_to_index = data.get('user_to_index', {})
        self.user_indices_in_graph = data.get('user_indices_in_graph', np.array([]))

        if len(self.labels) == 0:
            print("Warning: MultiModalDataset initialized with no labels.")
            # Depending on the use case, you might want to raise an error here
            # or ensure that __len__ returns 0 correctly.
            # For now, length will be 0, which might be handled by DataLoader.

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {}
        
        # Labels (must exist if len > 0)
        if len(self.labels) > 0 :
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            # This case should ideally not be reached if DataLoader uses __len__ correctly.
            # If __len__ is 0, __getitem__ shouldn't be called.
            # However, as a fallback:
            item['labels'] = torch.tensor(-1, dtype=torch.long) # Placeholder label

        # Behavior Sequences
        if hasattr(self.behavior_sequences, 'shape') and self.behavior_sequences.shape[0] > 0:
            item['behavior_sequences'] = torch.tensor(self.behavior_sequences[idx], dtype=torch.float32)
        else:
            seq_len = getattr(self.model_config, 'sequence_length', 128) # Default if not in config
            feat_dim = getattr(self.data_config, 'feature_dim', 256)    # Default if not in config
            item['behavior_sequences'] = torch.zeros((seq_len, feat_dim), dtype=torch.float32)

        # Text Content
        if self.text_content and idx < len(self.text_content): # Check if list is not empty and idx is valid
            item['text_content'] = self.text_content[idx] 
        else:
            item['text_content'] = "" # Placeholder for missing text

        # Structured Features
        if hasattr(self.structured_features, 'shape') and self.structured_features.shape[0] > 0:
            item['structured_features'] = torch.tensor(self.structured_features[idx], dtype=torch.float32)
        else:
            # Ensure structure_feat_dim is an attribute of model_config or provide a sensible default
            struct_dim = getattr(self.model_config, 'structure_feat_dim', 50) # Default if not in config
            item['structured_features'] = torch.zeros(struct_dim, dtype=torch.float32)

        # User indices in graph (per-sample data for GNN batching)
        if hasattr(self.user_indices_in_graph, 'shape') and self.user_indices_in_graph.shape[0] > 0:
            item['batch_user_indices_in_graph'] = torch.tensor(self.user_indices_in_graph[idx], dtype=torch.long)
        else:
            item['batch_user_indices_in_graph'] = torch.tensor(-1, dtype=torch.long) # Placeholder if no graph or user not in graph

        # Global graph features - node_features
        if hasattr(self.node_features, 'shape') and self.node_features.size > 0 and self.node_features.ndim > 1:
            # Ensure it's at least 2D (e.g., num_nodes, num_features)
            # self.node_features comes from data.get('node_features', np.array([]))
            # If it was np.array([]), .size is 0. If it was np.zeros((0, D)), .size is 0.
            # If it was np.zeros((1, D)), .size is D.
            item['node_features'] = torch.tensor(self.node_features, dtype=torch.float32)
        else:
            # Placeholder for node_features. 
            # The GNN input dimension will be determined by create_model based on this.
            # Use a reasonably small default, e.g., model_config.gnn_hidden_dim or a fixed small number.
            # Using a fixed number like 10 for placeholder if gnn_hidden_dim is too large or not set.
            default_node_feat_dim = getattr(self.model_config, 'gnn_input_dim_placeholder', getattr(self.model_config, 'gnn_hidden_dim', 10))
            if default_node_feat_dim <= 0: default_node_feat_dim = 10 # Ensure positive dimension
            item['node_features'] = torch.zeros((1, default_node_feat_dim), dtype=torch.float32) # Min 1 node, N features

        # Global graph features - adjacency_matrix
        if hasattr(self.adjacency_matrix, 'shape') and self.adjacency_matrix.size > 0 and self.adjacency_matrix.ndim > 1:
            item['adjacency_matrix'] = torch.tensor(self.adjacency_matrix, dtype=torch.float32)
        else:
            # Placeholder for adjacency_matrix
            item['adjacency_matrix'] = torch.zeros((1, 1), dtype=torch.float32) # Min 1x1 matrix

        # user_to_index is a dict, not typically returned per item unless specifically needed by model per batch.
        # item['user_to_index'] = self.user_to_index

        return item

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_score: 验证分数
            model: 模型
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False

class MultiModalTrainer:
    """多模态异常检测训练器"""
    
    def __init__(self, config: Config = None, output_dir: str = './outputs'):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            output_dir: 输出目录
        """
        self.config = config or Config()
        self.output_dir = output_dir
        
        # 修改设备选择逻辑
        device_setting = self.config.training.device
        if device_setting == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available(): # 检查 MPS (Apple Silicon GPUs)
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_setting)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练历史
        self.train_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
        }
        
        print(f"多模态训练器初始化完成")
        
        # 修改设备指示的打印方式
        device_type = self.device.type
        if device_type == 'cuda':
            print(f"  设备: {self.device} (已启用 GPU 加速)")
        elif device_type == 'mps':
            print(f"  设备: {self.device} (已启用 Apple Silicon GPU 加速)")
        else:
            print(f"  设备: {self.device} (使用 CPU)")
            
        print(f"  输出目录: {os.path.abspath(output_dir)}")
    
    def prepare_data_loaders(self, training_data: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备数据加载器
        
        Args:
            training_data: 训练数据
            
        Returns:
            训练、验证、测试数据加载器
        """
        logger = logging.getLogger(__name__) # 获取logger实例
        logger.info("准备数据加载器...")
        
        # 创建数据集 (不再传递 device)
        dataset = MultiModalDataset(training_data, self.config.model, self.config.data)
        
        # 划分数据集
        total_size = len(dataset)
        test_size = int(total_size * self.config.training.test_split)
        val_size = int(total_size * self.config.training.val_split)
        train_size = total_size - test_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        logger.info(f"数据集划分: 总计={total_size}, 训练集={train_size}, 验证集={val_size}, 测试集={test_size}")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"数据加载器准备完成:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, sample_data: Dict[str, Any]) -> MultiModalAnomalyDetector:
        """
        创建多模态模型
        
        Args:
            sample_data: 样本数据用于确定输入维度
            
        Returns:
            多模态异常检测模型
        """
        print("创建多模态模型...")
        
        # 获取输入维度 (这些是实际数据维度)
        behavior_seq_dim = sample_data['behavior_sequences'].shape[-1]
        node_feat_dim = sample_data['node_features'].shape[-1]
        struct_feat_dim = sample_data['structured_features'].shape[-1]
        
        # 模型组件的具体配置字典
        # 这些配置将基于 self.config.model (ModelConfig 对象) 和实际输入维度来构建
        
        # Transformer配置
        _transformer_config = {
            'input_dim': behavior_seq_dim, # 实际数据维度
            'hidden_dim': self.config.model.hidden_dim,
            'num_heads': self.config.model.num_heads,
            'num_layers': self.config.model.num_layers,
            'dropout': self.config.model.dropout
        }
        
        # GNN配置  
        _gnn_config = {
            'input_dim': node_feat_dim, # 实际数据维度
            'hidden_dim': self.config.model.gnn_hidden_dim,
            'output_dim': self.config.model.hidden_dim, # GNN输出对齐
            'num_layers': self.config.model.gnn_num_layers,
            'dropout': self.config.model.gnn_dropout
        }
        
        # BERT配置
        _bert_config = {
            'bert_model_name': self.config.model.bert_model_name,
            'max_length': self.config.model.bert_max_length,
            'output_dim': self.config.model.hidden_dim, # BERT输出对齐
            'dropout': self.config.model.dropout
            # BERT input_dim is implicit (vocab size handled by BERTTextEncoder)
        }
        
        # LightGBM配置
        _lgbm_config = { # LightGBMBranch 可能需要不同的参数名
            'input_dim': struct_feat_dim, # 实际数据维度
            'output_dim': self.config.model.hidden_dim, # LGBM输出对齐
            'num_leaves': self.config.model.lgbm_num_leaves, # 从ModelConfig获取
            'max_depth': self.config.model.lgbm_max_depth,
            'learning_rate': self.config.model.lgbm_learning_rate,
            'feature_fraction': self.config.model.lgbm_feature_fraction,
            'dropout': self.config.model.dropout # Assuming LightGBMBranch might use a general dropout passed this way
        }
        
        # 融合配置
        # input_dims for fusion will be set within MultiModalAnomalyDetector based on enabled modalities
        _fusion_config = {
            'embed_dim': self.config.model.hidden_dim,
            'dropout': self.config.model.dropout,
            # 'use_gating' from self.config.model.fusion_type or similar
        }
        
        # 分类头配置
        _head_config = {
                'input_dim': self.config.model.hidden_dim,
            'num_classes': self.config.model.num_classes,
            'dropout': self.config.model.head_dropout
        }
        
        # 创建模型，传入 ModelConfig 对象和各组件的详细配置
        model = MultiModalAnomalyDetector(
            model_config_obj=self.config.model, # 传递 ModelConfig 对象
            transformer_config=_transformer_config,
            gnn_config=_gnn_config,
            bert_config=_bert_config,
            lgbm_config=_lgbm_config,
            fusion_config=_fusion_config,
            head_config=_head_config
            # embed_dim and dropout are now primarily taken from model_config_obj if provided
        )
        model = model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型创建完成:")
        print(f"  总参数数: {total_params:,}")
        print(f"  可训练参数数: {trainable_params:,}")
        print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        return model
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        for batch_idx, batch_cpu in enumerate(train_loader):
            # 将批次中的张量移动到目标设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_cpu.items()}

            # 准备输入数据
            inputs = {
                'behavior_sequences': batch['behavior_sequences'],
                'node_features': batch['node_features'][0],  # 取第一个样本的节点特征（所有样本共享）
                'adjacency_matrix': batch['adjacency_matrix'][0],  # 取第一个样本的邻接矩阵
                'text_content': batch['text_content'],  # 直接使用批处理后的文本列表
                'structured_features': batch['structured_features']
            }
            
            # 添加批处理用户在图中的索引 (如果存在)
            if 'batch_user_indices_in_graph' in batch:
                inputs['batch_user_indices_in_graph'] = batch['batch_user_indices_in_graph']
            
            labels = batch['labels']
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"[DEBUG MMTrain train_epoch] After model call - outputs keys: {list(outputs.keys())}")
            if 'probabilities' in outputs:
                # print(f"[DEBUG MMTrain train_epoch] After model call - outputs['probabilities'] shape: {outputs['probabilities'].shape}")
                pass # 保持结构
            else:
                # print(f"[DEBUG MMTrain train_epoch] After model call - 'probabilities' not in outputs")
                pass # 保持结构
            
            # 计算损失
            loss = criterion(outputs['logits'], labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 预测和标签
            predictions = torch.argmax(outputs['probabilities'], dim=1)
            
            # 调试信息：检查 probabilities 的形状
            # print(f"[DEBUG MultiModalTrainer] Batch {batch_idx}: probabilities shape: {outputs['probabilities'].shape}")
            
            # 安全地获取异常概率
            if outputs['probabilities'].shape[1] > 1:
                probabilities = outputs['probabilities'][:, 1]  # 异常概率
            else:
                # 如果只有一个类别，使用第一个（也是唯一的）概率
                probabilities = outputs['probabilities'][:, 0]
                # print(f"[DEBUG MultiModalTrainer] Warning: probabilities only has 1 class, using [:, 0]")
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.detach().cpu().numpy())
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # 计算指标
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """验证一个epoch"""
        model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # 获取logger实例，用于可能的消融实验特定日志
        logger = logging.getLogger(__name__)
        is_ablation_single_modality_run = len(self.config.model.enabled_modalities) == 1

        with torch.no_grad():
            for batch_cpu in val_loader:
                # 将批次中的张量移动到目标设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch_cpu.items()}

                # 准备输入数据
                inputs = {
                    'behavior_sequences': batch['behavior_sequences'],
                    'node_features': batch['node_features'][0],
                    'adjacency_matrix': batch['adjacency_matrix'][0],
                    'text_content': batch['text_content'], # 直接使用批处理后的文本列表
                    'structured_features': batch['structured_features']
                }
                # 添加批处理用户在图中的索引 (如果存在)
                if 'batch_user_indices_in_graph' in batch:
                    inputs['batch_user_indices_in_graph'] = batch['batch_user_indices_in_graph']
                
                labels = batch['labels']
                
                # 前向传播
                outputs = model(inputs)
                # print(f"[DEBUG MMTrain validate_epoch] After model call - outputs keys: {list(outputs.keys())}")
                if 'probabilities' in outputs:
                    # print(f"[DEBUG MMTrain validate_epoch] After model call - outputs['probabilities'] shape: {outputs['probabilities'].shape}")
                    pass # 保持结构
                else:
                    # print(f"[DEBUG MMTrain validate_epoch] After model call - 'probabilities' not in outputs")
                    pass # 保持结构
                
                loss = criterion(outputs['logits'], labels)
                
                # 统计
                total_loss += loss.item()
                
                # 预测和标签
                predictions = torch.argmax(outputs['probabilities'], dim=1)
                
                # 调试信息：检查 probabilities 的形状
                # print(f"[DEBUG MultiModalTrainer validate] Batch {len(all_predictions)//16}: probabilities shape: {outputs['probabilities'].shape}")
                
                # 安全地获取异常概率
                if outputs['probabilities'].shape[1] > 1:
                    probabilities = outputs['probabilities'][:, 1]
                else:
                    # 如果只有一个类别，使用第一个（也是唯一的）概率
                    probabilities = outputs['probabilities'][:, 0]
                    # print(f"[DEBUG MultiModalTrainer validate] Warning: probabilities only has 1 class, using [:, 0]")
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        if is_ablation_single_modality_run:
            logger.info(f"[Ablation Val] Enabled Modalities: {self.config.model.enabled_modalities}")
            logger.info(f"[Ablation Val] Validation Labels Counter: {Counter(all_labels)}")
            logger.info(f"[Ablation Val] Validation Predictions Counter: {Counter(all_predictions)}")
            logger.info(f"[Ablation Val] Validation F1 (weighted): {f1:.5f}, Accuracy: {accuracy:.5f}")
            
            # 🔧 添加更详细的调试信息
            logger.info(f"[Ablation Val] Unique prediction values: {set(all_predictions)}")
            logger.info(f"[Ablation Val] Probability distribution - Mean: {np.mean(all_probabilities):.4f}, Std: {np.std(all_probabilities):.4f}")
            logger.info(f"[Ablation Val] Probability range: [{np.min(all_probabilities):.4f}, {np.max(all_probabilities):.4f}]")
            
            # 检查是否所有预测都相同
            if len(set(all_predictions)) == 1:
                logger.warning(f"[Ablation Val] ⚠️  所有预测都是相同的值: {all_predictions[0]}！这表明模型没有学到有用的模式。")
            
            # 检查概率分布是否过于集中
            prob_std = np.std(all_probabilities)
            if prob_std < 0.01:
                logger.warning(f"[Ablation Val] ⚠️  概率分布过于集中 (std={prob_std:.6f})，可能表明模型输出缺乏多样性。")

        # 计算AUC（如果有正负样本）
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            auc = 0.0
        
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def train(self, training_data: Dict[str, Any]) -> Tuple[MultiModalAnomalyDetector, Dict[str, float]]:
        """
        完整的训练流程
        
        Args:
            training_data: 训练数据
            
        Returns:
            训练好的模型和最终的测试指标
        """
        print(f"\n{'='*60}")
        print(f"开始多模态异常检测模型训练")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 准备数据加载器
        train_loader, val_loader, test_loader = self.prepare_data_loaders(training_data)
        
        # 创建模型
        sample_batch = next(iter(train_loader))
        model = self.create_model(sample_batch)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        # 早停
        early_stopping = EarlyStopping(
            patience=self.config.training.patience,
            restore_best_weights=True
        )
        
        # 训练循环
        best_val_score = -float('inf')
        
        for epoch in range(self.config.training.num_epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print("-" * 50)
            
            # 训练
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # 验证
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # 学习率调度
            scheduler.step(val_metrics['f1'])
            
            # 记录历史
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['train_acc'].append(train_metrics['accuracy'])
            self.train_history['train_f1'].append(train_metrics['f1'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            self.train_history['val_f1'].append(val_metrics['f1'])
            self.train_history['val_auc'].append(val_metrics['auc'])
            
            # 打印结果
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1} 完成 ({epoch_time:.2f}s)")
            print(f"训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # 保存最佳模型
            val_score = val_metrics['f1']  # 使用F1作为主要指标
            if val_score > best_val_score:
                best_val_score = val_score
                model_path = os.path.join(self.output_dir, 'best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': self.config,
                    'val_score': val_score,
                    'epoch': epoch,
                    'train_history': self.train_history
                }, model_path)
                print(f"保存最佳模型到: {model_path}")
            
            # 早停检查
            if early_stopping(val_score, model):
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        # 最终测试
        print(f"\n{'='*60}")
        print(f"在测试集上评估最佳模型")
        print(f"{'='*60}")
        
        test_metrics = self.validate_epoch(model, test_loader, criterion)
        
        print(f"测试结果:")
        print(f"  准确率: {test_metrics['accuracy']:.4f}")
        print(f"  精确率: {test_metrics['precision']:.4f}")
        print(f"  召回率: {test_metrics['recall']:.4f}")
        print(f"  F1分数: {test_metrics['f1']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        
        # 保存测试结果
        test_results = {
            'test_metrics': test_metrics,
            'train_history': self.train_history,
            'config': self.config.__dict__
        }
        
        results_path = os.path.join(self.output_dir, 'test_results.json')
        
        # Helper function to convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            # Add other type conversions if necessary (e.g., for Config objects)
            elif hasattr(obj, '__dict__'): # For dataclasses like Config sections
                 return {k: convert_numpy_types(v) for k, v in obj.__dict__.items()}
            return obj

        serializable_results = convert_numpy_types(test_results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"训练完成！总耗时: {total_time:.2f} 秒")
        print(f"{'-'*60}")
        
        return model, test_metrics
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss曲线
        axes[0, 0].plot(self.train_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.train_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.train_history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.train_history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1分数曲线
        axes[1, 0].plot(self.train_history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.train_history['val_f1'], label='Val F1')
        axes[1, 0].set_title('F1 Score Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC曲线
        axes[1, 1].plot(self.train_history['val_auc'], label='Val AUC')
        axes[1, 1].set_title('AUC Curve')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线保存到: {os.path.join(self.output_dir, 'training_curves.png')}")
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"混淆矩阵保存到: {os.path.join(self.output_dir, 'confusion_matrix.png')}")

def main():
    """主函数示例"""
    # 创建配置
    config = Config()
    config.training.num_epochs = 20
    config.training.batch_size = 16
    config.training.learning_rate = 1e-4
    
    # 创建多模态数据流水线
    pipeline = MultiModalDataPipeline(
        config=config,
        data_version='r4.2',
        feature_dim=256,
        num_cores=8
    )
    
    # 运行数据处理流水线
    training_data = pipeline.run_full_multimodal_pipeline(
        start_week=0,
        end_week=5,
        max_users=100,
        sequence_length=128
    )
    
    # 创建训练器
    trainer = MultiModalTrainer(config=config, output_dir='./multimodal_outputs')
    
    # 开始训练
    model, test_metrics = trainer.train(training_data)
    
    print("多模态异常检测模型训练完成！")

if __name__ == "__main__":
    main() 