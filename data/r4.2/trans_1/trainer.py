#!/usr/bin/env python
# coding: utf-8

"""
训练和评估模块
支持多任务学习、早停、模型保存和可视化
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report, confusion_matrix
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from data_processor import CERTDataProcessor, ThreatDetectionDataset, collate_fn
from transformer_model import TransformerThreatDetector, ModelConfig

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
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        检查是否应该早停
        
        参数:
        - score: 当前验证分数（越高越好）
        - model: 模型
        
        返回:
        - 是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False

class ThreatDetectionTrainer:
    """威胁检测模型训练器"""
    
    def __init__(self, config, output_dir: str = './outputs'):
        self.config = config
        self.output_dir = output_dir
        self.device = config.device
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练历史
        self.train_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }
        
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备训练、验证和测试数据"""
        print("准备数据...")
        
        # 数据预处理
        processor = CERTDataProcessor(self.config)
        processed_data = processor.process_data(
            data_path, few_shot_samples=self.config.few_shot_samples
        )
        
        sequences = processed_data['sequences']
        contexts = processed_data['contexts']
        labels = processed_data['labels']
        
        # 划分数据集
        train_val_indices, test_indices = train_test_split(
            range(len(sequences)), test_size=self.config.test_split,
            stratify=labels, random_state=self.config.random_seed
        )
        
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=self.config.val_split,
            stratify=[labels[i] for i in train_val_indices],
            random_state=self.config.random_seed
        )
        
        # 创建数据集
        train_dataset = ThreatDetectionDataset(
            [sequences[i] for i in train_indices],
            [contexts[i] for i in train_indices],
            [labels[i] for i in train_indices],
            mask_prob=self.config.mask_prob
        )
        
        val_dataset = ThreatDetectionDataset(
            [sequences[i] for i in val_indices],
            [contexts[i] for i in val_indices],
            [labels[i] for i in val_indices],
            mask_prob=0.0  # 验证时不使用掩蔽
        )
        
        test_dataset = ThreatDetectionDataset(
            [sequences[i] for i in test_indices],
            [contexts[i] for i in test_indices],
            [labels[i] for i in test_indices],
            mask_prob=0.0  # 测试时不使用掩蔽
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        print(f"测试集: {len(test_dataset)} 样本")
        
        # 保存数据统计信息
        self.feature_dim = sequences[0].shape[-1]
        self.context_dim = len(contexts[0])
        
        return train_loader, val_loader, test_loader
    
    def create_model(self) -> TransformerThreatDetector:
        """创建模型"""
        model_config = ModelConfig(
            input_dim=self.feature_dim,
            context_dim=self.context_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            sequence_length=self.config.sequence_length,
            classification_weight=self.config.classification_weight,
            masked_lm_weight=self.config.masked_lm_weight
        )
        
        model = TransformerThreatDetector(model_config)
        model = model.to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        
        total_loss = 0.0
        total_classification_loss = 0.0
        total_mlm_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移到设备
            sequences = batch['sequences'].to(self.device)
            masked_sequences = batch['masked_sequences'].to(self.device)
            contexts = batch['contexts'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            mask_positions = batch['mask_positions']
            
            # 前向传播
            optimizer.zero_grad()
            
            outputs = model(
                sequences=sequences,
                masked_sequences=masked_sequences,
                contexts=contexts,
                attention_mask=attention_mask
            )
            
            # 计算损失
            losses = model.compute_loss(outputs, labels, mask_positions)
            
            # 反向传播
            losses['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            
            optimizer.step()
            
            # 统计
            total_loss += losses['total_loss'].item()
            total_classification_loss += losses['classification_loss'].item()
            if 'mlm_loss' in losses:
                total_mlm_loss += losses['mlm_loss'].item()
            
            # 预测和标签
            predictions = torch.argmax(outputs['classification_logits'], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 打印进度
            if batch_idx % self.config.log_interval == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {losses["total_loss"].item():.4f}')
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)
        avg_mlm_loss = total_mlm_loss / len(train_loader) if total_mlm_loss > 0 else 0.0
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return {
            'loss': avg_loss,
            'classification_loss': avg_classification_loss,
            'mlm_loss': avg_mlm_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移到设备
                sequences = batch['sequences'].to(self.device)
                contexts = batch['contexts'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 前向传播
                outputs = model(
                    sequences=sequences,
                    contexts=contexts,
                    attention_mask=attention_mask
                )
                
                # 计算损失
                losses = model.compute_loss(outputs, labels)
                total_loss += losses['total_loss'].item()
                
                # 预测和概率
                probabilities = torch.softmax(outputs['classification_logits'], dim=1)
                predictions = torch.argmax(outputs['classification_logits'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 异常类别的概率
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probabilities)
            precision = precision_score(all_labels, all_predictions, zero_division=0)
            recall = recall_score(all_labels, all_predictions, zero_division=0)
            f1 = f1_score(all_labels, all_predictions, zero_division=0)
        else:
            auc = precision = recall = f1 = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def train(self, data_path: str) -> TransformerThreatDetector:
        """完整的训练流程"""
        print("开始训练...")
        
        # 准备数据
        train_loader, val_loader, test_loader = self.prepare_data(data_path)
        
        # 创建模型
        model = self.create_model()
        
        # 优化器和调度器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # 早停
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            restore_best_weights=True
        )
        
        # 训练循环
        best_val_score = -float('inf')
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # 验证
            val_metrics = self.validate_epoch(model, val_loader)
            
            # 学习率调度
            scheduler.step(val_metrics['auc'])
            
            # 记录历史
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['train_acc'].append(train_metrics['accuracy'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            self.train_history['val_auc'].append(val_metrics['auc'])
            
            # 打印结果
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} ({epoch_time:.2f}s)")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            val_score = val_metrics['auc']  # 使用AUC作为主要指标
            if val_score > best_val_score:
                best_val_score = val_score
                if self.config.save_model:
                    model_path = os.path.join(self.output_dir, 'best_model.pth')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': self.config,
                        'val_score': val_score,
                        'epoch': epoch
                    }, model_path)
                    print(f"保存最佳模型到: {model_path}")
            
            # 早停检查
            if early_stopping(val_score, model):
                print(f"早停在第 {epoch+1} 轮")
                break
        
        # 最终测试
        print("\n开始最终测试...")
        test_metrics = self.validate_epoch(model, test_loader)
        
        print(f"\n最终测试结果:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"AUC: {test_metrics['auc']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")
        
        # 生成详细报告
        if len(np.unique(test_metrics['labels'])) > 1:
            print(f"\n详细分类报告:")
            print(classification_report(
                test_metrics['labels'], test_metrics['predictions'],
                target_names=['Normal', 'Anomaly']
            ))
        
        # 保存训练历史和测试结果
        self.save_results(test_metrics)
        
        return model
    
    def save_results(self, test_metrics: Dict[str, float]):
        """保存训练结果和可视化"""
        # 保存训练历史
        history_path = os.path.join(self.output_dir, 'training_history.npz')
        np.savez(history_path, **self.train_history)
        
        # 保存测试结果
        results = {
            'accuracy': test_metrics['accuracy'],
            'auc': test_metrics['auc'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1']
        }
        
        results_path = os.path.join(self.output_dir, 'test_results.json')
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # 绘制训练曲线
        if self.config.plot_results:
            self.plot_training_curves()
            self.plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
            self.plot_roc_curve(test_metrics['labels'], test_metrics['probabilities'])
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.train_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(self.train_history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(self.train_history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC曲线
        axes[1, 0].plot(self.train_history['val_auc'], label='Val AUC', color='green')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 移除空白子图
        axes[1, 1].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, labels: List[int], predictions: List[int]):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, labels: List[int], probabilities: List[float]):
        """绘制ROC曲线"""
        if len(np.unique(labels)) <= 1:
            return
        
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'),
                   dpi=300, bbox_inches='tight')
        plt.close() 