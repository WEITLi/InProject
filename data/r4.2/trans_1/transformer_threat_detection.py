#!/usr/bin/env python
# coding: utf-8

"""
上下文增强的 Transformer 内部威胁检测模型
基于 CERT r4.2 数据集，支持多任务学习和小样本训练

主要特性：
1. 基于 Transformer 的序列建模
2. 用户上下文信息融合
3. 多任务学习（异常检测 + 掩蔽预测）
4. 支持小样本和自监督训练
5. 全面的评估指标和可视化
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, classification_report, confusion_matrix
)

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
from trainer import ThreatDetectionTrainer

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    """设置所有随机种子以确保实验可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    """配置类，包含所有超参数和设置"""
    
    def __init__(self):
        # 数据配置
        self.data_path = None
        self.sequence_length = 30  # 序列长度（天数）
        self.min_sequence_length = 5  # 最小序列长度
        self.test_split = 0.2
        self.val_split = 0.1
        self.random_seed = 42
        
        # 模型配置
        self.hidden_dim = 128
        self.num_layers = 4
        self.num_heads = 8
        self.dropout = 0.1
        self.context_dim = 32  # 上下文特征维度
        self.vocab_size = 1000  # 事件类型词汇表大小
        
        # 训练配置
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.early_stopping_patience = 10
        self.gradient_clip = 1.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 多任务学习权重
        self.classification_weight = 1.0
        self.masked_lm_weight = 0.5
        self.mask_prob = 0.15
        
        # 小样本学习配置
        self.few_shot_samples = None  # None表示使用全部数据
        self.pseudo_label_threshold = 0.9
        self.use_self_training = False
        
        # 输出配置
        self.output_dir = './outputs'
        self.save_model = True
        self.plot_results = True
        self.log_interval = 10
        
    def save_config(self, path):
        """保存配置到JSON文件"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, path):
        """从JSON文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if key == 'device':
                setattr(config, key, torch.device(value))
            else:
                setattr(config, key, value)
        
        return config

def train_model(config):
    """训练模型"""
    print("="*60)
    print("开始训练上下文增强 Transformer 模型")
    print("="*60)
    
    # 创建训练器
    trainer = ThreatDetectionTrainer(config, config.output_dir)
    
    # 开始训练
    model = trainer.train(config.data_path)
    
    print("训练完成！")
    return model

def evaluate_model(config, model_path):
    """评估已训练的模型"""
    print("="*60)
    print("开始评估已训练的模型")
    print("="*60)
    
    # TODO: 实现模型加载和评估
    print("评估功能待实现...")

def run_experiment(config):
    """运行实验（包括不同的配置对比）"""
    print("="*60)
    print("运行对比实验")
    print("="*60)
    
    # 定义不同的实验配置
    experiments = [
        {"name": "baseline", "few_shot_samples": None, "masked_lm_weight": 0.0},
        {"name": "few_shot_100", "few_shot_samples": 100, "masked_lm_weight": 0.0},
        {"name": "few_shot_100_mlm", "few_shot_samples": 100, "masked_lm_weight": 0.5},
        {"name": "full_data_mlm", "few_shot_samples": None, "masked_lm_weight": 0.5},
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n运行实验: {exp['name']}")
        print("-" * 40)
        
        # 复制配置并修改实验参数
        exp_config = Config()
        for key, value in config.__dict__.items():
            setattr(exp_config, key, value)
        
        # 应用实验设置
        for key, value in exp.items():
            if key != 'name':
                setattr(exp_config, key, value)
        
        # 设置输出目录
        exp_config.output_dir = os.path.join(config.output_dir, f"experiment_{exp['name']}")
        
        try:
            # 训练模型
            trainer = ThreatDetectionTrainer(exp_config, exp_config.output_dir)
            model = trainer.train(exp_config.data_path)
            
            # 加载测试结果
            results_path = os.path.join(exp_config.output_dir, 'test_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    exp_results = json.load(f)
                results[exp['name']] = exp_results
                
        except Exception as e:
            print(f"实验 {exp['name']} 失败: {e}")
            results[exp['name']] = None
    
    # 保存对比结果
    comparison_path = os.path.join(config.output_dir, 'experiment_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 打印对比结果
    print("\n" + "="*60)
    print("实验对比结果")
    print("="*60)
    
    for exp_name, exp_results in results.items():
        if exp_results:
            print(f"{exp_name:20s} - AUC: {exp_results['auc']:.4f}, "
                  f"F1: {exp_results['f1']:.4f}, "
                  f"Precision: {exp_results['precision']:.4f}, "
                  f"Recall: {exp_results['recall']:.4f}")
        else:
            print(f"{exp_name:20s} - 失败")

def main():
    """主函数，解析命令行参数并启动训练/评估"""
    parser = argparse.ArgumentParser(description='Transformer-based Insider Threat Detection')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True, 
                       help='路径到预处理的数据文件')
    parser.add_argument('--sequence_length', type=int, default=30,
                       help='输入序列长度（天数）')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Transformer隐藏维度')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    
    # 实验设置
    parser.add_argument('--few_shot_samples', type=int, default=None,
                       help='小样本学习的样本数量')
    parser.add_argument('--use_self_training', action='store_true',
                       help='是否使用自训练')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    
    # 模式选择
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both', 'experiment'], 
                       default='train', help='运行模式')
    parser.add_argument('--model_path', type=str, default=None,
                       help='预训练模型路径（用于评估）')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    # 更新配置
    for key, value in vars(args).items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(config.random_seed)
    
    # 保存配置
    config_path = os.path.join(config.output_dir, 'config.json')
    config.save_config(config_path)
    
    print("="*60)
    print("上下文增强 Transformer 内部威胁检测")
    print("="*60)
    print(f"数据路径: {config.data_path}")
    print(f"序列长度: {config.sequence_length}")
    print(f"隐藏维度: {config.hidden_dim}")
    print(f"设备: {config.device}")
    print(f"输出目录: {config.output_dir}")
    print(f"运行模式: {args.mode}")
    if config.few_shot_samples:
        print(f"小样本学习: {config.few_shot_samples} 样本")
    print("="*60)
    
    # 运行主要流程
    if args.mode == 'train':
        train_model(config)
        
    elif args.mode == 'eval':
        if args.model_path is None:
            raise ValueError("评估模式需要提供 --model_path 参数")
        evaluate_model(config, args.model_path)
        
    elif args.mode == 'both':
        model = train_model(config)
        # 评估刚训练好的模型
        print("\n评估刚训练好的模型...")
        
    elif args.mode == 'experiment':
        run_experiment(config)
    
    print("\n任务完成！")

if __name__ == "__main__":
    main() 