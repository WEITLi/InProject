#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态异常检测主训练脚本
整合原有项目框架，支持多种训练模式和配置
"""

import os
import sys
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core_logic'))

try:
    # 尝试从core_logic包导入
    from core_logic.multimodal_pipeline import MultiModalDataPipeline
    from core_logic.train_pipeline_multimodal.multimodal_trainer import MultiModalTrainer
    from core_logic.config import Config, ModelConfig, TrainingConfig, DataConfig
except ImportError:
    # 如果包导入失败，尝试直接导入
    from multimodal_pipeline import MultiModalDataPipeline
    from train_pipeline.multimodal_trainer import MultiModalTrainer
    from config import Config, ModelConfig, TrainingConfig, DataConfig

def set_seed(seed: int = 42):
    """设置随机种子确保实验可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_config_from_args(args) -> Config:
    """从命令行参数创建配置"""
    config = Config()
    
    # 模型配置
    config.model.hidden_dim = args.hidden_dim
    config.model.num_heads = args.num_heads
    config.model.num_layers = args.num_layers
    config.model.sequence_length = args.sequence_length
    config.model.enable_gnn = args.enable_gnn
    config.model.enable_bert = args.enable_bert
    config.model.enable_lgbm = args.enable_lgbm
    config.model.enable_transformer = args.enable_transformer
    
    # 训练配置
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.num_epochs = args.num_epochs
    config.training.patience = args.patience
    config.training.test_split = args.test_split
    config.training.val_split = args.val_split
    config.training.device = args.device
    
    # 数据配置
    config.data.data_version = args.data_version
    config.data.feature_dim = args.feature_dim
    config.data.start_week = args.start_week
    config.data.end_week = args.end_week
    config.data.max_users = args.max_users
    
    # 环境配置
    config.seed = args.seed
    config.num_workers = args.num_workers
    config.output_dir = args.output_dir
    config.experiment_name = args.experiment_name
    config.debug = args.debug
    
    return config

def train_multimodal_model(config: Config) -> Dict[str, any]:
    """
    训练多模态异常检测模型
    
    Args:
        config: 配置对象
        
    Returns:
        训练结果字典
    """
    print(f"\n{'='*80}")
    print(f"开始多模态异常检测模型训练")
    print(f"实验名称: {config.experiment_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config.output_dir, f"{config.experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        # 转换配置为可序列化的字典
        config_dict = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    print(f"实验目录: {experiment_dir}")
    print(f"配置保存到: {config_path}")
    
    try:
        # Step 1: 创建多模态数据流水线
        print(f"\n{'='*60}")
        print(f"Step 1: 初始化多模态数据流水线")
        print(f"{'='*60}")
        
        pipeline = MultiModalDataPipeline(
            config=config,
            data_version=config.data.data_version,
            feature_dim=config.data.feature_dim,
            num_cores=config.num_workers
        )
        
        # Step 2: 运行数据处理流水线
        print(f"\n{'='*60}")
        print(f"Step 2: 运行数据处理流水线")
        print(f"{'='*60}")
        
        training_data = pipeline.run_full_multimodal_pipeline(
            start_week=config.data.start_week,
            end_week=config.data.end_week,
            max_users=config.data.max_users,
            sequence_length=config.model.sequence_length
        )
        
        # 保存训练数据信息
        data_info = {
            'total_samples': len(training_data['labels']),
            'normal_samples': int(np.sum(training_data['labels'] == 0)),
            'anomaly_samples': int(np.sum(training_data['labels'] == 1)),
            'behavior_sequences_shape': training_data['behavior_sequences'].shape,
            'node_features_shape': training_data['node_features'].shape,
            'adjacency_matrix_shape': training_data['adjacency_matrix'].shape,
            'structured_features_shape': training_data['structured_features'].shape,
            'text_samples': len(training_data['text_content'])
        }
        
        data_info_path = os.path.join(experiment_dir, 'data_info.json')
        with open(data_info_path, 'w') as f:
            json.dump(data_info, f, indent=4)
        
        print(f"数据信息保存到: {data_info_path}")
        
        # Step 3: 创建训练器并开始训练
        print(f"\n{'='*60}")
        print(f"Step 3: 创建训练器并开始训练")
        print(f"{'='*60}")
        
        trainer = MultiModalTrainer(config=config, output_dir=experiment_dir)
        model = trainer.train(training_data)
        
        # Step 4: 保存最终结果
        total_time = time.time() - start_time
        
        final_results = {
            'experiment_name': config.experiment_name,
            'experiment_dir': experiment_dir,
            'total_time': total_time,
            'data_info': data_info,
            'config': config_dict,
            'success': True
        }
        
        results_path = os.path.join(experiment_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        
        print(f"\n{'='*80}")
        print(f"多模态异常检测模型训练完成！")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"实验结果保存到: {experiment_dir}")
        print(f"{'='*80}")
        
        return final_results
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        
        # 保存错误信息
        error_results = {
            'experiment_name': config.experiment_name,
            'experiment_dir': experiment_dir,
            'error': str(e),
            'success': False
        }
        
        error_path = os.path.join(experiment_dir, 'error_results.json')
        with open(error_path, 'w') as f:
            json.dump(error_results, f, indent=4)
        
        raise e

def run_experiment_comparison(base_config: Config, experiment_configs: List[Dict]) -> Dict[str, any]:
    """
    运行对比实验
    
    Args:
        base_config: 基础配置
        experiment_configs: 实验配置列表
        
    Returns:
        对比实验结果
    """
    print(f"\n{'='*80}")
    print(f"开始运行对比实验")
    print(f"实验数量: {len(experiment_configs)}")
    print(f"{'='*80}")
    
    comparison_results = {}
    
    for i, exp_config in enumerate(experiment_configs):
        print(f"\n{'='*60}")
        print(f"运行实验 {i+1}/{len(experiment_configs)}: {exp_config['name']}")
        print(f"{'='*60}")
        
        # 复制基础配置
        config = Config()
        config.__dict__.update(base_config.__dict__)
        
        # 应用实验特定配置
        for key, value in exp_config.items():
            if key != 'name':
                if hasattr(config, key):
                    setattr(config, key, value)
                elif hasattr(config.model, key):
                    setattr(config.model, key, value)
                elif hasattr(config.training, key):
                    setattr(config.training, key, value)
                elif hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        # 设置实验名称
        config.experiment_name = exp_config['name']
        
        try:
            # 运行训练
            results = train_multimodal_model(config)
            comparison_results[exp_config['name']] = results
            
        except Exception as e:
            print(f"实验 {exp_config['name']} 失败: {e}")
            comparison_results[exp_config['name']] = {
                'success': False,
                'error': str(e)
            }
    
    # 保存对比结果
    comparison_dir = os.path.join(base_config.output_dir, 'comparison_results')
    os.makedirs(comparison_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(comparison_dir, f'comparison_{timestamp}.json')
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"对比实验完成！")
    print(f"结果保存到: {comparison_path}")
    print(f"{'='*80}")
    
    return comparison_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态异常检测模型训练')
    
    # 数据参数
    parser.add_argument('--data_version', type=str, default='r4.2',
                       help='数据集版本 (默认: r4.2)')
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='特征维度 (默认: 256)')
    parser.add_argument('--start_week', type=int, default=0,
                       help='开始周数 (默认: 0)')
    parser.add_argument('--end_week', type=int, default=None,
                       help='结束周数 (默认: None, 使用全部数据)')
    parser.add_argument('--max_users', type=int, default=None,
                       help='最大用户数 (默认: None, 使用全部用户)')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度 (默认: 256)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数 (默认: 8)')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Transformer层数 (默认: 6)')
    parser.add_argument('--sequence_length', type=int, default=128,
                       help='序列长度 (默认: 128)')
    
    # 模块启用控制
    parser.add_argument('--enable_gnn', action='store_true', default=True,
                       help='启用GNN用户图嵌入')
    parser.add_argument('--enable_bert', action='store_true', default=True,
                       help='启用BERT文本编码')
    parser.add_argument('--enable_lgbm', action='store_true', default=True,
                       help='启用LightGBM结构化特征')
    parser.add_argument('--enable_transformer', action='store_true', default=True,
                       help='启用Transformer序列建模')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小 (默认: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率 (默认: 1e-4)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停patience (默认: 10)')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='测试集比例 (默认: 0.2)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='验证集比例 (默认: 0.2)')
    
    # 环境参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备类型 (默认: cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数 (默认: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录 (默认: ./outputs)')
    parser.add_argument('--experiment_name', type=str, default='multimodal_anomaly_detection',
                       help='实验名称 (默认: multimodal_anomaly_detection)')
    
    # 运行模式
    parser.add_argument('--mode', type=str, choices=['train', 'experiment', 'comparison'], 
                       default='train', help='运行模式 (默认: train)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='配置文件路径 (可选)')
    
    # 调试模式
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='快速开发模式（只运行少量数据）')
    
    args = parser.parse_args()
    
    # 快速开发模式设置
    if args.fast_dev_run:
        args.end_week = 3
        args.max_users = 50
        args.num_epochs = 5
        args.debug = True
        print("🚀 快速开发模式启用")
    
    # 从配置文件加载（如果提供）
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        print(f"从配置文件加载: {args.config_file}")
    
    # 创建配置
    config = create_config_from_args(args)
    
    print(f"\n{'='*80}")
    print(f"多模态异常检测模型训练")
    print(f"{'='*80}")
    print(f"运行模式: {args.mode}")
    print(f"数据版本: {config.data.data_version}")
    print(f"特征维度: {config.data.feature_dim}")
    print(f"隐藏维度: {config.model.hidden_dim}")
    print(f"设备: {config.training.device}")
    print(f"输出目录: {config.output_dir}")
    if args.debug:
        print("🐛 调试模式启用")
    print(f"{'='*80}")
    
    # 根据模式运行
    if args.mode == 'train':
        # 单次训练
        results = train_multimodal_model(config)
        
    elif args.mode == 'experiment':
        # 运行预定义实验
        experiment_configs = [
            {'name': 'baseline', 'enable_gnn': False, 'enable_bert': False, 'enable_lgbm': False},
            {'name': 'transformer_only', 'enable_gnn': False, 'enable_bert': False, 'enable_lgbm': False},
            {'name': 'transformer_gnn', 'enable_bert': False, 'enable_lgbm': False},
            {'name': 'transformer_bert', 'enable_gnn': False, 'enable_lgbm': False},
            {'name': 'transformer_lgbm', 'enable_gnn': False, 'enable_bert': False},
            {'name': 'full_multimodal', 'enable_gnn': True, 'enable_bert': True, 'enable_lgbm': True}
        ]
        
        results = run_experiment_comparison(config, experiment_configs)
        
    elif args.mode == 'comparison':
        # 运行超参数对比实验
        comparison_configs = [
            {'name': 'small_model', 'hidden_dim': 128, 'num_layers': 4},
            {'name': 'medium_model', 'hidden_dim': 256, 'num_layers': 6},
            {'name': 'large_model', 'hidden_dim': 512, 'num_layers': 8},
        ]
        
        results = run_experiment_comparison(config, comparison_configs)
    
    print(f"\n🎉 所有任务完成！")

if __name__ == "__main__":
    main() 