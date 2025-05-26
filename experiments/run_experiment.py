#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主实验运行脚本
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import time

# 动态添加包含core_logic的父目录 (feature_extraction_scenario) 到sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_of_core_logic = current_script_dir # feature_extraction_scenario 目录
if parent_of_core_logic not in sys.path:
    sys.path.insert(0, parent_of_core_logic)

from core_logic.multimodal_pipeline import MultiModalDataPipeline
from core_logic.train_pipeline import MultiModalAnomalyDetector, AnomalyDataset, train_model
from core_logic.evaluation_utils import plot_roc_curve, plot_pr_curve, calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score
from core_logic.config import Config

def set_seed(seed_value=42):
    """设置随机种子以保证可复现性"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # 对于MPS (Apple Silicon GPUs)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value) 
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)

    # --- 0. 设置设备 ---
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else 
                              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else 
                              "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- 1. 创建输出目录 ---
    experiment_output_dir = os.path.join(args.output_dir_base, args.experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    model_save_dir = os.path.join(experiment_output_dir, 'models')
    plots_save_dir = os.path.join(experiment_output_dir, 'plots')
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(plots_save_dir, exist_ok=True)
    
    print(f"Experiment output will be saved to: {experiment_output_dir}")

    # --- 2. 数据准备 ("improved" 模式) ---
    print("\nPhase 1: Data Preparation (Improved Mode)")
    # 获取默认配置
    pipeline_config = Config()
    
    # 更新config对象：从命令行参数覆盖
    # 数据相关路径和参数 (使用属性赋值)
    pipeline_config.data.data_version = args.data_version
    pipeline_config.data.source_dir = args.source_data_dir
    pipeline_config.data.work_dir_base = os.path.join(args.output_dir_base, args.experiment_name, "pipeline_workdir")
    pipeline_config.data.start_week = args.start_week_data
    pipeline_config.data.end_week = args.end_week_data
    pipeline_config.data.max_users = args.max_users
    pipeline_config.data.sequence_length = args.sequence_length
    pipeline_config.data.feature_dim = args.feature_dim # Encoder feature dim, used by pipeline
    pipeline_config.data.num_cores = args.num_cores # For CERTDatasetPipeline parallel processing

    # 控制是否强制重新生成 DataByWeek_parquet/* 目录中的文件
    # 当 args.force_regenerate_base_data 为 True 时，会删除已存在的对应 Parquet 目录并从原始CSV重新合并。
    # 这主要影响 sample_ratio=1.0 的情况。对于 sample_ratio < 1.0 的情况，
    # 如果对应的采样数据目录已存在，通常会直接使用，除非此标志也为True。
    # 具体逻辑见 CERTDatasetPipeline.step1_combine_raw_data。
    pipeline_config.data.force_regenerate_combined_weeks = args.force_regenerate_base_data
    pipeline_config.data.force_regenerate_analysis_levels = args.force_regenerate_analysis_levels
    
    # 设置采样比例 (确保命令行优先)
    if args.data_sample_ratio < 1.0:
        print(f"Pipeline data sample_ratio explicitly set to: {args.data_sample_ratio} from command line (overriding any config value).")
        pipeline_config.data.sample_ratio = args.data_sample_ratio
    elif not hasattr(pipeline_config.data, 'sample_ratio') or pipeline_config.data.sample_ratio is None: # 未在命令行指定 (<1.0)，且配置中也没有或为None
        pipeline_config.data.sample_ratio = 1.0 # 默认值为1.0
        print(f"Pipeline data sample_ratio defaulted to: 1.0 (not in cmd line or config).")
    else: # 使用配置中的值 (因为命令行是1.0或者没有指定<1.0的值)
        # No change needed if command line is 1.0 and config value should be kept
        print(f"Pipeline data sample_ratio using value from config: {pipeline_config.data.sample_ratio}")

    # 打印最终的pipeline数据配置以供调试
    print(f"Final Pipeline Config data section: {pipeline_config.data}")

    # 训练相关参数 (也使用属性赋值)
    # 确保 TrainingConfig 中有这些属性
    if hasattr(pipeline_config.training, 'max_users'):
        pipeline_config.training.max_users = args.max_users 
    if hasattr(pipeline_config.training, 'num_cores'): # num_cores 可能更多是数据处理的
        pipeline_config.training.num_cores = args.num_cores 
    if hasattr(pipeline_config.training, 'feature_dim'):
        pipeline_config.training.feature_dim = args.feature_dim

    # 1. 初始化并运行数据流水线
    print(f"\n{'='*30} 1. Initializing Data Pipeline {'='*30}")
    data_pipeline = MultiModalDataPipeline(config=pipeline_config)
    
    # 获取训练数据字典
    # 调用 run_full_multimodal_pipeline 时，直接从 args 传递参数以确保命令行设置优先
    training_data_dict = data_pipeline.run_full_multimodal_pipeline(
        start_week=args.start_week_data,
        end_week=args.end_week_data,
        max_users=args.max_users,
        sequence_length=args.sequence_length
    )
    # 检查返回的数据
    if not training_data_dict or not training_data_dict.get('users'):
        print("Error: Training data dictionary is empty or missing 'users' key.")
        return

    print("Data preparation complete.")

    # --- 3. 划分训练集和验证集 ---
    # 从 all_processed_data 中提取需要的字段，并进行划分
    # 假设我们基于用户进行划分，以确保同一用户的数据不会同时出现在训练集和验证集
    print("\nPhase 2: Data Splitting")
    user_ids = np.array(training_data_dict['users'])
    labels_for_stratify = np.array(training_data_dict['labels'])
    
    # 确保即使只有一个类别，也能进行划分 ( stratify 在单类别时可能无意义)
    try:
        train_users, val_users, _, _ = train_test_split(
            user_ids, 
            labels_for_stratify, # 用于分层抽样
            test_size=args.val_split_ratio, 
            random_state=args.seed,
            stratify=labels_for_stratify if len(np.unique(labels_for_stratify)) > 1 else None
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
        train_users, val_users = train_test_split(
            user_ids, 
            test_size=args.val_split_ratio, 
            random_state=args.seed
        )

    print(f"Total users: {len(user_ids)}, Training users: {len(train_users)}, Validation users: {len(val_users)}")

    def create_data_dict_for_split(original_data, selected_users_list):
        selected_indices = [original_data['user_to_index'][user] for user in selected_users_list]
        
        split_data = {}
        split_data['behavior_sequences'] = original_data['behavior_sequences'][selected_indices]
        split_data['structured_features'] = original_data['structured_features'][selected_indices]
        split_data['labels'] = original_data['labels'][selected_indices]
        split_data['users'] = list(selected_users_list)
        # 文本和图数据目前是全局的，或者需要更复杂的切分逻辑，暂时保持原样或选择性传递
        split_data['text_content'] = [original_data['text_content'][i] for i in selected_indices]
        
        # 对于GNN，图结构通常是固定的，但节点特征需要对应选择的用户
        # 简化处理：如果GNN模型需要用户子图，这里需要额外逻辑
        # 目前的 MultiModalAnomalyDetector 还不直接使用它们，所以可以先传递完整的
        split_data['node_features'] = original_data['node_features'] 
        split_data['adjacency_matrix'] = original_data['adjacency_matrix']
        split_data['user_to_index'] = {user: i for i, user in enumerate(selected_users_list)} # 更新user_to_index
        return split_data

    train_data_dict = create_data_dict_for_split(training_data_dict, train_users)
    val_data_dict = create_data_dict_for_split(training_data_dict, val_users)

    train_dataset = AnomalyDataset(train_data_dict)
    val_dataset = AnomalyDataset(val_data_dict)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.dataloader_workers)
    print("Data loaders created.")

    # --- 4. 模型初始化 ---
    print("\nPhase 3: Model Initialization (Improved Mode)")
    input_dimensions = {
        'behavior_sequences': args.feature_dim, # 来自MultiModalDataPipeline
        'structured_features': training_data_dict['structured_features'].shape[1] if training_data_dict['structured_features'].ndim == 2 else 50
    }
    model = MultiModalAnomalyDetector(
        input_dims=input_dimensions, 
        hidden_dim=args.hidden_dim
    ).to(device)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Model, criterion, and optimizer initialized.")

    # --- 5. 模型训练与评估 ---
    print("\nPhase 4: Model Training and Evaluation")
    start_train_time = time.time()
    trained_model, training_metrics = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        num_epochs=args.epochs, 
        device=device,
        model_save_path=model_save_dir,
        experiment_name=args.experiment_name
    )
    total_training_time = time.time() - start_train_time
    print(f"Total training and validation time: {total_training_time:.2f} seconds")

    # --- 6. 最终评估与结果保存 ---
    print("\nPhase 5: Final Evaluation and Saving Results")
    trained_model.eval()
    final_val_labels = []
    final_val_scores = []
    final_val_preds = []

    with torch.no_grad():
        for batch_data in val_loader:
            behavior_seq = batch_data['behavior_seq'].to(device)
            structured_features = batch_data['structured_features'].to(device)
            labels = batch_data['label'].to(device)
            outputs = trained_model(behavior_seq, structured_features)
            
            final_val_labels.extend(labels.cpu().numpy().flatten())
            final_val_scores.extend(outputs.cpu().numpy().flatten())
            final_val_preds.extend((outputs.cpu().numpy().flatten() > 0.5).astype(int))

    final_val_labels = np.array(final_val_labels)
    final_val_scores = np.array(final_val_scores)
    final_val_preds = np.array(final_val_preds)

    # 计算最终指标
    final_accuracy = calculate_accuracy(final_val_labels, final_val_preds)
    final_precision = calculate_precision(final_val_labels, final_val_preds)
    final_recall = calculate_recall(final_val_labels, final_val_preds)
    final_f1 = calculate_f1_score(final_val_labels, final_val_preds)

    print("\nFinal Validation Metrics on Best Model:")
    print(f"  Accuracy:  {final_accuracy:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall:    {final_recall:.4f}")
    print(f"  F1 Score:  {final_f1:.4f}")

    # 保存图表
    plot_roc_curve(final_val_labels, final_val_scores, 
                     title=f'ROC Curve ({args.experiment_name})', 
                     output_path=os.path.join(plots_save_dir, "final_roc_curve.png"))
    plot_pr_curve(final_val_labels, final_val_scores, 
                    title=f'PR Curve ({args.experiment_name})', 
                    output_path=os.path.join(plots_save_dir, "final_pr_curve.png"))

    # 保存实验总结
    summary = {
        'experiment_name': args.experiment_name,
        'args': vars(args),
        'device_used': str(device),
        'total_training_time_seconds': total_training_time,
        'avg_epoch_train_time_seconds': training_metrics['avg_train_time_epoch'],
        'avg_val_inference_time_seconds': training_metrics['avg_val_inference_time'],
        'best_validation_f1_during_training': training_metrics['best_val_f1'],
        'final_validation_metrics': {
            'accuracy': final_accuracy,
            'precision': final_precision,
            'recall': final_recall,
            'f1_score': final_f1
        },
        'model_path': os.path.join(model_save_dir, f'{args.experiment_name}_best_model.pth')
    }

    summary_file_path = os.path.join(experiment_output_dir, 'experiment_summary.json')
    with open(summary_file_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Experiment summary saved to: {summary_file_path}")
    print("\nExperiment finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Multimodal Anomaly Detection Experiment')
    
    # 数据参数
    parser.add_argument('--data_version', type=str, default='r4.2', help='Dataset version for MultiModalDataPipeline')
    parser.add_argument('--start_week_data', type=int, default=0, help='Start week for data generation pipeline')
    parser.add_argument('--end_week_data', type=int, default=3, help='End week for data generation pipeline (e.g., 3 for 0,1,2 weeks)') # Quick test default
    parser.add_argument('--max_users', type=int, default=100, help='Max users for data pipeline') # Quick test default
    parser.add_argument('--sequence_length', type=int, default=128, help='Sequence length for behavior sequences')
    parser.add_argument('--source_data_dir', type=str, default='../data/r4.2', help="原始CERT数据集的路径")
    parser.add_argument('--work_data_dir', type=str, default='workdata_experiment', help="数据处理的工作目录")
    parser.add_argument('--val_split_ratio', type=float, default=0.2, help="验证集划分比例")
    parser.add_argument('--data_sample_ratio', type=float, default=1.0, help="数据采样比例，用于MultiModalDataPipeline")

    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=256, help="行为序列特征维度") # 这个值需要与 BasePipeline 的 encoder 匹配
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the main model')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataloader_workers', type=int, default=0, help='Number of workers for DataLoader (0 for main process)')

    # 系统参数
    parser.add_argument('--num_cores', type=int, default=8, help='Number of CPU cores for data pipeline parallel tasks')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to use for training')
    parser.add_argument('--force_regenerate_base_data', action='store_true', help='Force regeneration of the base Parquet data (sample_ratio=1.0 or specific sampled data if this flag is set).')
    parser.add_argument('--force_regenerate_analysis_levels', action='store_true', help='Force regeneration of WeekLevelFeatures, DayLevelFeatures, and SessionLevelFeatures CSV files.')

    # 输出参数
    parser.add_argument('--output_dir_base', type=str, default='./experiment_results', help='Base directory to save experiment outputs')
    parser.add_argument('--experiment_name', type=str, default='improved_v1_test', help='Name of this specific experiment run')
    
    # 模式 (预留)
    # parser.add_argument('--experiment_mode', type=str, default='improved', choices=['baseline', 'improved'], help='Experiment mode')

    args = parser.parse_args()
    main(args) 