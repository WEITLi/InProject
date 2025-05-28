#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态内部威胁检测系统 - 主实验控制脚本
Main Experiment Controller for Multimodal Internal Threat Detection System

支持的实验类型：
1. baseline - 传统ML方法对比实验 (RandomForest vs XGBoost vs 多模态)
2. tune - 超参数优化实验 (使用Optuna)
3. ablation - 消融实验 (不同模态组合)
4. imbalance - 数据不平衡适应性实验
5. realtime - 实时检测实验 (待实现)
6. generalization - 模型泛化能力评估实验
"""

import os
import sys
import copy

# --- Start of sys.path modification for robust imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__)) # .../experiments
parent_dir = os.path.dirname(current_script_dir) # .../InProject

# Add parent_dir to sys.path so 'experiments' can be treated as a package
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Clean up: remove current_script_dir itself if it was added, to avoid confusion
if current_script_dir in sys.path:
    sys.path.remove(current_script_dir)
# --- End of sys.path modification ---

import time
import argparse
import logging
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd #确保导入pandas

# 导入核心模块 (relative to parent_dir)
try:
    try:
        from experiments.core_logic.config import Config, ModelConfig, TrainingConfig, DataConfig
        from experiments.core_logic.multimodal_pipeline import MultiModalDataPipeline
        from experiments.core_logic.train_pipeline_multimodal import MultiModalTrainer
        # from experiments.core_logic.evaluation_utils import ExperimentEvaluator
    except ImportError as e:
        # print(f"⚠️ 核心模块未找到，使用模拟模式: {e}")
        class Config: 
            def __init__(self):
                self.data = type('DataConfig', (), { 'max_users': 20, 'max_weeks': 2, 'sample_ratio': 1.0, 'sequence_length': 10, 'num_cores': 1, 'data_version': 'mock', 'feature_dim': 8 })()
                self.training = type('TrainingConfig', (), { 'epochs': 1, 'batch_size': 4, 'learning_rate': 0.001, 'device': 'cpu' })()
                self.model = type('ModelConfig', (), { 'hidden_dim': 32, 'num_heads': 2, 'num_layers': 1, 'dropout': 0.1, 'enabled_modalities': ['behavior', 'structured'], 'sequence_length': 10 })()
                self.n_trials = 2
                self.seed = 42
        
        class MultiModalDataPipeline:
            def __init__(self, config): self.config = config
            def run_base_feature_extraction(self, **kwargs): print("Mock: Running base feature extraction...")
            def prepare_training_data(self, **kwargs): 
                print("Mock: Preparing training data for baseline...")
                num_samples = self.config.data.max_users
                if num_samples >= 2:
                    labels = [0]*(num_samples // 2) + [1]*(num_samples - num_samples // 2)
                elif num_samples == 1:
                    labels = [0]
                else:
                    labels = []
                np.random.shuffle(labels)
                print(f"Mock labels generated: {labels} (Total: {len(labels)}, Users: {num_samples})")
                return {
                    'labels': np.array(labels),
                    'users': [f'mock_user_{i}' for i in range(num_samples)],
                    'behavior_sequences': np.random.rand(num_samples, self.config.model.sequence_length, 8) if num_samples > 0 else np.array([]),
                    'structured_features': np.random.rand(num_samples, 5) if num_samples > 0 else np.array([]),
                    'node_features': np.random.rand(num_samples, 10) if num_samples > 0 else np.array([]),
                    'text_content': [f"mock text {i}" for i in range(num_samples)] if num_samples > 0 else []
                    }
        
        class MultiModalTrainer:
            def __init__(self, config, output_dir): 
                self.config = config
                self.output_dir = output_dir
                self.train_history = {'val_f1': [0.5+np.random.rand()*0.1], 'val_auc': [0.6+np.random.rand()*0.1], 'train_loss': [0.5], 'val_loss': [0.5]}
            def train(self, data): 
                print("Mock: Training MultiModal model...")
                num_test_samples = 0
                if 'labels' in data and hasattr(data['labels'], '__len__'):
                    num_test_samples = len(data['labels']) // 5 
                    if num_test_samples == 0 and len(data['labels']) > 0:
                        num_test_samples = 1
                
                mock_y_true = np.random.randint(0, 2, num_test_samples) if num_test_samples > 0 else np.array([])
                # Ensure mock_y_pred_proba has shape (num_test_samples, 2) for binary classification
                mock_y_pred_proba = np.random.rand(num_test_samples, 2) if num_test_samples > 0 else np.array([])
                if num_test_samples > 0 and mock_y_pred_proba.shape[0] > 0 : # Normalize probabilities to sum to 1 per sample
                    mock_y_pred_proba = mock_y_pred_proba / np.sum(mock_y_pred_proba, axis=1, keepdims=True)


                mock_test_metrics = {
                    'f1': 0.55 + np.random.rand() * 0.1 if num_test_samples > 0 else 0.0,
                    'auc': 0.65 + np.random.rand() * 0.1 if num_test_samples > 0 else 0.0,
                    'precision': 0.5 + np.random.rand() * 0.1 if num_test_samples > 0 else 0.0,
                    'recall': 0.6 + np.random.rand() * 0.1 if num_test_samples > 0 else 0.0,
                    'accuracy': 0.6 + np.random.rand() * 0.1 if num_test_samples > 0 else 0.0,
                    'fpr': 0.2 + np.random.rand() * 0.1 if num_test_samples > 0 else 0.0, # Mock FPR
                    'y_true': mock_y_true,
                    'y_pred_proba': mock_y_pred_proba
                }
                return self, mock_test_metrics
    
    # 导入工具模块 (relative to parent_dir)
    from experiments.utils.wandb_utils import init_wandb
    from experiments.utils.baseline_models import BaselineModelTrainer, run_baseline_comparison
    from experiments.utils.improved_baseline_models import run_improved_baseline_comparison  # 新增改进版baseline
    from experiments.utils.imbalance_utils import run_imbalance_experiment as utils_run_imbalance_experiment, ImbalanceHandler
    from experiments.utils.optuna_tuning import run_optuna_tuning, get_multimodal_search_space
    
except ImportError as e:
    print(f"❌ 最终导入模块失败: {e}. 请检查PYTHONPATH和脚本位置。")
    print("   PYTHONPATH: ", os.environ.get('PYTHONPATH'))
    print("   当前 sys.path:")
    for i, p in enumerate(sys.path):
        print(f"     {i}: {p}")
    print(f"   预期父目录 (InProject): {parent_dir}")
    print(f"   当前脚本目录 (experiments): {current_script_dir}")
    sys.exit(1)

def setup_logging(output_dir: str, experiment_name: str) -> logging.Logger:
    """
    设置日志配置
    
    Args:
        output_dir: 输出目录
        experiment_name: 实验名称
        
    Returns:
        配置好的logger
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件路径
    log_file = os.path.join(output_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('MainExperiment')
    logger.info(f"实验日志已配置: {log_file}")
    
    return logger

def load_config(config_file: Optional[str] = None, **kwargs) -> Config:
    """
    加载实验配置
    
    Args:
        config_file: 配置文件路径 (JSON/YAML)
        **kwargs: 命令行参数覆盖 (通常来自 argparse)
        
    Returns:
        配置对象
    """
    # 创建默认配置
    config = Config()
    
    # argparse 的默认值，用于判断命令行参数是否被用户显式设置
    # 注意：如果 argparse 的默认值改变，这里也需要更新
    argparse_defaults = {
        'max_users': 100,
        'data_version': 'r4.2',
        'sample_ratio': 1.0,
        'epochs': 3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': 'auto',
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'n_trials': 20,
        'num_cores': 8,
        'seed': 42,
        'use_improved_baseline': False,
        'baseline_cv_folds': 5
    }

    file_values_loaded = {}

    # 如果提供了配置文件，加载并合并
    if config_file and os.path.exists(config_file):
        print(f"📄 加载配置文件: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.endswith('.json'):
                file_config = json.load(f)
            elif config_file.endswith( ('.yaml', '.yml')):
                file_config = yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_file}")
        
        # 合并配置
        for section_name, section_values in file_config.items():
            if hasattr(config, section_name) and isinstance(section_values, dict):
                section_obj = getattr(config, section_name)
                for key, value in section_values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        file_values_loaded[key] = value # 记录从文件加载的值
                # 特殊处理：如果是training配置且设置了epochs，同步num_epochs
                if section_name == 'training' and hasattr(section_obj, '__post_init__'):
                    section_obj.__post_init__()
            elif section_name in ['n_trials', 'seed', 'num_workers']: # 处理直接在Config根上的属性
                 if hasattr(config, section_name):
                    setattr(config, section_name, section_values)
                    file_values_loaded[section_name] = section_values # 包括 num_workers
            elif key in ['use_improved_baseline', 'baseline_cv_folds']: # 改进版baseline参数
                target_section_obj = config
            else:
                # 对于配置文件中存在，但Config类中完全没有对应属性或节的键，可以选择忽略或警告
                pass

    # 应用命令行参数覆盖
    for key, value_from_cmd in kwargs.items():
        if value_from_cmd is not None:
            is_default_argparse_value = (key in argparse_defaults and value_from_cmd == argparse_defaults[key])
            
            target_section_obj = None
            target_key_in_section = key

            if key in ['max_users', 'data_version', 'feature_dim', 'sample_ratio', 'sequence_length']: # num_cores is already in data section
                target_section_obj = config.data
            elif key in ['epochs', 'batch_size', 'learning_rate', 'device', 'test_split', 'val_split']:
                target_section_obj = config.training
            elif key in ['hidden_dim', 'num_heads', 'num_layers', 'dropout', 'enabled_modalities', 
                         'gnn_hidden_dim', 'gnn_num_layers', 'gnn_dropout', 
                         'bert_model_name', 'bert_max_length', 
                         'lgbm_num_leaves', 'lgbm_max_depth', 'lgbm_learning_rate', 'lgbm_feature_fraction',
                         'fusion_type', 'num_classes', 'head_dropout']:
                target_section_obj = config.model
            elif key in ['n_trials', 'seed', 'num_workers']: # 直接在Config根上的属性，包括 num_workers
                target_section_obj = config
            elif key in ['use_improved_baseline', 'baseline_cv_folds']: # 改进版baseline参数
                target_section_obj = config
            else:
                if hasattr(config, key):
                    if not (is_default_argparse_value and key in file_values_loaded):
                        setattr(config, key, value_from_cmd)
                continue

            if target_section_obj is not None and hasattr(target_section_obj, target_key_in_section):
                # 如果命令行参数是默认值 且 文件中已加载该值，则不覆盖
                if not (is_default_argparse_value and target_key_in_section in file_values_loaded):
                    setattr(target_section_obj, target_key_in_section, value_from_cmd)
                    # 特殊处理：如果设置了epochs，同步num_epochs
                    if target_key_in_section == 'epochs' and hasattr(target_section_obj, '__post_init__'):
                        target_section_obj.__post_init__()
            elif hasattr(config, key): # 再次检查根级别，以防上面未匹配但根级别存在
                if not (is_default_argparse_value and key in file_values_loaded):
                     setattr(config, key, value_from_cmd)
    
    # data.num_cores 的特殊处理：确保命令行 --num_cores 优先于文件中的 data.num_cores
    # 并确保 config.num_cores (用于DataLoader) 与 config.data.num_cores (用于数据流水线) 可以独立设置或同步
    # 当前的命令行 --num_cores 会通过上面的逻辑直接设置到 config.num_workers (如果 Config 类有 num_workers 属性)
    # 或 config.num_cores (如果 Config 类有 num_cores 属性且它被映射到那里)。

    # 当前逻辑会将 --num_cores (命令行) 设置到 config.num_cores (根级别)
    # 如果YAML中 data: num_cores: X 存在, 它会设置到 config.data.num_cores
    # 如果YAML中 num_workers: Y 存在, 它会设置到 config.num_workers (根级别)
    
    # 确保 config.data.num_cores 可以被命令行 --num_cores 覆盖，如果用户期望这种行为
    # 鉴于 `--num_cores` 的帮助文本是 "CPU核心数"，它可能同时指代数据处理和数据加载
    # 这里我们假设命令行的 --num_cores 主要目标是 config.num_cores (或我们将其视为 config.num_workers)
    # 而文件中的 data.num_cores 控制数据预处理的核心数。

    # 如果您希望命令行的 --num_cores 同时覆盖 config.num_workers 和 config.data.num_cores:
    # if 'num_cores' in kwargs and kwargs['num_cores'] is not None:
    #     num_cores_cmd_val = kwargs['num_cores']
    #     is_default_nc = ('num_cores' in argparse_defaults and num_cores_cmd_val == argparse_defaults['num_cores'])
    #     if not (is_default_nc and 'num_cores' in file_values_loaded): # 检查根 num_cores
    #         config.num_workers = num_cores_cmd_val # 假设 config.num_workers 是DataLoader用的
    #     if not (is_default_nc and 'num_cores' in file_values_loaded.get('data', {})):
    #         config.data.num_cores = num_cores_cmd_val # 数据预处理用的

    # 确保 config.num_workers (DataLoader用的) 和 config.data.num_cores (Pipeline用的) 被正确设置
    # 命令行 --num_cores 应该更新 config.num_workers (假设Config类中有此属性 for DataLoader)
    # YAML 中根级别的 num_workers 更新 config.num_workers
    # YAML 中 data.num_cores 更新 config.data.num_cores

    # 整理：
    # 1. YAML 根 num_workers -> config.num_workers (已由上面代码处理)
    # 2. YAML data.num_cores -> config.data.num_cores (已由上面代码处理)
    # 3. 命令行 --num_cores -> config.num_workers (如果 Config 顶级属性是 num_workers)
    #                         或 config.num_cores (如果 Config 顶级属性是 num_cores)
    # 假设我们希望 `--num_cores` 更新用于 DataLoader 的 worker 数量，即 `config.num_workers`。
    # 而 `config.data.num_cores` 用于数据流水线中的并行处理。

    # 如果 Config 类直接有 num_workers 属性 (推荐用于 DataLoader)
    if 'num_cores' in kwargs and hasattr(config, 'num_workers'):
        cmd_val = kwargs['num_cores']
        if cmd_val is not None:
            is_default_argparse_val_for_nc = ('num_cores' in argparse_defaults and cmd_val == argparse_defaults['num_cores'])
            if not (is_default_argparse_val_for_nc and 'num_workers' in file_values_loaded): # 比较时用 num_workers 因为这是目标
                config.num_workers = cmd_val
    
    # 如果 Config 类直接有 num_cores 属性 (作为 DataLoader worker 数的旧方式或通用CPU数)
    # 并且也希望命令行 --num_cores 更新它 (覆盖文件中的根级别 num_cores)
    elif 'num_cores' in kwargs and hasattr(config, 'num_cores'):
        cmd_val = kwargs['num_cores']
        if cmd_val is not None:
            is_default_argparse_val_for_nc = ('num_cores' in argparse_defaults and cmd_val == argparse_defaults['num_cores'])
            if not (is_default_argparse_val_for_nc and 'num_cores' in file_values_loaded): # 比较时用 num_cores
                config.num_cores = cmd_val

    return config

def run_baseline_experiment(config: Config, output_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    运行基线方法对比实验
    比较传统机器学习方法与多模态模型的检测性能
    
    Args:
        config: 实验配置
        output_dir: 输出目录
        logger: 日志器
        
    Returns:
        实验结果
    """
    logger.info("🔬 开始基线方法对比实验")
    
    # 初始化WandB
    wandb_logger = init_wandb(
        project_name="threat_detection_experiments",
        experiment_type="baseline",
        model_type="comparison",
        config=config.__dict__,
        tags=["baseline", "comparison", "traditional_ml", "multimodal"]
    )
    
    try:
        # 创建数据流水线
        pipeline = MultiModalDataPipeline(config=config)
        
        # 运行基础特征提取
        pipeline.run_base_feature_extraction(
            start_week=0,
            end_week=config.data.max_weeks,
            max_users=config.data.max_users,
            sample_ratio=config.data.sample_ratio
        )
        
        # 准备训练数据
        training_data = pipeline.prepare_training_data(
            start_week=0,
            end_week=config.data.max_weeks,
            max_users=config.data.max_users,
            sequence_length=config.model.sequence_length
        )
        
        results = {'experiment_type': 'baseline', 'models': {}}
        
        # 1. 传统机器学习基线
        logger.info("🔧 训练传统机器学习基线模型...")
        
        # 检查是否使用改进版baseline模型
        use_improved_baseline = getattr(config, 'use_improved_baseline', False)
        
        if use_improved_baseline:
            logger.info("🔧 使用改进版基线模型...")
            traditional_results = run_improved_baseline_comparison(
                training_data, 
                output_dir,
                models=["random_forest", "xgboost"],
                cv_folds=getattr(config, 'baseline_cv_folds', 5)
            )
            
            # 记录改进版传统ML结果到WandB
            for model_type, model_results in traditional_results.items():
                if 'error' not in model_results and 'cv_results' in model_results:
                    cv_results = model_results['cv_results']
                    wandb_logger.log_metrics({
                        f"{model_type}_cv_f1_mean": cv_results.get('f1_test_mean', 0.0),
                        f"{model_type}_cv_f1_std": cv_results.get('f1_test_std', 0.0),
                        f"{model_type}_cv_auc_mean": cv_results.get('roc_auc_test_mean', 0.0),
                        f"{model_type}_cv_auc_std": cv_results.get('roc_auc_test_std', 0.0),
                        f"{model_type}_cv_precision_mean": cv_results.get('precision_test_mean', 0.0),
                        f"{model_type}_cv_recall_mean": cv_results.get('recall_test_mean', 0.0),
                        f"{model_type}_cv_pr_auc_mean": cv_results.get('average_precision_test_mean', 0.0),
                        f"{model_type}_n_features": model_results.get('n_features', 0)
                    })
                    
                    # 记录特征重要性
                    if 'feature_importance' in model_results and 'feature_names' in model_results:
                        wandb_logger.log_feature_importance(
                            model_results['feature_names'],
                            model_results['feature_importance'],
                            title=f"{model_type.title()} Feature Importance (Improved)"
                        )
        else:
            logger.info("📊 使用原始基线模型...")
            traditional_results = run_baseline_comparison(
                training_data, 
                output_dir,
                models=["random_forest", "xgboost"]
            )
            
            # 记录传统ML结果到WandB
            for model_type, model_results in traditional_results.items():
                if 'error' not in model_results and 'test_metrics' in model_results:
                    tm = model_results['test_metrics']
                    wandb_logger.log_metrics({
                        f"{model_type}_test_f1": tm.get('f1', 0.0),
                        f"{model_type}_test_auc": tm.get('auc', 0.0),
                        f"{model_type}_test_accuracy": tm.get('accuracy', 0.0),
                        f"{model_type}_test_precision": tm.get('precision', 0.0),
                        f"{model_type}_test_recall": tm.get('recall', 0.0),
                        f"{model_type}_test_fpr": tm.get('fpr', 0.0) # 假设fpr已计算
                    })
                    
                    # 记录特征重要性
                    if 'feature_importance' in model_results and 'feature_names' in model_results:
                        wandb_logger.log_feature_importance(
                            model_results['feature_names'],
                            model_results['feature_importance'],
                            title=f"{model_type.title()} Feature Importance"
                        )
                    
                    # 记录ROC曲线 (如果数据存在)
                    if 'y_true' in tm and 'y_pred_proba' in tm and \
                       isinstance(tm['y_true'], np.ndarray) and isinstance(tm['y_pred_proba'], np.ndarray) and \
                       len(tm['y_true']) > 0 and len(tm['y_pred_proba']) > 0 and \
                       tm['y_pred_proba'].ndim == 2 and tm['y_pred_proba'].shape[1] >=2: # Ensure proba has at least 2 classes for ROC
                        try:
                            wandb_logger.log_roc_curve(tm['y_true'], tm['y_pred_proba'], title=f"{model_type.title()} ROC Curve")
                        except Exception as e_roc:
                            logger.warning(f"WandB: 绘制 {model_type} ROC曲线失败: {e_roc}")
        
        results['models'].update(traditional_results)
        
        # 2. 多模态模型
        logger.info("🧠 训练多模态模型...")
        multimodal_trainer = MultiModalTrainer(config=config, output_dir=output_dir)
        multimodal_model, multimodal_test_metrics = multimodal_trainer.train(training_data)
        
        # 获取多模态模型结果
        multimodal_results = {
            'model_type': 'multimodal',
            'train_history': multimodal_trainer.train_history,
            'model_path': os.path.join(output_dir, 'best_model.pth'),
            'test_metrics': multimodal_test_metrics
        }
        
        # 记录多模态结果到WandB
        if multimodal_trainer.train_history:
            history = multimodal_trainer.train_history
            if 'val_f1' in history and len(history['val_f1']) > 0:
                best_val_f1 = max(history['val_f1'])
                wandb_logger.log_metrics({
                    'multimodal_best_val_f1': best_val_f1,
                    'multimodal_final_train_loss': history.get('train_loss', [0])[-1],
                    'multimodal_final_val_loss': history.get('val_loss', [0])[-1]
                })
            
            # 记录训练曲线
            wandb_logger.log_training_curves(history, "Multimodal Training Curves")

        if multimodal_test_metrics: # 记录测试集指标
            mm_tm = multimodal_test_metrics
            wandb_logger.log_metrics({
                f"multimodal_test_f1": mm_tm.get('f1', 0.0),
                f"multimodal_test_auc": mm_tm.get('auc', 0.0),
                f"multimodal_test_accuracy": mm_tm.get('accuracy', 0.0),
                f"multimodal_test_precision": mm_tm.get('precision', 0.0),
                f"multimodal_test_recall": mm_tm.get('recall', 0.0),
                f"multimodal_test_fpr": mm_tm.get('fpr', 0.0)
            })
            # 记录ROC曲线 (如果数据存在)
            if 'y_true' in mm_tm and 'y_pred_proba' in mm_tm and \
               isinstance(mm_tm['y_true'], np.ndarray) and isinstance(mm_tm['y_pred_proba'], np.ndarray) and \
               len(mm_tm['y_true']) > 0 and len(mm_tm['y_pred_proba']) > 0 and \
               mm_tm['y_pred_proba'].ndim == 2 and mm_tm['y_pred_proba'].shape[1] >=2:
                try:
                    wandb_logger.log_roc_curve(mm_tm['y_true'], mm_tm['y_pred_proba'], title="Multimodal ROC Curve")
                except Exception as e_roc_mm:
                    logger.warning(f"WandB: 绘制多模态ROC曲线失败: {e_roc_mm}")

        results['models']['multimodal'] = multimodal_results
        
        # 3. 性能对比总结
        logger.info("📊 生成性能对比报告...")
        comparison_summary = {}
        
        for model_name, model_result in results['models'].items():
            if 'error' not in model_result:
                metrics_to_log = {}
                if model_name == 'multimodal':
                    # 多模态模型使用测试集分数 (如果存在)，否则回退到验证集分数
                    test_metrics = model_result.get('test_metrics', {})
                    if test_metrics: # 优先使用测试集指标
                        metrics_to_log['f1_score'] = test_metrics.get('f1', 0.0)
                        metrics_to_log['auc_score'] = test_metrics.get('auc', 0.0)
                        metrics_to_log['precision'] = test_metrics.get('precision', 0.0)
                        metrics_to_log['recall'] = test_metrics.get('recall', 0.0)
                        metrics_to_log['accuracy'] = test_metrics.get('accuracy', 0.0)
                        metrics_to_log['fpr'] = test_metrics.get('fpr', 0.0) 
                    else: # 回退到验证集指标 (主要用于F1和AUC)
                        history = model_result.get('train_history', {})
                        metrics_to_log['f1_score'] = max(history.get('val_f1', [0.0])) if history.get('val_f1') else 0.0
                        metrics_to_log['auc_score'] = max(history.get('val_auc', [0.0])) if history.get('val_auc') else 0.0
                        # 对于验证集，通常不直接计算precision/recall/fpr，除非显式提供
                        metrics_to_log['precision'] = 0.0 
                        metrics_to_log['recall'] = 0.0
                        metrics_to_log['accuracy'] = 0.0
                        metrics_to_log['fpr'] = 0.0
                elif 'cv_results' in model_result:
                    # 改进版基线模型使用交叉验证结果
                    cv_results = model_result['cv_results']
                    metrics_to_log['f1_score'] = cv_results.get('f1_test_mean', 0.0)
                    metrics_to_log['auc_score'] = cv_results.get('roc_auc_test_mean', 0.0)
                    metrics_to_log['precision'] = cv_results.get('precision_test_mean', 0.0)
                    metrics_to_log['recall'] = cv_results.get('recall_test_mean', 0.0)
                    metrics_to_log['accuracy'] = cv_results.get('accuracy_test_mean', 0.0)
                    metrics_to_log['fpr'] = 0.0  # 改进版模型暂不计算FPR
                    metrics_to_log['pr_auc'] = cv_results.get('average_precision_test_mean', 0.0)
                    metrics_to_log['n_features'] = model_result.get('n_features', 0)
                    metrics_to_log['model_type'] = 'improved_baseline'
                else:
                    # 传统ML模型使用测试分数
                    test_metrics = model_result.get('test_metrics', {})
                    metrics_to_log['f1_score'] = test_metrics.get('f1', 0.0)
                    metrics_to_log['auc_score'] = test_metrics.get('auc', 0.0)
                    metrics_to_log['precision'] = test_metrics.get('precision', 0.0)
                    metrics_to_log['recall'] = test_metrics.get('recall', 0.0)
                    metrics_to_log['accuracy'] = test_metrics.get('accuracy', 0.0)
                    metrics_to_log['fpr'] = test_metrics.get('fpr', 0.0)
                    metrics_to_log['model_type'] = 'original_baseline'
                
                comparison_summary[model_name] = metrics_to_log
        
        results['comparison_summary'] = comparison_summary
        
        # 记录对比结果到WandB (这里保留F1，但可扩展)
        model_names_for_chart = list(comparison_summary.keys())
        f1_scores_for_chart = [comparison_summary[name]['f1_score'] for name in model_names_for_chart]
        
        wandb_logger.log_ablation_results(model_names_for_chart, f1_scores_for_chart)
        # 可以添加更多图表，例如 AUC comparison_summary[name]['auc_score'] 等
        
        logger.info("✅ 基线对比实验完成")
        logger.info("📈 模型性能对比:")
        for model_name, metrics_dict in comparison_summary.items():
            model_type_info = metrics_dict.get('model_type', 'unknown')
            log_str = f"   {model_name} ({model_type_info}): "
            log_str += f"F1={metrics_dict.get('f1_score', 0.0):.4f}, "
            log_str += f"AUC={metrics_dict.get('auc_score', 0.0):.4f}, "
            log_str += f"Precision={metrics_dict.get('precision', 0.0):.4f}, "
            log_str += f"Recall={metrics_dict.get('recall', 0.0):.4f}, "
            log_str += f"Accuracy={metrics_dict.get('accuracy', 0.0):.4f}"
            
            # 为改进版baseline添加额外信息
            if model_type_info == 'improved_baseline':
                log_str += f", PR-AUC={metrics_dict.get('pr_auc', 0.0):.4f}"
                log_str += f", Features={metrics_dict.get('n_features', 0)}"
            elif model_type_info == 'original_baseline':
                log_str += f", FPR={metrics_dict.get('fpr', 0.0):.4f}"
            
            logger.info(log_str)
        
        return results
        
    finally:
        wandb_logger.finish()

def run_tune_experiment(config: Config, output_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    运行超参数优化实验 (使用Optuna)
    
    Args:
        config: 实验配置
        output_dir: 输出目录
        logger: 日志器
        
    Returns:
        实验结果
    """
    logger.info("🎯 开始超参数优化实验")
    
    # 初始化WandB
    wandb_logger = init_wandb(
        project_name="threat_detection_experiments",
        experiment_type="tuning",
        model_type="multimodal",
        config=config.__dict__,
        tags=["tuning", "optuna", "hyperparameter_optimization"]
    )
    
    try:
        # 准备数据
        pipeline = MultiModalDataPipeline(config=config)
        pipeline.run_base_feature_extraction(
            start_week=0,
            end_week=config.data.max_weeks,
            max_users=config.data.max_users,
            sample_ratio=config.data.sample_ratio
        )
        
        training_data = pipeline.prepare_training_data(
            start_week=0,
            end_week=config.data.max_weeks,
            max_users=config.data.max_users,
            sequence_length=config.model.sequence_length
        )
        
        # 创建训练器
        trainer = MultiModalTrainer(config=config, output_dir=output_dir)
        
        # 定义搜索空间
        search_space = get_multimodal_search_space()
        
        # 运行Optuna优化
        logger.info("🔍 开始Optuna超参数搜索...")
        tuning_results = run_optuna_tuning(
            model_type="multimodal",
            data=training_data,
            trainer=trainer,
            output_dir=output_dir,
            n_trials=getattr(config, 'n_trials', 20),  # 默认20次试验
            wandb_logger=wandb_logger
        )
        
        # 使用最佳参数重新训练
        logger.info("🏆 使用最佳参数重新训练模型...")
        best_params = tuning_results['best_params']
        
        # 更新配置
        for param_name, param_value in best_params.items():
            if param_name in ['learning_rate', 'batch_size', 'weight_decay']:
                setattr(config.training, param_name, param_value)
            elif param_name in ['hidden_dim', 'num_heads', 'num_layers', 'dropout']:
                setattr(config.model, param_name, param_value)
        
        # 重新训练最佳模型
        best_trainer = MultiModalTrainer(config=config, output_dir=os.path.join(output_dir, 'best_model'))
        best_model, best_test_metrics = best_trainer.train(training_data)
        
        results = {
            'experiment_type': 'tuning',
            'tuning_results': tuning_results,
            'best_params': best_params,
            'best_score': tuning_results['best_value'],
            'final_model_path': os.path.join(output_dir, 'best_model', 'best_model.pth'),
            'test_metrics': best_test_metrics
        }
        
        # 记录最终结果到WandB
        wandb_logger.log_metrics({
            'best_hyperopt_score': tuning_results['best_value'],
            'total_trials': tuning_results['n_trials']
        })
        
        logger.info("✅ 超参数优化实验完成")
        logger.info(f"🏆 最佳参数: {best_params}")
        logger.info(f"🎯 最佳分数: {tuning_results['best_value']:.4f}")
        
        return results
        
    finally:
        wandb_logger.finish()

def run_ablation_experiment(config: Config, output_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    运行消融实验
    研究多模态模型中各分支模块的独立贡献
    
    Args:
        config: 实验配置
        output_dir: 输出目录
        logger: 日志器
        
    Returns:
        实验结果
    """
    logger.info("🔬 开始消融实验")
    
    # 初始化WandB
    wandb_logger = init_wandb(
        project_name="threat_detection_experiments",
        experiment_type="ablation",
        model_type="multimodal",
        config=config.__dict__,
        tags=["ablation", "modality_analysis", "feature_contribution"]
    )
    
    try:
        # 定义消融实验的模态组合
        modality_combinations = [
            {'name': 'behavior_only', 'modalities': ['behavior']},
            {'name': 'graph_only', 'modalities': ['graph']},
            {'name': 'text_only', 'modalities': ['text']},
            {'name': 'structured_only', 'modalities': ['structured']},
            {'name': 'behavior_graph', 'modalities': ['behavior', 'graph']},
            {'name': 'behavior_text', 'modalities': ['behavior', 'text']},
            {'name': 'behavior_structured', 'modalities': ['behavior', 'structured']},
            {'name': 'all_modalities', 'modalities': ['behavior', 'graph', 'text', 'structured']}
        ]
        
        results = {'experiment_type': 'ablation', 'combinations': {}}
        combination_f1_scores = []
        combination_names = []
        
        for combination in modality_combinations:
            logger.info(f"🧪 测试模态组合: {combination['name']}")
            
            # 为每个组合创建子目录
            combo_output_dir = os.path.join(output_dir, combination['name'])
            os.makedirs(combo_output_dir, exist_ok=True)
            
            # 修改配置以只使用特定模态
            combo_config = copy.deepcopy(config)
            combo_config.model.enabled_modalities = combination['modalities']
            
            try:
                # 创建数据流水线
                pipeline = MultiModalDataPipeline(config=combo_config)
                pipeline.run_base_feature_extraction(
                    start_week=0,
                    end_week=combo_config.data.max_weeks,
                    max_users=combo_config.data.max_users,
                    sample_ratio=combo_config.data.sample_ratio
                )
                
                training_data = pipeline.prepare_training_data(
                    start_week=0,
                    end_week=combo_config.data.max_weeks,
                    max_users=combo_config.data.max_users,
                    sequence_length=combo_config.model.sequence_length
                )
                
                # 训练模型
                trainer = MultiModalTrainer(config=combo_config, output_dir=combo_output_dir)
                model, test_metrics = trainer.train(training_data)
                
                # 获取最佳验证F1分数
                train_history = trainer.train_history
                best_f1 = max(train_history.get('val_f1', [0.0])) if train_history.get('val_f1') else 0.0
                
                combo_result = {
                    'modalities': combination['modalities'],
                    'best_val_f1': best_f1,
                    'train_history': train_history,
                    'model_path': os.path.join(combo_output_dir, 'best_model.pth'),
                    'test_metrics': test_metrics
                }
                
                results['combinations'][combination['name']] = combo_result
                combination_f1_scores.append(best_f1)
                combination_names.append(combination['name'])
                
                # 记录到WandB
                wandb_logger.log_metrics({
                    f"{combination['name']}_f1": best_f1,
                    f"{combination['name']}_num_modalities": len(combination['modalities'])
                })
                
                logger.info(f"✅ {combination['name']} 完成 - F1: {best_f1:.4f}")
                
            except Exception as e:
                logger.error(f"❌ 模态组合 {combination['name']} 实验失败: {e}")
                results['combinations'][combination['name']] = {'error': str(e)}
                combination_f1_scores.append(0.0)
                combination_names.append(combination['name'])
        
        # 记录消融结果到WandB
        wandb_logger.log_ablation_results(combination_names, combination_f1_scores)
        
        # 分析结果
        results['analysis'] = {
            'best_combination': combination_names[combination_f1_scores.index(max(combination_f1_scores))],
            'best_f1_score': max(combination_f1_scores),
            'modality_contribution': dict(zip(combination_names, combination_f1_scores))
        }
        
        logger.info("✅ 消融实验完成")
        logger.info("📊 模态组合性能排序:")
        sorted_combinations = sorted(zip(combination_names, combination_f1_scores), 
                                   key=lambda x: x[1], reverse=True)
        for name, score in sorted_combinations:
            logger.info(f"   {name}: F1={score:.4f}")
        
        return results
        
    finally:
        wandb_logger.finish()

def run_imbalance_experiment(config: Config, output_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    运行数据不平衡适应性实验
    评估模型在不同程度的数据失衡下的鲁棒性表现
    
    Args:
        config: 实验配置
        output_dir: 输出目录
        logger: 日志器
        
    Returns:
        实验结果
    """
    logger.info("⚖️ 开始数据不平衡适应性实验")
    
    # 初始化WandB
    wandb_logger = init_wandb(
        project_name="threat_detection_experiments",
        experiment_type="imbalance",
        model_type="analysis",
        config=config.__dict__,
        tags=["imbalance", "robustness", "sampling_strategies"]
    )
    
    try:
        # 准备数据
        pipeline = MultiModalDataPipeline(config=config)
        pipeline.run_base_feature_extraction(
            start_week=0,
            end_week=config.data.max_weeks,
            max_users=config.data.max_users,
            sample_ratio=config.data.sample_ratio
        )
        
        training_data = pipeline.prepare_training_data(
            start_week=0,
            end_week=config.data.max_weeks,
            max_users=config.data.max_users,
            sequence_length=config.model.sequence_length
        )
        
        # 创建基线模型训练器用于不平衡实验
        baseline_trainer = BaselineModelTrainer(model_type="random_forest")
        
        # 定义不平衡比例和采样策略
        ratios = [1.0, 2.0, 3.0, 4.0, 5.0]  # 正常:恶意比例
        sampling_strategies = ['none', 'smote', 'adasyn', 'random_undersample']
        
        # 运行不平衡实验
        imbalance_results = utils_run_imbalance_experiment(
            training_data,
            output_dir,
            baseline_trainer,
            ratios=ratios,
            sampling_strategies=sampling_strategies
        )
        
        # 记录结果到WandB
        strategy_comparison = imbalance_results.get('strategy_comparison', {})
        
        for strategy, metrics in strategy_comparison.items():
            wandb_logger.log_metrics({
                f"{strategy}_avg_f1": metrics['avg_f1'],
                f"{strategy}_avg_auc": metrics['avg_auc']
            })
        
        # 生成不平衡分析图表
        if strategy_comparison:
            # 选择最佳策略的结果进行可视化
            best_strategy = max(strategy_comparison.keys(), 
                              key=lambda x: strategy_comparison[x]['avg_f1'])
            best_metrics = strategy_comparison[best_strategy]
            
            wandb_logger.log_imbalance_analysis(
                ratios,
                best_metrics['f1_scores'],
                best_metrics['auc_scores']
            )
        
        results = {
            'experiment_type': 'imbalance',
            'imbalance_results': imbalance_results,
            'best_strategy': max(strategy_comparison.keys(), 
                               key=lambda x: strategy_comparison[x]['avg_f1']) if strategy_comparison else None
        }
        
        logger.info("✅ 数据不平衡实验完成")
        if strategy_comparison:
            logger.info("📈 采样策略性能对比:")
            for strategy, metrics in strategy_comparison.items():
                logger.info(f"   {strategy}: 平均F1={metrics['avg_f1']:.4f}, 平均AUC={metrics['avg_auc']:.4f}")
        
        return results
        
    finally:
        wandb_logger.finish()

def run_realtime_experiment(config: Config, output_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    运行实时检测实验
    
    Args:
        config: 实验配置
        output_dir: 输出目录
        logger: 日志器
        
    Returns:
        实验结果
    """
    logger.info("⚡ 开始实时检测实验")
    
    # TODO: 实现实时检测逻辑
    # 这里应该包括：
    # 1. 模拟实时数据流
    # 2. 增量学习
    # 3. 在线检测
    # 4. 性能监控
    
    results = {
        'experiment_type': 'realtime',
        'status': 'not_implemented',
        'message': '实时检测实验尚未实现'
    }
    
    logger.warning("⚠️ 实时检测实验尚未实现")
    return results

def run_generalization_eval(config: Config, output_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    运行泛化能力评估实验
    在不同用户子集下重复训练和验证，观测模型性能是否稳定。

    Args:
        config: 实验配置 (gen_config.yaml)
        output_dir: 输出目录
        logger: 日志器

    Returns:
        实验结果
    """
    logger.info("🧬 开始泛化能力评估实验")

    num_runs = getattr(config.generalization, 'num_gen_runs', 5) # 从配置中获取运行次数，默认为5
    base_seed = getattr(config.generalization, 'base_seed', 42)  # 基础种子，默认为42
    
    fixed_max_users = 200
    fixed_sample_ratio = 1.0

    all_run_metrics = []
    wandb_project_name = "threat_detection_experiments" # 与prompt一致

    for i in range(num_runs):
        current_seed = base_seed + i
        run_name = f"gen_eval_seed_{current_seed}"
        logger.info(f"\n--- 泛化实验轮次 {i+1}/{num_runs} (种子: {current_seed}) ---")

        # 更新配置以用于当前轮次
        current_run_config = Config() # 创建新的Config实例以避免修改原始config对象
        current_run_config.__dict__.update(config.__dict__) # 深拷贝或选择性拷贝可能更好
        
        # 更新DataConfig部分
        current_run_config.data.max_users = fixed_max_users
        current_run_config.data.sample_ratio = fixed_sample_ratio
        current_run_config.seed = current_seed # 设置全局种子，会被流水线使用
        
        # MultiModalDataPipeline的 __init__ 会从 config.data 读取 num_cores, data_version, feature_dim
        # 也会从 config 读取 seed

        # WandB 初始化
        wandb_run_logger = init_wandb(
            project_name=wandb_project_name,
            experiment_type="generalization", # group
            model_type="multimodal", # 可以是固定的，或者从config读取
            config={**current_run_config.__dict__, "experiment_name": "generalization", "random_seed": current_seed, "max_users": fixed_max_users, "run_iteration": i+1}, # 附加字段
            tags=["generalization", f"seed_{current_seed}"],
            run_name_override=run_name # 使用特定名称
        )

        try:
            logger.info(f"  使用配置: max_users={current_run_config.data.max_users}, sample_ratio={current_run_config.data.sample_ratio}, seed={current_run_config.seed}")
            
            # 1. 数据准备
            pipeline = MultiModalDataPipeline(config=current_run_config) # MMP会使用config中的seed来初始化CERTDatasetPipeline
            
            # 注意: run_base_feature_extraction 的 max_users 和 sample_ratio 参数会覆盖 CERTDatasetPipeline 中
            # 通过 config 设置的同名参数。为了让泛化实验的种子和用户/采样率设置生效，
            # 我们需要确保 CERTDatasetPipeline 在创建时就拿到了正确的种子，
            # 并且 run_base_feature_extraction 调用时不传递 max_users 和 sample_ratio (或传递None)，
            # 这样它就会使用 CERTDatasetPipeline 实例内部已经通过config设置好的值。
            # 或者，直接在这里传递正确的值。
            # 当前 MultiModalDataPipeline.__init__ 会将 config.data.max_users 等传递给 CERTDatasetPipeline
            # 所以 CERTDatasetPipeline 应该已经有了正确的 max_users 和 sample_ratio (如果它们在 config.data 中设置)。
            # 然而，泛化实验要求 max_users=200, sample_ratio=1.0，这些值已在 current_run_config.data 中设置。
            
            # MultiModalDataPipeline.run_base_feature_extraction 现在会从 self.config.data.sample_ratio 读取采样率
            # 它传递给 CERTDatasetPipeline.run_full_pipeline 的 max_users 是 None (因为原始调用中是None)，
            # sample_ratio 是从 config.data.sample_ratio 来的。
            # CERTDatasetPipeline.run_full_pipeline 中的 max_users 和 sample_ratio 如果为None，则不会覆盖实例内的值。
            # 为了确保 CERTDatasetPipeline 使用我们为当前轮次设置的 fixed_max_users 和 fixed_sample_ratio，
            # 最好的方式是确保 CERTDatasetPipeline 初始化时就从 config.data 接收到这些值。
            # 而 current_run_config.data 已经被正确设置了。

            pipeline.run_base_feature_extraction(
                start_week=0, # 或从config读取
                end_week=current_run_config.data.max_weeks, # 或从config读取
                # max_users 和 sample_ratio 不在这里传递，让 CERTDatasetPipeline 使用其内部通过 config 设置的值
            )
            
            training_data = pipeline.prepare_training_data(
                start_week=0, # 或从config读取
                end_week=current_run_config.data.max_weeks, # 或从config读取
                max_users=current_run_config.data.max_users, # 确保 MultiModalDataPipeline 也使用正确的 max_users
                sequence_length=current_run_config.model.sequence_length
            )

            if not training_data or len(training_data.get('labels', [])) == 0:
                logger.warning(f"轮次 {i+1} (种子 {current_seed}): 未生成训练数据，跳过此轮。")
                all_run_metrics.append({'seed': current_seed, 'error': 'No training data'})
                wandb_run_logger.finish()
                continue
            
            # 检查数据集中是否至少有两个类别
            unique_labels = np.unique(training_data['labels'])
            if len(unique_labels) < 2:
                logger.warning(f"轮次 {i+1} (种子 {current_seed}): 训练数据只包含一个类别 ({unique_labels})，无法进行有意义的训练/评估。跳过此轮。")
                all_run_metrics.append({'seed': current_seed, 'error': 'Single class in training data'})
                wandb_run_logger.finish()
                continue

            # 2. 模型训练与评估 (只训练多模态模型)
            logger.info(f"  轮次 {i+1}: 训练多模态模型...")
            run_output_dir = os.path.join(output_dir, f"run_{i+1}_seed_{current_seed}")
            os.makedirs(run_output_dir, exist_ok=True)
            
            # 确保 MultiModalTrainer 也使用当前轮次的种子，以便其内部的train/val/test划分是一致的（如果它也用了随机种子）
            # MultiModalTrainer 的 __init__ 接收 config，它内部的 _prepare_dataloaders 会使用 config.seed
            multimodal_trainer = MultiModalTrainer(config=current_run_config, output_dir=run_output_dir)
            # 修改：接收模型和测试指标
            trained_model, final_test_metrics = multimodal_trainer.train(training_data)

            # 收集指标 - 直接使用 final_test_metrics
            # final_test_metrics 字典的键如 'f1', 'auc' 等，而不是 'test_f1'
            current_run_results = {
                'seed': current_seed,
                'f1_score': final_test_metrics.get('f1', np.nan),
                'auc': final_test_metrics.get('auc', np.nan),
                'precision': final_test_metrics.get('precision', np.nan),
                'recall': final_test_metrics.get('recall', np.nan),
                'accuracy': final_test_metrics.get('accuracy', np.nan),
                'error': None
            }
            all_run_metrics.append(current_run_results)

            # 记录到WandB
            wandb_run_logger.log_metrics({
                'generalization_test_f1': current_run_results['f1_score'],
                'generalization_test_auc': current_run_results['auc'],
                'generalization_test_precision': current_run_results['precision'],
                'generalization_test_recall': current_run_results['recall'],
                'generalization_test_accuracy': current_run_results['accuracy']
            })
            logger.info(f"  轮次 {i+1} 结果: F1={current_run_results['f1_score']:.4f}, AUC={current_run_results['auc']:.4f}")

        except Exception as e:
            logger.error(f"❌ 泛化实验轮次 {i+1} (种子 {current_seed}) 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_run_metrics.append({'seed': current_seed, 'f1_score': np.nan, 'auc': np.nan, 'precision': np.nan, 'recall': np.nan, 'accuracy': np.nan, 'error': str(e)})
        finally:
            wandb_run_logger.finish()

    # 结果聚合与保存
    results_df = pd.DataFrame(all_run_metrics)
    summary_stats = results_df[['f1_score', 'auc', 'precision', 'recall', 'accuracy']].agg(['mean', 'std']).reset_index()

    logger.info("\n--- 泛化实验总结 ---")
    logger.info("详细轮次结果:")
    logger.info(results_df.to_string())
    logger.info("\n性能指标统计 (平均值 ± 标准差):")
    for col in ['f1_score', 'auc', 'precision', 'recall', 'accuracy']:
        mean_val = results_df[col].mean()
        std_val = results_df[col].std()
        logger.info(f"  {col.replace('_', ' ').title()}: {mean_val:.4f} ± {std_val:.4f}")

    # 保存结果
    summary_file_path = os.path.join(output_dir, "gen_eval_summary.csv")
    results_df.to_csv(summary_file_path, index=False)
    logger.info(f"详细结果已保存到: {summary_file_path}")
    
    summary_stats_path = os.path.join(output_dir, "gen_eval_mean_std.csv")
    summary_stats.to_csv(summary_stats_path, index=False)
    logger.info(f"统计摘要已保存到: {summary_stats_path}")

    # 可选：生成箱线图 (需要 matplotlib 和 seaborn)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=results_df['f1_score'])
        plt.title('F1 Score Distribution (Generalization Runs)')
        plt.ylabel('F1 Score')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=results_df['auc'])
        plt.title('AUC Distribution (Generalization Runs)')
        plt.ylabel('AUC')
        
        boxplot_path = os.path.join(output_dir, "gen_eval_f1_auc_boxplot.png")
        plt.tight_layout()
        plt.savefig(boxplot_path)
        logger.info(f"F1和AUC箱线图已保存到: {boxplot_path}")
        plt.close()

    except ImportError:
        logger.warning("matplotlib 或 seaborn 未安装，跳过生成箱线图。")
    except Exception as e_plot:
        logger.error(f"生成箱线图时出错: {e_plot}")


    return {
        'experiment_type': 'generalization',
        'num_runs': num_runs,
        'all_run_metrics': all_run_metrics,
        'summary_statistics': summary_stats.to_dict('records')
    }

def save_results(results: Dict[str, Any], output_dir: str, experiment_name: str):
    """
    保存实验结果
    
    Args:
        results: 实验结果
        output_dir: 输出目录
        experiment_name: 实验名称
    """
    results_file = os.path.join(output_dir, f"{experiment_name}_results.json")
    
    # 处理不可序列化的对象
    def json_serializable(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_serializable)
    
    print(f"📁 实验结果已保存: {results_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多模态内部威胁检测系统 - 主实验控制脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验类型说明:
  baseline  - 传统ML方法对比实验 (RandomForest vs XGBoost vs 多模态)
  tune      - 超参数优化实验 (使用Optuna)
  ablation  - 消融实验，测试不同模态组合的效果
  imbalance - 数据不平衡适应性实验
  realtime  - 实时检测实验 (待实现)
  generalization - 模型泛化能力评估实验

示例:
  python main_experiment.py --run_type baseline --max_users 100 --epochs 5
  python main_experiment.py --run_type tune --config_file configs/tune_config.yaml --n_trials 30
  python main_experiment.py --run_type ablation --output_dir ./results/ablation_exp
  python main_experiment.py --run_type imbalance --max_users 200
  python main_experiment.py --run_type generalization --config_file configs/gen_config.yaml
        """
    )
    
    # 基本参数
    parser.add_argument('--run_type', type=str, required=True,
                       choices=['baseline', 'tune', 'ablation', 'imbalance', 'realtime', 'generalization'],
                       help='实验类型')
    parser.add_argument('--config_file', type=str, default=None,
                       help='配置文件路径 (JSON/YAML)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='输出目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称 (默认使用run_type + 时间戳)')
    
    # 数据参数
    parser.add_argument('--max_users', type=int, default=100,
                       help='最大用户数')
    parser.add_argument('--data_version', type=str, default='r4.2',
                       help='数据集版本')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='数据采样比例')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='训练设备')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Transformer层数')
    
    # 超参数优化参数
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Optuna优化试验次数')
    
    # 系统参数
    parser.add_argument('--num_cores', type=int, default=8,
                       help='CPU核心数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 改进版基线模型参数
    parser.add_argument('--use_improved_baseline', action='store_true',
                       help='使用改进版基线模型（差异化特征工程和交叉验证）')
    parser.add_argument('--baseline_cv_folds', type=int, default=5,
                       help='基线模型交叉验证折数')
    
    args = parser.parse_args()
    
    # 生成实验名称
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{args.run_type}_{timestamp}"
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir, args.experiment_name)
    
    try:
        # 加载配置
        config = load_config(
            config_file=args.config_file,
            max_users=args.max_users,
            data_version=args.data_version,
            sample_ratio=args.sample_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_cores=args.num_cores,
            seed=args.seed,
            n_trials=args.n_trials,
            use_improved_baseline=args.use_improved_baseline,
            baseline_cv_folds=args.baseline_cv_folds
        )
        
        logger.info(f"🎯 开始实验: {args.experiment_name}")
        logger.info(f"📋 实验类型: {args.run_type}")
        logger.info(f"📁 输出目录: {output_dir}")
        
        # 根据实验类型运行相应的实验
        if args.run_type == 'baseline':
            results = run_baseline_experiment(config, output_dir, logger)
        elif args.run_type == 'tune':
            results = run_tune_experiment(config, output_dir, logger)
        elif args.run_type == 'ablation':
            results = run_ablation_experiment(config, output_dir, logger)
        elif args.run_type == 'imbalance':
            results = run_imbalance_experiment(config, output_dir, logger)
        elif args.run_type == 'realtime':
            results = run_realtime_experiment(config, output_dir, logger)
        elif args.run_type == 'generalization':
            results = run_generalization_eval(config, output_dir, logger)
        else:
            raise ValueError(f"不支持的实验类型: {args.run_type}")
        
        # 保存结果
        save_results(results, output_dir, args.experiment_name)
        
        logger.info(f"🎉 实验 {args.experiment_name} 完成!")
        
    except Exception as e:
        logger.error(f"❌ 实验失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 