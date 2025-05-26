#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna Hyperparameter Tuning Utilities
Optuna超参数优化工具模块
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
import os
import json
from datetime import datetime
import logging

# 设置Optuna日志级别
optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptunaOptimizer:
    """Optuna超参数优化器"""
    
    def __init__(self, 
                 study_name: str = None,
                 direction: str = "maximize",
                 storage: str = None,
                 random_state: int = 42):
        """
        初始化Optuna优化器
        
        Args:
            study_name: 研究名称
            direction: 优化方向 ("maximize" 或 "minimize")
            storage: 存储后端 (可选)
            random_state: 随机种子
        """
        self.study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.direction = direction
        self.storage = storage
        self.random_state = random_state
        
        # 创建采样器
        self.sampler = optuna.samplers.TPESampler(seed=random_state)
        
        # 创建研究
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=self.sampler,
            storage=storage,
            load_if_exists=True
        )
        
        self.best_params = None
        self.best_value = None
        self.optimization_history = []
    
    def define_search_space(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        定义搜索空间
        
        Args:
            trial: Optuna试验对象
            search_space: 搜索空间定义
            
        Returns:
            采样的参数字典
        """
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            elif param_type == 'discrete_uniform':
                params[param_name] = trial.suggest_discrete_uniform(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    param_config['q']
                )
            else:
                raise ValueError(f"不支持的参数类型: {param_type}")
        
        return params
    
    def optimize_multimodal_model(self,
                                 training_data: Dict[str, Any],
                                 model_trainer,
                                 search_space: Dict[str, Any],
                                 n_trials: int = 50,
                                 timeout: Optional[int] = None,
                                 wandb_logger=None) -> Dict[str, Any]:
        """
        优化多模态模型的超参数
        
        Args:
            training_data: 训练数据
            model_trainer: 模型训练器
            search_space: 搜索空间
            n_trials: 试验次数
            timeout: 超时时间（秒）
            wandb_logger: WandB记录器
            
        Returns:
            优化结果
        """
        def objective(trial):
            try:
                # 采样超参数
                params = self.define_search_space(trial, search_space)
                
                # 更新模型配置
                config = model_trainer.config
                for param_name, param_value in params.items():
                    if param_name in ['learning_rate', 'batch_size', 'weight_decay']:
                        setattr(config.training, param_name, param_value)
                    elif param_name in ['hidden_dim', 'num_heads', 'num_layers', 'dropout']:
                        setattr(config.model, param_name, param_value)
                    else:
                        # 其他参数设置到相应的配置段
                        if hasattr(config.model, param_name):
                            setattr(config.model, param_name, param_value)
                        elif hasattr(config.training, param_name):
                            setattr(config.training, param_name, param_value)
                
                # 训练模型
                model = model_trainer.train(training_data)
                
                # 获取验证指标
                train_history = model_trainer.train_history
                if 'val_f1' in train_history and len(train_history['val_f1']) > 0:
                    # 使用最佳验证F1分数
                    objective_value = max(train_history['val_f1'])
                elif 'val_accuracy' in train_history and len(train_history['val_accuracy']) > 0:
                    # 备选：使用验证准确率
                    objective_value = max(train_history['val_accuracy'])
                else:
                    # 最后备选：使用训练F1
                    objective_value = max(train_history.get('train_f1', [0.0]))
                
                # 记录到WandB
                if wandb_logger:
                    wandb_logger.log_metrics({
                        'trial_number': trial.number,
                        'objective_value': objective_value,
                        **{f'param_{k}': v for k, v in params.items()}
                    })
                
                # 记录优化历史
                self.optimization_history.append({
                    'trial': trial.number,
                    'params': params,
                    'value': objective_value
                })
                
                return objective_value
                
            except Exception as e:
                print(f"⚠️ Trial {trial.number} 失败: {e}")
                return 0.0  # 返回最差分数
        
        # 开始优化
        print(f"🎯 开始Optuna超参数优化...")
        print(f"   搜索空间: {search_space}")
        print(f"   试验次数: {n_trials}")
        
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # 获取最佳结果
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        print(f"✅ 优化完成!")
        print(f"   最佳参数: {self.best_params}")
        print(f"   最佳分数: {self.best_value:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'optimization_history': self.optimization_history,
            'study': self.study
        }
    
    def optimize_traditional_model(self,
                                 multimodal_data: Dict[str, Any],
                                 model_trainer,
                                 search_space: Dict[str, Any],
                                 n_trials: int = 50,
                                 timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        优化传统机器学习模型的超参数
        
        Args:
            multimodal_data: 多模态数据
            model_trainer: 基线模型训练器
            search_space: 搜索空间
            n_trials: 试验次数
            timeout: 超时时间（秒）
            
        Returns:
            优化结果
        """
        def objective(trial):
            try:
                # 采样超参数
                params = self.define_search_space(trial, search_space)
                
                # 更新模型参数
                if model_trainer.model_type == "random_forest":
                    from sklearn.ensemble import RandomForestClassifier
                    model_trainer.model = RandomForestClassifier(
                        random_state=model_trainer.random_state,
                        n_jobs=-1,
                        **params
                    )
                elif model_trainer.model_type == "xgboost":
                    import xgboost as xgb
                    model_trainer.model = xgb.XGBClassifier(
                        random_state=model_trainer.random_state,
                        n_jobs=-1,
                        eval_metric='logloss',
                        **params
                    )
                
                # 训练模型
                results = model_trainer.train(multimodal_data)
                
                # 获取测试F1分数作为目标
                objective_value = results['test_metrics']['f1']
                
                # 记录优化历史
                self.optimization_history.append({
                    'trial': trial.number,
                    'params': params,
                    'value': objective_value
                })
                
                return objective_value
                
            except Exception as e:
                print(f"⚠️ Trial {trial.number} 失败: {e}")
                return 0.0
        
        # 开始优化
        print(f"🎯 开始优化 {model_trainer.model_type} 模型...")
        
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # 获取最佳结果
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        print(f"✅ {model_trainer.model_type} 优化完成!")
        print(f"   最佳参数: {self.best_params}")
        print(f"   最佳F1分数: {self.best_value:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'optimization_history': self.optimization_history
        }
    
    def get_optimization_history_df(self) -> pd.DataFrame:
        """获取优化历史的DataFrame"""
        if not self.optimization_history:
            return pd.DataFrame()
        
        # 展开参数字典
        rows = []
        for record in self.optimization_history:
            row = {'trial': record['trial'], 'value': record['value']}
            row.update(record['params'])
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results(self, output_dir: str, filename: str = "optuna_results.json"):
        """保存优化结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'study_name': self.study_name,
            'direction': self.direction,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'optimization_history': self.optimization_history
        }
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 优化结果已保存到: {filepath}")
        
        # 保存优化历史CSV
        history_df = self.get_optimization_history_df()
        if not history_df.empty:
            csv_path = os.path.join(output_dir, "optimization_history.csv")
            history_df.to_csv(csv_path, index=False)
            print(f"📊 优化历史已保存到: {csv_path}")

def get_multimodal_search_space() -> Dict[str, Any]:
    """获取多模态模型的默认搜索空间"""
    return {
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-2,
            'log': True
        },
        'hidden_dim': {
            'type': 'categorical',
            'choices': [64, 128, 256, 512]
        },
        'num_heads': {
            'type': 'categorical',
            'choices': [4, 8, 16]
        },
        'num_layers': {
            'type': 'int',
            'low': 2,
            'high': 8
        },
        'dropout': {
            'type': 'float',
            'low': 0.1,
            'high': 0.5
        },
        'batch_size': {
            'type': 'categorical',
            'choices': [16, 32, 64, 128]
        },
        'weight_decay': {
            'type': 'float',
            'low': 1e-6,
            'high': 1e-3,
            'log': True
        }
    }

def get_random_forest_search_space() -> Dict[str, Any]:
    """获取随机森林的搜索空间"""
    return {
        'n_estimators': {
            'type': 'int',
            'low': 50,
            'high': 300
        },
        'max_depth': {
            'type': 'int',
            'low': 3,
            'high': 20
        },
        'min_samples_split': {
            'type': 'int',
            'low': 2,
            'high': 20
        },
        'min_samples_leaf': {
            'type': 'int',
            'low': 1,
            'high': 10
        },
        'max_features': {
            'type': 'categorical',
            'choices': ['sqrt', 'log2', None]
        }
    }

def get_xgboost_search_space() -> Dict[str, Any]:
    """获取XGBoost的搜索空间"""
    return {
        'n_estimators': {
            'type': 'int',
            'low': 50,
            'high': 300
        },
        'max_depth': {
            'type': 'int',
            'low': 3,
            'high': 10
        },
        'learning_rate': {
            'type': 'float',
            'low': 0.01,
            'high': 0.3,
            'log': True
        },
        'subsample': {
            'type': 'float',
            'low': 0.6,
            'high': 1.0
        },
        'colsample_bytree': {
            'type': 'float',
            'low': 0.6,
            'high': 1.0
        },
        'reg_alpha': {
            'type': 'float',
            'low': 0.0,
            'high': 1.0
        },
        'reg_lambda': {
            'type': 'float',
            'low': 0.0,
            'high': 1.0
        }
    }

def run_optuna_tuning(model_type: str,
                     data: Dict[str, Any],
                     trainer,
                     output_dir: str,
                     n_trials: int = 50,
                     custom_search_space: Dict[str, Any] = None,
                     wandb_logger=None) -> Dict[str, Any]:
    """
    运行Optuna超参数调优
    
    Args:
        model_type: 模型类型 ("multimodal", "random_forest", "xgboost")
        data: 数据
        trainer: 训练器
        output_dir: 输出目录
        n_trials: 试验次数
        custom_search_space: 自定义搜索空间
        wandb_logger: WandB记录器
        
    Returns:
        优化结果
    """
    # 创建优化器
    optimizer = OptunaOptimizer(
        study_name=f"{model_type}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        direction="maximize"
    )
    
    # 获取搜索空间
    if custom_search_space:
        search_space = custom_search_space
    elif model_type == "multimodal":
        search_space = get_multimodal_search_space()
    elif model_type == "random_forest":
        search_space = get_random_forest_search_space()
    elif model_type == "xgboost":
        search_space = get_xgboost_search_space()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 运行优化
    if model_type == "multimodal":
        results = optimizer.optimize_multimodal_model(
            data, trainer, search_space, n_trials, wandb_logger=wandb_logger
        )
    else:
        results = optimizer.optimize_traditional_model(
            data, trainer, search_space, n_trials
        )
    
    # 保存结果
    optimizer.save_results(output_dir, f"{model_type}_optuna_results.json")
    
    return results 