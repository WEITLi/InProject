#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WandB Integration Utilities
WandB集成工具模块
"""

import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

class WandBLogger:
    """WandB日志记录器"""
    
    def __init__(self, 
                 project_name: str = "threat_detection_experiments",
                 experiment_type: str = "baseline",
                 model_type: str = "multimodal",
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 run_name_override: Optional[str] = None):
        """
        初始化WandB记录器
        
        Args:
            project_name: 项目名称
            experiment_type: 实验类型
            model_type: 模型类型
            config: 配置字典
            tags: 标签列表
            run_name_override: 运行名称覆盖
        """
        self.project_name = project_name
        self.experiment_type = experiment_type
        self.model_type = model_type
        
        # 生成运行名称
        if run_name_override:
            run_name = run_name_override
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"{experiment_type}_{model_type}_{timestamp}"
        
        # 准备标签
        if tags is None:
            tags = [experiment_type, model_type]
        
        # 设置WandB初始化参数，增加超时时间和重试机制
        wandb_settings = wandb.Settings(
            init_timeout=180,  # 增加超时时间到3分钟
            start_method="thread"  # 使用线程启动方式，可能更稳定
        )
        
        # 重试机制
        max_retries = 3
        retry_delay = 10  # 秒
        
        for attempt in range(max_retries):
            try:
                print(f"尝试初始化 WandB (第 {attempt + 1}/{max_retries} 次)...")
                self.run = wandb.init(
                    project=project_name,
                    group=experiment_type,
                    name=run_name,
                    config=config,
                    tags=tags,
                    reinit=True,  # 允许重新初始化
                    settings=wandb_settings
                )
                print(f"✅ WandB 初始化成功: {self.run.url}")
                break
                
            except Exception as e:
                print(f"❌ WandB 初始化失败 (第 {attempt + 1} 次): {e}")
                
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    print("⚠️ WandB 初始化最终失败，将使用离线模式")
                    # 设置为离线模式
                    os.environ["WANDB_MODE"] = "offline"
                    try:
                        self.run = wandb.init(
                            project=project_name,
                            name=run_name,
                            config=config,
                            tags=tags,
                            mode="offline",
                            settings=wandb_settings
                        )
                        print("✅ WandB 离线模式初始化成功")
                    except Exception as offline_e:
                        print(f"❌ WandB 离线模式也失败: {offline_e}")
                        print("⚠️ 将禁用 WandB 日志记录")
                        self.run = None
                    break
        
        # 添加实验类型到配置
        if self.run.config:
            wandb.config.update({"experiment_type": experiment_type})
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置信息"""
        wandb.config.update(config)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录指标"""
        if self.run is not None:
            try:
                self.run.log(metrics, step=step)
            except Exception as e:
                print(f"⚠️ WandB 指标记录失败: {e}")
    
    def log_model_metrics(self, 
                         train_metrics: Dict[str, float],
                         val_metrics: Dict[str, float],
                         epoch: int):
        """记录训练和验证指标"""
        log_dict = {}
        
        # 添加训练指标
        for key, value in train_metrics.items():
            log_dict[f"train_{key}"] = value
        
        # 添加验证指标
        for key, value in val_metrics.items():
            log_dict[f"val_{key}"] = value
        
        self.log_metrics(log_dict, step=epoch)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str] = None):
        """记录混淆矩阵"""
        if class_names is None:
            class_names = ["Normal", "Malicious"]
        
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
    
    def log_feature_importance(self, 
                             feature_names: List[str], 
                             importance_scores: np.ndarray,
                             title: str = "Feature Importance"):
        """记录特征重要性"""
        if self.run is None:
            return
        
        try:
            # 创建特征重要性图表
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 排序特征重要性
            sorted_idx = np.argsort(importance_scores)[-20:]  # 显示前20个重要特征
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_scores = importance_scores[sorted_idx]
            
            ax.barh(range(len(sorted_features)), sorted_scores)
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel('Importance Score')
            ax.set_title(title)
            plt.tight_layout()
            
            # 记录到WandB
            wandb.log({f"{title.lower().replace(' ', '_')}": wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ WandB 特征重要性记录失败: {e}")
    
    def log_attention_heatmap(self, 
                            attention_weights: np.ndarray,
                            feature_names: List[str],
                            title: str = "Attention Heatmap"):
        """记录注意力热图"""
        if self.run is None:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
            
            # 设置标签
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_ylabel('Samples')
            ax.set_title(title)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            # 记录到WandB
            wandb.log({f"{title.lower().replace(' ', '_')}": wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ WandB 注意力热图记录失败: {e}")
    
    def log_training_curves(self, 
                          train_history: Dict[str, List[float]],
                          title: str = "Training Curves"):
        """记录训练曲线"""
        if self.run is None:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            metrics = ['loss', 'accuracy', 'f1', 'auc']
            
            for i, metric in enumerate(metrics):
                if i < len(axes):
                    ax = axes[i]
                    
                    # 绘制训练曲线
                    if f'train_{metric}' in train_history:
                        ax.plot(train_history[f'train_{metric}'], 
                               label=f'Train {metric.upper()}', marker='o')
                    
                    # 绘制验证曲线
                    if f'val_{metric}' in train_history:
                        ax.plot(train_history[f'val_{metric}'], 
                               label=f'Val {metric.upper()}', marker='s')
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric.upper())
                    ax.set_title(f'{metric.upper()} Curves')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 记录到WandB
            wandb.log({f"{title.lower().replace(' ', '_')}": wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ WandB 训练曲线记录失败: {e}")
    
    def log_imbalance_analysis(self, 
                             ratios: List[float], 
                             f1_scores: List[float],
                             auc_scores: List[float]):
        """记录数据不平衡分析结果"""
        if self.run is None:
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # F1分数曲线
            ax1.plot(ratios, f1_scores, marker='o', linewidth=2, markersize=8)
            ax1.set_xlabel('Imbalance Ratio (Normal:Malicious)')
            ax1.set_ylabel('F1 Score')
            ax1.set_title('F1 Score vs Imbalance Ratio')
            ax1.grid(True, alpha=0.3)
            
            # AUC分数曲线
            ax2.plot(ratios, auc_scores, marker='s', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel('Imbalance Ratio (Normal:Malicious)')
            ax2.set_ylabel('AUC Score')
            ax2.set_title('AUC Score vs Imbalance Ratio')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 记录到WandB
            wandb.log({"imbalance_analysis": wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ WandB 不平衡分析记录失败: {e}")
    
    def log_ablation_results(self, 
                           modality_combinations: List[str],
                           f1_scores: List[float]):
        """记录消融实验结果"""
        if self.run is None:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 创建柱状图
            bars = ax.bar(range(len(modality_combinations)), f1_scores, 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'][:len(modality_combinations)])
            
            # 设置标签
            ax.set_xticks(range(len(modality_combinations)))
            ax.set_xticklabels(modality_combinations, rotation=45, ha='right')
            ax.set_ylabel('F1 Score')
            ax.set_title('Ablation Study: Modality Combinations vs F1 Score')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 记录到WandB
            wandb.log({"ablation_results": wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ WandB 消融实验结果记录失败: {e}")
    
    def finish(self):
        """结束WandB运行"""
        if self.run is not None:
            try:
                self.run.finish()
                print("✅ WandB 运行已结束")
            except Exception as e:
                print(f"⚠️ WandB 运行结束失败: {e}")

def init_wandb(project_name: str = "threat_detection_experiments",
               experiment_type: str = "baseline",
               model_type: str = "multimodal",
               config: Optional[Dict[str, Any]] = None,
               tags: Optional[List[str]] = None,
               run_name_override: Optional[str] = None) -> WandBLogger:
    """
    初始化WandB记录器的便捷函数
    
    Args:
        project_name: 项目名称
        experiment_type: 实验类型
        model_type: 模型类型
        config: 配置字典
        tags: 标签列表
        run_name_override: 运行名称覆盖
    
    Returns:
        WandBLogger实例
    """
    return WandBLogger(
        project_name=project_name,
        experiment_type=experiment_type,
        model_type=model_type,
        config=config,
        tags=tags,
        run_name_override=run_name_override
    ) 