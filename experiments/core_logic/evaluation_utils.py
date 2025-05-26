#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算模块
包含常用的分类任务评估指标。
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os
import json

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算准确率 (Accuracy)

    Args:
        y_true: 真实标签 (0或1)
        y_pred: 预测标签 (0或1)

    Returns:
        float: 准确率
    """
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1, average: str = 'binary') -> float:
    """
    计算精确率 (Precision)

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        pos_label: 指定正类的标签 (默认为1, 代表异常)
        average: 多分类问题的平均策略 (默认为 'binary' 用于二分类)

    Returns:
        float: 精确率
    """
    return precision_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=0)

def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1, average: str = 'binary') -> float:
    """
    计算召回率 (Recall)

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        pos_label: 指定正类的标签 (默认为1, 代表异常)
        average: 多分类问题的平均策略 (默认为 'binary' 用于二分类)

    Returns:
        float: 召回率
    """
    return recall_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=0)

def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1, average: str = 'binary') -> float:
    """
    计算 F1 分数 (F1 Score)

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        pos_label: 指定正类的标签 (默认为1, 代表异常)
        average: 多分类问题的平均策略 (默认为 'binary' 用于二分类)

    Returns:
        float: F1 分数
    """
    return f1_score(y_true, y_pred, pos_label=pos_label, average=average, zero_division=0)

def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, title: str = 'ROC Curve', output_path: str = None):
    """
    绘制并可选择保存ROC曲线图

    Args:
        y_true: 真实标签 (0或1)
        y_scores: 模型预测的概率分数 (目标类的正类概率)
        title: 图表标题
        output_path: (可选) 图片保存路径. 如果为None, 则显示图片.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"ROC curve saved to {output_path}")
        plt.close() # 关闭图像，避免在某些环境中重复显示
    else:
        plt.show()

def plot_pr_curve(y_true: np.ndarray, y_scores: np.ndarray, title: str = 'Precision-Recall Curve', output_path: str = None):
    """
    绘制并可选择保存Precision-Recall曲线图

    Args:
        y_true: 真实标签 (0或1)
        y_scores: 模型预测的概率分数 (目标类的正类概率)
        title: 图表标题
        output_path: (可选) 图片保存路径. 如果为None, 则显示图片.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision) # 注意这里 x是recall, y是precision

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
        print(f"Precision-Recall curve saved to {output_path}")
        plt.close()
    else:
        plt.show()

class ExperimentEvaluator:
    """
    实验评估器
    用于评估多模态异常检测实验的结果
    """
    
    def __init__(self, output_dir: str = './results'):
        """
        初始化评估器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估预测结果
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_scores: 预测概率分数 (可选)
            
        Returns:
            评估指标字典
        """
        metrics = {
            'accuracy': calculate_accuracy(y_true, y_pred),
            'precision': calculate_precision(y_true, y_pred),
            'recall': calculate_recall(y_true, y_pred),
            'f1_score': calculate_f1_score(y_true, y_pred)
        }
        
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                metrics['pr_auc'] = average_precision_score(y_true, y_scores)
            except ValueError as e:
                print(f"Warning: Could not calculate AUC scores: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = 'Confusion Matrix', 
                            output_path: Optional[str] = None) -> None:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            title: 图表标题
            output_path: 输出路径
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(title, fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {output_path}")
            plt.close()
        else:
            plt.show()
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            分类报告字符串
        """
        return classification_report(y_true, y_pred, 
                                   target_names=['Normal', 'Anomaly'])
    
    def compare_experiments(self, experiment_results: Dict[str, Dict[str, Any]], 
                          output_path: Optional[str] = None) -> pd.DataFrame:
        """
        比较多个实验结果
        
        Args:
            experiment_results: 实验结果字典
            output_path: 输出路径
            
        Returns:
            比较结果DataFrame
        """
        comparison_data = []
        
        for exp_name, results in experiment_results.items():
            if 'metrics' in results:
                row = {'experiment': exp_name}
                row.update(results['metrics'])
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Experiment comparison saved to {output_path}")
        
        return df
    
    def plot_training_curves(self, training_history: Dict[str, List[float]], 
                           title: str = 'Training Curves',
                           output_path: Optional[str] = None) -> None:
        """
        绘制训练曲线
        
        Args:
            training_history: 训练历史
            title: 图表标题
            output_path: 输出路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if 'train_loss' in training_history and 'val_loss' in training_history:
            axes[0, 0].plot(training_history['train_loss'], label='Training Loss')
            axes[0, 0].plot(training_history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy curves
        if 'train_acc' in training_history and 'val_acc' in training_history:
            axes[0, 1].plot(training_history['train_acc'], label='Training Accuracy')
            axes[0, 1].plot(training_history['val_acc'], label='Validation Accuracy')
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # F1 curves
        if 'train_f1' in training_history and 'val_f1' in training_history:
            axes[1, 0].plot(training_history['train_f1'], label='Training F1')
            axes[1, 0].plot(training_history['val_f1'], label='Validation F1')
            axes[1, 0].set_title('F1 Score Curves')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # AUC curves
        if 'val_auc' in training_history:
            axes[1, 1].plot(training_history['val_auc'], label='Validation AUC')
            axes[1, 1].set_title('AUC Curves')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {output_path}")
            plt.close()
        else:
            plt.show()
    
    def save_evaluation_report(self, experiment_name: str, 
                             metrics: Dict[str, float],
                             classification_report: str,
                             config: Optional[Dict] = None) -> None:
        """
        保存评估报告
        
        Args:
            experiment_name: 实验名称
            metrics: 评估指标
            classification_report: 分类报告
            config: 实验配置
        """
        report = {
            'experiment_name': experiment_name,
            'metrics': metrics,
            'classification_report': classification_report,
            'config': config
        }
        
        report_path = os.path.join(self.output_dir, f"{experiment_name}_evaluation_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Evaluation report saved to {report_path}")

if __name__ == '__main__':
    # 示例用法
    y_true_sample = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred_sample = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 1]) # 二进制预测
    y_scores_sample = np.array([0.1, 0.6, 0.8, 0.3, 0.2, 0.9, 0.7, 0.95, 0.4, 0.75]) # 概率分数

    print(f"Accuracy: {calculate_accuracy(y_true_sample, y_pred_sample):.4f}")
    print(f"Precision (for class 1): {calculate_precision(y_true_sample, y_pred_sample):.4f}")
    print(f"Recall (for class 1): {calculate_recall(y_true_sample, y_pred_sample):.4f}")
    print(f"F1 Score (for class 1): {calculate_f1_score(y_true_sample, y_pred_sample):.4f}")

    # 测试 ExperimentEvaluator
    evaluator = ExperimentEvaluator()
    metrics = evaluator.evaluate_predictions(y_true_sample, y_pred_sample, y_scores_sample)
    print(f"\nEvaluator metrics: {metrics}")

    # 绘制ROC曲线 (显示)
    plot_roc_curve(y_true_sample, y_scores_sample, title='Sample ROC Curve')
    # 绘制PR曲线 (保存到文件)
    # os.makedirs("temp_plots", exist_ok=True) # 确保目录存在
    # plot_pr_curve(y_true_sample, y_scores_sample, title='Sample PR Curve', output_path="temp_plots/sample_pr_curve.png")
    
    print("\n--- Example with different pos_label for minority class (e.g., pos_label=0) ---")
    # 假设0是少数类，我们想看预测0的指标
    y_true_minority = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 1]) # 0是少数
    y_pred_minority = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    print(f"Precision (for class 0): {calculate_precision(y_true_minority, y_pred_minority, pos_label=0):.4f}")
    print(f"Recall (for class 0):    {calculate_recall(y_true_minority, y_pred_minority, pos_label=0):.4f}")
    print(f"F1 Score (for class 0):  {calculate_f1_score(y_true_minority, y_pred_minority, pos_label=0):.4f}") 