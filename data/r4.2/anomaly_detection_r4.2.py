#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import paired_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import argparse
import os

# 设置matplotlib后端，避免在无显示器环境中的问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

# 设置seaborn样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

'''
这个脚本的主要功能是使用简单的自编码器（autoencoder）在 CERT r4.2 数据集上进行异常检测。
适配用户生成的时间表示数据文件，支持多种数据格式和参数配置。

核心算法：
1. 自编码器重构：在正常数据上训练，学习正常行为模式
2. 重构误差：异常行为的重构误差通常更大
3. 阈值检测：根据预算设置不同的检测阈值
4. 性能评估：AUC、检测率、精确率、召回率等指标
'''

def load_and_preprocess_data(file_path, test_ratio=0.5, random_seed=42):
    """
    加载和预处理数据
    
    参数:
    - file_path: 数据文件路径
    - test_ratio: 测试集比例
    - random_seed: 随机种子
    
    返回:
    - 训练和测试数据
    """
    print(f"正在加载数据: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    data = pd.read_pickle(file_path)
    print(f"数据形状: {data.shape}")
    print(f"数据列: {data.columns.tolist()[:10]}...")
    
    # 检查必要的列
    if 'user' not in data.columns:
        raise ValueError("数据中缺少 'user' 列")
    if 'insider' not in data.columns:
        raise ValueError("数据中缺少 'insider' 列")
    
    # 定义不需要的列（信息列，不用于训练）
    removed_cols = ['user', 'day', 'week', 'starttime', 'endtime', 'sessionid', 'insider',
                   'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin', 'project',
                   'O', 'C', 'E', 'A', 'N', 'subs_ind']
    
    # 只保留数据中实际存在的信息列
    actual_removed_cols = [col for col in removed_cols if col in data.columns]
    x_cols = [col for col in data.columns if col not in actual_removed_cols]
    
    print(f"特征列数量: {len(x_cols)}")
    print(f"移除的信息列: {actual_removed_cols}")
    
    # 检查是否有时间列进行分割
    if 'week' in data.columns:
        time_col = 'week'
    elif 'day' in data.columns:
        time_col = 'day'
    else:
        print("警告: 没有找到时间列，将随机分割数据")
        time_col = None
    
    # 数据分割
    np.random.seed(random_seed)
    
    if time_col:
        # 按时间分割
        max_time = data[time_col].max()
        split_time = max_time * (1 - test_ratio)
        
        train_data = data[data[time_col] <= split_time]
        test_data = data[data[time_col] > split_time]
        
        print(f"按{time_col}分割: 训练集 <= {split_time:.1f}, 测试集 > {split_time:.1f}")
    else:
        # 随机分割
        train_indices = np.random.choice(len(data), int(len(data) * (1-test_ratio)), replace=False)
        test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
        
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]
        
        print(f"随机分割: 训练集 {len(train_data)}, 测试集 {len(test_data)}")
    
    # 提取特征和标签
    X_train = train_data[x_cols].values
    y_train = train_data['insider'].values
    y_train_bin = y_train > 0
    
    X_test = test_data[x_cols].values  
    y_test = test_data['insider'].values
    y_test_bin = y_test > 0
    
    print(f"训练集: {X_train.shape}, 异常样本: {np.sum(y_train_bin)} ({100*np.mean(y_train_bin):.2f}%)")
    print(f"测试集: {X_test.shape}, 异常样本: {np.sum(y_test_bin)} ({100*np.mean(y_test_bin):.2f}%)")
    
    return X_train, y_train_bin, X_test, y_test_bin, x_cols

def train_autoencoder(X_train, hidden_layers=None, max_iter=25, random_seed=42):
    """
    训练自编码器
    
    参数:
    - X_train: 训练数据
    - hidden_layers: 隐藏层配置
    - max_iter: 最大迭代次数
    - random_seed: 随机种子
    
    返回:
    - 训练好的自编码器模型
    """
    print("开始训练自编码器...")
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 默认隐藏层配置
    if hidden_layers is None:
        input_dim = X_train.shape[1]
        hidden_layers = (input_dim//4, input_dim//8, input_dim//4)
    
    print(f"网络结构: {X_train.shape[1]} -> {hidden_layers} -> {X_train.shape[1]}")
    
    # 创建和训练自编码器
    autoencoder = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        random_state=random_seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5
    )
    
    autoencoder.fit(X_train_scaled, X_train_scaled)
    
    print(f"训练完成，最终损失: {autoencoder.loss_:.6f}")
    
    return autoencoder, scaler

def detect_anomalies(autoencoder, scaler, X_test, y_test_bin):
    """
    使用训练好的自编码器进行异常检测
    
    参数:
    - autoencoder: 训练好的自编码器
    - scaler: 数据标准化器
    - X_test: 测试数据
    - y_test_bin: 测试标签
    
    返回:
    - 重构误差和评估结果
    """
    print("开始异常检测...")
    
    # 标准化测试数据
    X_test_scaled = scaler.transform(X_test)
    
    # 计算重构误差
    X_reconstructed = autoencoder.predict(X_test_scaled)
    reconstruction_errors = paired_distances(X_test_scaled, X_reconstructed.reshape(X_test_scaled.shape))
    
    print(f"重构误差统计:")
    print(f"  均值: {np.mean(reconstruction_errors):.6f}")
    print(f"  标准差: {np.std(reconstruction_errors):.6f}")
    print(f"  最小值: {np.min(reconstruction_errors):.6f}")
    print(f"  最大值: {np.max(reconstruction_errors):.6f}")
    
    # 计算AUC
    if len(np.unique(y_test_bin)) > 1:
        auc_score = roc_auc_score(y_test_bin, reconstruction_errors)
        print(f"\nAUC Score: {auc_score:.4f}")
    else:
        auc_score = np.nan
        print("\n警告: 测试集中只有一个类别，无法计算AUC")
    
    return reconstruction_errors, auc_score

def evaluate_detection_rates(reconstruction_errors, y_test_bin, budgets=[0.001, 0.01, 0.05, 0.1, 0.2]):
    """
    在不同预算下评估检测率
    
    参数:
    - reconstruction_errors: 重构误差
    - y_test_bin: 真实标签
    - budgets: 预算列表（检测比例）
    
    返回:
    - 检测率结果
    """
    print("\n不同预算下的检测率:")
    print("预算\t阈值\t\t检测率\t精确率\t检测数量")
    print("-" * 60)
    
    results = []
    
    for budget in budgets:
        # 计算阈值（预算百分位数）
        threshold = np.percentile(reconstruction_errors, 100 - 100 * budget)
        
        # 标记为异常的样本
        flagged = reconstruction_errors > threshold
        
        # 计算指标
        if np.sum(y_test_bin) > 0:
            detection_rate = np.sum(y_test_bin[flagged]) / np.sum(y_test_bin)
        else:
            detection_rate = 0
            
        if np.sum(flagged) > 0:
            precision = np.sum(y_test_bin[flagged]) / np.sum(flagged)
        else:
            precision = 0
        
        num_detected = np.sum(flagged)
        
        results.append({
            'budget': budget,
            'threshold': threshold,
            'detection_rate': detection_rate,
            'precision': precision,
            'num_detected': num_detected
        })
        
        print(f"{budget*100:4.1f}%\t{threshold:.6f}\t{detection_rate*100:6.2f}%\t{precision*100:6.2f}%\t{num_detected:6d}")
    
    return results

def plot_results(reconstruction_errors, y_test_bin, output_dir="."):
    """
    绘制结果图表
    
    参数:
    - reconstruction_errors: 重构误差
    - y_test_bin: 真实标签
    - output_dir: 输出目录
    """
    print("\n生成结果图表...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: 重构误差分布
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    
    # 只绘制正常样本的直方图（因为可能没有异常样本）
    plt.hist(reconstruction_errors[~y_test_bin], bins=50, alpha=0.7, 
             label='Normal', 
             density=True, color='skyblue', edgecolor='black', linewidth=0.5)
    
    # 如果有异常样本，绘制异常样本的直方图
    if np.sum(y_test_bin) > 0:
        plt.hist(reconstruction_errors[y_test_bin], bins=50, alpha=0.7, 
                 label='Anomaly', 
                 density=True, color='salmon', edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Reconstruction Error Distribution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 图2: ROC曲线（如果有异常样本）
    plt.subplot(1, 2, 2)
    if len(np.unique(y_test_bin)) > 1 and np.sum(y_test_bin) > 0:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test_bin, reconstruction_errors)
        auc_score = roc_auc_score(y_test_bin, reconstruction_errors)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}', color='blue')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, 
                 label='Random', alpha=0.7)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    else:
        # 绘制重构误差的箱线图
        plt.boxplot(reconstruction_errors, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        plt.ylabel('Reconstruction Error', fontsize=12)
        plt.title('Reconstruction Error Distribution', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Mean: {np.mean(reconstruction_errors):.3f}\n'
        stats_text += f'Std: {np.std(reconstruction_errors):.3f}\n'
        stats_text += f'Min: {np.min(reconstruction_errors):.3f}\n'
        stats_text += f'Max: {np.max(reconstruction_errors):.3f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'anomaly_detection_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # 不显示图片，避免在服务器环境中的显示问题
    # plt.show()
    plt.close()
    
    print(f"图表已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='CERT r4.2 异常检测')
    parser.add_argument('--input', type=str, required=True, help='输入的pickle文件路径')
    parser.add_argument('--test_ratio', type=float, default=0.5, help='测试集比例 (默认: 0.5)')
    parser.add_argument('--max_iter', type=int, default=25, help='最大迭代次数 (默认: 25)')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--output_dir', type=str, default='.', help='输出目录 (默认: 当前目录)')
    parser.add_argument('--plot', action='store_true', help='生成结果图表')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CERT r4.2 异常检测 - 自编码器方法")
    print("="*60)
    print(f"输入文件: {args.input}")
    print(f"测试集比例: {args.test_ratio}")
    print(f"最大迭代次数: {args.max_iter}")
    print(f"随机种子: {args.random_seed}")
    print("="*60)
    
    try:
        # 1. 加载和预处理数据
        X_train, y_train_bin, X_test, y_test_bin, feature_cols = load_and_preprocess_data(
            args.input, args.test_ratio, args.random_seed
        )
        
        # 2. 训练自编码器
        autoencoder, scaler = train_autoencoder(
            X_train, max_iter=args.max_iter, random_seed=args.random_seed
        )
        
        # 3. 异常检测
        reconstruction_errors, auc_score = detect_anomalies(
            autoencoder, scaler, X_test, y_test_bin
        )
        
        # 4. 评估检测率
        detection_results = evaluate_detection_rates(reconstruction_errors, y_test_bin)
        
        # 5. 生成图表（可选）
        if args.plot:
            plot_results(reconstruction_errors, y_test_bin, args.output_dir)
        
        print("\n" + "="*60)
        print("异常检测完成！")
        print("="*60)
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 