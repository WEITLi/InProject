#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练与评估流水线
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os

# 尝试相对导入评估工具，如果失败，则假定在同一级别（用于直接运行此脚本进行测试）
try:
    from .evaluation_utils import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score, plot_roc_curve, plot_pr_curve
    from .multimodal_pipeline import MultiModalDataPipeline # 用于主函数中的示例数据加载
except ImportError:
    print("ImportError: Running train_pipeline.py directly or evaluation_utils not found in relative path. Trying direct import for evaluation_utils.")
    # This fallback is mainly for allowing direct execution of this script for testing evaluation utils if they were in the same folder
    # For a proper project structure, the relative imports should work when called from a main script.
    from evaluation_utils import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score, plot_roc_curve, plot_pr_curve
    # MultiModalDataPipeline might not be available for direct run, placeholder for data needed

class MultiModalAnomalyDetector(nn.Module):
    """
    多模态异常检测器模型 (占位符)
    后续将集成GNN和Transformer等模块。
    """
    def __init__(self, input_dims: dict, hidden_dim: int = 128, output_dim: int = 1):
        super(MultiModalAnomalyDetector, self).__init__()
        
        self.seq_feat_dim = input_dims.get('behavior_sequences', 256)
        self.struct_feat_dim = input_dims.get('structured_features', 50)
        # self.text_embedding_dim = input_dims.get('text_embedding', 768) # 预留
        # self.gnn_output_dim = input_dims.get('gnn_output', 64) # 预留

        # 行为序列处理 (简化: LSTM -> last hidden state)
        self.lstm = nn.LSTM(self.seq_feat_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # 结构化特征处理 (简化: Linear layer)
        self.linear_struct = nn.Linear(self.struct_feat_dim, hidden_dim // 2) # 使得拼接后维度合理
        
        # 组合层 (简化: 拼接后通过线性层)
        # 假设行为序列输出hidden_dim, 结构化特征输出hidden_dim/2
        combined_input_dim = hidden_dim + (hidden_dim // 2)
        self.fc1 = nn.Linear(combined_input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, behavior_seq, structured_features, node_features=None, adj_matrix=None, text_input=None):
        #行为序列处理
        # behavior_seq: (batch_size, seq_len, self.seq_feat_dim)
        _, (h_n, _) = self.lstm(behavior_seq)
        seq_embed = h_n[-1] # (batch_size, hidden_dim)
        
        #结构化特征处理
        # structured_features: (batch_size, self.struct_feat_dim)
        struct_embed = self.relu(self.linear_struct(structured_features)) # (batch_size, hidden_dim/2)
        
        # 拼接特征 (在此简化版本中，我们只使用了序列和结构化特征)
        combined_embed = torch.cat((seq_embed, struct_embed), dim=1)
        
        x = self.relu(self.fc1(combined_embed))
        x = self.fc2(x)
        return self.sigmoid(x) # 输出概率

class AnomalyDataset(Dataset):
    """
    多模态异常数据的数据集类。
    """
    def __init__(self, data_dict: dict):
        self.behavior_sequences = torch.FloatTensor(data_dict['behavior_sequences'])
        self.structured_features = torch.FloatTensor(data_dict['structured_features'])
        self.labels = torch.FloatTensor(data_dict['labels']).unsqueeze(1) # (batch, 1) for BCE Loss
        
        # 全局图数据 (如果GNN模型需要，应在训练循环中直接传递，而不是每个item都返回)
        self.node_features = data_dict.get('node_features') # 可能为None
        if self.node_features is not None:
            self.node_features = torch.FloatTensor(self.node_features)
        self.adjacency_matrix = data_dict.get('adjacency_matrix') # 可能为None
        if self.adjacency_matrix is not None:
            self.adjacency_matrix = torch.FloatTensor(self.adjacency_matrix)
            
        # 文本数据 (需要预处理和向量化)
        self.text_content = data_dict.get('text_content', []) # 原始文本列表
        
        self.users = data_dict.get('users', [])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        sample = {
            'behavior_seq': self.behavior_sequences[idx],
            'structured_features': self.structured_features[idx],
            'label': self.labels[idx],
            'user': self.users[idx] if self.users else f'user_{idx}'
            # 'text_content': self.text_content[idx] if self.text_content else "" # 需进一步处理
        }
        return sample

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                num_epochs: int, 
                device: torch.device,
                model_save_path: str = './models',
                experiment_name: str = 'experiment'):
    """
    训练模型并进行评估。
    """
    os.makedirs(model_save_path, exist_ok=True)
    best_val_f1 = -1.0
    best_model_path = os.path.join(model_save_path, f'{experiment_name}_best_model.pth')

    train_times_epoch = []
    inference_times_val = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train() # 设置模型为训练模式
        running_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            behavior_seq = batch_data['behavior_seq'].to(device)
            structured_features = batch_data['structured_features'].to(device)
            labels = batch_data['label'].to(device)
            
            # TODO: 预留GNN和文本数据的处理位置
            # node_features_global = train_loader.dataset.node_features.to(device) if hasattr(train_loader.dataset, 'node_features') and train_loader.dataset.node_features is not None else None
            # adj_matrix_global = train_loader.dataset.adjacency_matrix.to(device) if hasattr(train_loader.dataset, 'adjacency_matrix') and train_loader.dataset.adjacency_matrix is not None else None
            # text_batch_processed = process_text(batch_data['text_content']) # 假设有文本处理函数

            optimizer.zero_grad()
            outputs = model(behavior_seq, structured_features) # 当前模型只用这两者
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_time = time.time() - epoch_start_time
        train_times_epoch.append(epoch_train_time)
        avg_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_epoch_loss:.4f}, Time: {epoch_train_time:.2f}s")

        # --- 验证阶段 ---
        model.eval() # 设置模型为评估模式
        val_running_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        all_val_scores = []
        
        val_epoch_start_time = time.time()
        with torch.no_grad():
            for batch_data in val_loader:
                behavior_seq = batch_data['behavior_seq'].to(device)
                structured_features = batch_data['structured_features'].to(device)
                labels = batch_data['label'].to(device)
                
                outputs = model(behavior_seq, structured_features)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                scores = outputs.cpu().numpy()
                preds = (scores > 0.5).astype(int)
                
                all_val_labels.extend(labels.cpu().numpy().flatten())
                all_val_preds.extend(preds.flatten())
                all_val_scores.extend(scores.flatten())
        
        val_epoch_inference_time = time.time() - val_epoch_start_time
        inference_times_val.append(val_epoch_inference_time)

        avg_val_loss = val_running_loss / len(val_loader)
        
        #确保转换成numpy array
        all_val_labels = np.array(all_val_labels)
        all_val_preds = np.array(all_val_preds)
        all_val_scores = np.array(all_val_scores)

        val_accuracy = calculate_accuracy(all_val_labels, all_val_preds)
        val_precision = calculate_precision(all_val_labels, all_val_preds)
        val_recall = calculate_recall(all_val_labels, all_val_preds)
        val_f1 = calculate_f1_score(all_val_labels, all_val_preds)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}, F1: {val_f1:.4f}, InfTime: {val_epoch_inference_time:.2f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} (F1: {best_val_f1:.4f})")
        
    avg_train_time = np.mean(train_times_epoch) if train_times_epoch else 0
    avg_inference_time = np.mean(inference_times_val) if inference_times_val else 0
    print(f"Finished Training. Avg Epoch Train Time: {avg_train_time:.2f}s, Avg Val Inference Time: {avg_inference_time:.2f}s") 
    
    # 加载最佳模型进行最终评估或返回
    model.load_state_dict(torch.load(best_model_path))
    return model, {'avg_train_time_epoch': avg_train_time, 'avg_val_inference_time': avg_inference_time, 'best_val_f1': best_val_f1}

if __name__ == '__main__':
    print("Running conceptual example for train_pipeline.py...")

    # --- 1. 准备数据 (使用MultiModalDataPipeline的模拟输出) ---
    # 在实际项目中，你会调用MultiModalDataPipeline来获取这些数据
    # 这里我们用随机数据模拟，确保维度和类型与流水线输出一致
    num_train_samples = 200
    num_val_samples = 50
    seq_len = 128
    behavior_dim = 256 # pipeline.feature_dim
    struct_dim = 50     # pipeline._integrate_multimodal_data中structured_feat_data.append(np.zeros(50))的默认维度
    num_nodes_graph = 100 # 假设图中有100个用户节点
    node_feat_dim = 14    # pipeline._construct_user_relationships中'node_features'的维度

    def get_dummy_data_dict(num_samples, is_train_data=True):
        user_prefix = "train_user_" if is_train_data else "val_user_"
        return {
            'behavior_sequences': np.random.rand(num_samples, seq_len, behavior_dim).astype(np.float32),
            'structured_features': np.random.rand(num_samples, struct_dim).astype(np.float32),
            'labels': np.random.randint(0, 2, num_samples).astype(np.float32),
            'users': [f'{user_prefix}{i}' for i in range(num_samples)],
            # 全局图数据，对于所有样本都一样 (如果需要)
            'node_features': np.random.rand(num_nodes_graph, node_feat_dim).astype(np.float32), 
            'adjacency_matrix': np.random.rand(num_nodes_graph, num_nodes_graph).astype(np.float32),
            'text_content': [f"Sample text for user {i}" for i in range(num_samples)],
            'user_to_index': {f'{user_prefix}{i}': i for i in range(num_samples)} # 简化版
        }

    train_data_dict = get_dummy_data_dict(num_train_samples, is_train_data=True)
    val_data_dict = get_dummy_data_dict(num_val_samples, is_train_data=False)
    # 在实际中，val_data_dict的node_features和adjacency_matrix应与train_data_dict共享或对应
    val_data_dict['node_features'] = train_data_dict['node_features'] 
    val_data_dict['adjacency_matrix'] = train_data_dict['adjacency_matrix']

    train_dataset = AnomalyDataset(train_data_dict)
    val_dataset = AnomalyDataset(val_data_dict)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # --- 2. 初始化模型、损失函数、优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dimensions = {
        'behavior_sequences': behavior_dim,
        'structured_features': struct_dim
    }
    model = MultiModalAnomalyDetector(input_dims=input_dimensions, hidden_dim=64).to(device)
    criterion = nn.BCELoss() # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 3. 训练模型 ---
    num_epochs_example = 2 # 示例运行5个epoch
    trained_model, metrics = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs_example, device, experiment_name='dummy_test')

    print("Conceptual training example finished.")
    print(f"Best validation F1: {metrics['best_val_f1']:.4f}")
    print(f"Avg epoch train time: {metrics['avg_train_time_epoch']:.2f}s, Avg val inference time: {metrics['avg_val_inference_time']:.2f}s")

    # --- 4. (可选) 绘制最终评估曲线 ---
    # 为了绘制曲线，需要重新在验证集上获取分数
    trained_model.eval()
    final_val_labels = []
    final_val_scores = []
    with torch.no_grad():
        for batch_data in val_loader:
            behavior_seq = batch_data['behavior_seq'].to(device)
            structured_features = batch_data['structured_features'].to(device)
            labels = batch_data['label'].to(device)
            outputs = trained_model(behavior_seq, structured_features)
            final_val_labels.extend(labels.cpu().numpy().flatten())
            final_val_scores.extend(outputs.cpu().numpy().flatten())
    
    final_val_labels = np.array(final_val_labels)
    final_val_scores = np.array(final_val_scores)

    # 创建保存绘图的目录
    plots_dir = "./plots/dummy_test"
    os.makedirs(plots_dir, exist_ok=True)

    plot_roc_curve(final_val_labels, final_val_scores, title='ROC Curve (Dummy Test)', output_path=os.path.join(plots_dir, "roc_curve.png"))
    plot_pr_curve(final_val_labels, final_val_scores, title='PR Curve (Dummy Test)', output_path=os.path.join(plots_dir, "pr_curve.png"))
    print(f"Sample plots saved to {plots_dir}") 