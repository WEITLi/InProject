#!/usr/bin/env python3
"""User Graph Neural Network for User Relationship Modeling"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict

class GraphConvolution(nn.Module):
    """图卷积层"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim] 节点特征
            adj: [num_nodes, num_nodes] 邻接矩阵
        Returns:
            output: [num_nodes, output_dim] 输出特征
        """
        # 图卷积: A * X * W
        support = self.linear(x)  # [num_nodes, output_dim]
        output = torch.mm(adj, support)  # [num_nodes, output_dim]
        return self.dropout(output)

class UserGNN(nn.Module):
    """
    用户图神经网络
    
    基于用户之间的关系（如同部门、协作历史等）构建图，
    学习用户的图嵌入表示
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 构建GCN层
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(GraphConvolution(input_dim, hidden_dim, dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim, dropout))
        
        # 最后一层
        if num_layers > 1:
            self.layers.append(GraphConvolution(hidden_dim, output_dim, dropout))
        
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self, 
        node_features: torch.Tensor, 
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: [num_users, input_dim] 用户初始特征
            adj_matrix: [num_users, num_users] 用户关系邻接矩阵
            
        Returns:
            user_embeddings: [num_users, output_dim] 用户图嵌入
        """
        x = node_features
        
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_matrix)
            
            # 除了最后一层，都应用激活函数
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        # 层归一化
        x = self.layer_norm(x)
        
        return x
    
    def build_adjacency_matrix(
        self, 
        user_info: Dict, 
        num_users: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        根据用户信息构建邻接矩阵
        
        Args:
            user_info: 包含用户信息的字典 {user_id: {dept, role, ...}}
            num_users: 用户总数
            device: 设备
            
        Returns:
            adj_matrix: [num_users, num_users] 邻接矩阵
        """
        adj_matrix = torch.zeros(num_users, num_users, device=device)
        
        user_ids = list(user_info.keys())
        
        for i, user_i in enumerate(user_ids):
            for j, user_j in enumerate(user_ids):
                if i != j:
                    # 基于部门相似性
                    dept_similarity = 1.0 if user_info[user_i].get('dept') == user_info[user_j].get('dept') else 0.0
                    
                    # 基于角色相似性
                    role_similarity = 1.0 if user_info[user_i].get('role') == user_info[user_j].get('role') else 0.0
                    
                    # 综合相似性（可以添加更多因素）
                    similarity = 0.6 * dept_similarity + 0.4 * role_similarity
                    
                    # 设置阈值
                    if similarity > 0.5:
                        adj_matrix[i, j] = similarity
        
        # 添加自连接
        adj_matrix.fill_diagonal_(1.0)
        
        # 归一化邻接矩阵（度归一化）
        degree = adj_matrix.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # 避免除零
        adj_matrix = adj_matrix / degree
        
        return adj_matrix

class UserGraphBuilder:
    """用户图构建器"""
    
    def __init__(self):
        self.user_to_idx = {}
        self.idx_to_user = {}
        
    def build_user_graph(
        self, 
        users_df,
        interaction_data=None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        构建用户图
        
        Args:
            users_df: 用户信息DataFrame
            interaction_data: 用户交互数据（可选）
            
        Returns:
            node_features: [num_users, feature_dim] 用户节点特征
            adj_matrix: [num_users, num_users] 邻接矩阵
            user_mapping: 用户ID到索引的映射
        """
        # 创建用户映射
        users = users_df.index.tolist()
        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.idx_to_user = {idx: user for idx, user in enumerate(users)}
        
        num_users = len(users)
        
        # 构建节点特征
        node_features = self._build_node_features(users_df)
        
        # 构建邻接矩阵
        adj_matrix = self._build_adjacency_matrix(users_df, interaction_data)
        
        return node_features, adj_matrix, self.user_to_idx
    
    def _build_node_features(self, users_df) -> torch.Tensor:
        """构建用户节点特征"""
        num_users = len(users_df)
        feature_dim = 10  # 基础特征维度
        
        features = torch.zeros(num_users, feature_dim)
        
        for idx, (user_id, user_data) in enumerate(users_df.iterrows()):
            # 部门 one-hot 编码（简化）
            dept_map = {'IT': 0, 'Finance': 1, 'HR': 2, 'Marketing': 3, 'Engineering': 4, 'Operations': 5}
            dept_idx = dept_map.get(user_data.get('dept', 'IT'), 0)
            features[idx, dept_idx] = 1.0
            
            # 角色编码
            role_map = {'Employee': 0, 'Manager': 1, 'Director': 2, 'Executive': 3}
            role_idx = role_map.get(user_data.get('role', 'Employee'), 0)
            features[idx, 6 + role_idx] = 1.0
        
        return features
    
    def _build_adjacency_matrix(self, users_df, interaction_data=None) -> torch.Tensor:
        """构建邻接矩阵"""
        num_users = len(users_df)
        adj_matrix = torch.zeros(num_users, num_users)
        
        users = users_df.index.tolist()
        
        for i, user_i in enumerate(users):
            for j, user_j in enumerate(users):
                if i != j:
                    user_i_data = users_df.loc[user_i]
                    user_j_data = users_df.loc[user_j]
                    
                    # 计算相似性
                    similarity = 0.0
                    
                    # 部门相似性
                    if user_i_data.get('dept') == user_j_data.get('dept'):
                        similarity += 0.5
                    
                    # 角色相似性
                    if user_i_data.get('role') == user_j_data.get('role'):
                        similarity += 0.3
                    
                    # IT管理员特殊连接
                    if user_i_data.get('ITAdmin', 0) and user_j_data.get('ITAdmin', 0):
                        similarity += 0.2
                    
                    adj_matrix[i, j] = similarity
        
        # 添加自连接
        adj_matrix.fill_diagonal_(1.0)
        
        # 度归一化
        degree = adj_matrix.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1
        adj_matrix = adj_matrix / degree
        
        return adj_matrix

def test_user_gnn():
    """测试用户GNN"""
    print("🧪 Testing User GNN...")
    
    # 创建模拟数据
    num_users = 50
    input_dim = 10
    
    # 模拟用户特征
    node_features = torch.randn(num_users, input_dim)
    
    # 模拟邻接矩阵（随机稀疏图）
    adj_matrix = torch.rand(num_users, num_users)
    adj_matrix = (adj_matrix > 0.8).float()  # 稀疏化
    adj_matrix.fill_diagonal_(1.0)  # 自连接
    
    # 度归一化
    degree = adj_matrix.sum(dim=1, keepdim=True)
    degree[degree == 0] = 1
    adj_matrix = adj_matrix / degree
    
    # 创建模型
    model = UserGNN(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=128,
        num_layers=3
    )
    
    print(f"  Node features shape: {node_features.shape}")
    print(f"  Adjacency matrix shape: {adj_matrix.shape}")
    print(f"  Adjacency matrix density: {(adj_matrix > 0).float().mean():.3f}")
    
    # 前向传播
    with torch.no_grad():
        user_embeddings = model(node_features, adj_matrix)
    
    print(f"  User embeddings shape: {user_embeddings.shape}")
    
    # 测试梯度
    model.train()
    user_embeddings = model(node_features, adj_matrix)
    loss = user_embeddings.sum()
    loss.backward()
    
    print("  ✅ Gradient computation successful")
    print("  ✅ User GNN test passed")

if __name__ == "__main__":
    test_user_gnn() 