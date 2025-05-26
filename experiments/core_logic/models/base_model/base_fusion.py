#!/usr/bin/env python3
"""Base Fusion Module for Multi-modal Feature Integration"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class BaseFusion(nn.Module):
    """
    基础融合模块
    
    支持多种融合策略：拼接、加权平均、注意力机制等
    """
    
    def __init__(
        self,
        input_dims: List[int],
        output_dim: int = 256,
        fusion_type: str = "concat",
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.num_modalities = len(input_dims)
        
        if fusion_type == "concat":
            self._build_concat_fusion()
        elif fusion_type == "weighted":
            self._build_weighted_fusion()
        elif fusion_type == "attention":
            self._build_attention_fusion(hidden_dim)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def _build_concat_fusion(self):
        """构建拼接融合"""
        total_input_dim = sum(self.input_dims)
        self.projection = nn.Sequential(
            nn.Linear(total_input_dim, self.output_dim * 2),
            nn.ReLU(),
            nn.Linear(self.output_dim * 2, self.output_dim)
        )
    
    def _build_weighted_fusion(self):
        """构建加权融合"""
        # 投影到相同维度
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.output_dim) for dim in self.input_dims
        ])
        
        # 学习权重
        self.fusion_weights = nn.Parameter(torch.ones(self.num_modalities))
        
    def _build_attention_fusion(self, hidden_dim: int):
        """构建注意力融合"""
        # 投影到相同维度
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.output_dim) for dim in self.input_dims
        ])
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 各模态特征列表 [modality1, modality2, ...]
                     每个元素形状为 [batch_size, feature_dim]
        
        Returns:
            fused_features: [batch_size, output_dim] 融合后的特征
        """
        if len(features) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(features)}")
        
        if self.fusion_type == "concat":
            return self._concat_fusion(features)
        elif self.fusion_type == "weighted":
            return self._weighted_fusion(features)
        elif self.fusion_type == "attention":
            return self._attention_fusion(features)
    
    def _concat_fusion(self, features: List[torch.Tensor]) -> torch.Tensor:
        """拼接融合"""
        # 拼接所有特征
        concatenated = torch.cat(features, dim=-1)  # [batch_size, sum(input_dims)]
        
        # 投影到输出维度
        output = self.projection(concatenated)
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output
    
    def _weighted_fusion(self, features: List[torch.Tensor]) -> torch.Tensor:
        """加权融合"""
        # 投影到相同维度
        projected_features = []
        for i, feature in enumerate(features):
            projected = self.projections[i](feature)
            projected_features.append(projected)
        
        # 堆叠特征
        stacked_features = torch.stack(projected_features, dim=1)  # [batch_size, num_modalities, output_dim]
        
        # 应用权重（使用softmax归一化）
        weights = F.softmax(self.fusion_weights, dim=0)  # [num_modalities]
        weights = weights.view(1, -1, 1)  # [1, num_modalities, 1]
        
        # 加权平均
        weighted_features = (stacked_features * weights).sum(dim=1)  # [batch_size, output_dim]
        
        output = self.dropout(weighted_features)
        output = self.layer_norm(output)
        
        return output
    
    def _attention_fusion(self, features: List[torch.Tensor]) -> torch.Tensor:
        """注意力融合"""
        # 投影到相同维度
        projected_features = []
        for i, feature in enumerate(features):
            projected = self.projections[i](feature)
            projected_features.append(projected)
        
        # 堆叠特征
        stacked_features = torch.stack(projected_features, dim=1)  # [batch_size, num_modalities, output_dim]
        
        # 计算注意力权重
        attention_scores = self.attention(stacked_features)  # [batch_size, num_modalities, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_modalities, 1]
        
        # 注意力加权
        attended_features = (stacked_features * attention_weights).sum(dim=1)  # [batch_size, output_dim]
        
        output = self.dropout(attended_features)
        output = self.layer_norm(output)
        
        return output
    
    def get_fusion_weights(self) -> Dict[str, torch.Tensor]:
        """获取融合权重（用于分析）"""
        if self.fusion_type == "weighted":
            return {"weights": F.softmax(self.fusion_weights, dim=0)}
        elif self.fusion_type == "attention":
            # 注意力权重需要在前向传播时计算
            return {"type": "attention", "message": "weights computed dynamically"}
        else:
            return {"type": self.fusion_type, "message": "no explicit weights"}

class ModalityGate(nn.Module):
    """模态门控机制"""
    
    def __init__(self, input_dim: int, num_modalities: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_modalities),
            nn.Sigmoid()
        )
    
    def forward(self, context: torch.Tensor, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        基于上下文信息对模态进行门控
        
        Args:
            context: [batch_size, input_dim] 上下文信息
            features: 各模态特征列表
            
        Returns:
            gated_features: 门控后的特征列表
        """
        gates = self.gate(context)  # [batch_size, num_modalities]
        
        gated_features = []
        for i, feature in enumerate(features):
            gate_weight = gates[:, i:i+1]  # [batch_size, 1]
            gated_feature = feature * gate_weight
            gated_features.append(gated_feature)
        
        return gated_features

def test_base_fusion():
    """测试基础融合模块"""
    print("🧪 Testing Base Fusion...")
    
    # 创建测试数据
    batch_size = 32
    features = [
        torch.randn(batch_size, 256),  # Transformer特征
        torch.randn(batch_size, 128),  # GNN特征
        torch.randn(batch_size, 512),  # BERT特征
        torch.randn(batch_size, 64),   # LGBM特征
    ]
    
    input_dims = [256, 128, 512, 64]
    output_dim = 256
    
    # 测试不同融合类型
    fusion_types = ["concat", "weighted", "attention"]
    
    for fusion_type in fusion_types:
        print(f"\n  Testing {fusion_type} fusion:")
        
        model = BaseFusion(
            input_dims=input_dims,
            output_dim=output_dim,
            fusion_type=fusion_type
        )
        
        # 前向传播
        with torch.no_grad():
            output = model(features)
        
        print(f"    Input shapes: {[f.shape for f in features]}")
        print(f"    Output shape: {output.shape}")
        
        # 检查梯度
        model.train()
        output = model(features)
        loss = output.sum()
        loss.backward()
        
        print(f"    ✅ {fusion_type} fusion test passed")
        
        # 显示权重信息
        weights_info = model.get_fusion_weights()
        if "weights" in weights_info:
            print(f"    Fusion weights: {weights_info['weights'].detach().numpy()}")
    
    # 测试模态门控
    print("\n  Testing Modality Gate:")
    context = torch.randn(batch_size, 100)
    gate = ModalityGate(input_dim=100, num_modalities=len(features))
    
    with torch.no_grad():
        gated_features = gate(context, features)
    
    print(f"    Gated features shapes: {[f.shape for f in gated_features]}")
    print("    ✅ Modality gate test passed")
    
    print("\n  ✅ All Base Fusion tests passed")

if __name__ == "__main__":
    test_base_fusion() 