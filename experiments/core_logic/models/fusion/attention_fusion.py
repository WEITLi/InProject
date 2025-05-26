#!/usr/bin/env python3
"""Simplified Attention Fusion for Multi-modal Features"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class AttentionFusion(nn.Module):
    """
    简化的注意力融合模块
    
    实现多模态特征的融合策略：
    1. 特征投影到统一空间
    2. 模态门控机制
    3. 加权融合
    """
    
    def __init__(
        self,
        input_dims: List[int],
        embed_dim: int = 256,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.embed_dim = embed_dim
        self.num_modalities = len(input_dims)
        self.use_gating = use_gating
        
        # 模态投影层
        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for dim in input_dims
        ])
        
        # 模态门控网络（可选）
        if use_gating:
            self.gate_network = nn.Sequential(
                nn.Linear(embed_dim * self.num_modalities, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, self.num_modalities),
                nn.Softmax(dim=-1)
            )
        
        # 注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, modality_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            modality_features: 各模态特征列表，每个元素形状为 [batch_size, feature_dim]
            
        Returns:
            融合结果字典
        """
        batch_size = modality_features[0].shape[0]
        
        # 1. 投影到统一空间
        projected_features = []
        for i, features in enumerate(modality_features):
            projected = self.modality_projections[i](features)
            projected_features.append(projected)
        
        # 堆叠特征 [batch_size, num_modalities, embed_dim]
        modality_stack = torch.stack(projected_features, dim=1)
        
        # 2. 计算注意力权重
        attention_scores = []
        for i in range(self.num_modalities):
            score = self.attention_weights(projected_features[i])  # [batch_size, 1]
            attention_scores.append(score)
        
        attention_scores = torch.cat(attention_scores, dim=1)  # [batch_size, num_modalities]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 3. 模态门控（可选）
        if self.use_gating:
            # 连接所有模态特征
            concatenated = torch.cat(projected_features, dim=1)  # [batch_size, embed_dim * num_modalities]
            gate_weights = self.gate_network(concatenated)  # [batch_size, num_modalities]
            
            # 结合注意力权重和门控权重
            final_weights = attention_weights * gate_weights
            final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)
        else:
            final_weights = attention_weights
            gate_weights = torch.ones_like(attention_weights)
        
        # 4. 加权融合
        weighted_features = torch.sum(
            modality_stack * final_weights.unsqueeze(-1), 
            dim=1
        )
        
        # 5. 最终变换
        fused_features = self.final_fusion(weighted_features)
        
        return {
            "fused_features": fused_features,           # [batch_size, embed_dim]
            "modality_features": modality_stack,        # [batch_size, num_modalities, embed_dim]
            "attention_weights": attention_weights,     # [batch_size, num_modalities]
            "gate_weights": gate_weights if self.use_gating else None,  # [batch_size, num_modalities]
            "final_weights": final_weights             # [batch_size, num_modalities]
        }

def test_attention_fusion():
    """测试注意力融合模块"""
    print("🧪 Testing Attention Fusion...")
    
    # 创建模拟多模态特征
    batch_size = 16
    modality_features = [
        torch.randn(batch_size, 256),  # Transformer特征
        torch.randn(batch_size, 128),  # GNN特征  
        torch.randn(batch_size, 512),  # BERT特征
        torch.randn(batch_size, 64),   # LGBM特征
    ]
    
    input_dims = [256, 128, 512, 64]
    
    print(f"  输入模态数量: {len(modality_features)}")
    print(f"  输入特征形状: {[f.shape for f in modality_features]}")
    
    # 创建注意力融合模型
    model = AttentionFusion(
        input_dims=input_dims,
        embed_dim=256,
        use_gating=True
    )
    
    # 前向传播
    with torch.no_grad():
        outputs = model(modality_features)
    
    print(f"  融合特征形状: {outputs['fused_features'].shape}")
    print(f"  模态特征形状: {outputs['modality_features'].shape}")
    print(f"  注意力权重形状: {outputs['attention_weights'].shape}")
    
    # 显示权重
    attention_weights = outputs['attention_weights'].mean(dim=0)
    final_weights = outputs['final_weights'].mean(dim=0)
    print(f"  平均注意力权重: {attention_weights.numpy()}")
    print(f"  最终融合权重: {final_weights.numpy()}")
    
    # 测试梯度
    model.train()
    outputs = model(modality_features)
    loss = outputs['fused_features'].sum()
    loss.backward()
    
    print("  ✅ Attention Fusion 测试通过")
    
    # 测试不同配置
    print("\n🧪 Testing Different Configurations...")
    
    # 简化配置（无门控）
    simple_model = AttentionFusion(
        input_dims=input_dims,
        embed_dim=128,
        use_gating=False
    )
    
    with torch.no_grad():
        simple_outputs = simple_model(modality_features)
    
    print(f"  简化模型融合特征形状: {simple_outputs['fused_features'].shape}")
    print(f"  门控权重: {simple_outputs['gate_weights']}")
    print("  ✅ 简化配置测试通过")

if __name__ == "__main__":
    test_attention_fusion() 