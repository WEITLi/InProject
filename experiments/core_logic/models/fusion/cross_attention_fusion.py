#!/usr/bin/env python3
"""Cross-Attention Fusion for Multi-modal Features"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Union # Union was missing

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim: int, key_value_dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(key_value_dim, num_heads * head_dim, bias=False)
        self.to_v = nn.Linear(key_value_dim, num_heads * head_dim, bias=False)

        self.to_out = nn.Linear(num_heads * head_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key_value.shape

        q = self.to_q(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(key_value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.num_heads * self.head_dim)
        
        output = self.to_out(context)
        return output

class CrossAttentionFusion(nn.Module):
    def __init__(self,
                 input_dims: Dict[str, int],
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 query_modality_name: str = 'behavior'):
        super().__init__()
        self.input_dims = input_dims
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = embed_dim // num_heads
        self.query_modality_name = query_modality_name
        self.modality_names = list(input_dims.keys())

        self.modality_projections = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.modality_projections[name] = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        self.cross_attention_layers = nn.ModuleDict()
        if self.query_modality_name in self.modality_names:
            for other_name in self.modality_names:
                if other_name != self.query_modality_name and other_name in input_dims: # Ensure other_name is valid
                    self.cross_attention_layers[f"{self.query_modality_name}_x_{other_name}"] = CrossAttentionLayer(
                        query_dim=embed_dim,
                        key_value_dim=embed_dim,
                        num_heads=num_heads,
                        head_dim=self.head_dim,
                        dropout=dropout
                    )
        
        # Optional: LayerNorm for enhanced_query_feat before final fusion input prep
        self.query_feat_norm = nn.LayerNorm(embed_dim)

        final_fusion_input_dim = len(self.modality_names) * embed_dim
        self.final_fusion_mlp = nn.Sequential(
            nn.Linear(final_fusion_input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout) # General dropout, if needed elsewhere

    def forward(self, modality_features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        projected_features = {}
        batch_size = 0 # Determine batch size from the first available feature

        for name, features in modality_features_dict.items():
            if name not in self.modality_projections:
                continue
            if batch_size == 0:
                batch_size = features.shape[0]

            proj_feat = self.modality_projections[name](features)
            if proj_feat.ndim == 2:
                proj_feat = proj_feat.unsqueeze(1)
            projected_features[name] = proj_feat
        
        if not projected_features: # No enabled or valid modalities found
             # Fallback: return zero tensor of expected output shape
            if batch_size == 0: batch_size = 1 # Default batch size if none could be inferred
            return {
                "fused_features": torch.zeros(batch_size, self.embed_dim, device=next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'),
                "modality_features_projected": {}
            }

        # 🔧 修改：只处理实际存在的模态，不填充零向量
        available_modalities = list(projected_features.keys())
        
        # Cross-attention enhancement (only if query modality is available)
        if self.query_modality_name in projected_features:
            query_feat_orig = projected_features[self.query_modality_name]
            
            contexts_from_others = []
            for other_name, other_feat in projected_features.items():
                if other_name != self.query_modality_name:
                    layer_name = f"{self.query_modality_name}_x_{other_name}"
                    if layer_name in self.cross_attention_layers:
                        context = self.cross_attention_layers[layer_name](query_feat_orig, other_feat)
                        contexts_from_others.append(context)
            
            enhanced_query_feat = query_feat_orig
            if contexts_from_others:
                for ctx in contexts_from_others:
                    enhanced_query_feat = enhanced_query_feat + ctx # Residual addition
                enhanced_query_feat = self.query_feat_norm(enhanced_query_feat) # Normalize after additions

            projected_features[self.query_modality_name] = enhanced_query_feat

        # 🔧 修改：只使用实际可用的模态进行融合
        final_fusion_inputs_list = []
        device = next(iter(projected_features.values())).device if projected_features else \
                 (next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu')

        # 只遍历实际存在的模态，而不是所有预定义的模态
        for name in available_modalities:
            feat = projected_features[name]
            if name == self.query_modality_name and feat.size(1) > 1:
                final_fusion_inputs_list.append(feat.mean(dim=1))
            else:
                final_fusion_inputs_list.append(feat.squeeze(1))

        if not final_fusion_inputs_list: # If somehow still empty
            if batch_size == 0: batch_size = 1 # Default batch size
            return {
                "fused_features": torch.zeros(batch_size, self.embed_dim, device=device),
                "modality_features_projected": {}
            }

        # 🔧 修改：动态调整融合层以适应不同数量的模态
        if len(final_fusion_inputs_list) == 1:
            # 单模态情况：直接使用该模态的特征，可选择性地通过一个简单的变换
            fused_features = final_fusion_inputs_list[0]
            # 可选：通过一个线性层进行变换以保持一致性
            if hasattr(self, 'single_modality_transform'):
                fused_features = self.single_modality_transform(fused_features)
        else:
            # 多模态情况：拼接后通过MLP
            concatenated_for_final_fusion = torch.cat(final_fusion_inputs_list, dim=1)
            
            # 动态创建或选择合适的融合层
            current_input_dim = concatenated_for_final_fusion.shape[1]
            expected_input_dim = len(self.modality_names) * self.embed_dim
            
            if current_input_dim != expected_input_dim:
                # 需要动态调整融合层
                if not hasattr(self, 'dynamic_fusion_mlps'):
                    self.dynamic_fusion_mlps = nn.ModuleDict()
                
                fusion_key = f"fusion_{len(available_modalities)}_{current_input_dim}"
                if fusion_key not in self.dynamic_fusion_mlps:
                    self.dynamic_fusion_mlps[fusion_key] = nn.Sequential(
                        nn.Linear(current_input_dim, self.embed_dim * 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(self.embed_dim * 2, self.embed_dim),
                        nn.LayerNorm(self.embed_dim)
                    ).to(device)
                
                fused_features = self.dynamic_fusion_mlps[fusion_key](concatenated_for_final_fusion)
            else:
                # 使用原始的融合层
                fused_features = self.final_fusion_mlp(concatenated_for_final_fusion)

        return {
            "fused_features": fused_features,
            "modality_features_projected": projected_features
        } 