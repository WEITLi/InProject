#!/usr/bin/env python
# coding: utf-8

"""
上下文增强的 Transformer 模型
支持序列建模、上下文融合和多任务学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class PositionalEncoding(nn.Module):
    """位置编码层"""
    
    def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
        - x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ContextFusionLayer(nn.Module):
    """上下文信息融合层"""
    
    def __init__(self, hidden_dim: int, context_dim: int, fusion_type: str = 'attention'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # 注意力机制融合
            self.context_projection = nn.Linear(context_dim, hidden_dim)
            self.attention_weights = nn.Linear(hidden_dim * 2, 1)
            
        elif fusion_type == 'gating':
            # 门控机制融合
            self.context_projection = nn.Linear(context_dim, hidden_dim)
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
            
        elif fusion_type == 'concat':
            # 直接拼接
            self.fusion_linear = nn.Linear(hidden_dim + context_dim, hidden_dim)
            
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
    
    def forward(self, sequence_features: torch.Tensor, 
                context_features: torch.Tensor) -> torch.Tensor:
        """
        参数:
        - sequence_features: [batch_size, seq_len, hidden_dim]
        - context_features: [batch_size, context_dim]
        
        返回:
        - fused_features: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = sequence_features.shape
        
        if self.fusion_type == 'attention':
            # 将上下文特征投影到序列特征空间
            projected_context = self.context_projection(context_features)  # [batch_size, hidden_dim]
            projected_context = projected_context.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
            
            # 计算注意力权重
            combined = torch.cat([sequence_features, projected_context], dim=-1)  # [batch_size, seq_len, hidden_dim*2]
            attention_weights = torch.sigmoid(self.attention_weights(combined))  # [batch_size, seq_len, 1]
            
            # 加权融合
            fused_features = attention_weights * sequence_features + (1 - attention_weights) * projected_context
            
        elif self.fusion_type == 'gating':
            # 门控融合
            projected_context = self.context_projection(context_features)
            projected_context = projected_context.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 计算门控值
            combined = torch.cat([sequence_features, projected_context], dim=-1)
            gate_values = torch.sigmoid(self.gate(combined))
            
            # 门控融合
            fused_features = gate_values * sequence_features + (1 - gate_values) * projected_context
            
        elif self.fusion_type == 'concat':
            # 直接拼接并线性变换
            expanded_context = context_features.unsqueeze(1).expand(-1, seq_len, -1)
            combined = torch.cat([sequence_features, expanded_context], dim=-1)
            fused_features = self.fusion_linear(combined)
        
        return fused_features

class TransformerThreatDetector(nn.Module):
    """上下文增强的 Transformer 威胁检测模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 输入投影层
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            config.hidden_dim, config.sequence_length, config.dropout
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        
        # 上下文融合层
        self.context_fusion = ContextFusionLayer(
            config.hidden_dim, config.context_dim, fusion_type='attention'
        )
        
        # 分类头
        self.classification_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 2)  # 二分类
        )
        
        # 掩蔽语言模型头（用于自监督学习）
        self.masked_lm_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim)  # 重构原始特征
        )
        
        # 池化层用于序列级别预测
        self.pooling_type = 'attention'  # 可选: 'mean', 'max', 'attention'
        
        if self.pooling_type == 'attention':
            self.attention_pooling = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, sequences: torch.Tensor, contexts: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                masked_sequences: Optional[torch.Tensor] = None,
                return_all_outputs: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
        - sequences: [batch_size, seq_len, input_dim] 原始序列
        - contexts: [batch_size, context_dim] 上下文特征
        - attention_mask: [batch_size, seq_len] 注意力掩码
        - masked_sequences: [batch_size, seq_len, input_dim] 掩蔽序列（用于MLM）
        - return_all_outputs: 是否返回所有中间输出
        
        返回:
        - 包含各种输出的字典
        """
        batch_size, seq_len, input_dim = sequences.shape
        
        # 选择输入序列（用于掩蔽语言模型或正常分类）
        input_sequences = masked_sequences if masked_sequences is not None else sequences
        
        # 输入投影
        x = self.input_projection(input_sequences)  # [batch_size, seq_len, hidden_dim]
        
        # 位置编码（需要转置以适应位置编码的输入格式）
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Transformer编码
        if attention_mask is not None:
            # 创建padding mask（True表示忽略的位置）
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        encoded_sequences = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # [batch_size, seq_len, hidden_dim]
        
        # 上下文信息融合
        fused_features = self.context_fusion(encoded_sequences, contexts)
        
        # 序列级别池化
        if self.pooling_type == 'mean':
            if attention_mask is not None:
                # 忽略padding位置的平均
                lengths = attention_mask.sum(dim=1, keepdim=True)
                pooled_output = (fused_features * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
            else:
                pooled_output = fused_features.mean(dim=1)
        
        elif self.pooling_type == 'max':
            if attention_mask is not None:
                # 将padding位置设为很小的值
                masked_features = fused_features.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )
                pooled_output = masked_features.max(dim=1)[0]
            else:
                pooled_output = fused_features.max(dim=1)[0]
        
        elif self.pooling_type == 'attention':
            # 注意力加权池化
            attention_weights = self.attention_pooling(fused_features)  # [batch_size, seq_len, 1]
            
            if attention_mask is not None:
                attention_weights = attention_weights.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )
            
            attention_weights = F.softmax(attention_weights, dim=1)
            pooled_output = (fused_features * attention_weights).sum(dim=1)
        
        # 分类预测
        classification_logits = self.classification_head(pooled_output)
        
        # 构建输出字典
        outputs = {
            'classification_logits': classification_logits,
            'pooled_output': pooled_output
        }
        
        # 掩蔽语言模型输出（如果提供了掩蔽序列）
        if masked_sequences is not None:
            mlm_logits = self.masked_lm_head(fused_features)
            outputs['mlm_logits'] = mlm_logits
            outputs['target_sequences'] = sequences
        
        # 返回中间结果（用于分析和可视化）
        if return_all_outputs:
            outputs.update({
                'encoded_sequences': encoded_sequences,
                'fused_features': fused_features,
                'attention_weights': attention_weights if self.pooling_type == 'attention' else None
            })
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    labels: torch.Tensor, 
                    mask_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        参数:
        - outputs: 模型输出
        - labels: 分类标签
        - mask_positions: 掩蔽位置（用于MLM损失）
        
        返回:
        - 损失字典
        """
        losses = {}
        
        # 分类损失
        classification_loss = F.cross_entropy(
            outputs['classification_logits'], labels
        )
        losses['classification_loss'] = classification_loss
        
        # 掩蔽语言模型损失
        if 'mlm_logits' in outputs and mask_positions is not None:
            mlm_logits = outputs['mlm_logits']
            target_sequences = outputs['target_sequences']
            
            # 只计算被掩蔽位置的损失
            batch_size, seq_len, input_dim = mlm_logits.shape
            
            # 展平张量以便计算损失
            mlm_logits_flat = mlm_logits.view(-1, input_dim)
            target_flat = target_sequences.view(-1, input_dim)
            
            # 创建掩码（只有被掩蔽的位置参与损失计算）
            if isinstance(mask_positions, list):
                # 处理批次中的掩码列表
                mask_flat = torch.zeros(batch_size * seq_len, dtype=torch.bool, device=mlm_logits.device)
                for i, mask_pos in enumerate(mask_positions):
                    start_idx = i * seq_len
                    end_idx = start_idx + len(mask_pos)
                    if len(mask_pos) > 0:
                        mask_flat[start_idx:start_idx + len(mask_pos)] = mask_pos
            else:
                mask_flat = mask_positions.view(-1)
            
            # 只对被掩蔽的位置计算MSE损失
            if mask_flat.any():
                masked_mlm_logits = mlm_logits_flat[mask_flat]
                masked_targets = target_flat[mask_flat]
                mlm_loss = F.mse_loss(masked_mlm_logits, masked_targets)
            else:
                mlm_loss = torch.tensor(0.0, device=mlm_logits.device)
            
            losses['mlm_loss'] = mlm_loss
        
        # 总损失
        total_loss = self.config.classification_weight * classification_loss
        if 'mlm_loss' in losses:
            total_loss += self.config.masked_lm_weight * losses['mlm_loss']
        
        losses['total_loss'] = total_loss
        
        return losses

class ModelConfig:
    """模型配置类"""
    
    def __init__(self, input_dim: int, context_dim: int, **kwargs):
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # 默认参数
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.num_layers = kwargs.get('num_layers', 4)
        self.num_heads = kwargs.get('num_heads', 8)
        self.dropout = kwargs.get('dropout', 0.1)
        self.sequence_length = kwargs.get('sequence_length', 30)
        
        # 多任务学习权重
        self.classification_weight = kwargs.get('classification_weight', 1.0)
        self.masked_lm_weight = kwargs.get('masked_lm_weight', 0.5)
        
    def __repr__(self):
        return f"ModelConfig(input_dim={self.input_dim}, context_dim={self.context_dim}, " \
               f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers})" 