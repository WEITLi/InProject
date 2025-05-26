#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Encoder for Behavioral Sequence Modeling
用于行为序列建模的Transformer编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    """
    Transformer编码器用于行为序列建模
    
    将用户的行为事件序列编码为高维表示，捕获时序依赖关系
    """
    
    def __init__(self, input_dim=256, hidden_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            pooled_output: [batch_size, hidden_dim]
        """
        x = self.input_projection(x)
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        x = self.pos_encoding(x)
        
        encoded = self.transformer_encoder(x)
        encoded = encoded.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        output = self.output_projection(encoded)
        pooled_output = output.mean(dim=1)  # [batch_size, hidden_dim]
        
        return pooled_output

def test_transformer_encoder():
    """测试Transformer编码器"""
    print("🧪 测试 Transformer Encoder...")
    
    model = TransformerEncoder(input_dim=128, hidden_dim=256)
    x = torch.randn(16, 32, 128)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ Transformer Encoder 测试通过")

if __name__ == "__main__":
    test_transformer_encoder() 