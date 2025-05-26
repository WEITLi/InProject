#!/usr/bin/env python3
"""BERT Text Encoder for Email and Web Content Processing using Hugging Face Transformers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import re
# import numpy as np # No longer needed for the core BERTTextEncoder
from transformers import AutoTokenizer, AutoModel # NEW IMPORT

class TextPreprocessor:
    """文本预处理器 (保留，因为其清理功能可能仍有用) """
    
    def __init__(self, max_length: int = 512): # max_length here is for TextPreprocessor's own logic if any
        self.max_length = max_length # This might be less relevant if BERT tokenizer handles truncation
        
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not isinstance(text, str):
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符，保留基本标点
        # This might be too aggressive for BERT, which handles punctuation. Consider revising.
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text) 
        
        return text.strip()
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """提取文本特征 (保留，可能用于其他目的或混合特征)"""
        if not text:
            return {
                'length': 0.0,
                'word_count': 0.0,
                'has_url': 0.0,
                'has_email': 0.0,
                'uppercase_ratio': 0.0,
                'special_char_ratio': 0.0
            }
        
        features = {}
        features['length'] = float(len(text))
        features['word_count'] = float(len(text.split()))
        features['has_url'] = 1.0 if re.search(r'http[s]?://|www\.', text.lower()) else 0.0
        features['has_email'] = 1.0 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text) else 0.0
        
        if len(text) > 0:
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        else:
            features['uppercase_ratio'] = 0.0
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        features['special_char_ratio'] = special_chars / len(text) if len(text) > 0 else 0.0
        
        return features

# class SimpleBERTEncoder(nn.Module): # Will be removed or commented out
#     """..."""
# class TextTokenizer: # Will be removed or commented out
#     """..."""

class BERTTextEncoder(nn.Module):
    """
    BERT文本编码器, 使用 Hugging Face Transformers 加载 'bert-base-uncased'.
    用于处理邮件内容、网页内容等文本数据.
    """
    
    def __init__(
        self,
        output_dim: int = 256, # Desired output dimension after projection
        bert_model_name: str = 'bert-base-uncased',
        max_length: int = 256, # Max sequence length for tokenizer
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        
        self.max_length = max_length
        self.bert_hidden_dim = self.bert_model.config.hidden_size # e.g., 768 for bert-base-uncased
        
        # 输出投影层，将BERT的输出调整到所需的output_dim
        self.output_projection = nn.Sequential(
            nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim // 2), # Intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.bert_hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            texts: 文本字符串列表
            
        Returns:
            text_embeddings: [batch_size, output_dim] 文本嵌入
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # 使用Hugging Face tokenizer进行编码
        # padding=True会填充到批次中最长序列的长度
        # truncation=True会截断超过max_length的序列
        # return_tensors='pt'返回PyTorch张量
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 将输入移动到BERT模型所在的设备 (如果BERT模型在GPU上)
        # 获取当前模块的设备，并确保输入数据在该设备上
        current_device = next(self.parameters()).device
        input_ids = encoded_input['input_ids'].to(current_device)
        attention_mask = encoded_input['attention_mask'].to(current_device)
        
        # 获取BERT模型输出
        with torch.no_grad(): # 通常在推理时不对预训练的BERT部分进行梯度计算，除非要微调
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 我们通常使用[CLS] token的输出来表示整个句子的嵌入
        # outputs.last_hidden_state 的形状是 [batch_size, seq_len, hidden_size]
        # [CLS] token 是序列的第一个token
        cls_embeddings = outputs.last_hidden_state[:, 0, :] # [batch_size, bert_hidden_dim]
        
        # 或者，也可以使用平均池化 (mean pooling)
        # attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        # sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, 1)
        # sum_mask = attention_mask_expanded.sum(1)
        # sum_mask = torch.clamp(sum_mask, min=1e-9) # Avoid division by zero
        # pooled_output = sum_embeddings / sum_mask # [batch_size, bert_hidden_dim]
        # current_embeddings = pooled_output

        current_embeddings = cls_embeddings # Using CLS token embedding
        
        # 通过输出投影层
        projected_output = self.output_projection(current_embeddings)
        
        # print(f"[DEBUG BERTTextEncoder] Input texts (first 2 if batch): {texts[:2]}")
        # print(f"[DEBUG BERTTextEncoder] Encoded input_ids shape: {encoded_input['input_ids'].shape}")
        # print(f"[DEBUG BERTTextEncoder] CLS embeddings shape: {cls_embeddings.shape}")
        # print(f"[DEBUG BERTTextEncoder] Projected output shape: {projected_output.shape}")
        if projected_output.numel() == 0:
            # print("[DEBUG BERTTextEncoder] WARNING: Projected output is empty!")
            pass #保留结构
        elif projected_output.shape[0] > 0 : # if batch not empty
            # print(f"[DEBUG BERTTextEncoder] Projected output (first sample if batch): {projected_output[0, :5]}") # Print first 5 features of first sample
            pass #保留结构

        return projected_output

def test_bert_encoder():
    """测试新的BERT编码器"""
    print("🧪 Testing BERT Text Encoder (Hugging Face)...")
    
    # 创建模型
    # output_dim可以根据你的系统需求调整
    # max_length可以根据你的文本特性调整
    try:
        model = BERTTextEncoder(
            output_dim=128, 
            max_length=64 
        )
        model.eval() # Set to evaluation mode for testing
    except Exception as e:
        print(f"❌ Error creating BERTTextEncoder: {e}")
        print("👉 Make sure you have the 'transformers' library installed (`pip install transformers`).")
        print("👉 Also, ensure an internet connection is available to download the model for the first time.")
        return

    # 测试文本
    test_texts = [
        "This is an urgent email about system maintenance.",
        "Please click on this link to reset your password.",
        "Meeting scheduled for tomorrow at 2 PM.",
        "Your account has been locked due to suspicious activity."
    ]
    
    print(f"  Input text count: {len(test_texts)}")
    if test_texts:
        print(f"  Sample text: '{test_texts[0]}'")
    
    # 前向传播
    try:
        with torch.no_grad():
            embeddings = model(test_texts)
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        return
    
    print(f"  Output embedding shape: {embeddings.shape}")
    if embeddings.numel() > 0 : # Check if tensor is not empty
        print(f"  Embedding mean: {embeddings.mean().item():.4f}")
        print(f"  Embedding std: {embeddings.std().item():.4f}")
    else:
        print("  Output embeddings are empty.")

    # 测试梯度 (如果需要微调BERT，可以去掉torch.no_grad()并测试梯度)
    # For now, we assume BERT is used for feature extraction without fine-tuning
    # model.train()
    # embeddings_for_grad = model(test_texts)
    # try:
    #     loss = embeddings_for_grad.sum()
    #     loss.backward()
    #     print("  ✅ Gradient test passed (simulated for projection layer)")
    # except Exception as e:
    #     print(f"❌ Error during gradient test: {e}")

    # 检查投影层是否有梯度
    model.train() # Switch to train mode to ensure gradients are computed for projection layer
    
    # Create a fresh input for grad test to avoid issues with graph already being freed
    # This ensures that the tokenizer and bert_model are called on the correct device if model was moved.
    current_device = next(model.parameters()).device
    
    encoded_input_grad = model.tokenizer(
        test_texts, padding=True, truncation=True, max_length=model.max_length, return_tensors='pt'
    )
    input_ids_grad = encoded_input_grad['input_ids'].to(current_device)
    attention_mask_grad = encoded_input_grad['attention_mask'].to(current_device)

    # BERT part (still no_grad if we are not fine-tuning BERT itself)
    # Ensure bert_model is also on the correct device if it was moved separately
    # (though it should be part of the module and move with model.to(device))
    with torch.no_grad():
         bert_outputs_grad = model.bert_model(input_ids=input_ids_grad, attention_mask=attention_mask_grad)
    cls_embeddings_grad = bert_outputs_grad.last_hidden_state[:, 0, :].to(current_device) 
    
    # Projection layer part (will have grads)
    projected_output_grad = model.output_projection(cls_embeddings_grad)
    
    try:
        loss = projected_output_grad.sum()
        loss.backward()
        
        grad_check_passed = False
        for param in model.output_projection.parameters():
            if param.grad is not None:
                grad_check_passed = True
                break
        if grad_check_passed:
            print("  ✅ Gradient test passed for the projection layer.")
        else:
            print("  ⚠️ Gradient test: No gradients found in the projection layer. Check setup.")
            
    except Exception as e:
        print(f"❌ Error during gradient test for projection layer: {e}")

    print("  ✅ BERT Text Encoder (Hugging Face) test attempt finished.")

if __name__ == "__main__":
    test_bert_encoder() 