#!/usr/bin/env python3
"""Classification Head for Anomaly Detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ClassificationHead(nn.Module):
    """
    分类头模块
    
    用于异常检测的二分类或多分类任务
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 2,
        hidden_dims: list = [128, 64],
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 构建分类器
        self.classifier = self._build_classifier(dropout)
        
        # 用于计算不确定性的额外头（可选）
        self.uncertainty_head = None
        
    def _build_classifier(self, dropout: float) -> nn.Module:
        """构建分类器网络"""
        layers = []
        
        # 输入层
        prev_dim = self.input_dim
        
        # 隐藏层
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: [batch_size, input_dim] 输入特征
            
        Returns:
            logits: [batch_size, num_classes] 分类logits
        """
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Args:
            features: [batch_size, input_dim] 输入特征
            
        Returns:
            probs: [batch_size, num_classes] 分类概率
        """
        logits = self.forward(features)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            features: [batch_size, input_dim] 输入特征
            
        Returns:
            predictions: [batch_size] 预测类别
        """
        logits = self.forward(features)
        predictions = torch.argmax(logits, dim=1)
        return predictions

class AnomalyDetectionHead(nn.Module):
    """
    异常检测头
    
    专门用于异常检测，包含异常分数和置信度
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [128, 64],
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.temperature = temperature
        
        # 异常分数分支
        self.anomaly_branch = self._build_branch(input_dim, hidden_dims, 1, dropout)
        
        # 置信度分支
        self.confidence_branch = self._build_branch(input_dim, hidden_dims, 1, dropout)
        
    def _build_branch(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float) -> nn.Module:
        """构建分支网络"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: [batch_size, input_dim] 输入特征
            
        Returns:
            outputs: 包含异常分数和置信度的字典
        """
        # 异常分数 (0-1, 越高越异常)
        anomaly_score = torch.sigmoid(self.anomaly_branch(features) / self.temperature)
        
        # 置信度 (0-1, 越高越确信)
        confidence = torch.sigmoid(self.confidence_branch(features))
        
        return {
            "anomaly_score": anomaly_score.squeeze(-1),  # [batch_size]
            "confidence": confidence.squeeze(-1),        # [batch_size]
        }
    
    def predict(self, features: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        预测异常
        
        Args:
            features: [batch_size, input_dim] 输入特征
            threshold: 异常判定阈值
            
        Returns:
            predictions: 包含预测结果的字典
        """
        outputs = self.forward(features)
        
        # 二值化预测
        is_anomaly = (outputs["anomaly_score"] > threshold).float()
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": outputs["anomaly_score"],
            "confidence": outputs["confidence"]
        }

class MultiTaskHead(nn.Module):
    """
    多任务头
    
    同时进行异常检测和风险等级分类
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [128, 64],
        num_risk_levels: int = 4,  # 低、中、高、极高
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_risk_levels = num_risk_levels
        
        # 共享特征提取
        self.shared_layers = self._build_shared_layers(input_dim, hidden_dims, dropout)
        
        # 异常检测头
        self.anomaly_head = nn.Linear(hidden_dims[-1], 1)
        
        # 风险等级分类头
        self.risk_head = nn.Linear(hidden_dims[-1], num_risk_levels)
        
        # 严重性回归头
        self.severity_head = nn.Linear(hidden_dims[-1], 1)
        
    def _build_shared_layers(self, input_dim: int, hidden_dims: list, dropout: float) -> nn.Module:
        """构建共享层"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: [batch_size, input_dim] 输入特征
            
        Returns:
            outputs: 包含多个任务输出的字典
        """
        # 共享特征
        shared_features = self.shared_layers(features)
        
        # 异常检测
        anomaly_logits = self.anomaly_head(shared_features)
        anomaly_probs = torch.sigmoid(anomaly_logits.squeeze(-1))
        
        # 风险等级分类
        risk_logits = self.risk_head(shared_features)
        risk_probs = F.softmax(risk_logits, dim=1)
        
        # 严重性回归
        severity_scores = torch.sigmoid(self.severity_head(shared_features).squeeze(-1))
        
        return {
            "anomaly_logits": anomaly_logits.squeeze(-1),
            "anomaly_probs": anomaly_probs,
            "risk_logits": risk_logits,
            "risk_probs": risk_probs,
            "severity_scores": severity_scores
        }
    
    def predict(self, features: torch.Tensor, anomaly_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """预测多个任务"""
        outputs = self.forward(features)
        
        # 异常检测预测
        is_anomaly = (outputs["anomaly_probs"] > anomaly_threshold).float()
        
        # 风险等级预测
        risk_predictions = torch.argmax(outputs["risk_logits"], dim=1)
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": outputs["anomaly_probs"],
            "risk_level": risk_predictions,
            "risk_probs": outputs["risk_probs"],
            "severity_score": outputs["severity_scores"]
        }

def test_classification_heads():
    """测试分类头模块"""
    print("🧪 Testing Classification Heads...")
    
    batch_size = 32
    input_dim = 256
    features = torch.randn(batch_size, input_dim)
    
    # 测试基础分类头
    print("\n  Testing ClassificationHead:")
    classifier = ClassificationHead(
        input_dim=input_dim,
        num_classes=2,
        hidden_dims=[128, 64]
    )
    
    with torch.no_grad():
        logits = classifier(features)
        probs = classifier.predict_proba(features)
        predictions = classifier.predict(features)
    
    print(f"    Input shape: {features.shape}")
    print(f"    Logits shape: {logits.shape}")
    print(f"    Probs shape: {probs.shape}")
    print(f"    Predictions shape: {predictions.shape}")
    print("    ✅ ClassificationHead test passed")
    
    # 测试异常检测头
    print("\n  Testing AnomalyDetectionHead:")
    anomaly_detector = AnomalyDetectionHead(
        input_dim=input_dim,
        hidden_dims=[128, 64]
    )
    
    with torch.no_grad():
        outputs = anomaly_detector(features)
        predictions = anomaly_detector.predict(features, threshold=0.5)
    
    print(f"    Anomaly scores shape: {outputs['anomaly_score'].shape}")
    print(f"    Confidence shape: {outputs['confidence'].shape}")
    print(f"    Binary predictions shape: {predictions['is_anomaly'].shape}")
    print("    ✅ AnomalyDetectionHead test passed")
    
    # 测试多任务头
    print("\n  Testing MultiTaskHead:")
    multitask_head = MultiTaskHead(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        num_risk_levels=4
    )
    
    with torch.no_grad():
        outputs = multitask_head(features)
        predictions = multitask_head.predict(features)
    
    print(f"    Anomaly probs shape: {outputs['anomaly_probs'].shape}")
    print(f"    Risk probs shape: {outputs['risk_probs'].shape}")
    print(f"    Severity scores shape: {outputs['severity_scores'].shape}")
    print(f"    Risk predictions shape: {predictions['risk_level'].shape}")
    print("    ✅ MultiTaskHead test passed")
    
    # 测试梯度
    print("\n  Testing gradient computation:")
    classifier.train()
    anomaly_detector.train()
    multitask_head.train()
    
    # 分类头梯度
    logits = classifier(features)
    loss1 = logits.sum()
    loss1.backward()
    
    # 异常检测头梯度
    outputs = anomaly_detector(features)
    loss2 = outputs["anomaly_score"].sum()
    loss2.backward()
    
    # 多任务头梯度
    outputs = multitask_head(features)
    loss3 = (outputs["anomaly_probs"].sum() + 
             outputs["risk_probs"].sum() + 
             outputs["severity_scores"].sum())
    loss3.backward()
    
    print("    ✅ All gradient computations successful")
    print("\n  ✅ All Classification Head tests passed")

if __name__ == "__main__":
    test_classification_heads() 