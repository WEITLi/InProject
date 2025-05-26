#!/usr/bin/env python3
"""测试所有base_model模块"""

import torch
import sys
import os

# 添加项目根目录到 sys.path
# Current script: .../feature_extraction_scenario/tests/test_models.py
# Project root for imports (InProject): .../feature_extraction_scenario/../../
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT) # Insert at the beginning

def test_transformer():
    """测试Transformer编码器"""
    print("🧪 测试 Transformer Encoder...")
    
    from feature_extraction_scenario.core_logic.models.base_model.transformer_encoder import TransformerEncoder
    
    model = TransformerEncoder(input_dim=128, hidden_dim=256)
    x = torch.randn(16, 32, 128)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ Transformer Encoder 测试通过")

def test_user_gnn():
    """测试用户GNN"""
    print("\n🧪 测试 User GNN...")
    
    from feature_extraction_scenario.core_logic.models.base_model.user_gnn import UserGNN
    
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
    
    print(f"  节点特征形状: {node_features.shape}")
    print(f"  邻接矩阵形状: {adj_matrix.shape}")
    print(f"  邻接矩阵密度: {(adj_matrix > 0).float().mean():.3f}")
    
    # 前向传播
    with torch.no_grad():
        user_embeddings = model(node_features, adj_matrix)
    
    print(f"  用户嵌入形状: {user_embeddings.shape}")
    print("  ✅ User GNN 测试通过")

def test_base_fusion():
    """测试基础融合模块"""
    print("\n🧪 测试 Base Fusion...")
    
    from feature_extraction_scenario.core_logic.models.base_model.base_fusion import BaseFusion
    
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
    
    # 测试拼接融合
    model = BaseFusion(
        input_dims=input_dims,
        output_dim=output_dim,
        fusion_type="concat"
    )
    
    with torch.no_grad():
        output = model(features)
    
    print(f"  输入形状: {[f.shape for f in features]}")
    print(f"  输出形状: {output.shape}")
    print("  ✅ Base Fusion 测试通过")

def test_classification_head():
    """测试分类头"""
    print("\n🧪 测试 Classification Head...")
    
    from feature_extraction_scenario.core_logic.models.base_model.head import ClassificationHead, AnomalyDetectionHead
    
    batch_size = 32
    input_dim = 256
    features = torch.randn(batch_size, input_dim)
    
    # 测试基础分类头
    classifier = ClassificationHead(
        input_dim=input_dim,
        num_classes=2,
        hidden_dims=[128, 64]
    )
    
    with torch.no_grad():
        logits = classifier(features)
        probs = classifier.predict_proba(features)
        predictions = classifier.predict(features)
    
    print(f"  分类logits形状: {logits.shape}")
    print(f"  分类概率形状: {probs.shape}")
    print(f"  预测结果形状: {predictions.shape}")
    
    # 测试异常检测头
    anomaly_detector = AnomalyDetectionHead(
        input_dim=input_dim,
        hidden_dims=[128, 64]
    )
    
    with torch.no_grad():
        outputs = anomaly_detector(features)
    
    print(f"  异常分数形状: {outputs['anomaly_score'].shape}")
    print(f"  置信度形状: {outputs['confidence'].shape}")
    print("  ✅ Classification Head 测试通过")

def main():
    """主测试函数"""
    print("🚀 开始测试所有 base_model 模块...\n")
    
    try:
        test_transformer()
        test_user_gnn()
        test_base_fusion()
        test_classification_head()
        
        print("\n🎉 所有模块测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 