#!/usr/bin/env python3
"""Complete System Test for Multi-modal Anomaly Detection"""

import torch
import torch.nn as nn
import numpy as np
import importlib
import sys
import os

# 添加项目根目录到 sys.path
# Current script: .../feature_extraction_scenario/tests/test_complete_system.py
# Project root for imports (InProject): .../feature_extraction_scenario/../../
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 清除模块缓存
def clear_cache():
    modules_to_remove = []
    for module_name in sys.modules:
        if 'models' in module_name:
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    importlib.invalidate_caches()

def test_individual_modules():
    """测试各个模块是否正常工作"""
    print("🔧 Testing Individual Modules...")
    
    clear_cache()
    
    # 1. 测试 Transformer Encoder
    from feature_extraction_scenario.core_logic.models.base_model.transformer_encoder import TransformerEncoder
    transformer = TransformerEncoder(input_dim=128, hidden_dim=256)
    x = torch.randn(8, 32, 128)
    output = transformer(x)
    print(f"✅ TransformerEncoder: {x.shape} -> {output.shape}")
    
    # 2. 测试 User GNN
    from feature_extraction_scenario.core_logic.models.base_model.user_gnn import UserGNN
    gnn = UserGNN(input_dim=10, output_dim=128)
    node_features = torch.randn(50, 10)
    adj_matrix = torch.randint(0, 2, (50, 50)).float()
    gnn_output = gnn(node_features, adj_matrix)
    print(f"✅ UserGNN: {node_features.shape} -> {gnn_output.shape}")
    
    # 3. 测试 BERT Encoder
    from feature_extraction_scenario.core_logic.models.text_encoder.bert_module import BERTTextEncoder
    bert = BERTTextEncoder(output_dim=128)
    texts = ["Test email content"] * 8
    bert_output = bert(texts)
    print(f"✅ BERTTextEncoder: {len(texts)} texts -> {bert_output.shape}")
    
    # 4. 测试 LightGBM Branch
    from feature_extraction_scenario.core_logic.models.structure_encoder.lightgbm_branch import LightGBMBranch
    lgbm = LightGBMBranch(input_dim=20, output_dim=128)
    struct_features = torch.randn(8, 20)
    lgbm_output = lgbm(struct_features)
    print(f"✅ LightGBMBranch: {struct_features.shape} -> {lgbm_output.shape}")
    
    # 5. 测试 Attention Fusion
    from feature_extraction_scenario.core_logic.models.fusion.attention_fusion import AttentionFusion
    fusion = AttentionFusion(input_dims=[256, 128, 128, 128], embed_dim=128)
    modality_features = [
        torch.randn(8, 256),  # Transformer
        torch.randn(8, 128),  # GNN
        torch.randn(8, 128),  # BERT  
        torch.randn(8, 128),  # LightGBM
    ]
    fusion_output = fusion(modality_features)
    print(f"✅ AttentionFusion: 4 modalities -> {fusion_output['fused_features'].shape}")
    
    # 6. 测试 Classification Head
    from feature_extraction_scenario.core_logic.models.base_model.head import ClassificationHead
    head = ClassificationHead(input_dim=128, num_classes=2)
    head_output = head(fusion_output['fused_features'])
    print(f"✅ ClassificationHead: {fusion_output['fused_features'].shape} -> logits: {head_output.shape}")
    
    return True

def test_simplified_multimodal_model():
    """测试简化的多模态模型"""
    print("\n🚀 Testing Simplified Multi-modal Model...")
    
    clear_cache()
    
    # 手动构建简化的多模态模型
    class SimplifiedMultiModalModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # 各模态编码器
            from feature_extraction_scenario.core_logic.models.base_model.transformer_encoder import TransformerEncoder
            from feature_extraction_scenario.core_logic.models.base_model.user_gnn import UserGNN
            from feature_extraction_scenario.core_logic.models.text_encoder.bert_module import BERTTextEncoder
            from feature_extraction_scenario.core_logic.models.structure_encoder.lightgbm_branch import LightGBMBranch
            from feature_extraction_scenario.core_logic.models.fusion.attention_fusion import AttentionFusion
            from feature_extraction_scenario.core_logic.models.base_model.head import ClassificationHead
            
            self.transformer = TransformerEncoder(input_dim=128, hidden_dim=128)
            self.gnn = UserGNN(input_dim=10, output_dim=128)
            self.bert = BERTTextEncoder(output_dim=128)
            self.lgbm = LightGBMBranch(input_dim=20, output_dim=128)
            self.fusion = AttentionFusion(input_dims=[128, 128, 128, 128], embed_dim=128)
            self.head = ClassificationHead(input_dim=128, num_classes=2)
        
        def forward(self, inputs):
            # 编码各模态
            behavior_features = self.transformer(inputs['behavior_sequences'])  # [batch, 128]
            
            user_features = self.gnn(inputs['node_features'], inputs['adjacency_matrix'])
            user_features = user_features[:inputs['behavior_sequences'].shape[0]]  # 匹配batch_size
            
            text_features = self.bert(inputs['text_content'])  # [batch, 128]
            struct_features = self.lgbm(inputs['structured_features'])  # [batch, 128]
            
            # 融合
            modality_features = [behavior_features, user_features, text_features, struct_features]
            fusion_output = self.fusion(modality_features)
            
            # 分类
            head_output = self.head(fusion_output['fused_features'])
            
            return {
                'logits': head_output,
                'probabilities': torch.softmax(head_output, dim=1),
                'anomaly_scores': torch.softmax(head_output, dim=1)[:, 1],
                'confidence': torch.max(torch.softmax(head_output, dim=1), dim=1)[0],
                'fused_features': fusion_output['fused_features'],
                'fusion_weights': fusion_output.get('final_weights')
            }
    
    # 创建模型和测试数据
    model = SimplifiedMultiModalModel()
    
    batch_size = 8
    inputs = {
        'behavior_sequences': torch.randn(batch_size, 32, 128),
        'node_features': torch.randn(50, 10),
        'adjacency_matrix': torch.randint(0, 2, (50, 50)).float(),
        'text_content': [f"Test email {i}" for i in range(batch_size)],
        'structured_features': torch.randn(batch_size, 20)
    }
    
    print(f"  输入数据:")
    print(f"    行为序列: {inputs['behavior_sequences'].shape}")
    print(f"    用户特征: {inputs['node_features'].shape}")
    print(f"    文本数量: {len(inputs['text_content'])}")
    print(f"    结构特征: {inputs['structured_features'].shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"\n  输出结果:")
    print(f"    logits: {outputs['logits'].shape}")
    print(f"    probabilities: {outputs['probabilities'].shape}")
    print(f"    anomaly_scores: {outputs['anomaly_scores'].shape}")
    print(f"    confidence: {outputs['confidence'].shape}")
    print(f"    fused_features: {outputs['fused_features'].shape}")
    
    # 预测结果
    print(f"\n  预测结果:")
    print(f"    平均异常分数: {outputs['anomaly_scores'].mean().item():.4f}")
    print(f"    平均置信度: {outputs['confidence'].mean().item():.4f}")
    
    anomaly_count = (outputs['anomaly_scores'] > 0.5).sum().item()
    print(f"    异常检测数量: {anomaly_count}/{batch_size}")
    
    if outputs['fusion_weights'] is not None:
        weights = outputs['fusion_weights'].mean(dim=0)
        print(f"    模态权重: Transformer={weights[0]:.3f}, GNN={weights[1]:.3f}, BERT={weights[2]:.3f}, LGBM={weights[3]:.3f}")
    
    # 测试梯度
    model.train()
    outputs = model(inputs)
    loss = outputs['logits'].sum()
    loss.backward()
    print(f"    梯度测试: ✅ 通过")
    
    return True

def main():
    """主测试函数"""
    print("🎯 多模态异常检测系统完整测试")
    print("=" * 50)
    
    try:
        # 测试各个模块
        if test_individual_modules():
            print("\n✅ 所有单个模块测试通过！")
        
        # 测试完整系统
        if test_simplified_multimodal_model():
            print("\n🎉 完整多模态系统测试通过！")
            
        print("\n" + "=" * 50)
        print("🏆 系统测试全部完成！")
        
        # 总结
        print("\n📋 系统组件总结:")
        print("  ✅ TransformerEncoder - 行为序列建模")
        print("  ✅ UserGNN - 用户关系图建模") 
        print("  ✅ BERTTextEncoder - 文本内容编码")
        print("  ✅ LightGBMBranch - 结构化特征处理")
        print("  ✅ AttentionFusion - 多模态融合")
        print("  ✅ ClassificationHead - 异常检测分类")
        print("  ✅ SimplifiedMultiModalModel - 完整系统集成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 