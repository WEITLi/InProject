#!/usr/bin/env python3
"""Multi-modal Anomaly Detection Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import sys
import os
import logging

# 添加模型路径
try:
    # 尝试相对导入
    from ..config import ModelConfig
    from ..models.base_model.transformer_encoder import TransformerEncoder
    from ..models.base_model.user_gnn import UserGNN
    from ..models.base_model.base_fusion import BaseFusion
    from ..models.base_model.head import ClassificationHead
    from ..models.text_encoder.bert_module import BERTTextEncoder
    from ..models.structure_encoder.lightgbm_branch import LightGBMBranch
    # from ..models.fusion.attention_fusion import AttentionFusion # Comment out old fusion
    from ..models.fusion.cross_attention_fusion import CrossAttentionFusion # Import new fusion
except ImportError:
    # 如果相对导入失败，添加路径并使用绝对导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from config import ModelConfig
    from models.base_model.transformer_encoder import TransformerEncoder
    from models.base_model.user_gnn import UserGNN
    from models.base_model.base_fusion import BaseFusion
    from models.base_model.head import ClassificationHead
    from models.text_encoder.bert_module import BERTTextEncoder
    from models.structure_encoder.lightgbm_branch import LightGBMBranch
    # from models.fusion.attention_fusion import AttentionFusion # Comment out old fusion
    from models.fusion.cross_attention_fusion import CrossAttentionFusion # Import new fusion

class MultiModalAnomalyDetector(nn.Module):
    """
    多模态异常检测模型
    
    整合所有模态编码器和融合机制，用于用户行为异常检测
    """
    
    def __init__(
        self,
        model_config_obj: ModelConfig, # 主配置对象
        # 各组件的配置字典，主要用于传递动态确定的input_dim
        transformer_config: Dict, 
        gnn_config: Dict,
        bert_config: Dict,
        lgbm_config: Dict,
        fusion_config: Dict,
        head_config: Dict,
        # embed_dim 和 dropout 可以从 model_config_obj 获取，这里作为参数是冗余的
    ):
        super().__init__()
        
        self.model_config = model_config_obj # 存储 ModelConfig
        self.embed_dim = self.model_config.hidden_dim # 主隐藏维度
        
        # 1. 行为序列编码器（Transformer）
        # 使用 model_config_obj 中的参数，并从传入的 transformer_config 更新 input_dim
        _actual_transformer_config = {
            'input_dim': transformer_config['input_dim'], # 来自 trainer
            'hidden_dim': self.model_config.hidden_dim,
            'num_heads': self.model_config.num_heads,
            'num_layers': self.model_config.num_layers
            # TransformerEncoder __init__ 不接受 dropout, 其内部 dropout 固定
        }
        self.behavior_encoder = TransformerEncoder(**_actual_transformer_config)
        
        # 2. 用户关系编码器（GNN）
        _actual_gnn_config = {
            'input_dim': gnn_config['input_dim'], # 来自 trainer
            'hidden_dim': self.model_config.gnn_hidden_dim,
            'output_dim': self.model_config.hidden_dim, # GNN 输出对齐到主 hidden_dim
            'num_layers': self.model_config.gnn_num_layers,
            'dropout': self.model_config.gnn_dropout
        }
        self.user_encoder = UserGNN(**_actual_gnn_config)
        
        # 3. 文本内容编码器（BERT）
        _actual_bert_config = {
            'bert_model_name': self.model_config.bert_model_name,
            'max_length': self.model_config.bert_max_length,
            'output_dim': self.model_config.hidden_dim, # BERT 输出对齐
            'dropout': self.model_config.dropout
            # input_dim for BERT is handled internally by tokenizer
        }
        self.text_encoder = BERTTextEncoder(**_actual_bert_config)
        
        # 4. 结构化特征编码器（LightGBM）
        _actual_lgbm_config = {
            'input_dim': lgbm_config['input_dim'], # 来自 trainer
            'output_dim': self.model_config.hidden_dim, # LGBM 输出对齐
            'dropout': self.model_config.dropout, # General dropout
            # Pass LightGBM specific params from ModelConfig
            'num_leaves': self.model_config.lgbm_num_leaves,
            'max_depth': self.model_config.lgbm_max_depth,
            'learning_rate': self.model_config.lgbm_learning_rate,
            'n_estimators': self.model_config.lgbm_n_estimators if hasattr(self.model_config, 'lgbm_n_estimators') else 100,
            'feature_fraction': self.model_config.lgbm_feature_fraction,
            'bagging_fraction': self.model_config.lgbm_bagging_fraction if hasattr(self.model_config, 'lgbm_bagging_fraction') else 0.8,
            'bagging_freq': self.model_config.lgbm_bagging_freq if hasattr(self.model_config, 'lgbm_bagging_freq') else 5
        }
        self.structure_encoder = LightGBMBranch(**_actual_lgbm_config)
        
        # 5. 多模态融合器
        # 动态构建 input_dims_for_fusion based on enabled_modalities and their actual output dimensions
        # self.fusion_input_dims = [] # Old list way
        self.fusion_input_dims_dict = {} # New dict way for CrossAttentionFusion

        if 'behavior' in self.model_config.enabled_modalities:
            # self.fusion_input_dims.append(_actual_transformer_config['hidden_dim'])
            # self.fusion_input_dims_dict['behavior'] = self.behavior_encoder.output_dim # Use actual output_dim from encoder
            self.fusion_input_dims_dict['behavior'] = _actual_transformer_config['hidden_dim'] # Transformer output_dim is its hidden_dim
        if 'graph' in self.model_config.enabled_modalities:
            # self.fusion_input_dims.append(_actual_gnn_config['output_dim'])
            self.fusion_input_dims_dict['graph'] = _actual_gnn_config['output_dim'] # GNN config already specifies output_dim aligned with hidden_dim
        if 'text' in self.model_config.enabled_modalities:
            # self.fusion_input_dims.append(_actual_bert_config['output_dim'])
            self.fusion_input_dims_dict['text'] = _actual_bert_config['output_dim'] # BERT config specifies output_dim aligned with hidden_dim
        if 'structured' in self.model_config.enabled_modalities:
            # self.fusion_input_dims.append(_actual_lgbm_config['output_dim'])
            self.fusion_input_dims_dict['structured'] = _actual_lgbm_config['output_dim'] # LGBM config specifies output_dim aligned with hidden_dim
        
        # 决定融合模块输出维度 (也是分类头输入维度)
        # For CrossAttentionFusion, the output_dim is self.model_config.hidden_dim (or passed as embed_dim to it)
        effective_fusion_embed_dim = self.model_config.hidden_dim

        # _actual_fusion_config = { # For old AttentionFusion
        #     'input_dims': self.fusion_input_dims,
        #     'embed_dim': effective_fusion_embed_dim,
        #     'dropout': self.model_config.dropout,
        #     'use_gating': True 
        # }
        
        _actual_cross_fusion_config = {
            'input_dims': self.fusion_input_dims_dict,
            'embed_dim': self.model_config.hidden_dim, # This is the output dim of CrossAttentionFusion's MLP
            'num_heads': getattr(self.model_config, 'fusion_num_heads', 4), # Default to 4 if not in config
            'dropout': self.model_config.dropout,
            'query_modality_name': getattr(self.model_config, 'fusion_query_modality', 'behavior') # Default
        }

        # self.fusion_module = AttentionFusion(**_actual_fusion_config) # Old
        self.fusion_module = CrossAttentionFusion(**_actual_cross_fusion_config) # New
        
        # 6. 分类头
        _actual_head_config = {
            'input_dim': effective_fusion_embed_dim, # Input to head is output of fusion
            'num_classes': self.model_config.num_classes,
            'dropout': self.model_config.head_dropout 
        }
        # 从ModelConfig获取可选参数（如果存在）
        if hasattr(self.model_config, 'head_hidden_dims'):
            _actual_head_config['hidden_dims'] = self.model_config.head_hidden_dims
        if hasattr(self.model_config, 'head_activation'):
            _actual_head_config['activation'] = self.model_config.head_activation

        # print(f"[DEBUG MultiModalAnomalyDetector] Fusion module embed_dim: {self.fusion_module.embed_dim}")
        # print(f"[DEBUG MultiModalAnomalyDetector] ClassificationHead _actual_head_config: {_actual_head_config}")

        self.classification_head = ClassificationHead(**_actual_head_config)
        
        self._feature_cache = {}
        
    def encode_behavior_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        """编码行为序列"""
        return self.behavior_encoder(sequences)
    
    def encode_user_relations(self, node_features: torch.Tensor, 
                            adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """编码用户关系"""
        return self.user_encoder(node_features, adjacency_matrix)
    
    def encode_text_content(self, texts: List[str]) -> torch.Tensor:
        """编码文本内容"""
        return self.text_encoder(texts)
    
    def encode_structured_features(self, features: Union[torch.Tensor, dict]) -> torch.Tensor:
        """编码结构化特征"""
        return self.structure_encoder(features)
    
    def forward(self, inputs: Dict[str, Union[torch.Tensor, List]]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            inputs: 输入字典，包含各模态数据
                - 'behavior_sequences': [batch_size, seq_len, feature_dim] 行为序列
                - 'node_features': [num_nodes, node_feat_dim] 节点特征
                - 'adjacency_matrix': [num_nodes, num_nodes] 邻接矩阵
                - 'text_content': List[str] 文本内容
                - 'structured_features': 结构化特征（tensor或dict）
                
        Returns:
            输出字典，包含预测结果和中间特征
        """
        batch_size = inputs['behavior_sequences'].shape[0]
        
        # 1. 各模态特征编码
        # modality_features = [] # Old list
        # modality_names = []    # Old list
        encoded_features_dict = {} # New dict
        
        # print(f"[DEBUG MMA Detector] Initial enabled_modalities: {self.model_config.enabled_modalities}")

        # 行为序列特征
        if 'behavior_sequences' in inputs and 'behavior' in self.model_config.enabled_modalities:
            behavior_features = self.encode_behavior_sequences(inputs['behavior_sequences'])
            # modality_features.append(behavior_features)
            # modality_names.append('behavior')
            encoded_features_dict['behavior'] = behavior_features
            # print(f"[DEBUG MMA Detector] Added 'behavior' features. Shape: {behavior_features.shape}")
            
        # 用户关系特征 (GNN)
        if (
            'node_features' in inputs and 
            'adjacency_matrix' in inputs and 
            # 'batch_user_indices_in_graph' in inputs and # GNN in this model takes full graph, then selects
            'graph' in self.model_config.enabled_modalities
        ): 
            # GNN processing needs to result in per-batch-item features
            # The current GNN (UserGNN) outputs features for ALL nodes in the graph.
            # We need to select the features for the users in the current batch.
            user_features_global = self.user_encoder(
                inputs['node_features'], 
                inputs['adjacency_matrix']
            ) # Shape: [num_total_graph_users, gnn_output_dim]

            if 'batch_user_indices_in_graph' in inputs: # This key is crucial for selecting batch-specific GNN features
                batch_indices = inputs['batch_user_indices_in_graph'] # [B]
                valid_mask = batch_indices >= 0
                
                gnn_output_dim = self.user_encoder.output_dim if hasattr(self.user_encoder, 'output_dim') else self.model_config.hidden_dim
                user_features_for_batch = torch.zeros(batch_size, gnn_output_dim, device=user_features_global.device)

                if torch.any(valid_mask):
                    valid_batch_indices_in_graph = batch_indices[valid_mask]
                    if valid_batch_indices_in_graph.max() < user_features_global.shape[0]:
                        selected_gnn_features = user_features_global[valid_batch_indices_in_graph]
                        user_features_for_batch[valid_mask] = selected_gnn_features
                    else:
                        # Log warning about out-of-bounds indices
                        pass 
                
                # modality_features.append(user_features_for_batch)
                # modality_names.append('graph')
                encoded_features_dict['graph'] = user_features_for_batch
                # print(f"[DEBUG MMA Detector] Added 'graph' features. Shape: {user_features_for_batch.shape}")
            else:
                # print("[DEBUG MMA Detector] 'batch_user_indices_in_graph' missing, cannot extract GNN features for batch.")
                # Fallback: add zeros or handle error, for now, 'graph' won't be in encoded_features_dict
                pass # Or add zeros: encoded_features_dict['graph'] = torch.zeros(...)

            
        # 文本内容特征
        if 'text_content' in inputs and 'text' in self.model_config.enabled_modalities:
            text_features = self.text_encoder(inputs['text_content'])
            # modality_features.append(text_features)
            # modality_names.append('text')
            encoded_features_dict['text'] = text_features
            # print(f"[DEBUG MMA Detector] Added 'text' features. Shape: {text_features.shape}")
            
        # 结构化特征
        if 'structured_features' in inputs and 'structured' in self.model_config.enabled_modalities:
            struct_features = self.encode_structured_features(inputs['structured_features'])
            # modality_features.append(struct_features)
            # modality_names.append('structured')
            encoded_features_dict['structured'] = struct_features
            # print(f"[DEBUG MMA Detector] Added 'structured' features. Shape: {struct_features.shape}")
        
        if not encoded_features_dict:
            # This means no modalities were enabled or no input provided for them.
            # Handle this gracefully: return zero logits/probabilities for the batch.
            # print("[DEBUG MMA Detector] Encoded features dictionary is empty!")
            dummy_logits = torch.zeros(batch_size, self.model_config.num_classes, device=inputs['behavior_sequences'].device if 'behavior_sequences' in inputs and inputs['behavior_sequences'].numel() > 0 else 'cpu')
            dummy_probs = F.softmax(dummy_logits, dim=1)
            return {
                'logits': dummy_logits,
                'probabilities': dummy_probs,
                'anomaly_scores': dummy_probs[:, 1] if self.model_config.num_classes > 1 else dummy_probs[:, 0],
                'confidence': torch.max(dummy_probs, dim=1)[0],
                'fused_features': torch.zeros(batch_size, self.model_config.hidden_dim, device=dummy_logits.device),
                'modality_features': {}
            }

        # 2. 多模态融合
        # if len(modality_features) > 1: # Old logic
        #     fusion_outputs = self.fusion_module(modality_features)
        # elif modality_features: # Old logic: only one modality
        #     fused_features = modality_features[0]
        #     fusion_outputs = { ... } # Mocked output
        # else: # No features, should be caught by the check above
            # pass

        # 🔧 添加调试信息（仅在消融实验中）
        if len(self.model_config.enabled_modalities) <= 2:  # 消融实验通常是单模态或双模态
            logger = logging.getLogger(__name__)
            logger.debug(f"[MMA Forward] Enabled modalities: {self.model_config.enabled_modalities}")
            logger.debug(f"[MMA Forward] Actually encoded modalities: {list(encoded_features_dict.keys())}")
            for name, feat in encoded_features_dict.items():
                logger.debug(f"[MMA Forward] {name} features shape: {feat.shape}, mean: {feat.mean().item():.4f}, std: {feat.std().item():.4f}")

        fusion_outputs = self.fusion_module(encoded_features_dict) # Pass dictionary
        fused_features = fusion_outputs['fused_features']
        
        # 🔧 添加融合后特征的调试信息
        if len(self.model_config.enabled_modalities) <= 2:
            logger.debug(f"[MMA Forward] Fused features shape: {fused_features.shape}, mean: {fused_features.mean().item():.4f}, std: {fused_features.std().item():.4f}")
        
        # 3. 异常检测分类
        logits = self.classification_head(fused_features)
        # print(f"[DEBUG MMA Detector] Inferred fused_features shape: {fused_features.shape}")
        # print(f"[DEBUG MMA Detector] Logits shape from ClassificationHead: {logits.shape}")
        probabilities = torch.softmax(logits, dim=1)
        
        # 调试信息：检查 probabilities 的形状
        # print(f"[DEBUG MMA Detector] Probabilities shape: {probabilities.shape}")
        
        # 安全地获取异常分数
        if probabilities.shape[1] > 1:
            anomaly_scores = probabilities[:, 1]
        else:
            # 如果只有一个类别，使用第一个（也是唯一的）概率
            anomaly_scores = probabilities[:, 0]
            # print(f"[DEBUG MMA Detector] Warning: probabilities only has 1 class, using [:, 0]")
        
        confidence = torch.max(probabilities, dim=1)[0]
        
        # 整合输出
        outputs = {
            'logits': logits,
            'probabilities': probabilities,
            'anomaly_scores': anomaly_scores,
            'confidence': confidence,
            'fused_features': fused_features,
            'modality_features': fusion_outputs.get('modality_features_projected', {}) # Get projected features from fusion output
            # outputs['modality_features'] = { # Old way
            #     name: feat for name, feat in zip(modality_names, modality_features)
            # }
        }
        
        # 添加融合权重信息 (CrossAttentionFusion might not return these in the same way)
        # if fusion_outputs.get('attention_weights') is not None:
        #     outputs['attention_weights'] = fusion_outputs['attention_weights']
        # if fusion_outputs.get('gate_weights') is not None:
        #     outputs['gate_weights'] = fusion_outputs['gate_weights']
            
        return outputs
    
    def predict_anomaly(self, inputs: Dict[str, Union[torch.Tensor, List]], 
                       threshold: float = 0.5) -> Dict[str, Union[torch.Tensor, List]]:
        """
        异常检测预测
        
        Args:
            inputs: 输入数据
            threshold: 异常检测阈值
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(inputs)
            
            # 异常判断
            anomaly_predictions = (outputs['anomaly_scores'] > threshold).long()
            
            # 置信度评估
            probs = outputs['probabilities']
            confidence = torch.max(probs, dim=1)[0]
            
            return {
                'is_anomaly': anomaly_predictions,
                'anomaly_scores': outputs['anomaly_scores'],
                'confidence': confidence,
                'probabilities': outputs['probabilities'],
                'class_predictions': torch.argmax(outputs['logits'], dim=1)
            }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取模态重要性"""
        importance = {}
        
        # 如果有融合权重，使用融合权重作为重要性
        if hasattr(self, '_last_fusion_weights'):
            weights = self._last_fusion_weights.mean(dim=0).cpu().numpy()
            modality_names = ['behavior', 'user_relations', 'text', 'structured']
            for i, name in enumerate(modality_names[:len(weights)]):
                importance[name] = float(weights[i])
        
        return importance

def test_multimodal_model():
    """测试多模态异常检测模型"""
    print("🧪 Testing Multi-modal Anomaly Detection Model...")
    
    # 创建模拟输入数据
    batch_size = 8
    
    inputs = {
        # 行为序列：[batch_size, seq_len, feature_dim]
        'behavior_sequences': torch.randn(batch_size, 32, 128),
        
        # 用户关系：节点特征和邻接矩阵
        'node_features': torch.randn(50, 10),
        'adjacency_matrix': torch.randint(0, 2, (50, 50)).float(),
        
        # 文本内容
        'text_content': [
            f"This is email content {i} about system alerts and notifications."
            for i in range(batch_size)
        ],
        
        # 结构化特征
        'structured_features': torch.randn(batch_size, 20)
    }
    
    print(f"  输入数据形状:")
    print(f"    行为序列: {inputs['behavior_sequences'].shape}")
    print(f"    节点特征: {inputs['node_features'].shape}")
    print(f"    邻接矩阵: {inputs['adjacency_matrix'].shape}")
    print(f"    文本数量: {len(inputs['text_content'])}")
    print(f"    结构化特征: {inputs['structured_features'].shape}")
    
    # 创建模型
    model = MultiModalAnomalyDetector(
        embed_dim=128,
        transformer_config={'input_dim': 128, 'hidden_dim': 128},
        gnn_config={'input_dim': 10, 'output_dim': 128},
        bert_config={'output_dim': 128},
        lgbm_config={'input_dim': 20, 'output_dim': 128},
        fusion_config={'embed_dim': 128}
    )
    
    # 前向传播
    with torch.no_grad():
        outputs = model(inputs)
    
    print(f"\n  输出结果:")
    print(f"    预测logits: {outputs['logits'].shape}")
    print(f"    概率分布: {outputs['probabilities'].shape}")
    print(f"    异常分数: {outputs['anomaly_scores'].shape}")
    print(f"    置信度: {outputs['confidence'].shape}")
    print(f"    融合特征: {outputs['fused_features'].shape}")
    
    # 显示预测结果
    print(f"    平均异常分数: {outputs['anomaly_scores'].mean().item():.4f}")
    print(f"    平均置信度: {outputs['confidence'].mean().item():.4f}")
    
    # 测试异常检测
    anomaly_results = model.predict_anomaly(inputs, threshold=0.5)
    print(f"    检测到异常数量: {anomaly_results['is_anomaly'].sum().item()}/{batch_size}")
    
    # 测试梯度
    model.train()
    outputs = model(inputs)
    loss = outputs['logits'].sum()
    loss.backward()
    
    print("  ✅ Multi-modal Model 测试通过")

if __name__ == "__main__":
    test_multimodal_model() 