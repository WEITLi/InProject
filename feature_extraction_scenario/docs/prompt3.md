1.请创建 models/base_model/ 模块，包括 transformer_encoder.py、user_gnn.py、base_fusion.py、head.py，实现行为序列建模 + 用户图嵌入融合的基本结构。
2.请在 models/text_encoder/ 中添加 bert_module.py，加载 huggingface transformers 的 BERT-base 模型，用于处理邮件/网页类文本，返回 [CLS] 表示。
3.请在 models/structure_encoder/ 中添加 lightgbm_branch.py，实现结构化特征输入的 LGBM 分支，并可导出向量用于融合。
4.请创建 models/fusion/attention_fusion.py，支持将多模态向量通过注意力机制加权融合，返回统一向量。
5.请创建 train_pipeline/multimodal_model.py 和 trainer.py，实现完整多模态模型组合、训练/验证/测试流程，支持 config.py 控制是否启用 GNN、BERT、LGBM 分支。
