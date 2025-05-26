#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试导入脚本
用于验证所有模块的导入是否正常工作
"""

import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core_logic'))

def test_imports():
    """测试所有关键模块的导入"""
    print("🔍 测试模块导入...")
    
    try:
        print("  测试配置模块...")
        from core_logic.config import Config
        print("  ✅ Config 导入成功")
        
        print("  测试数据流水线...")
        try:
            from core_logic.multimodal_pipeline import MultiModalDataPipeline
            print("  ✅ MultiModalDataPipeline 导入成功")
        except Exception as e:
            print(f"  ❌ MultiModalDataPipeline 导入失败: {e}")
            raise e
        
        print("  测试多模态模型...")
        from core_logic.train_pipeline_multimodal.multimodal_model import MultiModalAnomalyDetector
        print("  ✅ MultiModalAnomalyDetector 导入成功")
        
        print("  测试训练器...")
        from core_logic.train_pipeline_multimodal.multimodal_trainer import MultiModalTrainer
        print("  ✅ MultiModalTrainer 导入成功")
        
        print("  测试基础模型组件...")
        from core_logic.models.base_model.transformer_encoder import TransformerEncoder
        from core_logic.models.base_model.user_gnn import UserGNN
        from core_logic.models.base_model.head import ClassificationHead
        print("  ✅ 基础模型组件导入成功")
        
        print("  测试编码器组件...")
        from core_logic.models.text_encoder.bert_module import BERTTextEncoder
        from core_logic.models.structure_encoder.lightgbm_branch import LightGBMBranch
        print("  ✅ 编码器组件导入成功")
        
        print("  测试融合组件...")
        from core_logic.models.fusion.attention_fusion import AttentionFusion
        print("  ✅ 融合组件导入成功")
        
        print("\n🎉 所有模块导入测试通过！")
        return True
        
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print(f"错误详情: {type(e).__name__}: {str(e)}")
        return False
    except Exception as e:
        print(f"\n💥 其他错误: {e}")
        print(f"错误详情: {type(e).__name__}: {str(e)}")
        return False

def test_basic_functionality():
    """测试基础功能"""
    print("\n🔧 测试基础功能...")
    
    try:
        # 测试配置创建
        from core_logic.config import Config
        config = Config()
        print("  ✅ 配置对象创建成功")
        
        # 测试数据流水线创建
        from core_logic.multimodal_pipeline import MultiModalDataPipeline
        pipeline = MultiModalDataPipeline(
            config=config,
            data_version='r4.2',
            feature_dim=128,
            num_cores=1
        )
        print("  ✅ 数据流水线创建成功")
        
        # 测试训练器创建
        from core_logic.train_pipeline_multimodal.multimodal_trainer import MultiModalTrainer
        trainer = MultiModalTrainer(config=config, output_dir='./test_outputs')
        print("  ✅ 训练器创建成功")
        
        print("\n🎉 基础功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 基础功能测试失败: {e}")
        print(f"错误详情: {type(e).__name__}: {str(e)}")
        return False

def main():
    """主函数"""
    print("🚀 多模态异常检测模块导入测试")
    print("="*60)
    
    # 测试导入
    import_success = test_imports()
    
    if import_success:
        # 测试基础功能
        functionality_success = test_basic_functionality()
        
        if functionality_success:
            print("\n✅ 所有测试通过！可以开始使用多模态异常检测系统。")
            return True
        else:
            print("\n⚠️  导入成功但基础功能测试失败，请检查依赖和配置。")
            return False
    else:
        print("\n❌ 导入测试失败，请检查模块路径和依赖。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 