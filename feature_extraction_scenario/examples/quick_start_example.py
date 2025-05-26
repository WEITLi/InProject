#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态异常检测快速开始示例
演示如何使用新的多模态框架进行训练
"""

import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core_logic'))

try:
    # 尝试从core_logic包导入
    from core_logic.multimodal_pipeline import MultiModalDataPipeline
    from core_logic.train_pipeline_multimodal.multimodal_trainer import MultiModalTrainer
    from core_logic.config import Config
except ImportError:
    # 如果包导入失败，尝试直接导入
    from multimodal_pipeline import MultiModalDataPipeline
    from train_pipeline.multimodal_trainer import MultiModalTrainer
    from config import Config

def quick_start_example():
    """快速开始示例"""
    print("🚀 多模态异常检测快速开始示例")
    print("="*60)
    
    # 1. 创建配置
    config = Config()
    
    # 设置为快速开发模式
    config.training.num_epochs = 3
    config.training.batch_size = 8
    config.training.learning_rate = 1e-3
    config.data.data_version = 'r4.2'
    config.data.feature_dim = 128  # 减小特征维度以加快训练
    config.model.hidden_dim = 128
    config.model.num_layers = 2
    config.experiment_name = "quick_start_example"
    
    print(f"配置设置完成:")
    print(f"  训练轮数: {config.training.num_epochs}")
    print(f"  批大小: {config.training.batch_size}")
    print(f"  特征维度: {config.data.feature_dim}")
    print(f"  隐藏维度: {config.model.hidden_dim}")
    
    try:
        # 2. 创建多模态数据流水线
        print(f"\n📊 创建多模态数据流水线...")
        pipeline = MultiModalDataPipeline(
            config=config,
            data_version=config.data.data_version,
            feature_dim=config.data.feature_dim,
            num_cores=2  # 减少核心数以避免资源竞争
        )
        
        # 3. 运行数据处理（使用少量数据）
        print(f"\n🔄 运行数据处理流水线...")
        training_data = pipeline.run_full_multimodal_pipeline(
            start_week=0,
            end_week=2,  # 只使用前2周数据
            max_users=20,  # 只使用20个用户
            sequence_length=32  # 减少序列长度
        )
        
        print(f"数据处理完成:")
        print(f"  总样本数: {len(training_data['labels'])}")
        print(f"  正常样本: {sum(training_data['labels'] == 0)}")
        print(f"  异常样本: {sum(training_data['labels'] == 1)}")
        
        # 4. 创建训练器并开始训练
        print(f"\n🎯 开始模型训练...")
        trainer = MultiModalTrainer(
            config=config, 
            output_dir='./quick_start_outputs'
        )
        
        # 开始训练
        model = trainer.train(training_data)
        
        print(f"\n✅ 快速开始示例完成！")
        print(f"模型和结果保存在: ./quick_start_outputs")
        
        return model, training_data
        
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        print(f"这可能是由于缺少数据文件或依赖包导致的")
        print(f"请确保:")
        print(f"  1. 数据文件存在于正确位置")
        print(f"  2. 所有依赖包已正确安装")
        print(f"  3. 有足够的内存和计算资源")
        raise e

def simple_training_example():
    """简单训练示例 - 仅使用Transformer"""
    print("🔧 简单训练示例 - 仅使用Transformer")
    print("="*60)
    
    # 创建简化配置
    config = Config()
    config.training.num_epochs = 2
    config.training.batch_size = 4
    config.data.feature_dim = 64
    config.model.hidden_dim = 64
    config.model.num_layers = 2
    
    # 禁用其他模态，只使用Transformer
    config.model.enable_gnn = False
    config.model.enable_bert = False
    config.model.enable_lgbm = False
    config.model.enable_transformer = True
    
    config.experiment_name = "simple_transformer_only"
    
    print(f"简化配置:")
    print(f"  仅启用Transformer模块")
    print(f"  特征维度: {config.data.feature_dim}")
    print(f"  训练轮数: {config.training.num_epochs}")
    
    try:
        # 创建流水线
        pipeline = MultiModalDataPipeline(
            config=config,
            data_version='r4.2',
            feature_dim=config.data.feature_dim,
            num_cores=1
        )
        
        # 处理数据
        training_data = pipeline.run_full_multimodal_pipeline(
            start_week=0,
            end_week=1,  # 只使用1周数据
            max_users=10,  # 只使用10个用户
            sequence_length=16
        )
        
        # 训练模型
        trainer = MultiModalTrainer(
            config=config,
            output_dir='./simple_outputs'
        )
        
        model = trainer.train(training_data)
        
        print(f"\n✅ 简单训练示例完成！")
        return model
        
    except Exception as e:
        print(f"\n❌ 简单训练失败: {e}")
        raise e

def compare_modalities_example():
    """模态对比示例"""
    print("📊 模态对比示例")
    print("="*60)
    
    # 定义不同的模态配置
    modality_configs = [
        {
            'name': 'transformer_only',
            'enable_gnn': False,
            'enable_bert': False,
            'enable_lgbm': False,
            'enable_transformer': True
        },
        {
            'name': 'transformer_gnn',
            'enable_gnn': True,
            'enable_bert': False,
            'enable_lgbm': False,
            'enable_transformer': True
        },
        {
            'name': 'full_multimodal',
            'enable_gnn': True,
            'enable_bert': True,
            'enable_lgbm': True,
            'enable_transformer': True
        }
    ]
    
    results = {}
    
    for mod_config in modality_configs:
        print(f"\n🔄 训练配置: {mod_config['name']}")
        
        # 创建配置
        config = Config()
        config.training.num_epochs = 2
        config.training.batch_size = 4
        config.data.feature_dim = 64
        config.model.hidden_dim = 64
        config.model.num_layers = 2
        
        # 应用模态配置
        for key, value in mod_config.items():
            if key != 'name':
                setattr(config.model, key, value)
        
        config.experiment_name = mod_config['name']
        
        try:
            # 创建流水线（复用数据）
            pipeline = MultiModalDataPipeline(
                config=config,
                data_version='r4.2',
                feature_dim=config.data.feature_dim,
                num_cores=1
            )
            
            # 处理数据
            training_data = pipeline.run_full_multimodal_pipeline(
                start_week=0,
                end_week=1,
                max_users=10,
                sequence_length=16
            )
            
            # 训练模型
            trainer = MultiModalTrainer(
                config=config,
                output_dir=f'./comparison_outputs/{mod_config["name"]}'
            )
            
            model = trainer.train(training_data)
            
            # 记录结果（这里简化，实际应该从训练历史中获取）
            results[mod_config['name']] = {
                'success': True,
                'config': mod_config
            }
            
            print(f"✅ {mod_config['name']} 训练完成")
            
        except Exception as e:
            print(f"❌ {mod_config['name']} 训练失败: {e}")
            results[mod_config['name']] = {
                'success': False,
                'error': str(e)
            }
    
    print(f"\n📊 模态对比结果:")
    for name, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"  {name}: {status}")
    
    return results

def main():
    """主函数"""
    print("🎯 多模态异常检测示例集合")
    print("="*80)
    
    examples = [
        ("1", "快速开始示例", quick_start_example),
        ("2", "简单训练示例", simple_training_example),
        ("3", "模态对比示例", compare_modalities_example),
    ]
    
    print("可用示例:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    
    print("\n选择要运行的示例 (1-3), 或按 Enter 运行快速开始示例:")
    choice = input().strip()
    
    if choice == "":
        choice = "1"
    
    # 查找并运行选择的示例
    for num, name, func in examples:
        if choice == num:
            print(f"\n🚀 运行示例: {name}")
            print("="*60)
            try:
                result = func()
                print(f"\n🎉 示例 '{name}' 运行成功！")
                return result
            except Exception as e:
                print(f"\n💥 示例 '{name}' 运行失败: {e}")
                return None
    
    print(f"❌ 无效选择: {choice}")
    return None

if __name__ == "__main__":
    main()