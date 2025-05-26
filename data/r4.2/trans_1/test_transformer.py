#!/usr/bin/env python
# coding: utf-8

"""
测试脚本 - 验证 Transformer 威胁检测项目
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

def test_imports():
    """测试所有必要的模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试 PyTorch
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        # 测试其他依赖
        import numpy as np
        import pandas as pd
        import sklearn
        import matplotlib
        import seaborn
        
        print("✓ 所有基础依赖导入成功")
        
        # 测试自定义模块
        from data_processor import CERTDataProcessor
        from transformer_model import TransformerThreatDetector, ModelConfig
        from trainer import ThreatDetectionTrainer
        
        print("✓ 自定义模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_config():
    """测试配置类"""
    print("\n测试配置类...")
    
    try:
        from transformer_threat_detection import Config
        
        config = Config()
        print(f"✓ 默认设备: {config.device}")
        print(f"✓ 序列长度: {config.sequence_length}")
        print(f"✓ 隐藏维度: {config.hidden_dim}")
        
        # 测试保存和加载配置
        config.save_config('./test_config.json')
        loaded_config = Config.load_config('./test_config.json')
        
        # 清理
        os.remove('./test_config.json')
        
        print("✓ 配置保存和加载成功")
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        from transformer_model import TransformerThreatDetector, ModelConfig
        
        # 创建模型配置
        model_config = ModelConfig(
            input_dim=50,
            context_dim=6,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            sequence_length=10
        )
        
        # 创建模型
        model = TransformerThreatDetector(model_config)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型创建成功，参数数量: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_data_processor():
    """测试数据处理器（使用模拟数据）"""
    print("\n测试数据处理器...")
    
    try:
        from data_processor import CERTDataProcessor
        from transformer_threat_detection import Config
        
        # 创建模拟数据
        n_samples = 100
        n_features = 20
        
        data = pd.DataFrame({
            'user': [f'user_{i%10}' for i in range(n_samples)],
            'day': list(range(n_samples)),
            'insider': [0] * 95 + [1] * 5,  # 5% 异常
            'role': ['employee'] * 80 + ['admin'] * 20,
            'dept': ['IT'] * 50 + ['Finance'] * 50,
            **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)}
        })
        
        # 保存模拟数据
        data.to_pickle('./test_data.pkl')
        
        # 测试数据处理器
        config = Config()
        config.sequence_length = 5  # 减小序列长度用于测试
        
        processor = CERTDataProcessor(config)
        processed_data = processor.process_data('./test_data.pkl')
        
        print(f"✓ 处理了 {len(processed_data['sequences'])} 个序列")
        print(f"✓ 特征维度: {processed_data['sequences'][0].shape}")
        print(f"✓ 上下文维度: {len(processed_data['contexts'][0])}")
        
        # 清理
        os.remove('./test_data.pkl')
        
        return True
        
    except Exception as e:
        print(f"✗ 数据处理器测试失败: {e}")
        # 清理
        if os.path.exists('./test_data.pkl'):
            os.remove('./test_data.pkl')
        return False

def test_forward_pass():
    """测试模型前向传播"""
    print("\n测试模型前向传播...")
    
    try:
        from transformer_model import TransformerThreatDetector, ModelConfig
        import torch
        
        # 创建模型
        model_config = ModelConfig(
            input_dim=20,
            context_dim=6,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            sequence_length=5
        )
        
        model = TransformerThreatDetector(model_config)
        model.eval()
        
        # 创建模拟输入
        batch_size = 4
        seq_len = 5
        
        sequences = torch.randn(batch_size, seq_len, 20)
        contexts = torch.randn(batch_size, 6)
        attention_mask = torch.ones(batch_size, seq_len)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(sequences, contexts, attention_mask)
        
        print(f"✓ 分类输出形状: {outputs['classification_logits'].shape}")
        print(f"✓ 池化输出形状: {outputs['pooled_output'].shape}")
        
        # 测试带掩蔽的前向传播
        masked_sequences = sequences.clone()
        outputs_masked = model(sequences, contexts, attention_mask, masked_sequences)
        
        print(f"✓ 掩蔽语言模型输出形状: {outputs_masked['mlm_logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        return False

def check_data_files():
    """检查可用的数据文件"""
    print("\n检查可用的数据文件...")
    
    data_files = [
        '../dayr4.2_u200_w0-3_mweekdaysession_s1-percentile14.pkl',
        '../dayr4.2_u200_w0-3_mweekdaysession_s1-meandiff14.pkl',
        '../dayr4.2_u200_w0-3_mweekdaysession_s1-meddiff14.pkl',
        '../dayr4.2_u200_w0-3_mweekdaysession_s1-concat5.pkl',
    ]
    
    available_files = []
    for file in data_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.1f} MB)")
            available_files.append(file)
        else:
            print(f"✗ {file} (未找到)")
    
    if available_files:
        print(f"\n推荐使用: {available_files[0]}")
        return available_files[0]
    else:
        print("\n警告: 未找到可用的数据文件")
        return None

def main():
    """运行所有测试"""
    print("="*60)
    print("Transformer 威胁检测项目测试")
    print("="*60)
    
    tests = [
        ("模块导入", test_imports),
        ("配置类", test_config),
        ("模型创建", test_model_creation),
        ("数据处理器", test_data_processor),
        ("前向传播", test_forward_pass),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✓ {test_name} 通过")
        else:
            print(f"✗ {test_name} 失败")
    
    print(f"\n{'='*60}")
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目可以正常运行。")
        
        # 检查数据文件
        data_file = check_data_files()
        
        if data_file:
            print(f"\n🚀 可以开始训练了！")
            print(f"示例命令:")
            print(f"python transformer_threat_detection.py \\")
            print(f"    --data_path {data_file} \\")
            print(f"    --mode train \\")
            print(f"    --few_shot_samples 100 \\")
            print(f"    --num_epochs 5 \\")
            print(f"    --output_dir ./test_outputs")
        
    else:
        print("❌ 存在失败的测试，请检查环境配置。")
    
    print("="*60)

if __name__ == "__main__":
    main() 