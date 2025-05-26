#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新实验框架功能测试脚本
Test script for new experiment framework features
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'utils'))

def test_wandb_utils():
    """测试WandB工具模块"""
    print("🧪 测试WandB工具模块...")
    
    try:
        from utils.wandb_utils import WandBLogger, init_wandb
        
        # 测试WandBLogger初始化
        logger = WandBLogger(
            project_name="test_project",
            experiment_type="test",
            model_type="test_model",
            config={"test_param": 42}
        )
        
        # 测试日志记录
        logger.log_metrics({"test_metric": 0.85})
        
        # 结束测试
        logger.finish()
        
        print("✅ WandB工具模块测试通过")
        return True
        
    except Exception as e:
        print(f"❌ WandB工具模块测试失败: {e}")
        return False

def test_baseline_models():
    """测试传统ML基线模型"""
    print("🧪 测试传统ML基线模型...")
    
    try:
        from utils.baseline_models import BaselineModelTrainer
        import numpy as np
        
        # 创建模拟数据（增加样本数量以满足训练要求）
        mock_data = {
            'users': [f'user{i}' for i in range(20)],
            'labels': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'behavior_sequences': np.random.rand(20, 10, 5),
            'structured_features': np.random.rand(20, 8),
            'text_content': [f'text{i}' for i in range(20)]
        }
        
        # 测试RandomForest训练器
        trainer = BaselineModelTrainer(model_type="random_forest")
        
        # 测试特征提取
        features_df = trainer.extract_traditional_features(mock_data)
        print(f"   提取特征维度: {features_df.shape}")
        
        # 测试训练
        results = trainer.train(mock_data)
        print(f"   训练完成，测试F1: {results['test_metrics']['f1']:.4f}")
        
        print("✅ 传统ML基线模型测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 传统ML基线模型测试失败: {e}")
        return False

def test_imbalance_utils():
    """测试数据不平衡处理工具"""
    print("🧪 测试数据不平衡处理工具...")
    
    try:
        from utils.imbalance_utils import ImbalanceHandler, create_balanced_dataset
        import numpy as np
        
        # 创建模拟不平衡数据
        mock_data = {
            'users': [f'user{i}' for i in range(20)],
            'labels': [0] * 15 + [1] * 5,  # 15个正常，5个恶意
            'behavior_sequences': np.random.rand(20, 10, 5),
            'structured_features': np.random.rand(20, 8)
        }
        
        print(f"   原始数据分布: 正常={sum(1 for x in mock_data['labels'] if x == 0)}, 恶意={sum(1 for x in mock_data['labels'] if x == 1)}")
        
        # 测试不平衡处理器
        handler = ImbalanceHandler()
        
        # 测试创建不同比例数据集
        datasets = handler.create_imbalanced_datasets(mock_data, ratios=[2.0, 3.0])
        
        for ratio_name, dataset in datasets.items():
            labels = dataset['labels']
            normal_count = sum(1 for x in labels if x == 0)
            malicious_count = sum(1 for x in labels if x == 1)
            print(f"   {ratio_name}: 正常={normal_count}, 恶意={malicious_count}")
        
        # 测试平衡数据集创建
        balanced_data = create_balanced_dataset(mock_data, target_ratio=1.0)
        balanced_labels = balanced_data['labels']
        balanced_normal = sum(1 for x in balanced_labels if x == 0)
        balanced_malicious = sum(1 for x in balanced_labels if x == 1)
        print(f"   平衡后: 正常={balanced_normal}, 恶意={balanced_malicious}")
        
        print("✅ 数据不平衡处理工具测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据不平衡处理工具测试失败: {e}")
        return False

def test_optuna_tuning():
    """测试Optuna超参数优化"""
    print("🧪 测试Optuna超参数优化...")
    
    try:
        from utils.optuna_tuning import OptunaOptimizer, get_multimodal_search_space
        
        # 测试搜索空间定义
        search_space = get_multimodal_search_space()
        print(f"   搜索空间参数数量: {len(search_space)}")
        
        # 测试优化器初始化
        optimizer = OptunaOptimizer(
            study_name="test_study",
            direction="maximize"
        )
        
        print(f"   优化器创建成功: {optimizer.study_name}")
        
        print("✅ Optuna超参数优化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Optuna超参数优化测试失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("🧪 测试配置文件加载...")
    
    try:
        # 测试YAML配置文件
        config_files = [
            'configs/quick_test.yaml',
            'configs/tune_config.yaml',
            'configs/imbalance_config.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"   ✅ {config_file} 加载成功")
            else:
                print(f"   ⚠️ {config_file} 不存在")
        
        print("✅ 配置文件加载测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载测试失败: {e}")
        return False

def test_main_experiment_help():
    """测试主实验脚本帮助信息"""
    print("🧪 测试主实验脚本...")
    
    try:
        import subprocess
        
        # 测试帮助信息
        result = subprocess.run([
            sys.executable, 'main_experiment.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ 主实验脚本帮助信息正常")
            
            # 检查是否包含新的实验类型
            help_text = result.stdout
            new_types = ['baseline', 'tune', 'ablation', 'imbalance']
            
            for exp_type in new_types:
                if exp_type in help_text:
                    print(f"   ✅ 实验类型 '{exp_type}' 已支持")
                else:
                    print(f"   ⚠️ 实验类型 '{exp_type}' 未找到")
        else:
            print(f"   ❌ 主实验脚本执行失败: {result.stderr}")
            return False
        
        print("✅ 主实验脚本测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 主实验脚本测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始新实验框架功能测试")
    print("=" * 60)
    
    test_results = []
    
    # 运行各项测试
    test_functions = [
        test_config_loading,
        test_baseline_models,
        test_imbalance_utils,
        test_optuna_tuning,
        test_main_experiment_help,
        # test_wandb_utils,  # 需要WandB登录，可选测试
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_func.__name__, result))
        except Exception as e:
            print(f"❌ {test_func.__name__} 测试异常: {e}")
            test_results.append((test_func.__name__, False))
        
        print("-" * 40)
    
    # 汇总测试结果
    print("\n📊 测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！新实验框架功能正常。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查相关模块。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 