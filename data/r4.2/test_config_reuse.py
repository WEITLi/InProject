#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试配置重用功能的脚本
验证不同配置之间的数据重用逻辑是否正确
"""

import os
import sys
import pandas as pd
import numpy as np

# 导入主脚本中的函数
sys.path.append('.')
from feature_extraction import parse_config_id, find_compatible_config, copy_compatible_data

def test_parse_config_id():
    """测试配置ID解析功能"""
    print("=== 测试配置ID解析功能 ===")
    
    test_cases = [
        "u200_w0-29_msession_s1",
        "u400_w0-29_msession_s1", 
        "uall_w0-74_mweekdaysession_s0",
        "u100_w5-15_mweek_s1"
    ]
    
    for config_id in test_cases:
        config = parse_config_id(config_id)
        print(f"配置ID: {config_id}")
        print(f"  解析结果: {config}")
        print()

def test_compatibility_logic():
    """测试兼容性逻辑"""
    print("=== 测试兼容性逻辑 ===")
    
    # 模拟一些配置场景
    scenarios = [
        {
            "target": "u200_w0-29_msession_s1",
            "candidates": [
                "u400_w0-29_msession_s1",  # 用户数更多，应该兼容
                "u200_w0-30_msession_s1",  # 周数范围更大，应该兼容
                "u100_w0-29_msession_s1",  # 用户数更少，不兼容
                "u200_w5-29_msession_s1",  # 周数范围不包含，不兼容
                "u200_w0-29_mweek_s1",     # 模式不匹配，不兼容
                "u200_w0-29_msession_s0",  # 子会话设置不同，不兼容
            ]
        },
        {
            "target": "u100_w10-20_mweek_s0",
            "candidates": [
                "uall_w0-74_mweekdaysession_s0",  # 全部用户，全部周，包含week模式，应该兼容
                "u200_w5-25_mweek_s0",            # 用户数更多，周数范围包含，应该兼容
                "u50_w10-20_mweek_s0",            # 用户数更少，不兼容
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"场景 {i+1}: 目标配置 {scenario['target']}")
        target_config = parse_config_id(scenario['target'])
        
        for candidate in scenario['candidates']:
            candidate_config = parse_config_id(candidate)
            
            # 手动检查兼容性
            is_compatible = True
            reasons = []
            
            # 检查用户数量
            if target_config['max_users'] != 'all':
                if candidate_config['max_users'] == 'all' or (isinstance(candidate_config['max_users'], int) and candidate_config['max_users'] >= target_config['max_users']):
                    pass
                else:
                    is_compatible = False
                    reasons.append("用户数不足")
            
            # 检查周数范围
            if (candidate_config['start_week'] <= target_config['start_week'] and candidate_config['end_week'] >= target_config['end_week']):
                pass
            else:
                is_compatible = False
                reasons.append("周数范围不包含")
            
            # 检查模式
            if target_config['modes'] in candidate_config['modes']:
                pass
            else:
                is_compatible = False
                reasons.append("模式不匹配")
            
            # 检查子会话设置
            if candidate_config['enable_subsession'] == target_config['enable_subsession']:
                pass
            else:
                is_compatible = False
                reasons.append("子会话设置不同")
            
            status = "✓ 兼容" if is_compatible else f"✗ 不兼容 ({', '.join(reasons)})"
            print(f"  候选配置 {candidate}: {status}")
        print()

def create_mock_data():
    """创建模拟数据用于测试"""
    print("=== 创建模拟数据 ===")
    
    # 创建测试目录
    os.makedirs("test_data", exist_ok=True)
    
    # 创建模拟的NumDataByWeek数据
    config_id = "u400_w0-29_msession_s1"
    
    for week in range(0, 5):  # 只创建前5周的数据用于测试
        # 模拟400个用户的数据
        n_activities = np.random.randint(100, 500)  # 每周100-500个活动
        
        data = {
            'actid': range(n_activities),
            'pcid': np.random.choice(['PC001', 'PC002', 'PC003'], n_activities),
            'time_stamp': pd.date_range('2010-01-01', periods=n_activities, freq='H'),
            'user': np.random.choice(range(400), n_activities),  # 400个用户
            'day': np.random.randint(0, 7, n_activities),
            'act': np.random.choice([1, 2, 3, 4, 5, 6, 7], n_activities),
            'pc': np.random.choice([0, 1, 2, 3], n_activities),
            'time': np.random.choice([1, 2, 3, 4], n_activities),
            'mal_act': np.random.choice([0, 1], n_activities, p=[0.95, 0.05]),
            'insider': np.random.choice([0, 1, 2, 3], n_activities, p=[0.9, 0.03, 0.04, 0.03])
        }
        
        # 添加其他特征列（简化版本）
        for i in range(20):  # 添加20个额外特征
            data[f'feature_{i}'] = np.random.randn(n_activities)
        
        df = pd.DataFrame(data)
        filename = f"test_data/{week}_num_{config_id}.pickle"
        df.to_pickle(filename)
        print(f"创建模拟数据: {filename} ({df.shape[0]} 行, {df.shape[1]} 列)")

def test_data_reuse():
    """测试数据重用功能"""
    print("=== 测试数据重用功能 ===")
    
    # 测试从u400配置复制到u200配置
    source_config = "u400_w0-29_msession_s1"
    target_config = "u200_w0-29_msession_s1"
    
    print(f"测试从 {source_config} 复制数据到 {target_config}")
    
    # 复制数据
    copied_weeks = copy_compatible_data(source_config, target_config, range(0, 5), "test_data")
    
    if copied_weeks:
        print(f"成功复制了 {len(copied_weeks)} 周的数据: {copied_weeks}")
        
        # 验证复制的数据
        for week in copied_weeks:
            source_file = f"test_data/{week}_num_{source_config}.pickle"
            target_file = f"test_data/{week}_num_{target_config}.pickle"
            
            if os.path.exists(source_file) and os.path.exists(target_file):
                source_df = pd.read_pickle(source_file)
                target_df = pd.read_pickle(target_file)
                
                source_users = len(source_df['user'].unique())
                target_users = len(target_df['user'].unique())
                
                print(f"  周 {week}: 源数据 {source_users} 用户 -> 目标数据 {target_users} 用户")
                
                if target_users <= 200:
                    print(f"    ✓ 用户数量限制正确")
                else:
                    print(f"    ✗ 用户数量限制失败")
    else:
        print("没有复制任何数据")

def cleanup_test_data():
    """清理测试数据"""
    print("=== 清理测试数据 ===")
    import shutil
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
        print("已删除测试数据目录")

def main():
    """主测试函数"""
    print("开始测试配置重用功能...\n")
    
    try:
        # 运行各项测试
        test_parse_config_id()
        test_compatibility_logic()
        create_mock_data()
        test_data_reuse()
        
        print("所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试数据
        cleanup_test_data()

if __name__ == "__main__":
    main() 