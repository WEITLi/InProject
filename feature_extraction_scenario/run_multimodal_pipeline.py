#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态数据流水线运行脚本
避免相对导入问题的独立启动脚本
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
core_logic_path = os.path.join(project_root, 'core_logic')
sys.path.insert(0, project_root)
sys.path.insert(0, core_logic_path)

# 导入多模态流水线
from core_logic.multimodal_pipeline import MultiModalDataPipeline

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行多模态异常检测数据流水线')
    
    # 数据参数
    parser.add_argument('--data_version', type=str, default='r4.2',
                       help='数据集版本 (默认: r4.2)')
    parser.add_argument('--start_week', type=int, default=0,
                       help='开始周数 (默认: 0)')
    parser.add_argument('--end_week', type=int, default=None,
                       help='结束周数 (默认: None, 表示处理到最大周数)')
    parser.add_argument('--max_users', type=int, default=100,
                       help='最大用户数限制 (默认: 100, None表示无限制)')
    
    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='特征向量维度 (默认: 256)')
    parser.add_argument('--sequence_length', type=int, default=128,
                       help='行为序列长度 (默认: 128)')
    
    # 系统参数
    parser.add_argument('--num_cores', type=int, default=8,
                       help='CPU核心数 (默认: 8)')
    
    # 路径参数
    parser.add_argument('--source_dir', type=str, default=None,
                       help='源数据目录覆盖路径')
    parser.add_argument('--work_dir', type=str, default=None,
                       help='工作目录覆盖路径')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'base_only', 'multimodal_only', 'training_only'],
                       help='运行模式: full(完整流水线), base_only(仅基础特征), multimodal_only(仅多模态), training_only(仅训练数据)')
    
    # 快速测试模式
    parser.add_argument('--quick_test', action='store_true',
                       help='快速测试模式 (处理前3周, 50个用户)')
    
    args = parser.parse_args()
    
    # 快速测试模式设置
    if args.quick_test:
        args.end_week = 3
        args.max_users = 50
        sample_ratio = 0.1  # 只读取10%的数据
        print("🚀 快速测试模式: 处理前3周, 50个用户, 10%数据采样")
    else:
        sample_ratio = None
    
    print(f"{'='*80}")
    print(f"多模态异常检测数据流水线")
    print(f"{'='*80}")
    print(f"数据版本: {args.data_version}")
    print(f"处理周数: {args.start_week} - {args.end_week or '最大周数'}")
    print(f"用户限制: {args.max_users or '无限制'}")
    print(f"特征维度: {args.feature_dim}")
    print(f"序列长度: {args.sequence_length}")
    print(f"CPU核心数: {args.num_cores}")
    print(f"运行模式: {args.mode}")
    print(f"{'='*80}")
    
    try:
        # 创建多模态数据流水线
        pipeline = MultiModalDataPipeline(
            data_version=args.data_version,
            feature_dim=args.feature_dim,
            num_cores=args.num_cores,
            source_dir_override=args.source_dir,
            work_dir_override=args.work_dir
        )
        
        # 根据模式运行不同的流水线
        if args.mode == 'full':
            # 运行完整流水线
            training_data = pipeline.run_full_multimodal_pipeline(
                start_week=args.start_week,
                end_week=args.end_week,
                max_users=args.max_users,
                sequence_length=args.sequence_length
            )
            
            print(f"\n🎉 完整流水线执行成功!")
            print(f"训练数据形状:")
            for key, value in training_data.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"  {key}: {len(value)} 项")
                else:
                    print(f"  {key}: {type(value)}")
                    
        elif args.mode == 'base_only':
            # 仅运行基础特征提取
            pipeline.run_base_feature_extraction(
                start_week=args.start_week,
                end_week=args.end_week,
                max_users=args.max_users,
                sample_ratio=sample_ratio
            )
            print(f"\n🎉 基础特征提取完成!")
            
        elif args.mode == 'multimodal_only':
            # 仅运行多模态数据提取
            pipeline.extract_multimodal_data(
                start_week=args.start_week,
                end_week=args.end_week,
                max_users=args.max_users
            )
            print(f"\n🎉 多模态数据提取完成!")
            
        elif args.mode == 'training_only':
            # 仅准备训练数据
            training_data = pipeline.prepare_training_data(
                start_week=args.start_week,
                end_week=args.end_week,
                max_users=args.max_users,
                sequence_length=args.sequence_length
            )
            print(f"\n🎉 训练数据准备完成!")
            print(f"数据样本数: {len(training_data['labels'])}")
        
    except Exception as e:
        print(f"\n❌ 流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 