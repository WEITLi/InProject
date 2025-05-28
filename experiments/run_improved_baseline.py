#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版基线模型运行脚本
Run Improved Baseline Models Script

这个脚本专门用于运行改进版的基线模型实验，
展示Random Forest和XGBoost在差异化特征工程和交叉验证下的性能差异。
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    parser = argparse.ArgumentParser(
        description="运行改进版基线模型实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认参数运行改进版baseline实验
  python run_improved_baseline.py
  
  # 指定更多用户和更多交叉验证折数
  python run_improved_baseline.py --max_users 200 --baseline_cv_folds 10
  
  # 使用配置文件
  python run_improved_baseline.py --config_file configs/baseline_config.yaml
  
  # 指定输出目录
  python run_improved_baseline.py --output_dir ./results/improved_baseline_test
        """
    )
    
    # 基本参数
    parser.add_argument('--config_file', type=str, default=None,
                       help='配置文件路径 (可选)')
    parser.add_argument('--output_dir', type=str, default='./results/improved_baseline',
                       help='输出目录')
    parser.add_argument('--max_users', type=int, default=100,
                       help='最大用户数')
    parser.add_argument('--baseline_cv_folds', type=int, default=5,
                       help='交叉验证折数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 生成实验名称
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"improved_baseline_{timestamp}"
    
    # 构建main_experiment.py的调用参数
    cmd_args = [
        'python', 'main_experiment.py',
        '--run_type', 'baseline',
        '--use_improved_baseline',  # 关键参数：启用改进版baseline
        '--experiment_name', experiment_name,
        '--output_dir', args.output_dir,
        '--max_users', str(args.max_users),
        '--baseline_cv_folds', str(args.baseline_cv_folds),
        '--seed', str(args.seed)
    ]
    
    # 如果指定了配置文件，添加到参数中
    if args.config_file:
        cmd_args.extend(['--config_file', args.config_file])
    
    print("🚀 启动改进版基线模型实验...")
    print(f"📋 实验名称: {experiment_name}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"👥 最大用户数: {args.max_users}")
    print(f"🔄 交叉验证折数: {args.baseline_cv_folds}")
    print(f"🎲 随机种子: {args.seed}")
    print(f"📄 配置文件: {args.config_file or '使用默认配置'}")
    print()
    print("执行命令:")
    print(" ".join(cmd_args))
    print()
    
    # 执行命令
    import subprocess
    try:
        result = subprocess.run(cmd_args, check=True, cwd=current_dir)
        print("\n✅ 改进版基线模型实验完成!")
        print(f"📁 结果保存在: {os.path.join(args.output_dir, experiment_name)}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 实验执行失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 实验被用户中断")
        sys.exit(1)

if __name__ == "__main__":
    main() 