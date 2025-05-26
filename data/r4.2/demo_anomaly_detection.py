#!/usr/bin/env python
# coding: utf-8

"""
CERT r4.2 异常检测演示脚本

这个脚本演示如何使用不同类型的特征数据进行异常检测：
1. 基础特征数据（原始CSV）
2. 时间表示数据（percentile, concat, meandiff, meddiff）
3. 不同粒度数据（day, week, session）

使用方法：
python demo_anomaly_detection.py
"""

import os
import subprocess
import glob
import pandas as pd

def find_available_data_files():
    """找到所有可用的数据文件"""
    print("正在扫描可用的数据文件...")
    
    # 扫描CSV文件
    csv_files = glob.glob("ExtractedData/*.csv")
    
    # 扫描pickle文件
    pkl_files = glob.glob("*.pkl")
    
    print(f"\n找到 {len(csv_files)} 个CSV文件:")
    for f in sorted(csv_files)[:5]:  # 只显示前5个
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"  {f} ({size_mb:.1f} MB)")
    if len(csv_files) > 5:
        print(f"  ... 还有 {len(csv_files)-5} 个文件")
    
    print(f"\n找到 {len(pkl_files)} 个时间表示文件:")
    for f in sorted(pkl_files)[:5]:  # 只显示前5个
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"  {f} ({size_mb:.1f} MB)")
    if len(pkl_files) > 5:
        print(f"  ... 还有 {len(pkl_files)-5} 个文件")
    
    return csv_files, pkl_files

def run_anomaly_detection(input_file, output_dir, description):
    """运行异常检测"""
    print(f"\n{'='*60}")
    print(f"开始测试: {description}")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令
    cmd = [
        'python', 'anomaly_detection_r4.2.py',
        '--input', input_file,
        '--output_dir', output_dir,
        '--plot',
        '--max_iter', '50'  # 增加迭代次数提高收敛性
    ]
    
    try:
        # 运行异常检测
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 异常检测成功完成")
            print("输出摘要:")
            # 提取关键信息
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if ('AUC Score:' in line or 
                    '训练集:' in line or 
                    '测试集:' in line or
                    '图表已保存到:' in line):
                    print(f"  {line.strip()}")
        else:
            print("❌ 异常检测失败")
            print("错误信息:", result.stderr[-500:])  # 只显示最后500个字符
            
    except subprocess.TimeoutExpired:
        print("⏱️ 异常检测超时（5分钟）")
    except Exception as e:
        print(f"❌ 运行异常: {e}")

def main():
    print("CERT r4.2 异常检测演示")
    print("="*60)
    
    # 1. 扫描可用文件
    csv_files, pkl_files = find_available_data_files()
    
    if not csv_files and not pkl_files:
        print("❌ 没有找到可用的数据文件")
        print("请先运行特征提取脚本生成数据文件")
        return
    
    # 2. 创建演示输出目录
    demo_dir = "anomaly_detection_demo"
    os.makedirs(demo_dir, exist_ok=True)
    
    # 3. 测试不同类型的数据
    test_cases = []
    
    # CSV文件测试（如果可用）
    if csv_files:
        # 选择一个day级别的文件
        day_files = [f for f in csv_files if 'day' in f]
        if day_files:
            test_cases.append({
                'file': day_files[0],
                'output_dir': f"{demo_dir}/day_features",
                'description': "Day级别原始特征"
            })
        
        # 选择一个session级别的文件
        session_files = [f for f in csv_files if 'session' in f and 'nact' not in f and 'time' not in f]
        if session_files:
            test_cases.append({
                'file': session_files[0],
                'output_dir': f"{demo_dir}/session_features", 
                'description': "Session级别原始特征"
            })
    
    # Pickle文件测试（时间表示）
    if pkl_files:
        # percentile特征
        percentile_files = [f for f in pkl_files if 'percentile' in f]
        if percentile_files:
            test_cases.append({
                'file': percentile_files[0],
                'output_dir': f"{demo_dir}/percentile_features",
                'description': "Percentile时间表示特征"
            })
        
        # concat特征
        concat_files = [f for f in pkl_files if 'concat' in f]
        if concat_files:
            test_cases.append({
                'file': concat_files[0],
                'output_dir': f"{demo_dir}/concat_features",
                'description': "Concat时间表示特征"
            })
    
    # 4. 运行测试
    if not test_cases:
        print("❌ 没有找到适合的测试文件")
        return
    
    print(f"\n将测试 {len(test_cases)} 种不同类型的数据:")
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. {case['description']}: {case['file']}")
    
    print(f"\n开始运行异常检测测试...")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*20} 测试 {i}/{len(test_cases)} {'='*20}")
        run_anomaly_detection(
            case['file'], 
            case['output_dir'], 
            case['description']
        )
    
    # 5. 生成总结报告
    print(f"\n{'='*60}")
    print("演示完成！生成的文件:")
    print(f"{'='*60}")
    
    for case in test_cases:
        output_dir = case['output_dir']
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"\n{case['description']}:")
            print(f"  目录: {output_dir}")
            for f in files:
                print(f"    {f}")
        else:
            print(f"\n{case['description']}: ❌ 未生成文件")
    
    print(f"\n💡 你可以查看各个目录中的图表文件来比较不同特征的检测效果")
    print(f"💡 建议尝试使用更大的数据集（更多周数和用户）来获得有意义的异常检测结果")

if __name__ == "__main__":
    main() 