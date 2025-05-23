#!/usr/bin/env python
# coding: utf-8

"""
数据清理脚本
用于删除之前生成的数据文件和文件夹

清理内容：
- DataByWeek/ 文件夹及其内容
- NumDataByWeek/ 文件夹及其内容  
- ExtractedData/ 文件夹及其内容
- 当前目录下的 .pkl 文件（保留CSV原始数据）
"""

import os
import shutil
import glob
import argparse
from pathlib import Path

def get_file_size(path):
    """获取文件或文件夹大小（MB）"""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
        return total_size / (1024 * 1024)
    return 0

def scan_cleanup_targets():
    """扫描需要清理的文件和文件夹"""
    current_dir = os.getcwd()
    print(f"扫描目录: {current_dir}")
    print("="*60)
    
    # 要清理的文件夹
    target_dirs = ['DataByWeek', 'NumDataByWeek', 'ExtractedData']
    
    # 要清理的pkl文件（排除一些重要的数据文件）
    pkl_patterns = [
        '*.pkl'
    ]
    
    # 排除的重要文件（不删除）
    exclude_files = [
        'week-r5.2-percentile30.pkl',  # 示例数据
        # 可以在这里添加其他不想删除的文件
    ]
    
    cleanup_items = []
    total_size = 0
    
    # 检查文件夹
    for dir_name in target_dirs:
        if os.path.exists(dir_name):
            size_mb = get_file_size(dir_name)
            total_size += size_mb
            
            # 统计文件数量
            file_count = 0
            for root, dirs, files in os.walk(dir_name):
                file_count += len(files)
            
            cleanup_items.append({
                'type': 'directory',
                'path': dir_name,
                'size_mb': size_mb,
                'file_count': file_count
            })
            
            print(f"📁 文件夹: {dir_name}/")
            print(f"   大小: {size_mb:.1f} MB")
            print(f"   文件数: {file_count}")
            print()
    
    # 检查pkl文件
    pkl_files = []
    for pattern in pkl_patterns:
        found_files = glob.glob(pattern)
        for file_path in found_files:
            # 检查是否在排除列表中
            if os.path.basename(file_path) not in exclude_files:
                pkl_files.append(file_path)
    
    if pkl_files:
        print("📄 PKL 文件:")
        for file_path in pkl_files:
            size_mb = get_file_size(file_path)
            total_size += size_mb
            
            cleanup_items.append({
                'type': 'file',
                'path': file_path,
                'size_mb': size_mb,
                'file_count': 1
            })
            
            print(f"   {file_path} ({size_mb:.1f} MB)")
        print()
    
    # 显示排除的文件
    excluded_existing = [f for f in exclude_files if os.path.exists(f)]
    if excluded_existing:
        print("🔒 排除的文件（将保留）:")
        for file_path in excluded_existing:
            size_mb = get_file_size(file_path)
            print(f"   {file_path} ({size_mb:.1f} MB)")
        print()
    
    print("="*60)
    print(f"总共可清理大小: {total_size:.1f} MB")
    print(f"总共可清理项目: {len(cleanup_items)}")
    
    return cleanup_items, total_size

def confirm_cleanup(cleanup_items, total_size_mb):
    """确认是否执行清理"""
    if not cleanup_items:
        print("✅ 没有找到需要清理的文件或文件夹")
        return False
    
    print("\n" + "="*60)
    print("⚠️  警告：即将删除以下内容")
    print("="*60)
    
    for item in cleanup_items:
        if item['type'] == 'directory':
            print(f"📁 {item['path']}/ ({item['size_mb']:.1f} MB, {item['file_count']} 文件)")
        else:
            print(f"📄 {item['path']} ({item['size_mb']:.1f} MB)")
    
    print(f"\n💾 总大小: {total_size_mb:.1f} MB")
    print("⚠️  注意：删除操作不可恢复！")
    
    while True:
        response = input("\n确认删除？(y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            return True
        elif response in ['n', 'no', '否']:
            return False
        else:
            print("请输入 y/yes 或 n/no")

def execute_cleanup(cleanup_items, dry_run=False):
    """执行清理操作"""
    if dry_run:
        print("\n🔍 模拟运行模式（不会实际删除）:")
    else:
        print("\n🗑️  开始清理...")
    
    print("="*60)
    
    success_count = 0
    error_count = 0
    
    for item in cleanup_items:
        try:
            if dry_run:
                print(f"[模拟] 删除 {item['path']}")
            else:
                if item['type'] == 'directory':
                    shutil.rmtree(item['path'])
                    print(f"✅ 已删除文件夹: {item['path']}/")
                else:
                    os.remove(item['path'])
                    print(f"✅ 已删除文件: {item['path']}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 删除失败 {item['path']}: {e}")
            error_count += 1
    
    print("="*60)
    if dry_run:
        print(f"模拟完成: {success_count} 个项目可以删除, {error_count} 个错误")
    else:
        print(f"清理完成: {success_count} 个项目已删除, {error_count} 个错误")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='清理生成的数据文件和文件夹')
    parser.add_argument('--dry-run', action='store_true', 
                       help='模拟运行，不实际删除文件')
    parser.add_argument('--force', action='store_true',
                       help='强制删除，不询问确认')
    parser.add_argument('--exclude', nargs='*', default=[],
                       help='排除特定文件（文件名）')
    
    args = parser.parse_args()
    
    print("🧹 数据清理脚本")
    print("="*60)
    
    # 扫描要清理的内容
    cleanup_items, total_size = scan_cleanup_targets()
    
    if not cleanup_items:
        return
    
    # 确认清理
    if not args.force:
        if not confirm_cleanup(cleanup_items, total_size):
            print("❌ 清理已取消")
            return
    
    # 执行清理
    execute_cleanup(cleanup_items, dry_run=args.dry_run)
    
    # 清理后再次扫描，显示结果
    if not args.dry_run:
        print("\n🔍 清理后状态:")
        remaining_items, remaining_size = scan_cleanup_targets()
        if not remaining_items:
            print("✅ 所有目标文件已清理完成")
        else:
            print(f"⚠️  仍有 {len(remaining_items)} 个项目未清理")

if __name__ == "__main__":
    main() 