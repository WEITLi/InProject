#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整流水线测试脚本
验证dataset_pipeline.py的功能是否正常
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import shutil

# 调整 sys.path 以便正确导入 core_logic 模块
# 当前脚本位于 tests/ 目录下, 我们需要添加到 feature_extraction_scenario 的父目录
# .../feature_extraction_scenario/tests/test_pipeline.py
# 我们需要 .../feature_extraction_scenario/../ (即 InProject 目录)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# 从新的位置导入 CERTDatasetPipeline
from feature_extraction_scenario.core_logic.dataset_pipeline import CERTDatasetPipeline

# 定义测试文件和目录常量
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA_DIR = os.path.join(TESTS_DIR, 'sample_test_data')
FEATURE_EXTRACTION_SCENARIO_DIR = os.path.dirname(TESTS_DIR)
TEST_OUTPUT_DIR = os.path.join(FEATURE_EXTRACTION_SCENARIO_DIR, 'test_output')
TEST_DATA_MARKER_FILE = os.path.join(SAMPLE_DATA_DIR, '.test_data_created_by_pipeline')
BACKUP_DIR_ROOT = os.path.join(TESTS_DIR, 'backup_data') # Backup directory inside tests

def _ensure_dir(directory_path: str):
    """确保目录存在"""
    os.makedirs(directory_path, exist_ok=True)

def create_test_data():
    """
    创建测试用的CERT数据文件到 SAMPLE_DATA_DIR
    """
    print(f"📁 创建测试数据文件到: {SAMPLE_DATA_DIR}")
    _ensure_dir(SAMPLE_DATA_DIR)
    
    # 检查是否已存在真实数据 (此逻辑在此处可能不太适用，因为测试数据是隔离的)
    # 可以考虑移除或调整，暂时保留但输出调整
    real_data_files = ['email.csv', 'file.csv', 'http.csv', 'logon.csv', 'device.csv']
    # This check should ideally point to a shared, non-test data location if strict separation is needed.
    # For now, let's assume 'real' data is NOT in SAMPLE_DATA_DIR.
    
    # 创建测试用户列表
    users = ['ACM2278', 'ACM1796', 'CMP2946', 'BTH8471', 'DYQ9624']
    pcs = ['PC-1234', 'PC-5678', 'PC-9012', 'PC-3456']
    
    # 基础时间
    base_date = datetime(2024, 1, 1)
    
    # 创建邮件数据
    email_data = []
    for week in range(3):
        for day in range(7):
            for user in users[:3]:
                current_date = base_date + timedelta(weeks=week, days=day, hours=np.random.randint(8, 18))
                email_data.append({
                    'date': current_date.strftime('%m/%d/%Y %H:%M:%S'), 'user': user, 'pc': np.random.choice(pcs),
                    'to': f'{np.random.choice(users)}@dtaa.com', 'cc': '', 'bcc': '', 'from': f'{user}@dtaa.com',
                    'size': np.random.randint(1000, 50000), 'content': f'Test email content from {user}', 'activity': 'Send'
                })
    save_test_data(pd.DataFrame(email_data), 'email.csv')
    print(f"   创建 email.csv: {len(email_data)} 条记录")
    
    # 创建文件数据
    file_data = []
    for week in range(3):
        for day in range(7):
            for user in users:
                current_date = base_date + timedelta(weeks=week, days=day, hours=np.random.randint(8, 18))
                file_data.append({
                    'date': current_date.strftime('%m/%d/%Y %H:%M:%S'), 'user': user, 'pc': np.random.choice(pcs),
                    'filename': f'document_{np.random.randint(1, 100)}.docx', 'content': f'File content from {user}',
                    'activity': np.random.choice(['file open', 'file copy', 'file write'])
                })
    save_test_data(pd.DataFrame(file_data), 'file.csv')
    print(f"   创建 file.csv: {len(file_data)} 条记录")
    
    # 创建HTTP数据
    http_data = []
    urls = ['https://www.company.com/reports', 'https://www.google.com/search', 'https://www.github.com/projects', 'https://www.stackoverflow.com/questions']
    for week in range(3):
        for day in range(7):
            for user in users:
                current_date = base_date + timedelta(weeks=week, days=day, hours=np.random.randint(8, 18))
                http_data.append({
                    'date': current_date.strftime('%m/%d/%Y %H:%M:%S'), 'user': user, 'pc': np.random.choice(pcs),
                    'url': np.random.choice(urls), 'content': f'Web page content for {user}', 'activity': 'www visit'
                })
    save_test_data(pd.DataFrame(http_data), 'http.csv')
    print(f"   创建 http.csv: {len(http_data)} 条记录")
    
    # 创建登录数据
    logon_data = []
    for week in range(3):
        for day in range(7):
            for user in users:
                for _ in range(np.random.randint(1, 3)):
                    current_date = base_date + timedelta(weeks=week, days=day, hours=np.random.randint(7, 9))
                    logon_data.append({
                        'date': current_date.strftime('%m/%d/%Y %H:%M:%S'), 'user': user, 'pc': np.random.choice(pcs), 'activity': 'Logon'
                    })
    save_test_data(pd.DataFrame(logon_data), 'logon.csv')
    print(f"   创建 logon.csv: {len(logon_data)} 条记录")
    
    # 创建设备数据
    device_data = []
    for week in range(3):
        for day in range(7):
            for user in users[:2]:
                current_date = base_date + timedelta(weeks=week, days=day, hours=np.random.randint(10, 16))
                device_data.append({
                    'date': current_date.strftime('%m/%d/%Y %H:%M:%S'), 'user': user, 'pc': np.random.choice(pcs),
                    'activity': 'Connect', 'content': f'USB device connected by {user}'
                })
    save_test_data(pd.DataFrame(device_data), 'device.csv')
    print(f"   创建 device.csv: {len(device_data)} 条记录")
    
    # 创建心理测量数据（可选）
    psycho_data = []
    for user in users:
        psycho_data.append({
            'user': user, 'O': np.random.uniform(0.3, 0.8), 'C': np.random.uniform(0.3, 0.8),
            'E': np.random.uniform(0.3, 0.8), 'A': np.random.uniform(0.3, 0.8), 'N': np.random.uniform(0.3, 0.8)
        })
    save_test_data(pd.DataFrame(psycho_data), 'psychometric.csv')
    print(f"   创建 psychometric.csv: {len(psycho_data)} 条记录")
    
    # 创建answers目录和恶意用户标签 (在SAMPLE_DATA_DIR下)
    answers_dir = os.path.join(SAMPLE_DATA_DIR, 'answers')
    _ensure_dir(answers_dir)
    insiders_data = [
        {'user': 'ACM2278', 'scenario': 1, 'start_week': 1, 'end_week': 2},
        {'user': 'CMP2946', 'scenario': 2, 'start_week': 0, 'end_week': 1}
    ]
    save_test_data(pd.DataFrame(insiders_data), os.path.join('answers', 'insiders.csv')) # filename is relative to SAMPLE_DATA_DIR
    print(f"   创建 answers/insiders.csv: {len(insiders_data)} 个恶意用户")
    
    # 创建测试数据标记文件
    with open(TEST_DATA_MARKER_FILE, 'w') as f:
        f.write(f"测试数据创建时间: {datetime.now()}\\n")
        f.write(f"此文件标记 SAMPLE_DATA_DIR ({SAMPLE_DATA_DIR}) 中的测试数据的存在\\n")
        test_files = ['email.csv', 'file.csv', 'http.csv', 'logon.csv', 'device.csv', 'psychometric.csv', 'answers/insiders.csv']
        for test_file in test_files:
            f.write(f"TEST_FILE:{test_file}\\n") # Store relative paths within SAMPLE_DATA_DIR
    
    print("✅ 测试数据创建完成")
    return True

def save_test_data(df: pd.DataFrame, filename: str):
    """
    保存测试数据到 SAMPLE_DATA_DIR 并添加标记
    filename is relative to SAMPLE_DATA_DIR
    """
    full_path = os.path.join(SAMPLE_DATA_DIR, filename)
    _ensure_dir(os.path.dirname(full_path)) # Ensure subdirectory (like 'answers') exists
    
    df.to_csv(full_path, index=False)
    
    with open(full_path, 'r+') as f: # Open in r+ to prepend
        content = f.read()
        f.seek(0, 0)
        test_marker = f"# TEST_DATA_CREATED_BY_PIPELINE at {datetime.now()}\\n"
        f.write(test_marker + content)

def is_test_data_file(file_path: str) -> bool: # Now takes full path
    """
    检查文件是否为测试数据
    """
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline()
            return 'TEST_DATA_CREATED_BY_PIPELINE' in first_line
    except:
        return False

def backup_real_data(files_to_backup: List[str], original_data_source_dir: str): # Added original_data_source_dir
    """
    备份真实数据文件 (if any were found outside SAMPLE_DATA_DIR)
    """
    _ensure_dir(BACKUP_DIR_ROOT)
    backup_dir_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    specific_backup_dir = os.path.join(BACKUP_DIR_ROOT, backup_dir_name)
    _ensure_dir(specific_backup_dir)
    
    print(f"🕵️ 正在检查 {original_data_source_dir} 中的文件进行备份...")
    backed_up_count = 0
    for file_rel_path in files_to_backup: # files_to_backup are relative paths like 'email.csv'
        original_file_path = os.path.join(original_data_source_dir, file_rel_path)
        if os.path.exists(original_file_path):
            backup_target_path = os.path.join(specific_backup_dir, file_rel_path)
            _ensure_dir(os.path.dirname(backup_target_path))
            shutil.copy2(original_file_path, backup_target_path)
            print(f"   备份 {original_file_path} -> {backup_target_path}")
            backed_up_count +=1
        else:
            print(f"   文件未在源目录找到，跳过备份: {original_file_path}")

    if backed_up_count > 0:
        print(f"✅ {backed_up_count} 个真实数据文件已备份到: {specific_backup_dir}")
    else:
        print("   未找到符合条件的真实数据文件进行备份。")


def get_test_data_files() -> List[str]:
    """
    获取 SAMPLE_DATA_DIR 中的测试数据文件列表 (full paths)
    """
    test_files_full_paths = []
    if os.path.exists(TEST_DATA_MARKER_FILE):
        with open(TEST_DATA_MARKER_FILE, 'r') as f:
            for line in f:
                if line.startswith('TEST_FILE:'):
                    # Paths in marker file are relative to SAMPLE_DATA_DIR
                    relative_test_file = line.replace('TEST_FILE:', '').strip()
                    full_path = os.path.join(SAMPLE_DATA_DIR, relative_test_file)
                    test_files_full_paths.append(full_path)
    else: # Fallback: scan SAMPLE_DATA_DIR if marker is missing
        if os.path.exists(SAMPLE_DATA_DIR):
            for root, _, files in os.walk(SAMPLE_DATA_DIR):
                for file in files:
                    full_path = os.path.join(root, file)
                    if file.endswith('.csv') and is_test_data_file(full_path):
                         # Add only if it's a marked test data CSV to avoid grabbing other files
                        test_files_full_paths.append(full_path)
    return list(set(test_files_full_paths)) # Ensure unique paths

def test_basic_functionality():
    """测试基础功能"""
    print("\\n🧪 测试基础功能...")
    
    # 测试导入 (already done at top level)
    print("✅ 模块导入已在脚本顶部处理")
    
    # 测试初始化
    try:
        # Initialize with overridden paths for testing
        pipeline = CERTDatasetPipeline(
            data_version='r4.2_test', 
            feature_dim=128,
            source_dir_override=SAMPLE_DATA_DIR,
            work_dir_override=TEST_OUTPUT_DIR
        )
        print("✅ 流水线初始化成功 (使用测试路径)")
    except Exception as e:
        print(f"❌ 流水线初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_step_by_step():
    """分步测试流水线"""
    print("\\n🔬 分步测试流水线...")
    
    # Ensure test data exists
    if not os.path.exists(TEST_DATA_MARKER_FILE) and not get_test_data_files():
        print("   测试数据不存在，正在创建...")
        if not create_test_data():
            print("❌ 创建测试数据失败，中止分步测试。")
            return False
            
    _ensure_dir(TEST_OUTPUT_DIR) # Ensure output directory exists
    
    pipeline = CERTDatasetPipeline(
        data_version='r4.2_test', 
        feature_dim=128,
        source_dir_override=SAMPLE_DATA_DIR,
        work_dir_override=TEST_OUTPUT_DIR
    )
    
    try:
        # Step 1: 数据合并
        print("\\n1️⃣ 测试数据合并...")
        pipeline.step1_combine_raw_data(start_week=0, end_week=3)
        
        # 检查输出 (paths relative to TEST_OUTPUT_DIR)
        week_files = [os.path.join(TEST_OUTPUT_DIR, f"DataByWeek/{i}.pickle") for i in range(3)]
        existing_files = [f for f in week_files if os.path.exists(f)]
        print(f"   生成周文件: {len(existing_files)}/3 in {os.path.join(TEST_OUTPUT_DIR, 'DataByWeek')}")
        
        # Step 2: 用户数据
        print("\\n2️⃣ 测试用户数据加载...")
        users_df = pipeline.step2_load_user_data() # This will use psychometric and answers from SAMPLE_DATA_DIR
        print(f"   用户数量: {len(users_df)}")
        print(f"   恶意用户: {len(users_df[users_df['malscene'] > 0])}")
        
        # Step 3: 特征提取
        print("\\n3️⃣ 测试特征提取...")
        pipeline.step3_extract_features(users_df, start_week=0, end_week=3, max_users=3)
        
        # 检查特征文件 (paths relative to TEST_OUTPUT_DIR)
        feature_files = [os.path.join(TEST_OUTPUT_DIR, f"NumDataByWeek/{i}_features.pickle") for i in range(3)]
        existing_features = [f for f in feature_files if os.path.exists(f)]
        print(f"   生成特征文件: {len(existing_features)}/3 in {os.path.join(TEST_OUTPUT_DIR, 'NumDataByWeek')}")
        
        # Step 4: 多级别分析
        print("\\n4️⃣ 测试多级别分析...")
        pipeline.step4_multi_level_analysis(start_week=0, end_week=3, modes=['week', 'day', 'session'])
        
        # 检查输出文件 (paths relative to TEST_OUTPUT_DIR)
        output_files_relative = [
            'WeekLevelFeatures/weeks_0_2.csv',
            'DayLevelFeatures/days_0_2.csv', 
            'SessionLevelFeatures/sessions_0_2.csv'
        ]
        
        for rel_path in output_files_relative:
            output_file_path = os.path.join(TEST_OUTPUT_DIR, rel_path)
            if os.path.exists(output_file_path):
                df = pd.read_csv(output_file_path)
                print(f"   ✅ {output_file_path}: {len(df)} 条记录")
            else:
                print(f"   ❌ {output_file_path}: 文件不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 分步测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """测试完整流水线"""
    print("\\n🚀 测试完整流水线...")

    if not os.path.exists(TEST_DATA_MARKER_FILE) and not get_test_data_files():
        print("   测试数据不存在，正在创建...")
        if not create_test_data():
            print("❌ 创建测试数据失败，中止完整流水线测试。")
            return False

    _ensure_dir(TEST_OUTPUT_DIR)
    
    pipeline = CERTDatasetPipeline(
        data_version='r4.2_test', 
        feature_dim=64,
        source_dir_override=SAMPLE_DATA_DIR,
        work_dir_override=TEST_OUTPUT_DIR
    )
    
    try:
        start_time = time.time()
        
        pipeline.run_full_pipeline(
            start_week=0,
            end_week=3,
            max_users=3,
            modes=['week', 'day']  # 只测试周和日级别
        )
        
        elapsed_time = (time.time() - start_time) / 60
        print(f"✅ 完整流水线测试成功，耗时: {elapsed_time:.1f} 分钟 (输出到 {TEST_OUTPUT_DIR})")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_results():
    """分析测试结果 from TEST_OUTPUT_DIR"""
    print("\\n📊 分析测试结果...")
    
    result_files_relative = {
        'week': 'WeekLevelFeatures/weeks_0_2.csv',
        'day': 'DayLevelFeatures/days_0_2.csv',
        'session': 'SessionLevelFeatures/sessions_0_2.csv' # This might not be created if modes in test_full_pipeline is limited
    }
    
    all_results_found = True
    for mode, rel_file_path in result_files_relative.items():
        full_file_path = os.path.join(TEST_OUTPUT_DIR, rel_file_path)
        if os.path.exists(full_file_path):
            df = pd.read_csv(full_file_path)
            print(f"\\n📈 {mode.capitalize()}级别分析结果 (来自 {full_file_path}):")
            print(f"   记录数: {len(df)}")
            print(f"   用户数: {df['user'].nunique()}")
            if 'n_events' in df.columns: print(f"   平均事件数: {df['n_events'].mean():.1f}")
            if 'malicious_ratio' in df.columns: print(f"   恶意比例: {df['malicious_ratio'].mean():.3f}")
            
            print(f"   样本数据:")
            display_cols = [col for col in ['user', 'week', 'mode', 'n_events', 'malicious_ratio'] if col in df.columns]
            print(df[display_cols].head(3).to_string(index=False))
        else:
            # Only mark as not found if it was expected based on test_full_pipeline modes
            if mode in ['week', 'day']: # Assuming these are always run by test_full_pipeline
                 print(f"⚠️  {mode}级别结果文件不存在: {full_file_path}")
                 all_results_found = False
            else:
                 print(f"ℹ️  {mode}级别结果文件不存在 (可能未在测试中运行): {full_file_path}")


    if not all_results_found and ('session' not in ['week', 'day']): # Check if session was the missing one
        print("提醒: 'session' 级别数据可能未生成，因为 `test_full_pipeline` 中 modes=['week', 'day']。分步测试会生成它。")


def cleanup_test_files():
    """安全清理 TEST_OUTPUT_DIR 和 SAMPLE_DATA_DIR"""
    print("\\n🧹 安全清理测试文件和目录...")
    
    # 1. 清理生成的输出目录
    if os.path.exists(TEST_OUTPUT_DIR):
        try:
            shutil.rmtree(TEST_OUTPUT_DIR)
            print(f"   删除测试输出目录: {TEST_OUTPUT_DIR}")
        except Exception as e:
            print(f"   删除测试输出目录失败 {TEST_OUTPUT_DIR}: {e}")
    else:
        print(f"   测试输出目录未找到，无需删除: {TEST_OUTPUT_DIR}")

    # 2. 清理样本数据目录 (including marker file)
    if os.path.exists(SAMPLE_DATA_DIR):
        try:
            shutil.rmtree(SAMPLE_DATA_DIR) # This will remove all files within, including the marker
            print(f"   删除样本数据目录: {SAMPLE_DATA_DIR}")
        except Exception as e:
            print(f"   删除样本数据目录失败 {SAMPLE_DATA_DIR}: {e}")
    else:
        print(f"   样本数据目录未找到，无需删除: {SAMPLE_DATA_DIR}")
        # Explicitly check and remove marker if it somehow exists outside SAMPLE_DATA_DIR (legacy)
        legacy_marker = '.test_data_created_by_pipeline'
        if os.path.exists(legacy_marker) and os.path.isfile(legacy_marker):
             try:
                os.remove(legacy_marker)
                print(f"   删除遗留的标记文件: {legacy_marker}")
             except Exception as e:
                print(f"   删除遗留标记文件失败 {legacy_marker}: {e}")


    # 3. 删除备份目录（可选, and now they are inside TESTS_DIR/backup_data)
    if os.path.exists(BACKUP_DIR_ROOT):
        backup_sub_dirs = [d for d in os.listdir(BACKUP_DIR_ROOT) if os.path.isdir(os.path.join(BACKUP_DIR_ROOT, d)) and d.startswith('backup_')]
        if backup_sub_dirs:
            print(f"\\n📦 发现测试相关的备份目录于 {BACKUP_DIR_ROOT}: {backup_sub_dirs}")
            # Non-interactive cleanup for automated tests, or prompt if desired
            # For now, let's make it non-interactive:
            print("   自动清理测试相关的备份目录...")
            for backup_dir_name in backup_sub_dirs:
                try:
                    shutil.rmtree(os.path.join(BACKUP_DIR_ROOT, backup_dir_name))
                    print(f"      删除备份目录: {backup_dir_name}")
                except Exception as e:
                    print(f"      删除备份目录失败 {backup_dir_name}: {e}")
            # Optionally remove BACKUP_DIR_ROOT if it's now empty
            if not os.listdir(BACKUP_DIR_ROOT):
                try:
                    os.rmdir(BACKUP_DIR_ROOT)
                    print(f"   删除空的备份根目录: {BACKUP_DIR_ROOT}")
                except Exception as e:
                    print(f"   删除空的备份根目录失败 {BACKUP_DIR_ROOT}: {e}")
        elif not os.listdir(BACKUP_DIR_ROOT): # If BACKUP_DIR_ROOT exists but is empty
            try:
                os.rmdir(BACKUP_DIR_ROOT)
                print(f"   删除空的备份根目录: {BACKUP_DIR_ROOT}")
            except Exception as e:
                print(f"   删除空的备份根目录失败 {BACKUP_DIR_ROOT}: {e}")
    
    print("✅ 测试文件和目录清理完成")


def check_data_safety(): # Primarily checks SAMPLE_DATA_DIR now
    """检查 SAMPLE_DATA_DIR 的数据安全性"""
    print("\\n🔒 测试数据安全性检查...")
    
    if os.path.exists(TEST_DATA_MARKER_FILE):
        print(f"✅ 找到测试数据标记文件: {TEST_DATA_MARKER_FILE}")
        # Reading from marker is now primary source for get_test_data_files
        marked_files = get_test_data_files() # These are full paths now
        print(f"   标记的测试文件 ({len(marked_files)}):")
        for f_path in marked_files:
            print(f"     - {f_path} (存在: {os.path.exists(f_path)})")
    else:
        print(f"⚠️  未找到测试数据标记文件: {TEST_DATA_MARKER_FILE}")
        # Check if there are any CSVs in SAMPLE_DATA_DIR anyway
        if os.path.exists(SAMPLE_DATA_DIR):
            csv_files_in_sample_dir = [os.path.join(SAMPLE_DATA_DIR, f) for f in os.listdir(SAMPLE_DATA_DIR) if f.endswith('.csv')]
            if csv_files_in_sample_dir:
                print(f"   在 {SAMPLE_DATA_DIR} 中发现以下未标记的CSV文件:")
                for csv_file in csv_files_in_sample_dir:
                     print(f"     - {csv_file} (是否为测试文件: {is_test_data_file(csv_file)})")
            else:
                print(f"   {SAMPLE_DATA_DIR} 中没有找到CSV文件。")
        else:
            print(f"   样本数据目录 {SAMPLE_DATA_DIR} 不存在。")
    
    # Backup directory check remains similar but points to BACKUP_DIR_ROOT
    if os.path.exists(BACKUP_DIR_ROOT):
        backup_sub_dirs = [d for d in os.listdir(BACKUP_DIR_ROOT) if os.path.isdir(os.path.join(BACKUP_DIR_ROOT, d)) and d.startswith('backup_')]
        if backup_sub_dirs:
            print(f"\\n📦 测试相关的备份目录位于 {BACKUP_DIR_ROOT}: {backup_sub_dirs}")
        else:
            print(f"\\n📦 {BACKUP_DIR_ROOT} 中没有找到测试相关的备份子目录。")
    else:
        print(f"\\n📦 测试相关的备份根目录 {BACKUP_DIR_ROOT} 不存在。")


def restore_real_data(): # This function might need rethinking in the new structure
    """
    从备份恢复真实数据. 
    NOTE: This function's utility is reduced if test data is strictly isolated.
    It might be used to restore to a *different* location if needed.
    For now, it attempts to restore to where `backup_real_data` might have sourced from,
    which is now more abstract (needs `original_data_source_dir` if used).
    Consider if this function is still needed or how it should behave.
    """
    print(f"⚠️  警告: `restore_real_data` 的行为已改变。它现在会从 {BACKUP_DIR_ROOT} 恢复。")
    print("   请确保您了解此操作的目标恢复位置。")

    if not os.path.exists(BACKUP_DIR_ROOT):
        print(f"❌ 没有找到备份根目录: {BACKUP_DIR_ROOT}")
        return

    backup_sub_dirs = [d for d in os.listdir(BACKUP_DIR_ROOT) if os.path.isdir(os.path.join(BACKUP_DIR_ROOT, d)) and d.startswith('backup_')]
    if not backup_sub_dirs:
        print(f"❌ {BACKUP_DIR_ROOT} 中没有找到备份子目录。")
        return
    
    print(f"📦 发现备份目录于 {BACKUP_DIR_ROOT}: {backup_sub_dirs}")
    
    # Simplistic: choose the latest by name. A more robust way might involve parsing dates.
    latest_backup_subdir_name = max(backup_sub_dirs)
    latest_backup_full_path = os.path.join(BACKUP_DIR_ROOT, latest_backup_subdir_name)
    print(f"使用最新备份: {latest_backup_full_path}")
    
    # Define a *TARGET* directory for restoration.
    # This is CRITICAL. Where should these files go?
    # For safety, let's default to a NEW directory unless specified.
    # This part needs careful consideration based on actual use case.
    # Let's assume for now we're restoring to a hypothetical 'restored_data' dir in TESTS_DIR
    restore_target_dir = os.path.join(TESTS_DIR, f"restored_from_{latest_backup_subdir_name}")
    _ensure_dir(restore_target_dir)
    print(f" 文件将恢复到: {restore_target_dir}")
    
    restored_count = 0
    for root, _, files in os.walk(latest_backup_full_path):
        for file in files:
            backup_file_path = os.path.join(root, file)
            # Path relative to the specific backup sub-directory (e.g., 'email.csv')
            relative_path_in_backup = os.path.relpath(backup_file_path, latest_backup_full_path)
            target_restore_path = os.path.join(restore_target_dir, relative_path_in_backup)
            
            _ensure_dir(os.path.dirname(target_restore_path))
            shutil.copy2(backup_file_path, target_restore_path)
            print(f"   恢复: {relative_path_in_backup} -> {target_restore_path}")
            restored_count += 1
            
    if restored_count > 0:
        print(f"✅ {restored_count} 个文件已恢复到 {restore_target_dir}")
    else:
        print(f"⚠️  备份目录 {latest_backup_full_path} 为空或无法恢复文件。")


def main():
    """主测试函数"""
    print("🎯 CERT数据集流水线测试 (新结构)")
    print("=" * 50)
    
    _ensure_dir(TEST_OUTPUT_DIR) # Ensure output directory for pipeline runs
    _ensure_dir(SAMPLE_DATA_DIR) # Ensure sample data dir
    _ensure_dir(BACKUP_DIR_ROOT) # Ensure backup root dir

    check_data_safety()
    
    print(f"\\n📋 可用选项:")
    print("1. 创建测试数据 (在 sample_test_data/)")
    print("2. 运行完整测试 (使用 sample_test_data/ 和 test_output/)")
    print("3. 检查数据安全性 (主要针对 sample_test_data/)")
    print("4. 清理测试文件和目录 (sample_test_data/, test_output/, backups)")
    print("5. 从备份恢复数据 (到 tests/restored_from_backup_xxx)")
    print("6. 退出")
    
    while True:
        choice = input("\\n请选择操作 (1-6): ").strip()
        
        if choice == '1':
            if create_test_data(): # Now creates in SAMPLE_DATA_DIR
                print("✅ 测试数据创建成功")
            else:
                print("❌ 测试数据创建失败")
            # break # Allow further actions after creating data
            
        elif choice == '2':
            run_full_test() # Uses SAMPLE_DATA_DIR and TEST_OUTPUT_DIR
            # break
            
        elif choice == '3':
            check_data_safety() # Checks SAMPLE_DATA_DIR
            
        elif choice == '4':
            response = input(f"确认清理 {SAMPLE_DATA_DIR}, {TEST_OUTPUT_DIR} 和 {BACKUP_DIR_ROOT} 中的相关内容? (y/n): ")
            if response.lower() == 'y':
                cleanup_test_files()
            # break
            
        elif choice == '5':
            restore_real_data() # Restores from BACKUP_DIR_ROOT to a new dir
            
        elif choice == '6':
            print("👋 退出")
            break
            
        else:
            print("❌ 无效选择，请输入 1-6")

def run_full_test():
    """运行完整测试流程"""
    print("\\n🚀 运行完整测试流程...")
    
    test_results = {}
    
    try:
        if not get_test_data_files(): # Checks SAMPLE_DATA_DIR
            print("⚠️  未找到测试数据，先创建测试数据...")
            if not create_test_data():
                print("❌ 测试数据创建失败，无法继续测试。")
                return
        
        _ensure_dir(TEST_OUTPUT_DIR) # Ensure output dir for this run

        test_results['basic_functionality'] = test_basic_functionality()
        if not test_results['basic_functionality']: return # Stop if basic init fails
        
        test_results['step_by_step'] = test_step_by_step()
        if not test_results['step_by_step']: print("⚠️  分步测试中出现问题，但将继续尝试完整流水线测试。")
        
        test_results['full_pipeline'] = test_full_pipeline()
        
        analyze_results() # Analyzes from TEST_OUTPUT_DIR
        
    except Exception as e:
        print(f"❌ 测试过程出现严重错误: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\\n📋 测试摘要:")
    print("=" * 30)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        if not result: all_passed = False
        print(f"{test_name}: {status}")
    
    if test_results:
        # success_rate = sum(1 for res in test_results.values() if res) / len(test_results) * 100
        # print(f"\\n🎯 总体成功率: {success_rate:.1f}%")
        if all_passed:
            print("\\n🎯 所有主要测试步骤通过！")
        else:
            print("\\n🎯 部分测试步骤失败。")

    
    print(f"\\n🧹 测试完成。")
    print("   输出文件位于: " + os.path.abspath(TEST_OUTPUT_DIR))
    print("   测试数据位于: " + os.path.abspath(SAMPLE_DATA_DIR))
    print("   备份文件位于: " + os.path.abspath(BACKUP_DIR_ROOT))
    print("\\n   请根据需要选择清理选项（主菜单中的选项4）。")


if __name__ == "__main__":
    main() 