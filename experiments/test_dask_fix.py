#!/usr/bin/env python3
"""
测试Dask read_csv修复
"""
import sys
import os
sys.path.insert(0, '/Users/weitao_li/CodeField/DCAI/Huawei/Anomaly_Detection/InProject')

import dask.dataframe as dd
import pandas as pd
import tempfile

def test_dask_read_csv():
    """测试Dask read_csv参数兼容性"""
    print('🧪 测试Dask read_csv参数兼容性...')

    # 创建测试CSV文件
    test_data = pd.DataFrame({
        'user': ['user1', 'user2', 'user3'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'content': ['test1', 'test2', 'test3']
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        test_file = f.name

    try:
        # 测试基本参数
        ddf = dd.read_csv(test_file, blocksize='1MB', low_memory=False)
        print('✅ 基本Dask read_csv参数测试通过')
        print(f'   分区数: {ddf.npartitions}')
        print(f'   数据行数: {len(ddf.compute())}')
        
        # 测试采样
        sampled_ddf = ddf.sample(frac=0.5, random_state=42)
        print('✅ Dask采样测试通过')
        print(f'   采样后行数: {len(sampled_ddf.compute())}')
        
        return True
        
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        return False
        
    finally:
        os.unlink(test_file)

def test_environment_detection():
    """测试环境检测"""
    print('\n🔍 测试环境检测...')
    
    # 模拟Colab环境检测
    is_colab = 'google.colab' in str(globals().get('get_ipython', lambda: ''))
    print(f'   Colab环境检测: {is_colab}')
    
    # 获取系统信息
    import psutil
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    print(f'   系统内存: {memory.total/1024**3:.1f}GB')
    print(f'   可用内存: {memory.available/1024**3:.1f}GB')
    print(f'   CPU核心数: {cpu_count}')
    
    return True

if __name__ == "__main__":
    print("🚀 开始Dask修复测试...")
    
    success = True
    success &= test_dask_read_csv()
    success &= test_environment_detection()
    
    if success:
        print('\n🎉 所有测试通过！修复成功。')
    else:
        print('\n❌ 部分测试失败，需要进一步调试。') 