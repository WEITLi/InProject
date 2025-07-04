#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的CERT数据集特征提取流水线
基于feature_extraction_scenario模块化系统实现周级别、日级别和会话级别的分析
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
from joblib import Parallel, delayed
import dask.dataframe as dd
import shutil # Added for rmtree
from dask.distributed import Client, LocalCluster
import dask
warnings.filterwarnings('ignore')

# 优化Dask全局配置
dask.config.set({
    # 内存管理优化
    'distributed.worker.memory.target': 0.95,  # 目标内存使用率95%
    'distributed.worker.memory.spill': 0.85,   # 85%时开始溢出到磁盘
    'distributed.worker.memory.pause': 0.90,   # 90%时暂停接收新任务
    'distributed.worker.memory.terminate': 0.98,  # 98%时终止worker
    
    # I/O优化
    'dataframe.query-planning': True,  # 启用查询计划优化
    'dataframe.shuffle.method': 'tasks',  # 使用任务级shuffle
    'array.chunk-size': '256MB',  # 增大chunk大小
    
    # 并行优化
    'distributed.worker.daemon': False,
    'distributed.worker.multiprocessing-method': 'spawn',  # 使用spawn方法
    
    # 网络优化
    'distributed.comm.compression': 'lz4',  # 使用LZ4压缩
    'distributed.comm.timeouts.connect': '60s',
    'distributed.comm.timeouts.tcp': '60s',
    
    # 调度器优化
    'distributed.scheduler.allowed-failures': 10,  # 允许更多失败重试
    'distributed.scheduler.work-stealing': True,   # 启用工作窃取
    'distributed.scheduler.work-stealing-interval': '100ms',
})

# 导入新的模块化系统
try:
    # 尝试相对导入
    from .encoder import EventEncoder
    from .utils import FeatureEncoder
    from .temporal import encode_session_temporal_features
    from .user_context import encode_behavioral_risk_profile
except ImportError:
    # 如果相对导入失败，使用绝对导入
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from encoder import EventEncoder
    from utils import FeatureEncoder
    from temporal import encode_session_temporal_features
    from user_context import encode_behavioral_risk_profile

class CERTDatasetPipeline:
    """
    CERT数据集完整特征提取流水线
    
    实现功能：
    1. 原始日志数据的按周合并
    2. 用户信息和恶意用户标记的提取
    3. 活动数据的数值化特征提取
    4. 多粒度特征的统计计算和CSV导出
    """
    
    @staticmethod
    def _monitor_memory_usage(stage_name: str = ""):
        """监控内存使用情况"""
        import psutil
        import gc
        
        # 强制垃圾回收
        gc.collect()
        
        # 获取内存信息
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        print(f"📊 内存监控 {stage_name}:")
        print(f"   系统内存: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent:.1f}%)")
        print(f"   可用内存: {memory.available/1024**3:.1f}GB")
        print(f"   进程内存: {process_memory.rss/1024**3:.1f}GB (RSS), {process_memory.vms/1024**3:.1f}GB (VMS)")
        
        return {
            'system_used_gb': memory.used/1024**3,
            'system_available_gb': memory.available/1024**3,
            'process_rss_gb': process_memory.rss/1024**3,
            'system_percent': memory.percent
        }
    
    def __init__(self, data_version: str = 'r4.2', feature_dim: int = 256, num_cores: int = 8,
                 source_dir_override: Optional[str] = None, 
                 work_dir_override: Optional[str] = None,
                 seed: int = 42): # 添加 seed 参数
        """
        初始化流水线
        
        Args:
            data_version: 数据集版本
            feature_dim: 特征向量维度
            num_cores: CPU核心数
            source_dir_override: (可选) 覆盖源数据目录路径，用于测试
            work_dir_override: (可选) 覆盖工作目录路径，用于测试
            seed: 随机种子
        """
        self.data_version = data_version
        self.feature_dim = feature_dim
        self.num_cores = num_cores
        self.seed = seed # 存储种子
        
        # 路径配置
        if source_dir_override:
            self.source_data_dir = os.path.abspath(source_dir_override) # 确保绝对路径
            print(f"⚠️  使用覆盖的源数据目录: {self.source_data_dir}")
        else:
            # 动态计算正确的数据目录路径
            current_script_dir = os.path.dirname(os.path.abspath(__file__)) # core_logic
            core_logic_parent_dir = os.path.dirname(current_script_dir) # experiments
            project_root_dir = os.path.dirname(core_logic_parent_dir) # InProject
            self.source_data_dir = os.path.abspath(os.path.join(project_root_dir, 'data', data_version)) # 确保绝对路径
        
        if work_dir_override:
            self.work_dir = os.path.abspath(work_dir_override) # 确保绝对路径
            print(f"⚠️  使用覆盖的工作目录: {os.path.abspath(self.work_dir)}")
        else:
            # 工作目录设置为 InProject 目录
            current_script_dir = os.path.dirname(os.path.abspath(__file__)) # core_logic
            core_logic_parent_dir = os.path.dirname(current_script_dir) # experiments
            project_root_dir = os.path.dirname(core_logic_parent_dir) # InProject
            self.work_dir = os.path.abspath(project_root_dir) # 确保绝对路径
        
        # 确定数据集总周数
        self.max_weeks = 73 if data_version in ['r4.1', 'r4.2'] else 75
        
        # 初始化编码器
        self.encoder = EventEncoder(feature_dim=feature_dim, data_version=data_version)
        
        # 当前的采样率和对应的 Parquet 目录名
        self.current_sample_ratio = None # 会在 run_full_pipeline 中设置
        self.current_parquet_dir_name = "DataByWeek_parquet" # 默认值

        # 创建必要目录
        self._create_directories()
        
        print(f"初始化CERT数据集流水线")
        print(f"  数据版本: {data_version}")
        print(f"  源数据目录: {os.path.abspath(self.source_data_dir)}")
        print(f"  工作目录: {os.path.abspath(self.work_dir)}")
        print(f"  最大周数: {self.max_weeks}")
    
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            "tmp", "ExtractedData", "DataByWeek", "NumDataByWeek",
            "WeekLevelFeatures", "DayLevelFeatures", "SessionLevelFeatures"
        ]
        for directory in directories:
            work_path = os.path.join(self.work_dir, directory)
            os.makedirs(work_path, exist_ok=True)
    
    def _get_source_file_path(self, filename: str) -> str:
        """获取源文件的完整路径"""
        return os.path.join(self.source_data_dir, filename)
    
    def _get_work_file_path(self, filename: str) -> str:
        """获取工作文件的完整路径"""
        return os.path.join(self.work_dir, filename)
    
    def _read_csv_file(self, event_type: str, filename: str, file_path: str, file_type: str, sample_ratio: float = None):
        """
        优化的CSV文件读取方法
        
        Args:
            event_type: 事件类型
            filename: 文件名
            file_path: 文件路径
            file_type: 文件类型描述
            sample_ratio: 数据采样比例 (0-1)
            
        Returns:
            tuple: (event_type, dataframe) 或 None
        """
        import time
        start_time = time.time()
        
        print(f"   读取 {filename} ({file_type})...")
        
        # 检查是否为测试数据文件
        if file_type == "测试数据":
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    if 'TEST_DATA_CREATED_BY_PIPELINE' in first_line:
                        print(f"     ⚠️  使用测试数据: {filename}")
            except:
                pass
        
        try:
            # 获取文件大小
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"     文件大小: {file_size:.1f} MB")
            
            # 优化的CSV读取参数
            csv_params = {
                'low_memory': False,  # 避免数据类型推断问题
                'engine': 'c',       # 使用C引擎，更快
            }
            
            # 根据文件大小和采样比例选择读取策略
            if sample_ratio and sample_ratio < 1.0:
                print(f"     采样模式 (分块): 读取 {sample_ratio*100:.1f}% 的数据")
                chunk_size = 50000  # 每次读取5万行，可以根据实际情况调整
                sampled_chunks = []
                
                try:
                    # 获取列名，用于空DataFrame的创建
                    header_df = pd.read_csv(file_path, nrows=0, **csv_params)
                    column_names = header_df.columns
                    
                    for chunk in pd.read_csv(file_path, chunksize=chunk_size, **csv_params):
                        # 对每个块进行采样
                        # frac=sample_ratio 更符合比例采样意图
                        # random_state 保证可复现性
                        sampled_chunk = chunk.sample(frac=sample_ratio, random_state=42, replace=False)
                        if not sampled_chunk.empty:
                            sampled_chunks.append(sampled_chunk)
                    
                    if sampled_chunks:
                        df = pd.concat(sampled_chunks, ignore_index=True)
                    else:
                        # 如果没有采样到任何数据，创建一个具有正确列名的空DataFrame
                        df = pd.DataFrame(columns=column_names)
                    print(f"     分块采样读取完成，总行数: {len(df)}")
                
                except pd.errors.EmptyDataError:
                    print(f"     ⚠️ 文件 {filename} 为空或在分块采样过程中未读取到数据。")
                    # 尝试获取列名，如果文件就是空的，这也可能失败
                    try:
                        header_df = pd.read_csv(file_path, nrows=0, **csv_params)
                        column_names = header_df.columns
                    except pd.errors.EmptyDataError:
                        column_names = [] # 无法获取列名
                    df = pd.DataFrame(columns=column_names)
                except Exception as e:
                    print(f"     ❌ 分块采样读取 {filename} 失败: {e}")
                    return None # 出现其他错误则返回None

            elif file_size > 500:  # 大于500MB的文件
                print(f"     大文件检测，使用分块读取...")
                chunk_size = 50000  # 每次读取5万行
                chunks = []
                
                for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, **csv_params)):
                    chunks.append(chunk)
                    if (i + 1) % 10 == 0:  # 每10个chunk显示一次进度
                        print(f"     已读取 {(i + 1) * chunk_size} 行...")
                
                df = pd.concat(chunks, ignore_index=True)
                print(f"     分块读取完成，总行数: {len(df)}")
            else:
                # 小文件直接读取
                df = pd.read_csv(file_path, **csv_params)
            
            # 添加事件类型标识
            df['type'] = event_type
            
            # 优化日期转换
            if 'date' in df.columns:
                print(f"     转换日期列...")
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            elapsed_time = time.time() - start_time
            print(f"     ✅ {filename} 读取完成: {len(df)} 行, 耗时 {elapsed_time:.1f} 秒")
            
            return (event_type, df)
            
        except Exception as e:
            print(f"     ❌ 读取 {filename} 失败: {e}")
            return None
    
    def step1_combine_raw_data(self, start_week: int = 0, end_week: int = None, sample_ratio: float = None, force_regenerate: bool = False):
        """
        Step 1: 按周合并原始数据
        
        Args:
            start_week: 开始周数
            end_week: 结束周数
            sample_ratio: 数据采样比例 (0-1)，用于快速测试
            force_regenerate: 是否强制重新生成周数据 Parquet 文件集
        """
        # 检测运行环境并优化Dask配置
        is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
        
        try:
            client = Client(timeout="2s", processes=False) 
            print(f"🎛️  Connected to existing Dask client: {client}")
        except (OSError, TimeoutError):
            print("🚀 启动优化的Dask LocalCluster...")
            
            # 获取系统内存信息
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            cpu_count = psutil.cpu_count()
            
            # 根据环境调整配置
            if is_colab:
                print("   🔬 检测到Google Colab环境，使用保守配置")
                memory_fraction = 0.6  # Colab更保守
                n_workers = min(2, self.num_cores)  # Colab最多2个worker
                threads_per_worker = 1
                use_processes = False  # Colab使用线程
                memory_target = 0.85
                memory_spill = 0.75
                memory_pause = 0.80
            else:
                print("   🖥️  检测到本地环境，使用积极配置")
                memory_fraction = 0.8
                n_workers = self.num_cores
                threads_per_worker = 2
                use_processes = True
                memory_target = 0.95
                memory_spill = 0.85
                memory_pause = 0.90
            
            # 计算每个worker的内存限制
            memory_per_worker_gb = (available_memory_gb * memory_fraction) / n_workers
            memory_per_worker = f"{memory_per_worker_gb:.1f}GB"
            
            print(f"   💾 系统总内存: {total_memory_gb:.1f}GB, 可用内存: {available_memory_gb:.1f}GB")
            print(f"   🔧 CPU核心数: {cpu_count}, 使用workers: {n_workers}")
            print(f"   ⚙️  配置: {n_workers} workers, 每个worker {memory_per_worker}")
            
            # 创建LocalCluster
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_per_worker,
                processes=use_processes,
                dashboard_address=':8787' if not is_colab else None,  # Colab可能不支持dashboard
                silence_logs=False,
                # 环境特定的worker配置
                worker_kwargs={
                    'memory_target_fraction': memory_target,
                    'memory_spill_fraction': memory_spill,
                    'memory_pause_fraction': memory_pause,
                }
            )
            client = Client(cluster)
            print(f"🎛️  New optimized Dask LocalCluster started: {cluster}")
        
        # This part should be outside the try-except for client/cluster initialization 
        # if 'client' is not guaranteed to be defined in all paths of the try-except.
        # However, the original logic had it after the try-except, implying client would be defined.
        # For robustness, it might be better to check if client was successfully initialized.
        if 'client' in locals() and client and hasattr(client, 'dashboard_link') and client.dashboard_link:
            print(f"🔗 Dask Dashboard: {client.dashboard_link}")
        elif 'client' in locals() and client: # Client exists but no dashboard link
            print(f"⚠️  Dask Dashboard link not available, but client ({client}) exists.")
        else: # Client likely not initialized
            print("⚠️  Dask client not initialized or dashboard link not available.")
        # Original Dask initialization logic ends here
        
        if end_week is None:
            end_week = self.max_weeks
            
        print(f"\n{'='*60}")
        print(f"Step 1: 按周合并原始数据 (Dask模式, 输出 Parquet) (周 {start_week} 到 {end_week-1})")
        print(f"{'='*60}")
        
        # 监控初始内存状态
        self._monitor_memory_usage("开始Step1")

        # 使用 self.current_parquet_dir_name 作为目标目录
        parquet_output_dir = self._get_work_file_path(self.current_parquet_dir_name)
        os.makedirs(parquet_output_dir, exist_ok=True) # 确保目录存在
        print(f"   目标Parquet目录: {parquet_output_dir}")

        # 重新生成逻辑调整：
        # 1. 如果是采样数据 (self.current_sample_ratio < 1.0)，并且目录已存在，则跳过 (不看 force_regenerate)。
        #    除非 force_regenerate 也为True，此时采样数据也会被强制重新生成。
        # 2. 如果是全量数据 (self.current_sample_ratio is None or == 1.0)，则遵循 force_regenerate 标志。
        should_skip_generation = False
        is_sampled_data = self.current_sample_ratio is not None and 0 < self.current_sample_ratio < 1.0

        is_parquet_dir_valid = False
        if os.path.exists(parquet_output_dir) and os.path.isdir(parquet_output_dir):
            if (os.path.exists(os.path.join(parquet_output_dir, "_metadata")) or
                os.path.exists(os.path.join(parquet_output_dir, "_common_metadata")) or
                any(fname.startswith("week=") for fname in os.listdir(parquet_output_dir))):
                is_parquet_dir_valid = True

        if is_parquet_dir_valid: # 目录存在且是有效的Parquet目录
            if is_sampled_data and not force_regenerate: # 情况1: 采样数据已存在，且未强制重新生成
                should_skip_generation = True
                print(f"   ✅ 采样数据Parquet目录 {parquet_output_dir} 已存在且有效，跳过生成。")
            elif not is_sampled_data and not force_regenerate: # 情况2: 全量数据已存在，且未强制重新生成
                should_skip_generation = True
                print(f"   ✅ 全量数据Parquet目录 {parquet_output_dir} 已存在且有效，跳过生成。")
            elif force_regenerate: # 任何情况下，只要强制重新生成，就需要删除现有目录
                print(f"   ⚠️  配置了强制重新合并数据 (force_regenerate=True)，将删除现有Parquet目录: {parquet_output_dir}...")
                try:
                    shutil.rmtree(parquet_output_dir)
                    print(f"     已删除: {parquet_output_dir}")
                    is_parquet_dir_valid = False # 删除后不再有效
                except OSError as e:
                    print(f"     ❌ 删除 {parquet_output_dir} 失败: {e}")
                    # 如果删除失败，可能不应该继续，或者根据情况决定是否跳过
                    # 为安全起见，如果删除失败且目录仍然是有效的，则跳过以避免写入不完整数据
                    if is_parquet_dir_valid:
                        print(f"     由于删除失败且目录仍有效，将跳过生成以避免潜在问题。")
                        should_skip_generation = True 
            # 如果是采样数据，但 force_regenerate 为True，则不会跳过，会继续执行删除和重新生成
            # 如果是全量数据，force_regenerate 为True，也会执行删除和重新生成
        
        if should_skip_generation:
            return # 直接返回，不执行后续的数据读写
        
        # 如果 is_parquet_dir_valid 为 False（目录不存在或已被删除），则需要创建
        if not is_parquet_dir_valid:
            os.makedirs(parquet_output_dir, exist_ok=True) # 再次确保目录存在，以防被删除
            print(f"   创建新的Parquet目录: {parquet_output_dir}")

        raw_files = {
            'http': 'http.csv',
            'email': 'email.csv', 
            'file': 'file.csv',
            'logon': 'logon.csv',
            'device': 'device.csv'
        }
        
        missing_files = []
        for filename in raw_files.values():
            source_path = self._get_source_file_path(filename)
            if not os.path.exists(source_path):
                missing_files.append(filename)
        
        if missing_files:
            print(f"⚠️  缺失源数据文件: {missing_files}")
            print(f"   查找位置: {os.path.abspath(self.source_data_dir)}")
            local_missing = []
            for filename in missing_files:
                local_path = self._get_work_file_path(filename)
                if not os.path.exists(local_path):
                    local_missing.append(filename)
            
            if local_missing:
                raise FileNotFoundError(f"在源目录和工作目录都未找到: {local_missing}")
            else:
                print(f"   在工作目录找到测试数据文件，将使用测试数据")

        print("📁 读取原始数据文件 (使用 Dask)...")
        self._monitor_memory_usage("开始读取文件")
        dask_dfs = []

        for event_type, filename in raw_files.items():
            source_path = self._get_source_file_path(filename)
            work_path = self._get_work_file_path(filename)
            actual_file_path = None
            file_kind = ""

            if os.path.exists(source_path):
                actual_file_path = source_path
                file_kind = "源数据"
            elif os.path.exists(work_path):
                actual_file_path = work_path
                file_kind = "测试数据"
            
            if actual_file_path:
                print(f"   计划读取 {filename} ({file_kind}) 使用Dask...")
                # Dask read_csv, blocksize可以调整以平衡内存和并行度
                # 对于非常大的文件，可以减小 blocksize，例如 "32MB"
                # low_memory=False 类似 pandas, engine='c' 通常不需要显式指定给 dask
                try:
                    # 获取文件大小来动态调整blocksize
                    file_size_mb = os.path.getsize(actual_file_path) / (1024 * 1024)
                    
                    # 根据环境和文件大小动态调整blocksize
                    if is_colab:
                        # Colab环境使用更小的blocksize以避免内存问题
                        if file_size_mb > 1000:  # 大于1GB的文件
                            blocksize = f"{max(16, int(file_size_mb / (n_workers * 8)))}MB"
                        elif file_size_mb > 100:  # 100MB-1GB的文件
                            blocksize = f"{max(8, int(file_size_mb / (n_workers * 4)))}MB"
                        else:  # 小文件
                            blocksize = "8MB"
                    else:
                        # 本地环境使用原来的逻辑
                        if file_size_mb > 1000:  # 大于1GB的文件
                            blocksize = f"{max(32, int(file_size_mb / (self.num_cores * 4)))}MB"
                        elif file_size_mb > 100:  # 100MB-1GB的文件
                            blocksize = f"{max(16, int(file_size_mb / (self.num_cores * 2)))}MB"
                        else:  # 小文件
                            blocksize = "16MB"
                    
                    print(f"     文件大小: {file_size_mb:.1f}MB, 使用blocksize: {blocksize}")
                    
                    ddf = dd.read_csv(
                        actual_file_path, 
                        blocksize=blocksize,
                        # 基本优化参数
                        low_memory=False
                        # 注意：Dask的read_csv参数与pandas不完全相同
                    )
                    
                    # 如果指定了 sample_ratio < 1.0, Dask也支持采样
                    if sample_ratio and sample_ratio < 1.0:
                        print(f"     Dask 采样模式: 读取 {sample_ratio*100:.1f}% 的数据")
                        ddf = ddf.sample(frac=sample_ratio, random_state=42)

                    ddf = ddf.assign(type=event_type)
                    if 'date' in ddf.columns:
                         # Dask的to_datetime可能需要meta信息或显式计算来确定类型
                         # 一个简单的方式是先计算这列，或者提供meta
                        ddf['date'] = dd.to_datetime(ddf['date'], errors='coerce')
                    dask_dfs.append(ddf)
                    print(f"     ✅ {filename} 已加入Dask处理队列。")
                except Exception as e:
                    print(f"     ❌ Dask 读取 {filename} 失败: {e}")
            else:
                print(f"   ⚠️  文件 {filename} 未找到，跳过。")

        if not dask_dfs:
            print("❌ 没有可处理的Dask DataFrames，中止合并步骤。")
            return

        print("🔗 使用 Dask 合并所有事件数据...")
        self._monitor_memory_usage("文件读取完成，开始合并")
        combined_ddf = dd.concat(dask_dfs, ignore_index=True, interleave_partitions=True)
        
        # 获取当前分区信息
        current_partitions = combined_ddf.npartitions
        print(f"   当前分区数: {current_partitions}")
        
        # 动态计算最优分区大小
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if is_colab:
            # Colab环境使用更小的分区
            target_partition_size_mb = min(128, max(64, int((available_memory_gb * 0.4 * 1024) / (n_workers * 2))))
            max_partitions = n_workers * 4
            min_partitions = n_workers
        else:
            # 本地环境使用原来的逻辑
            target_partition_size_mb = min(500, max(200, int((available_memory_gb * 0.6 * 1024) / (self.num_cores * 2))))
            max_partitions = self.num_cores * 8
            min_partitions = self.num_cores
            
        target_partition_size = f"{target_partition_size_mb}MB"
        
        print(f"⚙️ 重新分区以优化内存使用...")
        print(f"   目标分区大小: {target_partition_size} (基于可用内存 {available_memory_gb:.1f}GB)")
        
        # 如果分区数太多或太少，进行重新分区
        if current_partitions > max_partitions or current_partitions < min_partitions:
            combined_ddf = combined_ddf.repartition(partition_size=target_partition_size)
            print(f"   重新分区完成: {combined_ddf.npartitions} 个分区")
        else:
            print(f"   分区数合理，跳过重新分区")

        print("⚙️ 使用 Dask 计算日期范围和周数...")
        # Dask需要 .compute() 来获取标量结果
        min_date_series = combined_ddf['date'].min()
        if isinstance(min_date_series, dd.Series) or isinstance(min_date_series, dd.core.Scalar):
             base_date = min_date_series.compute()
        else: # 如果已经是计算好的值 (不太可能在此流程中，但作为保险)
             base_date = min_date_series
        
        combined_ddf['week'] = ((combined_ddf['date'] - base_date).dt.days // 7).astype(int)
        
        # 移除昂贵的全局排序操作，因为原始CSV文件已经按时间排序
        # print("⚙️ 使用 Dask 显式按 'date' 列排序数据 (这可能需要较长时间)...")
        # combined_ddf = combined_ddf.sort_values('date')

        # 移除了 set_index('date') 操作，因为它也是性能瓶颈
        # print("⚙️ 将 'date' 列设置为 Dask DataFrame 的索引 (Dask 将计算或调整分区)...") # Removing set_index for now
        # combined_ddf = combined_ddf.set_index('date') 

        total_events_computed = combined_ddf.shape[0].compute() 
        min_week_computed = combined_ddf['week'].min().compute()
        max_week_computed = combined_ddf['week'].max().compute()
        print(f"📊 Dask 处理后总事件数: {total_events_computed}, 周数范围: {min_week_computed} - {max_week_computed}")
        
        print(f"💾 将Dask DataFrame直接保存为按周分区的Parquet文件集到: {parquet_output_dir}")
        print("   ⚡ 跳过全局排序，利用原始CSV文件已按时间排序的特性")
        try:
            # Ensure 'week' column exists for partitioning
            if 'week' not in combined_ddf.columns:
                raise ValueError("The 'week' column is missing from combined_ddf and is required for partitioning.")

            # 优化的Parquet写入配置
            to_parquet_kwargs = {
                'partition_on': ['week'],
                'engine': 'pyarrow',
                'write_index': False,
                'compression': 'snappy',
                'write_metadata_file': True,
                'row_group_size': 100000,
                'data_page_size': 1024*1024,
                'use_dictionary': True,
                'write_batch_size': 10000
            }

            # Snappy 不支持 compression_level
            # if to_parquet_kwargs['compression'] != 'snappy':
            #     to_parquet_kwargs['compression_level'] = 1 
            #  上面的逻辑是错误的，应该是如果不是snappy才设置，或者更准确地说，只有某些编解码器支持它
            #  为了安全起见，如果用snappy，就不设置compression_level

            if to_parquet_kwargs.get('compression') != 'snappy':
                to_parquet_kwargs['compression_level'] = 1 # 仅为非snappy编解码器设置（如果它们支持）

            combined_ddf.to_parquet(
                parquet_output_dir,
                **to_parquet_kwargs
            )
            print(f"   ✅ Dask DataFrame成功保存到Parquet目录: {parquet_output_dir}")
            self._monitor_memory_usage("Parquet保存完成")
        except Exception as e:
            print(f"   ❌ 保存Dask DataFrame到Parquet失败: {e}")
            # Potentially re-raise or handle more gracefully if this is critical
            raise

        # The old loop for saving individual pickle files is now replaced by to_parquet
        
        print("✅ Step 1 (Dask模式, Parquet输出, 无全局排序) 完成")
        self._monitor_memory_usage("Step1完成")
    
    def step2_load_user_data(self):
        """
        Step 2: 加载用户信息和恶意用户标记
        
        Returns:
            users_df: 用户信息DataFrame
        """
        print(f"\n{'='*60}")
        print("Step 2: 加载用户信息和恶意用户标记")
        print(f"{'='*60}")
        
        # 模拟用户数据（实际应从LDAP/psychometric.csv等文件读取）
        users_data = self._load_or_create_user_data()
        
        # 加载恶意用户标记（从answers目录）
        malicious_users = self._load_malicious_user_labels()
        
        # 合并用户数据
        users_df = pd.DataFrame(users_data).set_index('user_id')
        
        # 标记恶意用户
        for user_id, mal_info in malicious_users.items():
            if user_id in users_df.index:
                users_df.loc[user_id, 'malscene'] = mal_info['scenario']
                users_df.loc[user_id, 'mstart'] = mal_info['start_week']
                users_df.loc[user_id, 'mend'] = mal_info['end_week']
        
        print(f"📊 总用户数: {len(users_df)}")
        print(f"📊 恶意用户数: {len(users_df[users_df['malscene'] > 0])}")
        
        return users_df
    
    def _load_or_create_user_data(self):
        """加载或创建用户数据"""
        # 尝试从源目录LDAP子目录读取
        source_ldap_dir = os.path.join(self.source_data_dir, 'LDAP')
        ldap_files = []
        if os.path.exists(source_ldap_dir):
            ldap_files = [f for f in os.listdir(source_ldap_dir) if f.endswith('.csv')]
            print(f"📁 发现源LDAP文件: {len(ldap_files)} 个文件")
            # 这里应该实现实际的LDAP文件读取逻辑
            # 现在使用模拟数据，但保留将来扩展的可能性
        
        # 尝试从源目录读取心理测量数据
        source_psychometric = self._get_source_file_path('psychometric.csv')
        work_psychometric = self._get_work_file_path('psychometric.csv')
        
        ocean_scores = {}
        psychometric_file = None
        
        if os.path.exists(source_psychometric):
            psychometric_file = source_psychometric
            print(f"📁 读取源心理测量数据: psychometric.csv")
        elif os.path.exists(work_psychometric):
            psychometric_file = work_psychometric
            print(f"📁 读取工作目录心理测量数据: psychometric.csv (测试数据)")
        
        if psychometric_file:
            try:
                psycho_df = pd.read_csv(psychometric_file)
                print(f"   心理测量文件列名: {list(psycho_df.columns)}")
                
                # 确定用户ID列名
                user_id_col = None
                for col in ['user_id', 'user', 'employee_name']:
                    if col in psycho_df.columns:
                        user_id_col = col
                        break
                
                if user_id_col is None:
                    print("⚠️  心理测量文件中未找到用户ID列")
                else:
                    ocean_cols = ['O', 'C', 'E', 'A', 'N']
                    available_cols = [col for col in ocean_cols if col in psycho_df.columns]
                    if available_cols:
                        ocean_scores = psycho_df.set_index(user_id_col)[available_cols].to_dict('index')
                        print(f"   加载 {len(ocean_scores)} 个用户的OCEAN特征 (使用列: {user_id_col})")
                    else:
                        print("⚠️  心理测量文件中未找到OCEAN特征列")
            except Exception as e:
                print(f"⚠️  读取心理测量数据失败: {e}")
        
        # 获取所有用户（从活动数据中提取）
        all_users = set()
        parquet_dir = self._get_work_file_path(self.current_parquet_dir_name)

        if not (os.path.exists(parquet_dir) and os.path.isdir(parquet_dir)):
            print(f"⚠️  Parquet数据目录 {parquet_dir} 不存在，无法提取用户列表。将尝试从旧pickle文件（如果存在）或生成空用户列表。")
            # Fallback or error, for now, let's try to see if old pickle files exist as a super fallback
            # This fallback should ideally be removed once Parquet is stable.
            for week in range(self.max_weeks):
                 week_file_pickle = self._get_work_file_path(f"DataByWeek/{week}.pickle")
                 if os.path.exists(week_file_pickle):
                     try:
                         week_data_pickle = pd.read_pickle(week_file_pickle)
                         if 'user' in week_data_pickle.columns:
                             all_users.update(week_data_pickle['user'].unique())
                     except Exception as e_pickle:
                         print(f"  周 {week}: 读取旧pickle文件 {week_file_pickle} 失败: {e_pickle}")
            if not all_users:
                 print("⚠️  无法从Parquet或旧pickle文件加载用户，将生成空用户列表或基于psychometric（如果存在）")


        else: # Parquet directory exists
            print(f"   从Parquet目录 {parquet_dir} 提取用户列表...")
            # Potentially list all "week=*" subdirectories to know which weeks have data
            # For simplicity, iterating up to self.max_weeks
            for week in range(self.max_weeks):
                try:
                    # Only read the 'user' column for efficiency
                    user_data_for_week_ddf = dd.read_parquet(
                        parquet_dir,
                        columns=['user'],
                        filters=[('week', '==', week)],
                        engine='pyarrow'
                    )
                    # Check if the ddf is empty before computing
                    if user_data_for_week_ddf.npartitions > 0:
                        # A more robust check for emptiness with Dask
                        is_empty = (user_data_for_week_ddf.map_partitions(len).compute().sum() == 0)
                        if not is_empty:
                            user_data_for_week_pd = user_data_for_week_ddf.compute()
                            if 'user' in user_data_for_week_pd.columns:
                                all_users.update(user_data_for_week_pd['user'].unique())
                        # else:
                            # print(f"  周 {week}: Parquet分区为空.")
                except FileNotFoundError:
                     # This might happen if a specific week partition doesn't exist (e.g. week=X dir is missing)
                     # print(f"  周 {week}: Parquet数据分区未找到, 跳过.")
                     pass 
                except Exception as e:
                    print(f"  周 {week}: 从Parquet提取用户时发生错误: {e}")
                    pass
        
        # 创建用户数据
        users_data = []
        roles = ['Employee', 'Manager', 'Director', 'Executive']
        departments = ['IT', 'Finance', 'HR', 'Marketing', 'Engineering', 'Operations']
        
        for user_id in sorted(all_users):
            if pd.isna(user_id):
                continue
                
            user_data = {
                'user_id': user_id,
                'role': np.random.choice(roles),
                'dept': np.random.choice(departments),
                'ITAdmin': np.random.choice([0, 1], p=[0.9, 0.1]),
                'pc_type': np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.05, 0.05]),
                'npc': np.random.randint(1, 4),
                'malscene': 0,  # 默认非恶意
                'mstart': 0,
                'mend': 0
            }
            
            # 添加OCEAN特征
            if user_id in ocean_scores:
                user_data.update(ocean_scores[user_id])
            else:
                # 随机生成OCEAN分数
                user_data.update({
                    'O': np.random.uniform(0.2, 0.8),
                    'C': np.random.uniform(0.2, 0.8),
                    'E': np.random.uniform(0.2, 0.8),
                    'A': np.random.uniform(0.2, 0.8),
                    'N': np.random.uniform(0.2, 0.8)
                })
            
            users_data.append(user_data)
        
        return users_data
    
    def _load_malicious_user_labels(self):
        """加载恶意用户标签"""
        ans_file = self._get_source_file_path("answers/insiders.csv")
        
        # 更详细的调试打印
        print(f"[DEBUG CERTDatasetPipeline] Raw ans_file path: {ans_file}")
        abs_ans_file = os.path.abspath(ans_file)
        print(f"[DEBUG CERTDatasetPipeline] Absolute ans_file path: {abs_ans_file}")
        print(f"[DEBUG CERTDatasetPipeline] Is ans_file a file? {os.path.isfile(abs_ans_file)}")
        parent_dir_of_ans_file = os.path.dirname(abs_ans_file)
        print(f"[DEBUG CERTDatasetPipeline] Parent directory of ans_file: {parent_dir_of_ans_file}")
        print(f"[DEBUG CERTDatasetPipeline] Is parent directory a dir? {os.path.isdir(parent_dir_of_ans_file)}")
        
        # 检查文件是否存在
        if not os.path.exists(ans_file):
            print(f"⚠️  恶意用户标签文件 {ans_file} 不存在，创建模拟数据.")
            return {}
        
        malicious_users = {}
        
        # 尝试从源目录answers子目录读取
        source_answers_dir = os.path.join(self.source_data_dir, 'answers')
        work_answers_dir = self._get_work_file_path('answers')
        
        insiders_file = None
        
        # 优先从源目录读取
        source_insiders = os.path.join(source_answers_dir, 'insiders.csv')
        work_insiders = os.path.join(work_answers_dir, 'insiders.csv')
        
        if os.path.exists(source_insiders):
            insiders_file = source_insiders
            print(f"📁 读取源内部威胁者列表: answers/insiders.csv")
        elif os.path.exists(work_insiders):
            insiders_file = work_insiders
            print(f"📁 读取工作目录内部威胁者列表: answers/insiders.csv (测试数据)")
        
        if insiders_file:
            try:
                insiders_df = pd.read_csv(insiders_file)
                print(f"   恶意用户文件列名: {list(insiders_df.columns)}")
                
                for _, row in insiders_df.iterrows():
                    # 处理时间列名的变化
                    start_week = 0
                    end_week = self.max_weeks
                    
                    # 尝试不同的时间列名
                    if 'start_week' in row:
                        start_week = row['start_week']
                    elif 'start' in row:
                        # 如果是日期格式，需要转换为周数
                        start_date = row['start']
                        if isinstance(start_date, str) and start_date != '':
                            try:
                                # 这里可以添加日期到周数的转换逻辑
                                start_week = 0  # 简化处理
                            except:
                                start_week = 0
                    
                    if 'end_week' in row:
                        end_week = row['end_week']
                    elif 'end' in row:
                        # 如果是日期格式，需要转换为周数
                        end_date = row['end']
                        if isinstance(end_date, str) and end_date != '':
                            try:
                                # 这里可以添加日期到周数的转换逻辑
                                end_week = self.max_weeks  # 简化处理
                            except:
                                end_week = self.max_weeks
                    
                    malicious_users[row['user']] = {
                        'scenario': row.get('scenario', 1),
                        'start_week': start_week,
                        'end_week': end_week
                    }
                print(f"   加载 {len(malicious_users)} 个恶意用户标签")
            except Exception as e:
                print(f"⚠️  读取恶意用户标签失败: {e}")
        
        # 如果没有找到真实标签，创建一些模拟的恶意用户
        if not malicious_users:
            print("⚠️  未找到恶意用户标签文件，创建模拟数据")
            sample_users = ['ACM2278', 'CMP2946', 'BTH8471']  # 示例用户
            for i, user in enumerate(sample_users):
                malicious_users[user] = {
                    'scenario': i + 1,
                    'start_week': np.random.randint(10, 30),
                    'end_week': np.random.randint(40, 60)
                }
        
        return malicious_users
    
    def step3_extract_features(self, users_df: pd.DataFrame, start_week: int = 0, 
                              end_week: int = None, max_users: int = None):
        """
        Step 3: 提取活动特征
        
        Args:
            users_df: 用户信息
            start_week: 开始周数  
            end_week: 结束周数
            max_users: 最大用户数限制
        """
        if end_week is None:
            end_week = self.max_weeks
            
        print(f"\n{'='*60}")
        print(f"Step 3: 提取活动特征 (周 {start_week} 到 {end_week-1})")
        print(f"{'='*60}")
        
        # 用户数量限制
        if max_users and len(users_df) > max_users:
            print(f"应用用户数量限制: {len(users_df)} -> {max_users}")
            
            malicious_users_df = users_df[users_df['malscene'] > 0]
            normal_users_df = users_df[users_df['malscene'] == 0]

            selected_malicious_indices = []
            selected_normal_indices = []

            if len(malicious_users_df) >= max_users:
                # 如果恶意用户就足够多或超过 max_users，则只从恶意用户中选
                np.random.seed(self.seed) # 使用 self.seed
                selected_malicious_indices = np.random.choice(malicious_users_df.index, size=max_users, replace=False).tolist()
            else:
                # 恶意用户不足 max_users，全部选中
                selected_malicious_indices = malicious_users_df.index.tolist()
                remaining_slots = max_users - len(selected_malicious_indices)
                if remaining_slots > 0 and not normal_users_df.empty:
                    np.random.seed(self.seed) # 使用 self.seed
                    num_to_select_normal = min(remaining_slots, len(normal_users_df))
                    selected_normal_indices = np.random.choice(normal_users_df.index,
                                                             size=num_to_select_normal,
                                                             replace=False).tolist()
            
            final_selected_user_indices = selected_malicious_indices + selected_normal_indices
            # 如果因为某种原因 final_selected_user_indices 为空但 max_users > 0，需要处理
            if not final_selected_user_indices and max_users > 0 and not users_df.empty:
                 print(f"⚠️ 优先筛选后用户列表为空，但 max_users={max_users} > 0。回退到从原始 users_df 随机采样 {max_users} 个用户。")
                 np.random.seed(self.seed) # 使用 self.seed
                 final_selected_user_indices = np.random.choice(users_df.index, size=min(max_users, len(users_df)), replace=False).tolist()
            
            users_df = users_df.loc[final_selected_user_indices]
            
            # 重新计算选出用户的恶意和正常数量
            num_malicious_selected = sum(users_df['malscene'] > 0) if not users_df.empty else 0
            num_normal_selected = sum(users_df['malscene'] == 0) if not users_df.empty else 0
            print(f"最终用户数: {len(users_df)} (恶意: {num_malicious_selected}, 正常: {num_normal_selected})")
        
        # 准备训练数据（收集所有周的数据样本）
        print("📚 准备编码器训练数据...")
        all_events = []
        parquet_dir = self._get_work_file_path(self.current_parquet_dir_name)
        
        # 检查 Parquet 目录是否存在
        if not os.path.exists(parquet_dir):
            print(f"⚠️  Parquet数据目录不存在: {parquet_dir}")
            print("⚠️  无法收集训练数据，编码器将使用默认配置")
        else:
            # 从 Parquet 文件收集训练数据
            for week in range(start_week, min(end_week, start_week + 5)):  # 只用前几周训练
                try:
                    week_data_ddf = dd.read_parquet(
                        parquet_dir,
                        filters=[('week', '==', week)],
                        engine='pyarrow'
                    )
                    
                    if week_data_ddf.npartitions > 0 and week_data_ddf.map_partitions(len).compute().sum() > 0:
                        week_data = week_data_ddf.compute()
                        if len(week_data) > 0:
                            # 只取样本数据用于训练
                            sample_size = min(1000, len(week_data))
                            sample_data = week_data.sample(n=sample_size, random_state=42)
                            all_events.append(sample_data)
                            print(f"   从周 {week} 收集 {len(sample_data)} 条训练样本")
                except Exception as e:
                    print(f"   ⚠️  从周 {week} 收集训练数据失败: {e}")
                    continue
        
        if all_events:
            training_data = pd.concat(all_events, ignore_index=True)
            print(f"📊 训练数据总计: {len(training_data)} 条事件")
            
            # 拟合编码器
            print("🔧 拟合编码器...")
            try:
                self.encoder.fit(training_data, users_df)
                print("✅ 编码器拟合完成")
            except Exception as e:
                print(f"❌ 编码器拟合失败: {e}")
                print("⚠️  将使用默认编码器配置")
        else:
            print("⚠️  没有找到训练数据，编码器将使用默认配置")
            # 即使没有训练数据，也要确保编码器有基本的配置
            try:
                # 创建一个最小的虚拟数据集来初始化编码器
                dummy_data = pd.DataFrame({
                    'user': ['dummy_user'],
                    'type': ['http'],
                    'date': [pd.Timestamp.now()],
                    'content': ['dummy_content']
                })
                dummy_users = pd.DataFrame({
                    'role': ['Employee'],
                    'dept': ['IT']
                }, index=['dummy_user'])
                self.encoder.fit(dummy_data, dummy_users)
                print("✅ 编码器使用虚拟数据初始化完成")
            except Exception as e:
                print(f"❌ 编码器初始化失败: {e}")
        
        # 处理每一周
        print(f"🔄 开始处理 {end_week - start_week} 周的数据...")
        
        # 使用 joblib.Parallel 并行处理每周数据
        Parallel(n_jobs=self.num_cores)(
            delayed(self._process_week_features)(week, users_df)
            for week in range(start_week, end_week)
        )
        
        print("✅ Step 3 完成")
    
    def _process_week_features(self, week: int, users_df: pd.DataFrame):
        """处理单周的特征提取"""
        # week_file = self._get_work_file_path(f"DataByWeek/{week}.pickle")
        
        # if not os.path.exists(week_file):
        #     print(f"⚠️  周 {week}: 数据文件不存在")
        #     return
        
        # # 读取周数据
        # week_data = pd.read_pickle(week_file)

        # 使用 self.current_parquet_dir_name 动态确定要读取的 Parquet 目录
        parquet_dir = self._get_work_file_path(self.current_parquet_dir_name)
        week_data = None # Initialize to None

        try:
            week_data_ddf = dd.read_parquet(
                parquet_dir,
                filters=[('week', '==', week)],
                engine='pyarrow' # or 'fastparquet'
            )
            if week_data_ddf.npartitions == 0 or week_data_ddf.map_partitions(len).compute().sum() == 0:
                 print(f"📭 周 {week}: 无活动数据 (Parquet分区为空或未找到).")
                 empty_features = pd.DataFrame()
                 # 保存空的 Parquet 文件以保持一致性
                 empty_features.to_parquet(self._get_work_file_path(f"NumDataByWeek/{week}_features.parquet"), engine='pyarrow', index=False)
                 return
            week_data = week_data_ddf.compute() # Convert Dask DataFrame to Pandas DataFrame
        except FileNotFoundError: # More specific exception for missing Parquet directory/files
            print(f"⚠️  周 {week}: Parquet数据目录或特定周分区未找到 ({parquet_dir}, week={week}).")
        except Exception as e:
            print(f"⚠️  周 {week}: 读取Parquet数据失败 ({e}).")
        
        if week_data is None or week_data.empty: # Check if week_data is None (due to error) or empty
            # Save empty features to avoid breaking downstream if one week is missing/fails
            print(f"⚠️  周 {week}: 最终无数据可处理，保存空特征文件。")
            empty_features = pd.DataFrame()
            empty_features.to_parquet(self._get_work_file_path(f"NumDataByWeek/{week}_features.parquet"), engine='pyarrow', index=False)
            return

        if len(week_data) == 0: # Should be caught by the above, but as a safeguard
            print(f"📭 周 {week}: 无活动数据")
            # Save empty features to avoid breaking downstream if one week is missing/fails
            empty_features = pd.DataFrame()
            empty_features.to_parquet(self._get_work_file_path(f"NumDataByWeek/{week}_features.parquet"), engine='pyarrow', index=False)
            return
        
        # 在周级别进行排序，模仿 feature_extraction.py 中 process_week_num() 的做法
        # 这比全局排序要高效得多，因为只对单周数据排序
        if 'date' in week_data.columns:
            print(f"   📅 对周 {week} 的 {len(week_data)} 条事件按日期排序...")
            week_data = week_data.sort_values('date').reset_index(drop=True)
        
        # 过滤用户
        valid_users = set(week_data['user'].unique()) & set(users_df.index)
        week_data = week_data[week_data['user'].isin(valid_users)]
        
        print(f"📊 周 {week}: {len(week_data)} 条事件, {len(valid_users)} 个用户")
        
        # 为每个用户提取特征
        user_features = []
        
        for user_id in valid_users:
            user_events = week_data[week_data['user'] == user_id]
            user_context = users_df.loc[user_id].to_dict()
            
            # 判断是否为恶意时期
            is_malicious_period = False
            if user_context['malscene'] > 0:
                if user_context['mstart'] <= week <= user_context['mend']:
                    is_malicious_period = True
            
            # 编码用户的所有事件
            event_features = []
            for _, event in user_events.iterrows():
                event_dict = event.to_dict()
                try:
                    features, mask = self.encoder.encode_event(event_dict, user_context)
                    event_features.append({
                        'user': user_id,
                        'week': week,
                        'event_type': event.get('type', 'unknown'),
                        'date': event.get('date'),
                        'features': features,
                        'mask': mask,
                        'is_malicious': is_malicious_period
                    })
                except Exception as e:
                    print(f"⚠️  编码事件失败 - 用户 {user_id}, 周 {week}: {e}")
                    continue
            
            if event_features:
                user_features.extend(event_features)
        
        # 保存周特征
        if user_features:
            features_df = pd.DataFrame(user_features)
            # 将 features 和 mask 列转换为适合 Parquet 的格式 (例如 list of floats)
            # Parquet 不直接支持存储复杂的 NumPy 数组对象，除非它们被转换为更简单的类型。
            # 如果 encoder.encode_event 返回的是 NumPy 数组，这里需要处理。
            # 假设 features 和 mask 是数值型或者可以被Parquet接受的列表
            # 如果 features 和 mask 是numpy arrays, to_parquet 可能需要特殊处理或转换为list
            try:
                # 尝试直接保存。如果features/mask是复杂对象，可能需要转换
                # features_df['features'] = features_df['features'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                # features_df['mask'] = features_df['mask'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                features_df.to_parquet(self._get_work_file_path(f"NumDataByWeek/{week}_features.parquet"), engine='pyarrow', index=False)
            except Exception as e_parquet:
                print(f"   ⚠️  周 {week}: 保存到 Parquet 失败 ({e_parquet})。尝试将 features/mask 转为 list 后保存...")
                try:
                    features_df_copy = features_df.copy()
                    if 'features' in features_df_copy.columns:
                        features_df_copy['features'] = features_df_copy['features'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                    if 'mask' in features_df_copy.columns:
                         features_df_copy['mask'] = features_df_copy['mask'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                    features_df_copy.to_parquet(self._get_work_file_path(f"NumDataByWeek/{week}_features.parquet"), engine='pyarrow', index=False)
                except Exception as e_list_save:
                    print(f"   ❌ 周 {week}: 转换为list后保存到 Parquet 仍然失败 ({e_list_save})。将回退到 Pickle。")
                    features_df.to_pickle(self._get_work_file_path(f"NumDataByWeek/{week}_features.pickle")) # Fallback
            print(f"💾 周 {week}: 保存 {len(user_features)} 个特征向量到 Parquet (或Pickle回退)")
        else:
            print(f"⚠️  周 {week}: 没有提取到有效特征")
    
    def step4_multi_level_analysis(self, start_week: int = 0, end_week: int = None, 
                                   modes: List[str] = ['week', 'day', 'session'],
                                   force_regenerate_analysis_levels: bool = False):
        """
        Step 4: 多级别分析
        
        Args:
            start_week: 开始周数
            end_week: 结束周数  
            modes: 分析模式列表
            force_regenerate_analysis_levels: 是否强制重新生成多级别分析的CSV文件
        """
        if end_week is None:
            end_week = self.max_weeks
            
        print(f"\n{'='*60}")
        print(f"Step 4: 多级别分析 - 模式: {modes}, 强制重新生成: {force_regenerate_analysis_levels}")
        print(f"{'='*60}")
        
        for mode in modes:
            if mode == 'week':
                self._week_level_analysis(start_week, end_week, force_regenerate_analysis_levels)
            elif mode == 'day':
                self._day_level_analysis(start_week, end_week, force_regenerate_analysis_levels)
            elif mode == 'session':
                self._session_level_analysis(start_week, end_week, force_regenerate_analysis_levels)
            else:
                print(f"⚠️  未知分析模式: {mode}")
        
        print("✅ Step 4 完成")
    
    def _week_level_analysis(self, start_week: int, end_week: int, force_regenerate_analysis_levels: bool):
        """周级别分析"""
        print("\n📅 执行周级别分析...")
        
        output_file = self._get_work_file_path(f"WeekLevelFeatures/weeks_{start_week}_{end_week-1}.csv")
        if os.path.exists(output_file) and not force_regenerate_analysis_levels:
            print(f"   ✅ 周级别分析结果 {output_file} 已存在且未强制重新生成，跳过。")
            return
            
        week_features = []
        
        for week in range(start_week, end_week):
            features_file_parquet = self._get_work_file_path(f"NumDataByWeek/{week}_features.parquet")
            features_file_pickle = self._get_work_file_path(f"NumDataByWeek/{week}_features.pickle") # Fallback
            
            week_data = None
            if os.path.exists(features_file_parquet):
                try:
                    week_data = pd.read_parquet(features_file_parquet, engine='pyarrow')
                except Exception as e_parquet_read:
                    print(f"   ⚠️ 周 {week}: 读取 Parquet 特征文件 {features_file_parquet} 失败 ({e_parquet_read}). 尝试 Pickle 回退...")
                    if os.path.exists(features_file_pickle):
                        try:
                            week_data = pd.read_pickle(features_file_pickle)
                            print(f"     ✅ 周 {week}: 成功从 Pickle 文件 {features_file_pickle} 回退读取。")
                        except Exception as e_pickle_read:
                            print(f"     ❌ 周 {week}: 读取 Pickle 回退文件 {features_file_pickle} 也失败 ({e_pickle_read}).")
                    else:
                        print(f"     周 {week}: Pickle 回退文件 {features_file_pickle} 不存在。")
            elif os.path.exists(features_file_pickle):
                print(f"   周 {week}: Parquet 特征文件不存在，尝试读取 Pickle 文件 {features_file_pickle}...")
                try:
                    week_data = pd.read_pickle(features_file_pickle)
                except Exception as e_pickle_read:
                    print(f"     ❌ 周 {week}: 读取 Pickle 文件 {features_file_pickle} 失败 ({e_pickle_read}).")
            
            if week_data is None or week_data.empty:
                continue
            
            # 按用户聚合周特征
            for user in week_data['user'].unique():
                user_week_data = week_data[week_data['user'] == user]
                
                # 计算用户该周的聚合特征
                user_features = self._aggregate_user_features(user_week_data, 'week')
                user_features.update({
                    'user': user,
                    'week': week,
                    'mode': 'week',
                    'n_events': len(user_week_data),
                    'malicious_ratio': user_week_data['is_malicious'].mean()
                })
                
                week_features.append(user_features)
        
        # 保存周级别特征
        if week_features:
            week_df = pd.DataFrame(week_features)
            week_df.to_csv(output_file, index=False)
            print(f"💾 周级别分析完成: {output_file} ({len(week_features)} 条记录)")
        else:
            print("⚠️  周级别分析: 没有数据")
    
    def _day_level_analysis(self, start_week: int, end_week: int, force_regenerate_analysis_levels: bool):
        """日级别分析"""
        print("\n📆 执行日级别分析...")
        
        output_file = self._get_work_file_path(f"DayLevelFeatures/days_{start_week}_{end_week-1}.csv")
        if os.path.exists(output_file) and not force_regenerate_analysis_levels:
            print(f"   ✅ 日级别分析结果 {output_file} 已存在且未强制重新生成，跳过。")
            return
            
        day_features = []
        
        for week in range(start_week, end_week):
            features_file_parquet = self._get_work_file_path(f"NumDataByWeek/{week}_features.parquet")
            features_file_pickle = self._get_work_file_path(f"NumDataByWeek/{week}_features.pickle") # Fallback

            week_data = None
            if os.path.exists(features_file_parquet):
                try:
                    week_data = pd.read_parquet(features_file_parquet, engine='pyarrow')
                except Exception as e_parquet_read:
                    print(f"   ⚠️ 周 {week}: 读取 Parquet 特征文件 {features_file_parquet} 失败 ({e_parquet_read}). 尝试 Pickle 回退...")
                    if os.path.exists(features_file_pickle):
                        try:
                            week_data = pd.read_pickle(features_file_pickle)
                            print(f"     ✅ 周 {week}: 成功从 Pickle 文件 {features_file_pickle} 回退读取。")
                        except Exception as e_pickle_read:
                            print(f"     ❌ 周 {week}: 读取 Pickle 回退文件 {features_file_pickle} 也失败 ({e_pickle_read}).")
                    else:
                        print(f"     周 {week}: Pickle 回退文件 {features_file_pickle} 不存在。")
            elif os.path.exists(features_file_pickle):
                print(f"   周 {week}: Parquet 特征文件不存在，尝试读取 Pickle 文件 {features_file_pickle}...")
                try:
                    week_data = pd.read_pickle(features_file_pickle)
                except Exception as e_pickle_read:
                    print(f"     ❌ 周 {week}: 读取 Pickle 文件 {features_file_pickle} 失败 ({e_pickle_read}).")

            if week_data is None or week_data.empty:
                continue
            
            # 添加日期列
            week_data['day'] = pd.to_datetime(week_data['date']).dt.date
            
            # 按用户和日期聚合
            for user in week_data['user'].unique():
                user_data = week_data[week_data['user'] == user]
                
                for day in user_data['day'].unique():
                    day_data = user_data[user_data['day'] == day]
                    
                    # 计算用户该日的聚合特征
                    day_feature = self._aggregate_user_features(day_data, 'day')
                    day_feature.update({
                        'user': user,
                        'week': week,
                        'day': day,
                        'mode': 'day',
                        'n_events': len(day_data),
                        'malicious_ratio': day_data['is_malicious'].mean()
                    })
                    
                    day_features.append(day_feature)
        
        # 保存日级别特征
        if day_features:
            day_df = pd.DataFrame(day_features)
            day_df.to_csv(output_file, index=False)
            print(f"💾 日级别分析完成: {output_file} ({len(day_features)} 条记录)")
        else:
            print("⚠️  日级别分析: 没有数据")
    
    def _session_level_analysis(self, start_week: int, end_week: int, force_regenerate_analysis_levels: bool):
        """会话级别分析"""
        print("\n🖥️  执行会话级别分析...")
        
        output_file = self._get_work_file_path(f"SessionLevelFeatures/sessions_{start_week}_{end_week-1}.csv")
        if os.path.exists(output_file) and not force_regenerate_analysis_levels:
            print(f"   ✅ 会话级别分析结果 {output_file} 已存在且未强制重新生成，跳过。")
            return
            
        session_features = []
        
        for week in range(start_week, end_week):
            features_file_parquet = self._get_work_file_path(f"NumDataByWeek/{week}_features.parquet")
            features_file_pickle = self._get_work_file_path(f"NumDataByWeek/{week}_features.pickle") # Fallback
            
            week_data = None
            if os.path.exists(features_file_parquet):
                try:
                    week_data = pd.read_parquet(features_file_parquet, engine='pyarrow')
                except Exception as e_parquet_read:
                    print(f"   ⚠️ 周 {week}: 读取 Parquet 特征文件 {features_file_parquet} 失败 ({e_parquet_read}). 尝试 Pickle 回退...")
                    if os.path.exists(features_file_pickle):
                        try:
                            week_data = pd.read_pickle(features_file_pickle)
                            print(f"     ✅ 周 {week}: 成功从 Pickle 文件 {features_file_pickle} 回退读取。")
                        except Exception as e_pickle_read:
                            print(f"     ❌ 周 {week}: 读取 Pickle 回退文件 {features_file_pickle} 也失败 ({e_pickle_read}).")
                    else:
                        print(f"     周 {week}: Pickle 回退文件 {features_file_pickle} 不存在。")
            elif os.path.exists(features_file_pickle):
                print(f"   周 {week}: Parquet 特征文件不存在，尝试读取 Pickle 文件 {features_file_pickle}...")
                try:
                    week_data = pd.read_pickle(features_file_pickle)
                except Exception as e_pickle_read:
                    print(f"     ❌ 周 {week}: 读取 Pickle 文件 {features_file_pickle} 失败 ({e_pickle_read}).")
            
            if week_data is None or week_data.empty:
                continue
            
            # 按用户分组提取会话
            for user in week_data['user'].unique():
                user_data = week_data[week_data['user'] == user]
                user_data = user_data.sort_values('date')
                
                # 识别会话（基于时间间隔）
                sessions = self._identify_sessions(user_data)
                
                for session_id, session_data in sessions.items():
                    # 计算会话特征
                    session_feature = self._aggregate_user_features(session_data, 'session')
                    
                    # 添加会话特定特征
                    if len(session_data) > 1:
                        start_time = pd.to_datetime(session_data['date'].min())
                        end_time = pd.to_datetime(session_data['date'].max())
                        duration_minutes = (end_time - start_time).total_seconds() / 60
                    else:
                        duration_minutes = 0
                    
                    session_feature.update({
                        'user': user,
                        'week': week,
                        'session_id': session_id,
                        'mode': 'session',
                        'n_events': len(session_data),
                        'duration_minutes': duration_minutes,
                        'malicious_ratio': session_data['is_malicious'].mean()
                    })
                    
                    session_features.append(session_feature)
        
        # 保存会话级别特征
        if session_features:
            session_df = pd.DataFrame(session_features)
            session_df.to_csv(output_file, index=False)
            print(f"💾 会话级别分析完成: {output_file} ({len(session_features)} 条记录)")
        else:
            print("⚠️  会话级别分析: 没有数据")
    
    def _identify_sessions(self, user_data: pd.DataFrame, gap_threshold: int = 60):
        """
        识别用户会话
        
        Args:
            user_data: 用户数据
            gap_threshold: 会话间隔阈值（分钟）
            
        Returns:
            sessions: 会话字典
        """
        sessions = {}
        current_session = 0
        last_time = None
        
        user_data = user_data.sort_values('date').reset_index(drop=True)
        
        for idx, row in user_data.iterrows():
            current_time = pd.to_datetime(row['date'])
            
            # 如果时间间隔超过阈值，开始新会话
            if last_time is not None:
                gap_minutes = (current_time - last_time).total_seconds() / 60
                if gap_minutes > gap_threshold:
                    current_session += 1
            
            if current_session not in sessions:
                sessions[current_session] = []
            
            sessions[current_session].append(row)
            last_time = current_time
        
        # 转换为DataFrame
        session_dfs = {}
        for session_id, session_events in sessions.items():
            session_dfs[session_id] = pd.DataFrame(session_events)
        
        return session_dfs
    
    def _aggregate_user_features(self, user_data: pd.DataFrame, mode: str) -> Dict:
        """
        聚合用户特征
        
        Args:
            user_data: 用户事件数据
            mode: 聚合模式
            
        Returns:
            aggregated_features: 聚合特征字典
        """
        if len(user_data) == 0:
            return {}
        
        # 提取所有特征向量
        feature_vectors = np.stack(user_data['features'].values)
        mask_vectors = np.stack(user_data['mask'].values)
        
        # 计算有效特征的统计量
        valid_features = feature_vectors * mask_vectors
        
        aggregated = {
            # 基础统计
            'mean_features': np.mean(valid_features, axis=0),
            'std_features': np.std(valid_features, axis=0),
            'max_features': np.max(valid_features, axis=0),
            'min_features': np.min(valid_features, axis=0),
            
            # 事件类型分布
            'email_ratio': (user_data['event_type'] == 'email').mean(),
            'file_ratio': (user_data['event_type'] == 'file').mean(),
            'http_ratio': (user_data['event_type'] == 'http').mean(),
            'device_ratio': (user_data['event_type'] == 'device').mean(),
            'logon_ratio': (user_data['event_type'] == 'logon').mean(),
            
            # 特征有效性
            'feature_completeness': mask_vectors.mean(),
            
            # 时间特征（如果有日期信息）
            'unique_event_types': user_data['event_type'].nunique()
        }
        
        # 平展多维特征为标量（用于CSV保存）
        for i, val in enumerate(aggregated['mean_features']):
            aggregated[f'mean_feature_{i}'] = val
        
        # 移除多维数组（避免CSV保存错误）
        del aggregated['mean_features']
        del aggregated['std_features']
        del aggregated['max_features']  
        del aggregated['min_features']
        
        return aggregated
    
    def run_full_pipeline(self, start_week: int = 0, end_week: int = None, 
                         max_users: int = None, modes: List[str] = ['week', 'day', 'session'],
                         sample_ratio: float = None, force_regenerate_combined_weeks: bool = False,
                         force_regenerate_analysis_levels: bool = False):
        """
        运行完整的特征提取流水线
        
        Args:
            start_week: 开始周数
            end_week: 结束周数
            max_users: 最大用户数限制
            modes: 分析模式
            sample_ratio: 数据采样比例 (0-1)，用于快速测试
            force_regenerate_combined_weeks: 是否强制重新生成周数据 Parquet 文件集
            force_regenerate_analysis_levels: 是否强制重新生成多级别分析的CSV文件
        """
        # Removing the try...finally block and calls to _initialize_dask and _shutdown_dask
        start_time = time.time()
        self.current_sample_ratio = sample_ratio 
        
        if sample_ratio is not None and 0 < sample_ratio < 1.0:
            self.current_parquet_dir_name = f"DataByWeek_parquet_r{str(sample_ratio).replace('.', '')}"
        else:
            self.current_parquet_dir_name = "DataByWeek_parquet"
        print(f"Target Parquet directory for this run: {self.current_parquet_dir_name}")

        print(f"🚀 启动CERT数据集特征提取流水线")
        print(f"📊 参数: 周 {start_week}-{end_week or self.max_weeks-1}, 用户限制: {max_users or '无'}, 模式: {modes}, 采样率: {sample_ratio or 1.0}")
        
        # Original core logic of the pipeline
        # Step 1: 合并原始数据
        self.step1_combine_raw_data(start_week, end_week, sample_ratio, force_regenerate=force_regenerate_combined_weeks)
        
        # Step 2: 加载用户数据  
        users_df = self.step2_load_user_data()
        
        # Step 3: 提取特征
        self.step3_extract_features(users_df, start_week, end_week, max_users)
        
        # Step 4: 多级别分析
        self.step4_multi_level_analysis(start_week, end_week, modes, force_regenerate_analysis_levels)
        
        total_time = (time.time() - start_time) / 60
        print(f"\n🎉 流水线执行完成! 总耗时: {total_time:.1f} 分钟")
        
        # 输出结果统计
        self._print_results_summary(start_week, end_week, modes)
            
    def _print_results_summary(self, start_week: int, end_week: int, modes: List[str]):
        """打印结果摘要"""
        print(f"\n📋 结果摘要:")
        print(f"{'='*40}")
        
        for mode in modes:
            if mode == 'week':
                pattern = self._get_work_file_path(f"WeekLevelFeatures/weeks_{start_week}_{end_week-1}.csv")
            elif mode == 'day':
                pattern = self._get_work_file_path(f"DayLevelFeatures/days_{start_week}_{end_week-1}.csv")
            elif mode == 'session':
                pattern = self._get_work_file_path(f"SessionLevelFeatures/sessions_{start_week}_{end_week-1}.csv")
            else:
                continue
            
            if os.path.exists(pattern):
                df = pd.read_csv(pattern)
                print(f"📊 {mode.capitalize()}级别: {len(df)} 条记录 -> {pattern}")
            else:
                print(f"⚠️  {mode.capitalize()}级别: 无输出文件")

def main():
    """主函数"""
    # 解析命令行参数
    import sys
    
    # 默认参数
    num_cores = 8
    start_week = 0  
    end_week = None
    max_users = None
    modes = ['week', 'day', 'session']
    data_version = 'r4.2'
    
    # 解析命令行
    if len(sys.argv) > 1:
        num_cores = int(sys.argv[1])
    if len(sys.argv) > 2:
        start_week = int(sys.argv[2])
    if len(sys.argv) > 3:
        end_week = int(sys.argv[3])
    if len(sys.argv) > 4:
        max_users = int(sys.argv[4])
    if len(sys.argv) > 5:
        modes = sys.argv[5].split(',')
    
    # 获取数据版本
    current_dir = os.getcwd().split('/')[-1]
    if current_dir in ['r4.1', 'r4.2', 'r5.1', 'r5.2', 'r6.1', 'r6.2']:
        data_version = current_dir
    
    print(f"🎯 配置参数:")
    print(f"   数据版本: {data_version}")
    print(f"   CPU核心数: {num_cores}")
    print(f"   处理周范围: {start_week} - {end_week or '最大周数'}")
    print(f"   用户限制: {max_users or '无限制'}")
    print(f"   分析模式: {modes}")
    
    # 创建并运行流水线
    pipeline = CERTDatasetPipeline(
        data_version=data_version,
        feature_dim=256,
        num_cores=num_cores
    )
    
    pipeline.run_full_pipeline(
        start_week=start_week,
        end_week=end_week,
        max_users=max_users,
        modes=modes
    )

if __name__ == "__main__":
    main() 