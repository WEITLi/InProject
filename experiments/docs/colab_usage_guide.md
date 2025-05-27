# Google Colab环境使用指南

## 🔬 Colab环境特点

Google Colab提供的计算资源有以下特点：
- **内存**: 12-16GB RAM
- **CPU**: 通常2个核心
- **存储**: 临时磁盘空间
- **时间限制**: 12小时连续运行限制
- **网络**: 较好的下载速度

## 🚀 快速开始

### 1. 环境检测和优化

系统会自动检测Colab环境并应用优化配置：

```python
# 自动检测到Colab环境时的输出
🔬 检测到Google Colab环境，使用保守配置
💾 系统总内存: 12.7GB, 可用内存: 11.2GB
🔧 CPU核心数: 2, 使用workers: 2
⚙️ 配置: 2 workers, 每个worker 3.4GB
```

### 2. Colab专用优化

- **内存使用率**: 60% (更保守，避免OOM)
- **Worker数量**: 最多2个
- **线程模式**: 使用线程而非进程
- **分区大小**: 64-128MB (更小的分区)
- **Blocksize**: 8-16MB (更小的读取块)

## 📊 内存监控

在Colab中，内存监控更加重要：

```
📊 内存监控 开始Step1:
   系统内存: 8.5GB / 12.7GB (66.9%)
   可用内存: 3.8GB
   进程内存: 1.2GB (RSS), 2.1GB (VMS)
```

**警告信号**:
- 系统内存使用率 > 85%
- 可用内存 < 2GB
- 进程内存增长过快

## 🛠️ Colab特定配置

### 推荐的数据规模

```python
# 小规模测试 (推荐)
python main_experiment.py --run_type baseline --max_users 50 --data_version r4.2

# 中等规模测试 (谨慎)
python main_experiment.py --run_type baseline --max_users 200 --data_version r4.2

# 大规模测试 (不推荐，可能OOM)
# python main_experiment.py --run_type baseline --max_users 1000 --data_version r4.2
```

### 采样策略

在Colab中强烈建议使用数据采样：

```python
# 使用10%的数据进行快速测试
python main_experiment.py --run_type baseline --max_users 100 --sample_ratio 0.1

# 使用50%的数据进行中等规模测试
python main_experiment.py --run_type baseline --max_users 200 --sample_ratio 0.5
```

## ⚠️ 常见问题和解决方案

### 1. 内存不足 (OOM)

**症状**:
```
RuntimeError: CUDA out of memory
或
MemoryError: Unable to allocate array
```

**解决方案**:
```python
# 减少用户数量
--max_users 50

# 增加采样率
--sample_ratio 0.2

# 使用更小的时间窗口
--max_weeks 2
```

### 2. 连接超时

**症状**:
```
TimeoutError: Dask client connection timeout
```

**解决方案**:
```python
# 重启运行时
Runtime -> Restart Runtime

# 清理内存
import gc
gc.collect()
```

### 3. 磁盘空间不足

**症状**:
```
OSError: [Errno 28] No space left on device
```

**解决方案**:
```python
# 清理临时文件
!rm -rf /tmp/*

# 使用更小的数据集
--sample_ratio 0.1
```

## 🔧 性能优化技巧

### 1. 分阶段运行

```python
# 第一阶段：只运行数据合并
pipeline.step1_combine_raw_data(start_week=0, end_week=2, sample_ratio=0.2)

# 第二阶段：用户数据加载
users_df = pipeline.step2_load_user_data()

# 第三阶段：特征提取
pipeline.step3_extract_features(users_df, start_week=0, end_week=2, max_users=50)
```

### 2. 监控资源使用

```python
# 定期检查内存使用
import psutil
memory = psutil.virtual_memory()
print(f"内存使用率: {memory.percent:.1f}%")
print(f"可用内存: {memory.available/1024**3:.1f}GB")
```

### 3. 使用检查点

```python
# 保存中间结果
import pickle
with open('intermediate_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# 恢复中间结果
with open('intermediate_results.pkl', 'rb') as f:
    results = pickle.load(f)
```

## 📈 预期性能

在Colab环境中的典型性能：

| 数据规模 | 用户数 | 采样率 | 预期时间 | 内存峰值 |
|----------|--------|--------|----------|----------|
| 小规模   | 50     | 0.1    | 5-10分钟 | 4-6GB    |
| 中等规模 | 200    | 0.2    | 15-25分钟| 8-10GB   |
| 大规模   | 500    | 0.5    | 45-60分钟| 12-14GB  |

## 🚨 注意事项

1. **时间限制**: Colab有12小时运行限制，大规模实验可能需要分批进行
2. **数据持久性**: Colab的文件系统是临时的，重要结果需要下载保存
3. **GPU资源**: 如果使用GPU，注意GPU内存限制
4. **网络稳定性**: 长时间运行可能遇到网络中断

## 🔄 故障恢复

如果实验中断：

```python
# 检查已完成的步骤
import os
if os.path.exists('DataByWeek_parquet'):
    print("Step 1 已完成")
if os.path.exists('NumDataByWeek'):
    print("Step 3 已完成")

# 从中断点继续
pipeline.run_full_pipeline(
    start_week=0, 
    end_week=2,
    force_regenerate_combined_weeks=False  # 不重新生成已有数据
)
```

## 📱 移动端访问

Colab支持移动端访问，但建议：
- 只用于监控进度
- 不要在移动端启动大规模实验
- 及时保存重要结果 