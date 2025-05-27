# Dask内存优化使用指南

## 🎯 优化目标

提高Step 1数据合并阶段的内存使用效率，充分利用系统RAM，加快数据处理速度。

## 🔧 已实施的优化

### 1. 动态内存配置
- **自动检测系统内存**：根据可用内存动态分配worker内存
- **智能分区大小**：基于内存容量计算最优分区大小
- **进程级并行**：使用多进程避免Python GIL限制

### 2. 文件读取优化
- **动态blocksize**：根据文件大小自动调整读取块大小
- **分层处理策略**：
  - 大文件(>1GB)：`blocksize = 文件大小 / (worker数 × 4)`
  - 中等文件(100MB-1GB)：`blocksize = 文件大小 / (worker数 × 2)`
  - 小文件(<100MB)：固定16MB

### 3. 内存管理优化
- **目标内存使用率**：95%
- **溢出阈值**：85%时开始写入磁盘
- **暂停阈值**：90%时暂停接收新任务
- **终止阈值**：98%时重启worker

### 4. Parquet写入优化
- **Snappy压缩**：平衡压缩率和速度
- **大行组**：100,000行/组，减少元数据开销
- **字典编码**：自动压缩重复值

## 📊 内存监控

系统会在关键阶段自动显示内存使用情况：

```
📊 内存监控 开始Step1:
   系统内存: 12.5GB / 32.0GB (39.1%)
   可用内存: 18.2GB
   进程内存: 2.1GB (RSS), 3.4GB (VMS)
```

## 🚀 使用方法

### 基本使用
```bash
cd /Users/weitao_li/CodeField/DCAI/Huawei/Anomaly_Detection/InProject/experiments
python main_experiment.py --run_type baseline --max_users 1000 --max_weeks 10
```

### 监控内存使用
在运行过程中，观察以下指标：

1. **系统内存使用率**：应该达到70-85%
2. **进程内存(RSS)**：应该稳定增长，不应该频繁波动
3. **Dask Dashboard**：访问 http://localhost:8787 查看实时状态

### 调优建议

#### 如果内存使用率过低(<50%)：
```python
# 在配置文件中增加内存使用比例
system:
  memory_usage_fraction: 0.9  # 从0.8增加到0.9

data_processing:
  max_partition_size_mb: 800  # 从500增加到800
```

#### 如果出现内存不足：
```python
# 减少内存使用
system:
  memory_usage_fraction: 0.6  # 从0.8减少到0.6
  threads_per_worker: 1       # 从2减少到1

data_processing:
  max_partition_size_mb: 300  # 从500减少到300
```

#### 如果处理速度慢：
```python
# 增加并行度
system:
  threads_per_worker: 4       # 从2增加到4

data_processing:
  min_partition_size_mb: 100  # 从200减少到100，增加分区数
```

## 🔍 性能测试

### 测试不同配置的效果：

1. **小规模测试**：
```bash
python main_experiment.py --run_type baseline --max_users 100 --max_weeks 2
```

2. **中等规模测试**：
```bash
python main_experiment.py --run_type baseline --max_users 500 --max_weeks 5
```

3. **大规模测试**：
```bash
python main_experiment.py --run_type baseline --max_users 2000 --max_weeks 10
```

### 性能指标对比

记录以下指标来评估优化效果：

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| 内存使用率 | ~30% | ~80% | +167% |
| 处理速度 | ? | ? | ? |
| 分区数量 | ? | ? | ? |
| 文件读取时间 | ? | ? | ? |

## 🛠️ 故障排除

### 常见问题

1. **内存不足错误**：
   - 减少 `memory_usage_fraction`
   - 增加 `spill_fraction` 提前溢出
   - 减少 `max_partition_size_mb`

2. **处理速度慢**：
   - 检查磁盘I/O是否成为瓶颈
   - 增加 `threads_per_worker`
   - 减少 `min_partition_size_mb` 增加并行度

3. **Dask worker崩溃**：
   - 检查 `terminate_fraction` 是否过低
   - 增加 `allowed_failures`
   - 检查系统内存是否真的不足

### 调试命令

```bash
# 查看系统内存
free -h

# 查看进程内存使用
ps aux | grep python

# 监控实时内存使用
watch -n 1 'free -h && ps aux | grep python | head -5'
```

## 📈 预期效果

经过优化后，您应该看到：

1. **内存使用率提升**：从30%提升到70-85%
2. **处理速度加快**：更多数据并行处理
3. **更稳定的性能**：减少内存碎片和GC压力
4. **更好的资源利用**：充分发挥多核CPU优势

## 🔄 进一步优化

如果还需要更高的性能，可以考虑：

1. **SSD存储**：确保数据存储在SSD上
2. **增加RAM**：如果经常达到内存上限
3. **分布式处理**：使用多台机器组成Dask集群
4. **数据预处理**：提前过滤不需要的列和行 