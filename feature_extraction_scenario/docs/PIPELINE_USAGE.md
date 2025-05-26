# CERT数据集完整特征提取流水线使用指南

## 🔒 数据安全警告

> **⚠️ 重要提醒**: 本流水线会操作CSV数据文件，请在使用前阅读 [`./DATA_SAFETY_GUIDE.md`](./DATA_SAFETY_GUIDE.md) (同目录下)
> 
> **测试建议**: 
> - 使用 `../tests/test_pipeline.py` 进行安全测试，它具有完整的数据保护机制
> - 真实数据环境下首次运行前，请手动备份重要文件
> - 优先在独立目录中测试流水线功能

## 🎯 概述

新的`dataset_pipeline.py`实现了完整的CERT数据集特征提取流水线，与原始`feature_extraction.py`功能对等，支持：

- ✅ **原始日志数据的按周合并**
- ✅ **用户信息和恶意用户标记的提取**  
- ✅ **活动数据的数值化特征提取**
- ✅ **多粒度特征的统计计算和CSV导出**
- ✅ **周级别/日级别/会话级别的分析**

## 🏗️ 系统架构对比

### 原始系统 vs 新系统

| 功能模块 | 原始系统 | 新系统 | 状态 |
|---------|----------|--------|------|
| 数据合并 | `combine_by_timerange_pandas()` | `step1_combine_raw_data()` | ✅ 已实现 |
| 用户信息 | `get_mal_userdata()` | `step2_load_user_data()` | ✅ 已实现 |
| 特征提取 | `process_week_num()` + `f_calc()` | `step3_extract_features()` | ✅ 已实现 |
| 多级别分析 | `to_csv()` | `step4_multi_level_analysis()` | ✅ 已实现 |
| 事件编码 | `email_process()`, `file_process()` 等 | 模块化系统 | ✅ 已升级 |

## 🚀 快速开始

### 🧪 安全测试（推荐首次使用）

```bash
# 1. 安全测试（推荐）
# 假设您在 InProject/ 目录下运行
python feature_extraction_scenario/tests/test_pipeline.py

# 系统会自动：
# - 检测现有数据并提供保护
# - 创建标记的测试数据  
# - 运行完整测试
# - 安全清理测试文件
```

测试系统功能：
- ✅ **数据保护**: 自动检测并保护真实数据
- ✅ **备份机制**: 可选择备份现有数据
- ✅ **标记系统**: 测试数据带有明确标记
- ✅ **安全清理**: 只删除确认的测试文件

### 🏭 生产环境使用

> **⚠️ 注意**: 仅在充分测试后在生产环境使用

```bash
# 2. 生产环境（谨慎使用）
# 假设您在 InProject/ 目录下运行
python feature_extraction_scenario/core_logic/dataset_pipeline.py

# 使用前确保：
# - 已备份重要数据
# - 理解输入输出文件
# - 测试过相同配置
```

### 基础使用

```bash
# 在CERT数据集目录下运行（如r4.2文件夹），假设 feature_extraction_scenario 在其子目录
cd /path/to/r4.2/

# 基础运行（处理所有周，所有用户，所有模式）
python feature_extraction_scenario/core_logic/dataset_pipeline.py

# 指定参数运行
python feature_extraction_scenario/core_logic/dataset_pipeline.py [CPU核心数] [开始周] [结束周] [最大用户数] [模式列表]
```

### 参数说明

```bash
# 假设 feature_extraction_scenario/core_logic/dataset_pipeline.py 可直接执行或通过包装脚本
python path_to/dataset_pipeline.py 16 0 10 100 \"week,day,session\"
#                          |  |  |   |   |
#                          |  |  |   |   └── 分析模式（逗号分隔）
#                          |  |  |   └────── 最大用户数限制
#                          |  |  └────────── 结束周数
#                          |  └───────────── 开始周数  
#                          └──────────────── CPU核心数
```

## 📊 多级别分析详解

### 1. 周级别分析 (Week Level)

```python
# 执行周级别分析
pipeline = CERTDatasetPipeline(data_version='r4.2')
pipeline.run_full_pipeline(start_week=0, end_week=10, modes=['week'])
```

**输出文件**: `WeekLevelFeatures/weeks_0_9.csv`

**特征内容**:
- 用户每周的聚合活动特征
- 事件类型分布比例
- 特征向量的统计量（均值、标准差等）
- 恶意活动比例

**数据格式**:
```csv
user,week,mode,n_events,malicious_ratio,email_ratio,file_ratio,http_ratio,mean_feature_0,mean_feature_1,...
ACM2278,0,week,45,0.0,0.3,0.2,0.4,0.123,0.456,...
ACM2278,1,week,52,0.1,0.25,0.3,0.35,0.234,0.567,...
```

### 2. 日级别分析 (Day Level)

```python
# 执行日级别分析
pipeline.run_full_pipeline(start_week=0, end_week=10, modes=['day'])
```

**输出文件**: `DayLevelFeatures/days_0_9.csv`

**特征内容**:
- 用户每日的活动模式
- 工作日 vs 周末行为差异
- 日内活动分布

**数据格式**:
```csv
user,week,day,mode,n_events,malicious_ratio,email_ratio,file_ratio,duration_minutes,...
ACM2278,0,2024-01-15,day,12,0.0,0.4,0.2,480,...
ACM2278,0,2024-01-16,day,18,0.0,0.3,0.3,510,...
```

### 3. 会话级别分析 (Session Level)

```python
# 执行会话级别分析
pipeline.run_full_pipeline(start_week=0, end_week=10, modes=['session'])
```

**输出文件**: `SessionLevelFeatures/sessions_0_9.csv`

**特征内容**:
- 用户会话的持续时间
- 会话内活动密度
- 会话级别的行为模式

**数据格式**:
```csv
user,week,session_id,mode,n_events,duration_minutes,malicious_ratio,unique_event_types,...
ACM2278,0,0,session,8,120,0.0,3,...
ACM2278,0,1,session,15,240,0.1,4,...
```

## 🔧 高级配置

### 自定义会话识别

```python
# 修改会话间隔阈值（默认60分钟）
def custom_session_analysis():
    pipeline = CERTDatasetPipeline()
    
    # 在_identify_sessions方法中修改gap_threshold参数
    # gap_threshold=30  # 30分钟无活动则分割会话
    # gap_threshold=120 # 2小时无活动则分割会话
```

### 选择性模式执行

```python
# 只执行特定分析模式
pipeline.run_full_pipeline(
    start_week=0, 
    end_week=5, 
    modes=['session']  # 只执行会话级别分析
)

# 执行多种模式
pipeline.run_full_pipeline(
    start_week=0, 
    end_week=10, 
    modes=['week', 'day']  # 执行周级别和日级别分析
)
```

### 用户子集分析

```python
# 限制分析用户数量（自动保留所有恶意用户）
pipeline.run_full_pipeline(
    start_week=0, 
    end_week=10, 
    max_users=100,  # 最多100个用户
    modes=['week', 'day', 'session']
)
```

## 📁 输出文件结构

(此结构为示例，具体路径取决于 `dataset_pipeline.py` 的 `work_dir` 配置)
```
feature_extraction_scenario/
├── core_logic/
│   ├── dataset_pipeline.py              # 完整流水线
│   ├── encoder.py                       # 统一编码器
│   └── ... (其他核心逻辑模块)
├── docs/                                # 文档 (本文件所在目录)
├── output/                              # (被 .gitignore 忽略)
│   ├── DataByWeek/                      # 按周合并的原始数据 (如果 work_dir 指向这里)
│   ├── NumDataByWeek/                   # 数值化特征数据 (如果 work_dir 指向这里)
│   ├── WeekLevelFeatures/               # 周级别分析结果 (如果 work_dir 指向这里)
│   ├── DayLevelFeatures/                # 日级别分析结果 (如果 work_dir 指向这里)
│   ├── SessionLevelFeatures/            # 会话级别分析结果 (如果 work_dir 指向这里)
│   └── ExtractedData/                   # 其他导出数据 (如果 work_dir 指向这里)
├── tests/
│   └── test_pipeline.py                 # 测试脚本，其输出在 feature_extraction_scenario/test_output/
└── README.md                            # 项目主README
```

## 🔍 特征对比分析

### 新系统优势

1. **模块化设计**: 每个功能模块独立，易于维护和扩展
2. **统一接口**: EventEncoder提供一致的编码接口
3. **版本兼容**: 自动适配不同CERT数据集版本
4. **mask机制**: 智能处理缺失数据
5. **异常检测**: 内置多种可疑模式识别

### 特征维度对比

| 数据版本 | 原始系统特征数 | 新系统特征数 | 提升 |
|---------|---------------|-------------|------|
| r4.2 | 27列 | 256维向量 | 9.5x |
| r5.2 | 45列 | 256维向量 | 5.7x |
| r6.2 | 46列 | 256维向量 | 5.6x |

## 💡 使用建议

### 1. 渐进式处理

```bash
# 先处理少量周数验证系统
python dataset_pipeline.py 8 0 3 50 "week"

# 再扩展到更大规模
python dataset_pipeline.py 16 0 10 200 "week,day"

# 最后执行完整分析
python dataset_pipeline.py 32 0 73 None "week,day,session"
```

### 2. 资源优化

```python
# 对于大规模数据集
pipeline = CERTDatasetPipeline(
    data_version='r5.2',
    feature_dim=128,  # 减少向量维度
    num_cores=16      # 增加并行度
)
```

### 3. 结果验证

```python
# 检查输出文件
import pandas as pd

# 读取周级别结果
week_df = pd.read_csv('WeekLevelFeatures/weeks_0_9.csv')
print(f"周级别数据: {len(week_df)} 条记录")
print(f"用户数: {week_df['user'].nunique()}")
print(f"恶意比例: {week_df['malicious_ratio'].mean():.3f}")

# 检查特征分布
feature_cols = [col for col in week_df.columns if col.startswith('mean_feature_')]
print(f"特征维度: {len(feature_cols)}")
```

## 🚨 故障排除

### 常见问题

1. **原始数据文件缺失**
```bash
FileNotFoundError: 缺失原始数据文件: ['email.csv', 'file.csv']
```
**解决**: 确保在正确的CERT数据集目录下运行，包含所有CSV文件

2. **内存不足**
```bash
MemoryError: Unable to allocate array
```
**解决**: 减少`max_users`参数或增加系统内存

3. **编码器拟合失败**
```bash
ValueError: 编码器未拟合，请先调用fit()方法
```
**解决**: 确保训练数据不为空，检查数据格式

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 小规模测试
pipeline = CERTDatasetPipeline(data_version='r4.2')
pipeline.run_full_pipeline(
    start_week=0, 
    end_week=2,      # 只处理2周
    max_users=10,    # 只处理10个用户
    modes=['week']   # 只执行周级别分析
)
```

## 🎉 总结

新的`dataset_pipeline.py`完整实现了原始`feature_extraction.py`的所有核心功能，并在以下方面实现了显著提升：

- **✅ 模块化架构** - 更好的代码组织和维护性
- **✅ 统一编码** - 一致的特征表示格式
- **✅ 多级别分析** - 周/日/会话级别的完整支持
- **✅ 智能处理** - 自动处理缺失数据和版本差异
- **✅ 增强特征** - 更丰富的特征维度和异常检测

现在你可以完整地替代原始特征提取系统，获得更强大和灵活的内部威胁检测特征提取能力！ 