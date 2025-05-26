# 🔒 数据安全保护指南

## 概述

本指南说明如何安全地使用测试流水线，确保真实数据不会被意外删除。

## 🛡️ 安全机制

### 1. 测试数据标记系统

测试系统使用以下机制标记测试数据：

- **标记文件**: `.test_data_created_by_pipeline` 记录所有测试文件
- **文件头标记**: 测试CSV文件开头包含 `# TEST_DATA_CREATED_BY_PIPELINE` 注释
- **双重验证**: 清理时会检查标记文件和文件头标记

### 2. 自动备份机制

当检测到真实数据时，系统会：

1. 提示用户存在真实数据
2. 询问是否备份
3. 创建带时间戳的备份目录
4. 复制真实数据到备份目录

### 3. 安全清理机制

清理功能的安全特性：

- ✅ 只删除确认的测试文件
- ✅ 保护所有未标记的真实数据
- ✅ 提供详细的清理报告
- ✅ 支持数据恢复

## 📋 使用指南

### 基本流程

```bash
# 运行测试系统 (test_pipeline.py 位于 tests/ 目录下)
# 假设从 InProject/ 目录运行
python feature_extraction_scenario/tests/test_pipeline.py

# 系统会自动检查数据安全性并提供选项菜单
```

### 选项说明

1. **创建测试数据** - 创建标记的测试数据
2. **运行完整测试** - 执行完整的测试流程
3. **检查数据安全性** - 查看当前数据状态
4. **清理测试文件** - 安全删除测试数据
5. **恢复真实数据** - 从备份恢复数据
6. **退出** - 退出程序

### 数据状态识别

系统会显示每个CSV文件的状态：

```
📄 当前CSV文件状态:
   🧪 测试数据: email.csv
   📋 真实数据: real_email.csv
```

## 🚨 安全警告

### ⚠️ 注意事项

1. **真实数据检测**: 首次运行时如果检测到未标记的CSV文件，会被视为真实数据
2. **备份重要性**: 强烈建议在创建测试数据前备份真实数据
3. **手动删除风险**: 不要手动删除标记文件，可能导致安全机制失效

### 🔄 最佳实践

1. **始终备份**: 处理重要数据前进行备份
2. **检查状态**: 定期使用"检查数据安全性"功能
3. **分离环境**: 在不同目录处理真实数据和测试数据
4. **版本控制**: 使用Git等工具管理代码变更

## 📁 文件结构

### 测试数据文件 (位于 tests/sample_test_data/)

```
tests/
  └── sample_test_data/
      ├── .test_data_created_by_pipeline    # 标记文件
      ├── email.csv                         # 测试数据（带标记）
      ├── file.csv                          # 测试数据（带标记）
      ├── http.csv                          # 测试数据（带标记）
      ├── logon.csv                         # 测试数据（带标记）
      ├── device.csv                        # 测试数据（带标记）
      ├── psychometric.csv                  # 测试数据（带标记）
      └── answers/
          └── insiders.csv              # 测试数据
```

### 备份目录结构 (位于 tests/backup_data/)

```
tests/
  └── backup_data/
      └── backup_YYYYMMDD_HHMMSS/         # 时间戳备份目录
          ├── email.csv                   # 备份的真实数据
├── file.csv
├── http.csv
├── logon.csv
├── device.csv
└── psychometric.csv
```

### 生成的文件（通常位于 test_output/ 或项目根目录，具体取决于如何运行 pipeline）

```
feature_extraction_scenario/test_output/  # 当通过 test_pipeline.py 运行时
  ├── DataByWeek/                       # 按周数据
  ├── NumDataByWeek/                    # 特征数据
  ├── WeekLevelFeatures/                # 周级别分析
  ├── DayLevelFeatures/                 # 日级别分析
  ├── SessionLevelFeatures/             # 会话级别分析
  └── ... (其他生成的文件)

# 或者，如果直接运行 core_logic/dataset_pipeline.py 且 work_dir 未覆盖，
# 这些目录可能在 feature_extraction_scenario/ 根目录下创建 (已被 .gitignore 忽略)
# DataByWeek/
# NumDataByWeek/
# ...
```

## 🛠️ 故障排除

### 问题1: 误标记为真实数据

**症状**: 测试数据被识别为真实数据
**解决**: 
```bash
# 手动删除有问题的测试数据文件 (注意路径，例如 tests/sample_test_data/email.csv)
rm feature_extraction_scenario/tests/sample_test_data/email.csv ...
# 重新创建测试数据
python feature_extraction_scenario/tests/test_pipeline.py
```

### 问题2: 标记文件丢失

**症状**: 无法识别测试数据
**解决**:
```bash
# 检查文件头是否有测试标记 (注意路径)
head -1 feature_extraction_scenario/tests/sample_test_data/email.csv
# 如果确认是测试数据，手动删除
rm feature_extraction_scenario/tests/sample_test_data/email.csv ...
```

### 问题3: 需要恢复数据

**症状**: 意外删除了真实数据
**解决**:
```bash
# 使用恢复功能
python feature_extraction_scenario/tests/test_pipeline.py
# 选择选项 5: 恢复真实数据
```

## 🔍 验证安全性

### 检查测试数据标记

```bash
# 检查标记文件 (位于 tests/sample_test_data/)
cat feature_extraction_scenario/tests/sample_test_data/.test_data_created_by_pipeline

# 检查CSV文件头 (位于 tests/sample_test_data/)
head -1 feature_extraction_scenario/tests/sample_test_data/email.csv
# 应该看到: # TEST_DATA_CREATED_BY_PIPELINE at ...
```

### 验证真实数据保护

```bash
# 运行安全检查 (确保 PYTHONPATH 或当前路径允许导入)
# 假设从 InProject/ 运行
python -c "
import sys
sys.path.append('.') # 添加 InProject 到路径
from feature_extraction_scenario.tests import test_pipeline
test_pipeline.check_data_safety()
"
```

## 📞 支持

如果遇到数据安全问题：

1. **立即停止操作** - 避免进一步的数据风险
2. **检查备份目录** - 查找以`backup_`开头的目录
3. **使用恢复功能** - 通过程序菜单恢复数据
4. **联系支持团队** - 提供详细的错误信息

## 🔮 更新记录

- **v1.0**: 初始版本，基础安全保护
- **v1.1**: 增加文件头标记系统
- **v1.2**: 添加备份和恢复功能
- **v1.3**: 完善安全检查和用户交互 