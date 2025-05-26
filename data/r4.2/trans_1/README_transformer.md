# 上下文增强 Transformer 内部威胁检测

基于 CERT r4.2 数据集的上下文增强 Transformer 内部威胁检测系统，支持多任务学习和小样本训练。

## 项目特性

- **基于 Transformer 的序列建模**: 使用多层 Transformer 编码器处理用户行为序列
- **上下文信息融合**: 集成用户背景信息（部门、职能、权限等）
- **多任务学习**: 结合异常检测和掩蔽语言模型任务
- **小样本学习支持**: 适配数据稀缺场景
- **全面评估指标**: AUC、Precision、Recall、F1-Score 等
- **可视化结果**: 训练曲线、混淆矩阵、ROC 曲线

## 项目结构

```
trans_1/
├── transformer_threat_detection.py    # 主程序入口
├── data_processor.py                  # 数据预处理模块
├── transformer_model.py               # Transformer模型定义
├── trainer.py                         # 训练和评估模块
├── test_transformer.py                # 测试脚本
├── requirements_transformer.txt       # 依赖包列表
└── README_transformer.md             # 使用说明
```

## 环境要求

### Python 版本
- Python 3.8+

### 依赖包
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install tqdm
```

或使用提供的要求文件：
```bash
pip install -r requirements_transformer.txt
```

## 快速开始

### 1. 基础训练

使用默认配置训练模型：

```bash
python transformer_threat_detection.py \
    --data_path ../dayr4.2_u200_w0-3_mweekdaysession_s1-percentile14.pkl \
    --mode train \
    --output_dir ./outputs_transformer
```

### 2. 小样本学习

使用100个样本进行小样本学习：

```bash
python transformer_threat_detection.py \
    --data_path ../dayr4.2_u200_w0-3_mweekdaysession_s1-percentile14.pkl \
    --mode train \
    --few_shot_samples 100 \
    --output_dir ./outputs_few_shot
```

### 3. 自定义模型配置

调整模型超参数：

```bash
python transformer_threat_detection.py \
    --data_path ../dayr4.2_u200_w0-3_mweekdaysession_s1-percentile14.pkl \
    --mode train \
    --hidden_dim 256 \
    --num_layers 6 \
    --num_heads 16 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --num_epochs 100
```

### 4. 对比实验

运行多种配置的对比实验：

```bash
python transformer_threat_detection.py \
    --data_path ../dayr4.2_u200_w0-3_mweekdaysession_s1-percentile14.pkl \
    --mode experiment \
    --output_dir ./experiments
```

## 主要参数说明

### 数据参数
- `--data_path`: 数据文件路径（必需）。注意：数据文件位于上级目录，使用 `../` 前缀
- `--sequence_length`: 输入序列长度，默认30天
- `--few_shot_samples`: 小样本学习的样本数量

### 模型参数
- `--hidden_dim`: Transformer隐藏维度，默认128
- `--num_layers`: Transformer层数，默认4
- `--num_heads`: 注意力头数，默认8
- `--dropout`: Dropout概率，默认0.1

### 训练参数
- `--batch_size`: 批次大小，默认32
- `--learning_rate`: 学习率，默认1e-4
- `--num_epochs`: 训练轮数，默认50

### 运行模式
- `--mode`: 运行模式
  - `train`: 只训练
  - `eval`: 只评估（需要提供模型路径）
  - `both`: 训练后评估
  - `experiment`: 运行对比实验

## 输出文件说明

训练完成后，输出目录将包含以下文件：

### 模型文件
- `best_model.pth`: 最佳模型权重
- `config.json`: 训练配置

### 结果文件
- `test_results.json`: 测试集评估结果
- `training_history.npz`: 训练历史数据

### 可视化图表
- `training_curves.png`: 训练曲线（损失、准确率、AUC）
- `confusion_matrix.png`: 混淆矩阵
- `roc_curve.png`: ROC曲线

## 模型架构

### 整体流程
1. **数据预处理**: 构建用户行为序列和上下文特征
2. **序列编码**: Transformer编码器处理行为序列
3. **上下文融合**: 注意力机制融合用户上下文信息
4. **多任务输出**: 异常分类 + 掩蔽语言模型

### 关键组件

#### 1. 位置编码
```python
# 为序列添加位置信息
positional_encoding = PositionalEncoding(hidden_dim, max_length)
```

#### 2. 上下文融合
```python
# 三种融合策略：注意力、门控、拼接
context_fusion = ContextFusionLayer(hidden_dim, context_dim, fusion_type='attention')
```

#### 3. 多任务学习
```python
# 分类损失 + 掩蔽语言模型损失
total_loss = classification_weight * classification_loss + masked_lm_weight * mlm_loss
```

## 数据格式要求

### 输入数据
数据应为 pandas DataFrame 格式的 pickle 文件，包含以下列：

#### 必需列
- `user`: 用户ID
- `day`/`week`: 时间信息
- `insider`: 异常标签（0: 正常, >0: 异常）

#### 上下文列（可选）
- `role`: 用户角色
- `dept`: 部门
- `team`: 团队
- `ITAdmin`: 是否IT管理员

#### 特征列
- 其他数值特征列（如行为统计特征）

### 数据预处理
系统会自动进行以下预处理：
1. 构建滑动窗口序列
2. 提取用户上下文信息
3. 特征标准化
4. 数据集划分

## 测试系统

运行完整的测试套件：

```bash
python test_transformer.py
```

测试包括：
- 模块导入检查
- 配置类测试
- 模型创建测试
- 数据处理器测试
- 前向传播测试
- 数据文件检查

## 性能调优建议

### 1. 内存优化
- 减少 `sequence_length` 和 `batch_size`
- 使用较小的 `hidden_dim`

### 2. 训练优化
- 使用梯度累积处理大批次
- 调整学习率调度策略
- 启用早停避免过拟合

### 3. 模型优化
- 调整 Transformer 层数和注意力头数
- 尝试不同的上下文融合策略
- 调整多任务学习权重

## 常见问题

### Q1: 内存不足错误
**A**: 降低批次大小或序列长度：
```bash
--batch_size 16 --sequence_length 20
```

### Q2: 训练过慢
**A**: 减少模型复杂度：
```bash
--hidden_dim 64 --num_layers 2 --num_heads 4
```

### Q3: 性能不佳
**A**: 尝试调整学习率和训练轮数：
```bash
--learning_rate 5e-5 --num_epochs 100
```

### Q4: 数据不平衡
**A**: 使用小样本学习或调整损失权重

### Q5: 数据文件未找到
**A**: 确保数据文件在上级目录中，使用相对路径 `../文件名.pkl`

## 可用数据文件

以下数据文件位于上级目录（`../`）：
- `dayr4.2_u200_w0-3_mweekdaysession_s1-percentile14.pkl` （推荐）
- `dayr4.2_u200_w0-3_mweekdaysession_s1-meandiff14.pkl`
- `dayr4.2_u200_w0-3_mweekdaysession_s1-meddiff14.pkl`
- `dayr4.2_u200_w0-3_mweekdaysession_s1-concat5.pkl`

## 扩展功能

### 1. 自定义损失函数
可以在 `transformer_model.py` 中修改损失计算逻辑

### 2. 添加新的评估指标
在 `trainer.py` 中的 `validate_epoch` 函数中添加

### 3. 集成其他模型
在项目中添加 LSTM、CNN 等对比模型 