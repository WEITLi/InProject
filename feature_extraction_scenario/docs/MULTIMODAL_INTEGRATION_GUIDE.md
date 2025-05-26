# 多模态异常检测集成指南

## 概述

本指南介绍如何将多模态数据处理方式集成到原有的华为内部威胁检测项目中。新的多模态架构在保持原有功能的基础上，增加了用户关系图、文本内容分析、结构化特征融合等多模态能力。

## 架构升级

### 原有架构 vs 新架构

| 组件 | 原有架构 | 新多模态架构 |
|------|----------|--------------|
| 数据处理 | 单一特征提取 | 多模态数据流水线 |
| 模型架构 | Transformer | Transformer + GNN + BERT + LightGBM |
| 特征类型 | 行为序列 | 行为序列 + 用户关系 + 文本内容 + 结构化特征 |
| 融合机制 | 无 | 注意力融合 + 门控机制 |
| 训练流程 | 基础训练 | 多模态联合训练 |

### 新增组件

1. **多模态数据流水线** (`MultiModalDataPipeline`)
   - 整合现有的CERT数据集处理能力
   - 新增用户关系图构建
   - 文本内容提取和预处理
   - 结构化特征工程

2. **多模态模型** (`MultiModalAnomalyDetector`)
   - Transformer编码器：行为序列建模
   - GNN编码器：用户关系图嵌入
   - BERT编码器：文本内容理解
   - LightGBM分支：结构化特征处理
   - 注意力融合：多模态特征融合

3. **多模态训练器** (`MultiModalTrainer`)
   - 统一的训练流程
   - 多模态数据加载
   - 联合损失函数
   - 性能评估和可视化

## 快速开始

### 1. 环境准备

确保已安装所需依赖：

```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install transformers
pip install lightgbm
pip install torch-geometric
pip install scikit-learn pandas numpy matplotlib seaborn

# 可选：GPU支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 快速训练

使用快速开发模式进行测试：

```bash
cd Huawei/Anomaly_Detection/InProject/feature_extraction_scenario

# 快速开发模式（少量数据，快速验证）
python scripts/train_multimodal.py --fast_dev_run

# 基础训练模式
python scripts/train_multimodal.py --mode train --end_week 10 --max_users 200

# 完整训练模式
python scripts/train_multimodal.py --mode train --num_epochs 50
```

### 3. 对比实验

运行不同模态组合的对比实验：

```bash
# 运行预定义的模态对比实验
python scripts/train_multimodal.py --mode experiment --fast_dev_run

# 运行超参数对比实验
python scripts/train_multimodal.py --mode comparison --fast_dev_run
```

## 详细使用说明

### 命令行参数

#### 数据参数
- `--data_version`: 数据集版本 (默认: r4.2)
- `--feature_dim`: 特征维度 (默认: 256)
- `--start_week`: 开始周数 (默认: 0)
- `--end_week`: 结束周数 (默认: None, 使用全部数据)
- `--max_users`: 最大用户数 (默认: None, 使用全部用户)

#### 模型参数
- `--hidden_dim`: 隐藏层维度 (默认: 256)
- `--num_heads`: 注意力头数 (默认: 8)
- `--num_layers`: Transformer层数 (默认: 6)
- `--sequence_length`: 序列长度 (默认: 128)

#### 模块控制
- `--enable_gnn`: 启用GNN用户图嵌入
- `--enable_bert`: 启用BERT文本编码
- `--enable_lgbm`: 启用LightGBM结构化特征
- `--enable_transformer`: 启用Transformer序列建模

#### 训练参数
- `--batch_size`: 批大小 (默认: 32)
- `--learning_rate`: 学习率 (默认: 1e-4)
- `--num_epochs`: 训练轮数 (默认: 100)
- `--patience`: 早停patience (默认: 10)

### 使用示例

#### 1. 基础单模态训练

```bash
# 仅使用Transformer
python scripts/train_multimodal.py \
    --mode train \
    --enable_gnn False \
    --enable_bert False \
    --enable_lgbm False \
    --experiment_name "transformer_only"
```

#### 2. 双模态训练

```bash
# Transformer + GNN
python scripts/train_multimodal.py \
    --mode train \
    --enable_bert False \
    --enable_lgbm False \
    --experiment_name "transformer_gnn"
```

#### 3. 完整多模态训练

```bash
# 所有模态
python scripts/train_multimodal.py \
    --mode train \
    --hidden_dim 512 \
    --num_epochs 100 \
    --batch_size 16 \
    --experiment_name "full_multimodal"
```

#### 4. 自定义配置文件

创建配置文件 `config.json`：

```json
{
    "model": {
        "hidden_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "enable_gnn": true,
        "enable_bert": true,
        "enable_lgbm": true
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 50
    },
    "data": {
        "data_version": "r4.2",
        "start_week": 0,
        "end_week": 20,
        "max_users": 500
    }
}
```

使用配置文件：

```bash
python scripts/train_multimodal.py --config_file config.json
```

## 数据流水线详解

### 1. 基础特征提取

保持原有的CERT数据集处理能力：
- 按周合并原始数据
- 用户信息和恶意用户标记提取
- 活动数据的数值化特征提取
- 多粒度特征统计计算

### 2. 多模态数据提取

#### 用户关系图构建
- 基于部门、角色、OCEAN心理特征构建用户节点特征
- 计算用户间关系强度（部门关系、角色关系、心理相似性、活动交互）
- 生成邻接矩阵表示用户关系网络

#### 文本数据提取
- 邮件内容文本提取
- 文件名和路径信息提取
- HTTP URL信息提取
- 按用户聚合文本内容

#### 行为序列构建
- 按时间顺序构建用户行为序列
- 使用现有编码器进行事件编码
- 支持序列截断和填充

#### 结构化特征准备
- 从周级别特征文件提取数值特征
- 支持特征聚合和标准化
- 处理缺失值和异常值

### 3. 训练数据整合

将各模态数据整合为统一的训练格式：
- 行为序列：`[batch_size, sequence_length, feature_dim]`
- 用户图：节点特征 + 邻接矩阵
- 文本内容：字符串列表
- 结构化特征：`[batch_size, struct_feature_dim]`
- 标签：`[batch_size]`

## 模型架构详解

### 1. 多模态编码器

#### Transformer编码器
- 输入：行为序列 `[batch_size, seq_len, feature_dim]`
- 输出：序列表示 `[batch_size, hidden_dim]`
- 特点：时序建模、位置编码、多头注意力

#### GNN编码器
- 输入：节点特征 + 邻接矩阵
- 输出：用户图嵌入 `[num_users, hidden_dim]`
- 特点：图卷积、邻居聚合、关系建模

#### BERT编码器
- 输入：文本内容列表
- 输出：文本表示 `[batch_size, hidden_dim]`
- 特点：预训练语言模型、上下文理解

#### LightGBM分支
- 输入：结构化特征 `[batch_size, struct_dim]`
- 输出：结构化表示 `[batch_size, hidden_dim]`
- 特点：梯度提升、特征重要性、非线性建模

### 2. 多模态融合

#### 注意力融合机制
- 特征投影：将各模态特征投影到统一空间
- 注意力权重：计算模态间的重要性权重
- 门控机制：控制各模态的贡献度
- 加权融合：生成最终的融合表示

#### 融合策略
- **注意力融合**：基于注意力机制的动态权重
- **拼接融合**：简单的特征拼接
- **加权融合**：固定权重的线性组合

### 3. 异常检测头

#### 分类头设计
- 输入：融合特征 `[batch_size, hidden_dim]`
- 输出：分类logits `[batch_size, num_classes]`
- 包含：全连接层、Dropout、激活函数

#### 异常检测头
- 双分支设计：异常分数 + 置信度
- 输出：异常概率、置信度分数
- 支持：阈值调整、不确定性估计

## 训练流程详解

### 1. 数据加载

```python
# 创建多模态数据集
dataset = MultiModalDataset(training_data, device='cuda')

# 划分训练/验证/测试集
train_loader, val_loader, test_loader = prepare_data_loaders(dataset)
```

### 2. 模型创建

```python
# 自动确定输入维度
model = MultiModalAnomalyDetector(
    embed_dim=256,
    transformer_config={...},
    gnn_config={...},
    bert_config={...},
    lgbm_config={...}
)
```

### 3. 训练循环

```python
for epoch in range(num_epochs):
    # 训练阶段
    train_metrics = train_epoch(model, train_loader, optimizer, criterion)
    
    # 验证阶段
    val_metrics = validate_epoch(model, val_loader, criterion)
    
    # 学习率调度
    scheduler.step(val_metrics['f1'])
    
    # 早停检查
    if early_stopping(val_metrics['f1'], model):
        break
```

### 4. 评估和保存

```python
# 测试集评估
test_metrics = validate_epoch(model, test_loader, criterion)

# 保存模型和结果
torch.save(model.state_dict(), 'best_model.pth')
save_results(test_metrics, 'test_results.json')

# 生成可视化
plot_training_curves()
plot_confusion_matrix()
```

## 性能优化建议

### 1. 内存优化

- 使用梯度累积减少内存占用
- 启用混合精度训练
- 合理设置batch_size

```bash
# 小内存设置
python scripts/train_multimodal.py \
    --batch_size 8 \
    --num_workers 2 \
    --hidden_dim 128
```

### 2. 计算优化

- 使用GPU加速
- 启用数据并行
- 优化数据加载

```bash
# GPU优化设置
python scripts/train_multimodal.py \
    --device cuda \
    --num_workers 4 \
    --batch_size 32
```

### 3. 训练优化

- 使用学习率调度
- 启用早停机制
- 梯度裁剪防止梯度爆炸

## 实验结果分析

### 1. 模态贡献分析

运行模态对比实验分析各模态的贡献：

```bash
python scripts/train_multimodal.py --mode experiment
```

预期结果：
- Transformer Only: 基线性能
- Transformer + GNN: 用户关系建模提升
- Transformer + BERT: 文本理解增强
- Transformer + LightGBM: 结构化特征补充
- Full Multimodal: 最佳综合性能

### 2. 超参数敏感性

运行超参数对比实验：

```bash
python scripts/train_multimodal.py --mode comparison
```

分析维度：
- 模型大小对性能的影响
- 训练时间与性能的权衡
- 内存占用与模型复杂度

### 3. 性能指标

关注的主要指标：
- **准确率 (Accuracy)**: 整体分类正确率
- **精确率 (Precision)**: 异常检测精确度
- **召回率 (Recall)**: 异常检测覆盖率
- **F1分数**: 精确率和召回率的调和平均
- **AUC**: ROC曲线下面积
- **训练时间**: 模型训练效率

## 故障排除

### 1. 常见错误

#### 内存不足
```
RuntimeError: CUDA out of memory
```
解决方案：
- 减少batch_size
- 降低模型维度
- 使用CPU训练

#### 数据维度不匹配
```
RuntimeError: size mismatch
```
解决方案：
- 检查特征维度配置
- 确认数据预处理正确
- 验证模型输入输出维度

#### 模块导入错误
```
ModuleNotFoundError: No module named 'xxx'
```
解决方案：
- 检查Python路径设置
- 确认依赖包安装
- 验证相对导入路径

### 2. 调试技巧

#### 启用调试模式
```bash
python scripts/train_multimodal.py --debug --fast_dev_run
```

#### 检查数据流
```python
# 在代码中添加调试信息
print(f"数据形状: {data.shape}")
print(f"标签分布: {np.bincount(labels)}")
```

#### 监控训练过程
```python
# 使用TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')
writer.add_scalar('Loss/Train', loss, epoch)
```

## 扩展和定制

### 1. 添加新模态

要添加新的模态（如音频、图像等）：

1. 在`MultiModalDataPipeline`中添加数据提取方法
2. 在`MultiModalAnomalyDetector`中添加对应编码器
3. 更新融合机制以支持新模态
4. 修改训练器以处理新数据格式

### 2. 自定义融合策略

实现新的融合机制：

```python
class CustomFusion(nn.Module):
    def __init__(self, input_dims, output_dim):
        super().__init__()
        # 自定义融合逻辑
        
    def forward(self, modality_features):
        # 实现融合算法
        return fused_features
```

### 3. 集成外部模型

集成预训练模型或外部API：

```python
class ExternalEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.external_model = load_external_model(model_path)
        
    def forward(self, inputs):
        return self.external_model(inputs)
```

## 最佳实践

### 1. 实验管理

- 使用有意义的实验名称
- 记录详细的配置信息
- 保存中间结果和检查点
- 使用版本控制管理代码

### 2. 数据管理

- 定期备份重要数据
- 使用数据版本控制
- 监控数据质量
- 处理数据隐私和安全

### 3. 模型部署

- 模型压缩和优化
- 推理性能测试
- 监控系统集成
- 持续学习和更新

## 总结

多模态异常检测集成方案成功地将原有的单一Transformer架构升级为综合的多模态系统，在保持原有功能的基础上显著增强了模型的表达能力和检测性能。通过统一的训练框架和灵活的配置系统，用户可以根据具体需求选择合适的模态组合和模型配置。

关键优势：
- **向后兼容**：保持原有数据处理流程
- **模块化设计**：支持灵活的模态组合
- **统一框架**：简化训练和部署流程
- **性能提升**：多模态融合增强检测能力
- **易于扩展**：支持新模态和算法集成

建议的使用流程：
1. 从快速开发模式开始验证环境
2. 运行模态对比实验了解各组件贡献
3. 根据数据特点选择合适的模态组合
4. 调优超参数获得最佳性能
5. 部署到生产环境并持续监控 