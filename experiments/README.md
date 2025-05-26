# 🧪 多模态内部威胁检测实验框架

这是一个完整的实验框架，用于多模态内部威胁检测系统的研究和评估。框架支持多种实验类型，集成了WandB实验跟踪、Optuna超参数优化、数据不平衡处理等先进功能。

## 📋 支持的实验类型

### 1. 🔬 基线对比实验 (`baseline`)
比较传统机器学习方法与多模态模型的检测性能：
- **传统ML模型**: RandomForest, XGBoost
- **多模态模型**: 完整的多模态异常检测模型
- **统一评估**: F1, AUC, Precision, Recall
- **特征重要性分析**: SHAP值和特征重要性可视化

### 2. 🎯 超参数优化实验 (`tune`)
使用Optuna进行智能超参数搜索：
- **搜索空间**: learning_rate, hidden_dim, num_heads, num_layers, dropout, batch_size
- **优化算法**: TPE (Tree-structured Parzen Estimator)
- **早停机制**: 自动识别最优配置
- **可视化**: 优化过程和参数重要性分析

### 3. 🧪 消融实验 (`ablation`)
研究多模态模型中各分支模块的独立贡献：
- **模态组合**: behavior, graph, text, structured的不同组合
- **性能对比**: 各组合的F1分数对比
- **注意力分析**: 不同模态的注意力权重热图
- **贡献度排序**: 模态重要性排序

### 4. ⚖️ 数据不平衡适应性实验 (`imbalance`)
评估模型在不同程度数据失衡下的鲁棒性：
- **不平衡比例**: 1:1, 2:1, 3:1, 4:1, 5:1 (正常:威胁)
- **采样策略**: SMOTE, ADASYN, 随机欠采样, 组合采样
- **性能曲线**: 不平衡比例 vs F1/AUC曲线
- **策略对比**: 不同采样策略效果对比



## 🚀 快速开始

首先需要在https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247中下载r4.2版本的数据集，并解压到`data/`目录下。data目录和`experiments`目录同级。  
### 安装依赖
```bash
cd experiments
pip install -r requirements.txt
```

### 基本使用
```bash
# 基线对比实验
python main_experiment.py --run_type baseline --max_users 100 --epochs 5

# 超参数优化
python main_experiment.py --run_type tune --config_file configs/tune_config.yaml --n_trials 30

# 消融实验
python main_experiment.py --run_type ablation --max_users 50 --epochs 3

# 数据不平衡实验
python main_experiment.py --run_type imbalance --config_file configs/imbalance_config.yaml
```

## 📁 目录结构

```
experiments/
├── main_experiment.py          # 主实验控制脚本
├── core_logic/                 # 核心逻辑模块
│   ├── config.py              # 配置管理
│   ├── dataset_pipeline.py    # 数据处理流水线
│   ├── multimodal_pipeline.py # 多模态数据流水线
│   ├── train_pipeline_multimodal/ # 多模态训练模块
│   └── models/                # 模型定义
├── utils/                     # 实验工具模块
│   ├── wandb_utils.py         # WandB集成
│   ├── baseline_models.py     # 传统ML基线
│   ├── imbalance_utils.py     # 数据不平衡处理
│   └── optuna_tuning.py       # Optuna超参数优化
├── configs/                   # 配置文件
│   ├── baseline.yaml          # 基线实验配置
│   ├── ablation.yaml          # 消融实验配置
│   ├── tune_config.yaml       # 超参数优化配置
│   ├── imbalance_config.yaml  # 不平衡实验配置
│   └── quick_test.yaml        # 快速测试配置
├── results/                   # 实验结果
└── requirements.txt           # 依赖包列表
```

## 🔧 配置说明

### 基本配置结构
```yaml
model:
  enabled_modalities: ["behavior", "graph", "text", "structured"]
  hidden_dim: 256
  num_heads: 8
  num_layers: 6
  dropout: 0.1

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  device: "auto"

data:
  max_users: 100
  max_weeks: 72
  sample_ratio: 1.0
```

### 实验特定配置

#### 超参数优化
```yaml
# tune_config.yaml
n_trials: 30        # Optuna试验次数
timeout: 3600       # 超时时间(秒)
```

#### 数据不平衡
```yaml
# imbalance_config.yaml
imbalance_ratios: [1.0, 2.0, 3.0, 4.0, 5.0]
sampling_strategies: ["none", "smote", "adasyn", "random_undersample"]
```

## 📊 WandB集成

框架完全集成了Weights & Biases (WandB) 用于实验跟踪：

### 自动记录内容
- **配置参数**: 所有实验配置
- **训练指标**: Loss, Accuracy, F1, AUC
- **模型性能**: 验证和测试结果
- **可视化图表**: 
  - 训练曲线
  - 混淆矩阵
  - 特征重要性
  - 注意力热图
  - 消融实验结果
  - 不平衡分析曲线

### WandB项目组织
- **项目名**: `threat_detection_experiments`
- **实验分组**: 按实验类型自动分组
- **标签系统**: 自动添加相关标签

## 🎯 实验最佳实践

### 1. 基线实验
```bash
# 完整基线对比
python main_experiment.py --run_type baseline \
    --max_users 200 \
    --epochs 10 \
    --config_file configs/baseline.yaml
```

### 2. 超参数优化
```bash
# 快速调优 (开发阶段)
python main_experiment.py --run_type tune \
    --max_users 50 \
    --n_trials 20 \
    --epochs 3

# 完整调优 (最终实验)
python main_experiment.py --run_type tune \
    --config_file configs/tune_config.yaml \
    --n_trials 100
```

### 3. 消融实验
```bash
# 快速消融测试
python main_experiment.py --run_type ablation \
    --max_users 50 \
    --epochs 3

# 完整消融实验
python main_experiment.py --run_type ablation \
    --config_file configs/ablation.yaml
```

### 4. 数据不平衡实验
```bash
# 标准不平衡实验
python main_experiment.py --run_type imbalance \
    --config_file configs/imbalance_config.yaml
```

## 📈 结果分析

### 实验结果文件
每次实验会生成以下文件：
- `{experiment_name}_results.json`: 完整实验结果
- `{experiment_name}_{timestamp}.log`: 详细日志
- 模型文件: `best_model.pth`
- Optuna结果: `*_optuna_results.json`, `optimization_history.csv`

### WandB仪表板
访问 [wandb.ai](https://wandb.ai) 查看：
- 实时训练进度
- 实验对比分析
- 超参数重要性
- 模型性能可视化

## 🔍 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减少用户数和数据采样比例
   --max_users 50 --sample_ratio 0.5
   ```

2. **训练时间过长**
   ```bash
   # 减少训练轮数和试验次数
   --epochs 3 --n_trials 10
   ```

3. **WandB登录问题**
   ```bash
   wandb login
   # 或设置环境变量
   export WANDB_API_KEY=your_api_key
   ```

4. **CUDA内存错误**
   ```bash
   # 强制使用CPU
   --device cpu
   ```

### 调试模式
```bash
# 使用快速测试配置
python main_experiment.py --run_type baseline \
    --config_file configs/quick_test.yaml
```

## 🤝 贡献指南

1. **添加新实验类型**: 在 `main_experiment.py` 中添加新的 `run_*_experiment` 函数
2. **扩展工具模块**: 在 `utils/` 目录下添加新的工具模块
3. **更新配置**: 在 `configs/` 目录下添加对应的配置文件
4. **文档更新**: 更新此README和相关文档

## 📚 参考资料

- [WandB文档](https://docs.wandb.ai/)
- [Optuna文档](https://optuna.readthedocs.io/)
- [Imbalanced-learn文档](https://imbalanced-learn.org/)
- [SHAP文档](https://shap.readthedocs.io/)

---

�� **开始你的威胁检测研究之旅吧！** 