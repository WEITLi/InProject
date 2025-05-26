# 🧪 多模态异常检测实验框架：四阶段结构化实验流程

请生成一个主控脚本 `main_experiment.py`，用于统一调度以下四类实验流程。脚本结构应通过 `--run_type` 参数区分运行模式：

---

## 1️⃣ run_baseline() — 基线方法对比实验

### 🎯 目标：
比较传统机器学习方法与多模态模型的检测性能

### 📌 要求：
- 基于 `feature_extraction.py` 提取传统手工特征
- 实现两个经典模型：`RandomForestClassifier` 和 `XGBoostClassifier`
- 使用统一数据划分（例如 CERT 子集）
- 统一评估指标：F1, AUC, Precision, Recall
- 使用 WandB 记录每轮运行（group=baseline）

---

## 2️⃣ run_tuning() — 主模型的超参数优化实验

### 🎯 目标：
在多模态主模型（如 MultiModalAnomalyDetector）上进行超参数搜索，寻找最优配置

### 📌 要求：
- 使用 Optuna 作为调参引擎
- 搜索空间包括：
  - `learning_rate`：1e-5 ~ 1e-2
  - `hidden_dim`：64 ~ 512
  - `dropout`：0.1 ~ 0.5
  - `num_heads`, `num_layers`（可选）
- 数据集为多个子集组合的采样（10%-20%覆盖多场景）
- 每轮结果记录到 wandb（group=tuning，name 自动包含超参组合）
- 输出调参结果表格并保存最优参数配置

---

## 3️⃣ run_ablation() — 模型架构消融实验

### 🎯 目标：
研究多模态模型中各分支模块（GNN/BERT/结构化）的独立贡献

### 📌 要求：
- 固定为主模型结构（如 GNN + Transformer + BERT）
- 在 config 中添加以下模块开关：
  - `use_gnn`、`use_bert`、`use_structured`、`use_temporal`
- 每轮实验移除一个模块，对比其性能变化
- 在 WandB 中记录每次模块组合
- 输出：
  - 模块组合 vs F1 表格
  - Attention 热图（不同特征的注意力强度）
  - 特征重要性排序（如 SHAP）

---

## 4️⃣ run_imbalance_eval() — 数据不平衡适应性实验

### 🎯 目标：
评估模型在不同程度的数据失衡下的鲁棒性表现

### 📌 要求：
- 构建 5 组不同正负样本比例数据集：
  - 1:1, 2:1, 3:1, 4:1, 5:1（正常:威胁）
- 采样策略：
  - 欠采样多数类 + SMOTE 过采样（可选）
- 每组训练一个模型，记录 F1 / AUC
- 绘制：
  - Imbalance Ratio vs F1 曲线
  - 不同采样策略对比柱状图

---

## ✅ 全局配置需求

- 所有实验通过 `--config config/xxx.json` 加载参数
- 所有实验记录 WandB：
  - `project="threat_detection_experiments"`
  - `group=experiment_type`
  - `name="{experiment_type}_{model_type}_{timestamp}"`
  - 每轮添加 `"experiment_type": "baseline/tuning/ablation/imbalance"`

---

## ✅ 脚本接口和目录结构建议

```bash
python main_experiment.py --run_type baseline --config configs/base_config.json
python main_experiment.py --run_type tune --config configs/tune_config.json
python main_experiment.py --run_type ablation --config configs/ablation_config.json
python main_experiment.py --run_type imbalance --config configs/imbalance_config.json
```

experiments/
├── main_experiment.py
├── configs/
│   ├── base_config.json
│   ├── tune_config.json
│   ├── ablation_config.json
│   └── imbalance_config.json
├── utils/
│   └── wandb_utils.py  ← 包含 init_wandb(), log_config()
├── results/
│   ├── baseline_results.csv
│   ├── tuning_trials.csv
│   ├── ablation_analysis.csv
│   └── imbalance_curve.png

# 补充
# 🧪 多模态异常检测：泛化能力评估实验设计（run_generalization_eval）

本实验用于评估当前主模型（如 MultiModalAnomalyDetector）在不同用户组合下的泛化能力是否稳定。

## 🎯 实验目标
> 在不同用户子集下重复训练和验证，观测模型性能是否稳定，是否具有良好的泛化能力。
---

## ✅ 实验流程函数：`run_generalization_eval()`

请在 `main_experiment.py` 中新增此实验函数：

```python
def run_generalization_eval():
    ...

数据采样逻辑

从完整 CERT 用户池中，每轮随机采样 max_users=200 个用户,采样率为1
进行 N 轮重复实验（建议 N=5）；
每轮构建训练/验证集，完整训练并评估模型；
每轮使用相同的模型结构和超参数配置（固定 config）；
评估指标

每轮记录以下指标：

F1 Score
AUC
Precision
Recall
最终输出：

所有轮次的指标表格
平均值 ± 标准差
评估指标

WandB 日志记录

project: "threat_detection_experiments"
group: "generalization"
name: "gen_eval_seed_{i}"
可附加字段:
{
  "experiment_name": "generalization",
  "random_seed": 123,
  "max_users": 200
}
输出文件建议:
结果保存到:
results/
├── gen_eval_summary.csv
├── gen_eval_f1_auc_boxplot.png  # 可选
目录结构部分:
experiments/
├── main_experiment.py
├── configs/
│   └── gen_config.json
├── results/
│   ├── gen_eval_summary.csv
│   └── gen_eval_f1_auc_boxplot.png
注意事项:
每轮采样可使用 random.seed(seed) 确保稳定性；
可复用已有的模型构建与评估模块；
输出标准差建议用 numpy.std；
可视化建议使用 matplotlib 或 seaborn。