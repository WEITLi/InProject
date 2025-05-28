# 改进版基线模型说明文档

## 问题背景

在原始的基线模型实验中，Random Forest和XGBoost的测试结果过于相似，这可能表明：

1. **相同的特征工程**：两个模型使用了完全相同的特征提取方法
2. **相似的参数设置**：都使用默认参数，缺乏差异化调优
3. **评估方法问题**：单次划分可能导致结果不稳定
4. **特征不够丰富**：简单的统计特征无法发挥不同算法的优势

## 改进方案

### 1. 差异化特征工程

#### Random Forest特征策略
- **丰富特征集**：提取100+特征，包括：
  - 详细统计特征（均值、标准差、分位数、偏度、峰度等）
  - 时间序列特征（趋势、自相关、突发检测等）
  - 交互特征（特征间的乘积和比值）
  - 多维度特征（每个原始维度的独立统计）
- **特征预处理**：标准化处理，填充缺失值
- **设计理念**：利用Random Forest对特征数量不敏感的特点

#### XGBoost特征策略
- **精简特征集**：60-80个核心特征，包括：
  - 基础统计特征
  - 部分原始序列特征（利用XGBoost的特征选择能力）
  - 保留缺失值（测试XGBoost的缺失值处理能力）
- **最小预处理**：不进行标准化，保持原始数值范围
- **设计理念**：让XGBoost自动进行特征选择和重要性评估

### 2. 差异化参数优化

#### Random Forest参数网格（保守策略）
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}
```

#### XGBoost参数网格（激进策略）
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10, 12],      # 更深的树
    'learning_rate': [0.1, 0.2, 0.3], # 更高的学习率
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}
```

### 3. 改进的评估方法

- **5折分层交叉验证**：替代单次train/test划分
- **多指标评估**：F1、AUC、PR-AUC、精确率、召回率
- **二分类专用指标**：使用`average='binary'`而非`average='weighted'`
- **统计显著性**：报告均值±标准差

## 使用方法

### 方法1：通过main_experiment.py使用

```bash
# 使用改进版baseline模型
python main_experiment.py \
    --run_type baseline \
    --use_improved_baseline \
    --baseline_cv_folds 5 \
    --max_users 100

# 对比原始和改进版
python main_experiment.py \
    --run_type baseline \
    --max_users 100  # 不加--use_improved_baseline使用原始版本
```

### 方法2：使用专用运行脚本

```bash
# 使用默认参数
python run_improved_baseline.py

# 自定义参数
python run_improved_baseline.py \
    --max_users 200 \
    --baseline_cv_folds 10 \
    --output_dir ./results/my_test

# 使用配置文件
python run_improved_baseline.py \
    --config_file configs/baseline_config.yaml
```

### 方法3：在代码中直接调用

```python
from experiments.utils.improved_baseline_models import run_improved_baseline_comparison

# 准备多模态数据
training_data = {...}  # 你的训练数据

# 运行改进版baseline对比
results = run_improved_baseline_comparison(
    multimodal_data=training_data,
    output_dir="./results",
    models=["random_forest", "xgboost"],
    cv_folds=5
)

# 查看结果
for model_name, result in results.items():
    if 'error' not in result:
        cv_results = result['cv_results']
        print(f"{model_name}:")
        print(f"  F1: {cv_results['f1_test_mean']:.4f} ± {cv_results['f1_test_std']:.4f}")
        print(f"  AUC: {cv_results['roc_auc_test_mean']:.4f} ± {cv_results['roc_auc_test_std']:.4f}")
        print(f"  特征数: {result['n_features']}")
```

## 预期效果

### 性能差异
- **F1分数差异**：预期>0.02的差异
- **特征数量**：Random Forest ~100+特征，XGBoost ~60-80特征
- **参数选择**：体现各算法的特性差异

### 结果解释
- **Random Forest**：通过丰富特征和集成学习获得稳定性能
- **XGBoost**：通过梯度提升和内置特征选择获得效率和精度
- **交叉验证**：提供更可靠的性能估计和统计显著性

### 输出示例
```
🔬 测试改进版基线模型: random_forest
📊 数据准备完成: 100 样本, 127 特征
🔧 为 random_forest 优化超参数...
✅ 最佳参数: {'max_depth': 15, 'n_estimators': 200, ...}
🔄 开始 5 折交叉验证...
✅ random_forest 交叉验证完成
   测试集 F1: 0.7234 ± 0.0456
   测试集 AUC: 0.8123 ± 0.0234

🔬 测试改进版基线模型: xgboost  
📊 数据准备完成: 100 样本, 73 特征
🔧 为 xgboost 优化超参数...
✅ 最佳参数: {'learning_rate': 0.2, 'max_depth': 8, ...}
🔄 开始 5 折交叉验证...
✅ xgboost 交叉验证完成
   测试集 F1: 0.7456 ± 0.0389
   测试集 AUC: 0.8267 ± 0.0198

📊 改进版基线模型对比结果
🔬 RANDOM_FOREST:
   F1 Score: 0.7234 ± 0.0456
   特征数:   127

🔬 XGBOOST:
   F1 Score: 0.7456 ± 0.0389  
   特征数:   73
```

## 技术细节

### 特征工程差异
1. **Random Forest特征**：
   - 交互特征：`seq_mean * struct_mean`
   - 比值特征：`seq_mean / (struct_mean + 1e-8)`
   - 时间序列特征：趋势、自相关、突发检测
   - 维度级特征：每个原始维度的独立统计

2. **XGBoost特征**：
   - 原始序列特征：部分展平的序列值
   - 基础统计特征：均值、标准差、最值
   - 保留缺失值：测试内置缺失值处理
   - 避免过度工程化：让算法自主学习

### 参数优化策略
- **网格搜索**：使用StratifiedKFold确保类别平衡
- **评分标准**：F1分数（二分类）
- **并行计算**：充分利用多核CPU

### 评估改进
- **分层交叉验证**：保持每折中的类别比例
- **多指标报告**：全面评估模型性能
- **统计分析**：均值和标准差提供可靠性评估

## 故障排除

### 常见问题
1. **内存不足**：减少`max_users`或特征数量
2. **计算时间过长**：减少`cv_folds`或参数网格大小
3. **特征提取失败**：检查输入数据格式
4. **交叉验证失败**：确保数据中有足够的正负样本

### 调试建议
1. 先用小数据集测试（`max_users=20`）
2. 检查特征提取的中间结果
3. 验证交叉验证的数据划分
4. 监控内存和CPU使用情况

## 扩展建议

1. **添加更多模型**：LightGBM、CatBoost等
2. **特征选择**：使用SelectKBest等方法
3. **集成学习**：结合多个基线模型
4. **超参数优化**：使用Optuna等更高级的优化方法 