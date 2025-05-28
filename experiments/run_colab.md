# Google Colab 运行指南：多模态异常检测项目

本文档提供了在 Google Colab 环境中设置和运行多模态异常检测项目的步骤。

---

## 零、准备工作：上传项目文件

请将您的项目文件上传到 Colab 的 `/content/` 目录，或从版本控制系统 (如 Git) 克隆。
建议的项目根目录为 `/content/Mycert`。

如果您的项目在 GitHub 上，可以使用以下命令克隆 (请替换为您的仓库地址):
```python
# !git clone <YOUR_REPOSITORY_URL> /content/Mycert
# %cd /content/Mycert
```

确保项目结构大致如下：
```
/content/Mycert/
├── main_experiment.py
├── requirements.txt
├── core_logic/
│   ├── __init__.py
│   ├── models/  # 存放模型的目录
│   │   └── # (例如: your_model_file.py, __init__.py)
│   └── ...      # (例如: multimodal_pipeline.py)
├── utils/
│   ├── __init__.py
│   └── ...      # (例如: wandb_utils.py)
├── configs/
│   └── gen_config.yaml  # 或其他 YAML 配置文件
└── data/  # 建议的数据存放位置
# 可选 (如果您的项目使用):
# ├── run_experiment.py
# ├── experiment_runners/
```

---

## 第一部分：环境准备与依赖安装

### 1.1 安装项目依赖

此步骤将根据 `requirements.txt` 文件安装所有必要的 Python 包。
```python
%cd /content/Mycert
!pip install -r requirements.txt
!pip install wandb optuna torch dask matplotlib # 确保这些核心库被安装
```

### 1.2 检查 CUDA (如果使用 GPU)
```python
import torch

if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("CUDA not available. Using CPU.")
    device = torch.device("cpu")
```

---

## 第二部分：项目文件与目录结构

### 2.1 核心文件检查 (可选)
您可以运行以下代码来验证关键文件是否存在。
```python
import os

# 定义项目根目录
base_path = "/content/Mycert/"

# 定义需要检查的核心文件和目录
# 类型可以是 "file" 或 "dir"
expected_items = {
    "main_experiment.py": "file",
    "requirements.txt": "file",
    "core_logic/": "dir",
    "core_logic/__init__.py": "file", # 确保 core_logic 是一个包
    "core_logic/models/": "dir",    # 确保模型目录存在
    "utils/": "dir",
    "utils/__init__.py": "file",      # 确保 utils 是一个包
    "configs/": "dir",
    "configs/gen_config.yaml": "file" # 示例配置文件，请根据您的实际情况修改
    # 如果您有 run_experiment.py 或 experiment_runners/ 目录且它们是必须的，也请加入检查
    # "run_experiment.py": "file",
    # "experiment_runners/": "dir",
}

print(f"检查项目根目录: {base_path}\n")

all_found = True
for item_path, item_type in expected_items.items():
    full_path = os.path.join(base_path, item_path)
    if item_type == "file":
        if os.path.isfile(full_path):
            print(f"Found file: {full_path}")
        else:
            print(f"Missing file: {full_path}")
            all_found = False
    elif item_type == "dir":
        if os.path.isdir(full_path):
            print(f"Found directory: {full_path}")
        else:
            print(f"Missing directory: {full_path}")
            all_found = False

print("\n--- 文件结构检查完成 ---")
if all_found:
    print("所有核心文件和目录均已找到。")
else:
    print("警告：部分核心文件或目录缺失，请检查上传的项目文件。")


if not os.path.exists(os.path.join(base_path, "data")):
    print("\nWarning: data/ directory not found at project root. Please ensure your data is accessible from the expected location for your scripts.")
```

### 2.2 .gitignore (供参考)
如果您是从本地上传项目，以下 `.gitignore` 内容可供参考，以避免上传不必要的文件：
```
results/
.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
*.log
*.csv
*.png
*.jpeg
*.jpg
wandb/
```
**注意**: 在 Colab 环境中，`.gitignore` 主要用于版本控制 (如 Git)。如果您是手动上传文件，则不需要此文件。

---

## 第三部分：挂载 Google Drive (可选)

如果您希望将实验结果（如配置文件、CSV 文件、图像）保存到 Google Drive，请运行以下代码挂载您的 Drive。

```python
from google.colab import drive
drive.mount('/content/drive')
```

创建用于保存实验结果的目录：
```python
import os

# 建议的 Drive 中的结果保存路径
drive_results_path = "/content/drive/MyDrive/experiment_results/Mycert_outputs" # 您可以修改此路径
os.makedirs(drive_results_path, exist_ok=True)
print(f"实验结果将保存到: {drive_results_path}")
```

---

## 第四部分：运行实验脚本

### 4.1 执行单个实验 (示例)
根据您的项目配置，运行主实验脚本。
请确保 `--config_path` 指向您在 Colab 环境中的配置文件路径。
同时，根据需要调整 `--run_type` (例如 `train`, `evaluate`, `train_and_evaluate`, `ablation_study`)。

```python
%env MPLBACKEND=Agg
%cd /content/Mycert

# 示例命令，请根据您的实际情况修改
# 确保配置文件路径正确，例如 configs/your_config.yaml (相对路径)
# 或 /content/Mycert/configs/your_config.yaml (绝对路径)
# 如果您的数据不在默认位置，可能需要通过参数指定数据路径

!python main_experiment.py \
    --config_path "configs/gen_config.yaml" \
    --run_type "train_and_evaluate" \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs" \
    # --data_dir "data" # 如果需要，取消注释并设置数据目录 (relative to Mycert)
```
**注意：**
1.  确保配置文件 (`.yaml` 或 `.json`) 中的路径是 Colab 环境中的绝对路径 (例如 `/content/Mycert/data/...`) 或者相对于 `main_experiment.py` 的正确相对路径。
2.  `--output_dir` 参数指定了实验输出的保存位置。请确保您的 `main_experiment.py` 脚本支持此参数。如果 Google Drive 未挂载或不想直接输出到 Drive，您可以注释掉此行或修改为本地路径 (如 `results/`)。
3.  如果脚本将结果保存在 `--output_dir` 指定的文件夹外（例如，固定的 `results/` 目录），您仍可能需要第六部分的步骤来将这些结果复制到 Google Drive。

### 4.2 更多实验示例

以下是一些运行不同类型实验的示例命令。请根据您的需求调整配置文件和参数。
所有示例均已添加 `--output_dir` 参数，将输出指向之前定义的 Google Drive 路径。

#### 1. 训练和评估 (通用配置)
```python
%env MPLBACKEND=Agg
%cd /content/Mycert
!python main_experiment.py \
    --config_path "configs/gen_config.yaml" \
    --run_type "train_and_evaluate" \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs"
```

#### 2. 消融研究 (使用消融配置)
```python
%env MPLBACKEND=Agg
%cd /content/Mycert
!python main_experiment.py \
    --config_path "configs/ablation_config.yaml" \
    --run_type "ablation_study" \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs"
```

#### 3. 超参数调优 (使用调优配置)
```python
%env MPLBACKEND=Agg
%cd /content/Mycert
!python main_experiment.py \
    --config_path "configs/tune_config.yaml" \
    --run_type "hyperparameter_tuning" \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs" # 假设有这个run_type，请根据您的脚本实现调整
```

#### 4. 基线模型实验 (使用基线配置)
```python
%env MPLBACKEND=Agg
%cd /content/Mycert
!python main_experiment.py \
    --config_path "configs/baseline_config.yaml" \
    --run_type "train_and_evaluate" \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs" # 或其他适合基线的run_type
```

#### 5. 改进版基线模型实验 (推荐)
```python
%env MPLBACKEND=Agg
%cd /content/Mycert
# 使用改进版baseline模型，具有差异化特征工程和交叉验证
!python main_experiment.py \
    --run_type "baseline" \
    --use_improved_baseline \
    --baseline_cv_folds 5 \
    --max_users 100 \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs"
```

#### 6. 改进版基线模型对比实验
```python
%env MPLBACKEND=Agg
%cd /content/Mycert
# 先运行原始baseline
!python main_experiment.py \
    --run_type "baseline" \
    --max_users 100 \
    --experiment_name "original_baseline_comparison" \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs"

# 再运行改进版baseline
!python main_experiment.py \
    --run_type "baseline" \
    --use_improved_baseline \
    --baseline_cv_folds 5 \
    --max_users 100 \
    --experiment_name "improved_baseline_comparison" \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs"
```

#### 7. 使用专用改进版baseline脚本
```python
%env MPLBACKEND=Agg
%cd /content/Mycert
# 使用专门的改进版baseline运行脚本
!python run_improved_baseline.py \
    --max_users 200 \
    --baseline_cv_folds 10 \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs"
```

#### 8. 不平衡数据处理实验 (使用不平衡配置)
```python
%env MPLBACKEND=Agg
%cd /content/Mycert
!python main_experiment.py \
    --config_path "configs/imbalance_config.yaml" \
    --run_type "train_and_evaluate" \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs" # 或其他适合的run_type
```

#### 9. 快速测试 (使用快速测试配置)
```python
%env MPLBACKEND=Agg
%cd /content/Mycert
!python main_experiment.py \
    --config_path "configs/quick_test.yaml" \
    --run_type "train_and_evaluate" \
    --output_dir "/content/drive/MyDrive/experiment_results/Mycert_outputs" # 通常用于快速验证
```
**提示** mobilizing:
*   请确保上述配置文件 (`gen_config.yaml`, `ablation_config.yaml`, `tune_config.yaml`, `baseline_config.yaml`, `imbalance_config.yaml`, `quick_test.yaml`) 存在于您的 `/content/Mycert/configs/` 目录中。
*   `--run_type` 的具体可用值取决于您的 `main_experiment.py` 脚本的实现。请参考您的脚本或相关文档。例如，超参数调优的 `run_type` 可能是 `tune` 或其他特定名称。
*   确保您的 `main_experiment.py` 脚本能够识别并使用 `--output_dir` 参数来保存所有相关输出 (如模型、日志、结果CSV、图像等)。
*   如果实验需要指定特定的数据子集或输出目录，您可能需要添加如 `--data_version_suffix "subset_xyz"` 或 `--output_sub_dir "experiment_abc"` 等额外参数 (具体参数名需根据您的脚本确定)。

---

## 第五部分：结果展示与可视化

### 5.1 浏览实验输出目录
在分析具体实验结果之前，建议先浏览一下 Google Drive 上的主要输出目录，了解实际生成了哪些子目录和文件。

```python
import os

# Google Drive 上的主要输出目录 (与第三部分和第四部分一致)
drive_results_base_path = "/content/drive/MyDrive/experiment_results/Mycert_outputs"

if os.path.exists(drive_results_base_path):
    print(f"主要实验输出目录: {drive_results_base_path}")
    print("目录内容:")
    for item in os.listdir(drive_results_base_path):
        print(f"  - {item}")
else:
    print(f"错误: 主要实验输出目录 {drive_results_base_path} 未找到。请确保实验已正确运行并将结果保存到此路径。")

# 您可以进一步列出特定子目录的内容
# expected_sub_dirs = ["baseline_experiment", "ablation_experiment", "tune_experiment", "imbalance_experiment"]
# for sub_dir_name in expected_sub_dirs:
#     sub_dir_path = os.path.join(drive_results_base_path, sub_dir_name)
#     if os.path.exists(sub_dir_path) and os.path.isdir(sub_dir_path):
#         print(f"\n内容 {sub_dir_path}:")
#         for item in os.listdir(sub_dir_path):
#             print(f"  - {item}")
#     else:
#         print(f"\n警告: 子目录 {sub_dir_path} 未找到或不是一个目录。")
```

### 5.2 加载和分析特定实验的结果 (示例)
以下代码演示了如何从 Google Drive 上的特定实验子目录加载结果 (例如 `experiment_name_results.json` 文件)，并进行可视化。
请根据您实际生成的文件名和 JSON 结构进行调整。

```python
import pandas as pd
import matplotlib.pyplot as plt
import os
import json # 新增导入 json 模块

# Google Drive 上的主要输出目录
drive_results_base_path = "/content/drive/MyDrive/experiment_results/Mycert_outputs"

# 定义要分析的实验子目录和预期的JSON结果文件名
# 您可以根据需要修改或扩展此列表
experiments_to_analyze = {
    "baseline_experiment": "baseline_experiment_results.json",
    "ablation_experiment": "ablation_experiment_results.json",
    "tune_experiment": "tune_experiment_results.json",
    "imbalance_experiment": "imbalance_experiment_results.json"
}

for exp_name, json_filename in experiments_to_analyze.items():
    exp_results_path = os.path.join(drive_results_base_path, exp_name)
    json_file_path = os.path.join(exp_results_path, json_filename)
    
    print(f"\n--- 分析实验: {exp_name} ---")
    
    if os.path.exists(json_file_path):
        print(f"找到结果文件: {json_file_path}")
        try:
            # 读取 JSON 文件
            with open(json_file_path, 'r') as f:
                results_data = json.load(f)
            
            df_results = None # Initialize df_results
            # 尝试将结果数据转换为 Pandas DataFrame 以方便处理
            if isinstance(results_data, list):
                df_results = pd.DataFrame(results_data)
            elif isinstance(results_data, dict):
                if exp_name == "baseline_experiment":
                    if "comparison_summary" in results_data and isinstance(results_data["comparison_summary"], dict):
                        df_results = pd.DataFrame([{
                            'model': model_name,
                            'f1_score': metrics.get('f1_score'),
                            'auc_score': metrics.get('auc_score')
                        } for model_name, metrics in results_data["comparison_summary"].items()])
                    else:
                        print(f"  'comparison_summary' key not found or is not a dict in {json_filename} for baseline_experiment.")
                        continue
                elif exp_name == "abalation_experiment":
                    if "combinations" in results_data and isinstance(results_data["combinations"], dict):
                        df_results = pd.DataFrame([{
                            'combination': combo_name,
                            'best_val_f1': combo_data.get('best_val_f1'),
                            'last_val_auc': combo_data.get('train_history', {}).get('val_auc', [None])[-1] if combo_data.get('train_history', {}).get('val_auc') else None,
                            'modalities': "_".join(combo_data.get('modalities', [])) # Join modalities for easier display
                        } for combo_name, combo_data in results_data["combinations"].items()])
                    else:
                        print(f"  'combinations' key not found or is not a dict in {json_filename} for ablation_experiment.")
                        continue
                elif exp_name == "tune_experiment":
                    score = None
                    if 'best_score' in results_data:
                        score = results_data['best_score']
                    elif 'tuning_results' in results_data and isinstance(results_data['tuning_results'], dict) and 'best_value' in results_data['tuning_results']:
                        score = results_data['tuning_results']['best_value']
                    
                    if score is not None:
                        df_results = pd.DataFrame([{
                            'experiment_name': exp_name, # Used for x-axis in bar plot
                            'best_score': score
                        }])
                    else:
                        print(f"  'best_score' or 'tuning_results.best_value' not found in {json_filename} for tune_experiment.")
                        continue
                elif exp_name == "imbalance_experiment":
                    if "summary_statistics" in results_data and isinstance(results_data["summary_statistics"], list):
                        df_results = pd.DataFrame(results_data["summary_statistics"])
                    elif "all_run_metrics" in results_data and isinstance(results_data["all_run_metrics"], list):
                        df_results = pd.DataFrame(results_data["all_run_metrics"])
                    else:
                        print(f"  Neither 'summary_statistics' nor 'all_run_metrics' found or not a list in {json_filename} for imbalance_experiment.")
                        continue
                else:
                    print(f"  Specific parsing logic for experiment type '{exp_name}' (when data is dict) is not defined. Trying generic dict to DataFrame conversion.")
                    try:
                        # Attempt a generic conversion if the structure is flat enough or a list of records under a known key
                        # This is a fallback and might need specific handling if it fails often
                        if len(results_data) == 1 and isinstance(list(results_data.values())[0], list):
                             df_results = pd.DataFrame(list(results_data.values())[0])
                        else:
                             df_results = pd.DataFrame([results_data]) # Wrap dict in a list
                    except Exception as e_conv:
                        print(f"  Could not convert dict to DataFrame for {exp_name}: {e_conv}")
                        continue
            else:
                print(f"  无法解析 {json_filename}，JSON 结构既不是列表也不是字典。")
                continue

            if df_results is None or df_results.empty:
                print(f"  DataFrame is empty or None after parsing {json_filename}. Skipping further processing for this file.")
                continue

            print("结果概览 (转换后的DataFrame):")
            print(df_results.head())

            # 示例：绘制某个指标的变化
            plot_made = False
            # Plot for Baseline Experiment
            if exp_name == "baseline_experiment" and 'model' in df_results.columns and 'f1_score' in df_results.columns and 'auc_score' in df_results.columns:
                plt.figure(figsize=(10, 6))
                # Ensure scores are numeric, coerce errors to NaN and fill with 0 for plotting
                df_results['f1_score'] = pd.to_numeric(df_results['f1_score'], errors='coerce').fillna(0)
                df_results['auc_score'] = pd.to_numeric(df_results['auc_score'], errors='coerce').fillna(0)
                
                bar_width = 0.35
                index = range(len(df_results['model']))

                plt.bar(index, df_results['f1_score'], bar_width, label='F1 Score')
                plt.bar([i + bar_width for i in index], df_results['auc_score'], bar_width, label='AUC Score', alpha=0.7)
                
                plt.title(f'{exp_name} - Model Performance')
                plt.xlabel('Model')
                plt.ylabel('Score')
                plt.xticks([i + bar_width / 2 for i in index], df_results['model'], rotation=45, ha="right")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                plot_filename = f"{exp_name}_performance_plot.png"
                plot_save_path = os.path.join(exp_results_path, plot_filename)
                plt.savefig(plot_save_path)
                print(f"性能图已保存到: {plot_save_path}")
                plt.show()
                plot_made = True
            # Plot for Tune Experiment
            elif exp_name == "tune_experiment" and 'experiment_name' in df_results.columns and 'best_score' in df_results.columns:
                plt.figure(figsize=(8, 6))
                df_results['best_score'] = pd.to_numeric(df_results['best_score'], errors='coerce').fillna(0)
                plt.bar(df_results['experiment_name'], df_results['best_score'], label='Best Score (F1/Value)')
                plt.title(f'{exp_name} - Best Hyperparameter Tuning Score')
                plt.xlabel('Experiment')
                plt.ylabel('Best Score')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                plot_filename = f"{exp_name}_best_score_plot.png"
                plot_save_path = os.path.join(exp_results_path, plot_filename)
                plt.savefig(plot_save_path)
                print(f"最佳分数图已保存到: {plot_save_path}")
                plt.show()
                plot_made = True
            # Plot for Ablation Experiment
            elif exp_name == "abalation_experiment" and 'combination' in df_results.columns and 'best_val_f1' in df_results.columns:
                plt.figure(figsize=(12, 8)) # Increased figure size
                df_results['best_val_f1'] = pd.to_numeric(df_results['best_val_f1'], errors='coerce').fillna(0)
                df_sorted = df_results.sort_values(by='best_val_f1', ascending=False) # Sort for better viz
                plt.bar(df_sorted['combination'], df_sorted['best_val_f1'], label='Best Validation F1 Score')
                plt.title(f'{exp_name} - Ablation Study Performance')
                plt.xlabel('Modality Combination')
                plt.ylabel('Best Validation F1 Score')
                plt.xticks(rotation=45, ha="right")
                plt.legend()
                plt.grid(axis='y') # Grid only on y-axis for bar charts
                plt.tight_layout()
                
                plot_filename = f"{exp_name}_ablation_f1_plot.png"
                plot_save_path = os.path.join(exp_results_path, plot_filename)
                plt.savefig(plot_save_path)
                print(f"消融实验F1分数图已保存到: {plot_save_path}")
                plt.show()
                plot_made = True
            # Plot for Imbalance Experiment (Summary Statistics)
            elif exp_name == "imbalance_experiment" and 'index' in df_results.columns and ('mean' in df_results['index'].values):
                df_mean_metrics = df_results[df_results['index'] == 'mean'].iloc[0] # Get the 'mean' row as a Series
                metrics_to_plot = {}
                if 'f1_score' in df_mean_metrics:
                     metrics_to_plot['Mean F1 Score'] = pd.to_numeric(df_mean_metrics['f1_score'], errors='coerce').fillna(0)
                if 'auc' in df_mean_metrics:
                     metrics_to_plot['Mean AUC'] = pd.to_numeric(df_mean_metrics['auc'], errors='coerce').fillna(0)
                if 'precision' in df_mean_metrics:
                     metrics_to_plot['Mean Precision'] = pd.to_numeric(df_mean_metrics['precision'], errors='coerce').fillna(0)
                if 'recall' in df_mean_metrics:
                     metrics_to_plot['Mean Recall'] = pd.to_numeric(df_mean_metrics['recall'], errors='coerce').fillna(0)

                if metrics_to_plot:
                    plt.figure(figsize=(10, 6))
                    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
                    plt.title(f'{exp_name} - Mean Performance Metrics (from Summary)')
                    plt.ylabel('Score')
                    plt.xticks(rotation=15, ha="right")
                    plt.grid(axis='y')
                    plt.tight_layout()
                    
                    plot_filename = f"{exp_name}_summary_metrics_plot.png"
                    plot_save_path = os.path.join(exp_results_path, plot_filename)
                    plt.savefig(plot_save_path)
                    print(f"不平衡实验总结指标图已保存到: {plot_save_path}")
                    plt.show()
                    plot_made = True
            # Plot for Imbalance Experiment (All Runs) - if summary wasn't plotted
            elif exp_name == "imbalance_experiment" and 'seed' in df_results.columns and 'f1_score' in df_results.columns:
                plt.figure(figsize=(12, 7))
                df_results['f1_score'] = pd.to_numeric(df_results['f1_score'], errors='coerce').fillna(0)
                if 'auc' in df_results.columns:
                    df_results['auc'] = pd.to_numeric(df_results['auc'], errors='coerce').fillna(0)
                    df_results.set_index('seed')[['f1_score', 'auc']].plot(kind='bar', ax=plt.gca()) # Use current axes
                    plt.legend(['F1 Score', 'AUC'])
                else:
                    df_results.set_index('seed')[['f1_score']].plot(kind='bar', ax=plt.gca())
                    plt.legend(['F1 Score'])
                
                plt.title(f'{exp_name} - Metrics per Run (Seed)')
                plt.xlabel('Run (Seed)')
                plt.ylabel('Score')
                plt.xticks(rotation=0)
                plt.grid(axis='y')
                plt.tight_layout()

                plot_filename = f"{exp_name}_all_runs_metrics_plot.png"
                plot_save_path = os.path.join(exp_results_path, plot_filename)
                plt.savefig(plot_save_path)
                print(f"不平衡实验各轮次指标图已保存到: {plot_save_path}")
                plt.show()
                plot_made = True
            
            if not plot_made:
                print(f"  在 {json_filename} 中未能找到预期的键组合用于绘图。DataFrame 列: {df_results.columns.tolist() if df_results is not None else 'df_results is None'}")

        except Exception as e:
            print(f"读取或处理 {json_file_path} 时出错: {e}")
    else:
        print(f"警告: 结果文件 {json_file_path} 未找到。请检查实验输出。")

```

### 5.3 其他可视化
根据您的项目需求和各个实验子目录中的具体输出文件，添加更多的可视化代码单元。例如：
*   比较不同实验的最终性能指标 (柱状图、表格)。
*   绘制特定实验的混淆矩阵、ROC曲线等。
*   分析模型权重、特征重要性 (如果已保存)。

---

## 第六部分：验证输出与补充保存到 Google Drive

由于实验脚本已配置为通过 `--output_dir` 直接将主要结果保存到 Google Drive，此部分侧重于验证输出的完整性，并补充保存一些可能未自动包含在 `--output_dir` 中的文件，例如运行实验时使用的最终配置文件。

```python
import shutil
import os

# 项目根目录 (Colab环境)
project_root_colab = "/content/Mycert/"
# Google Drive 上的主要输出目录 (与第三部分一致)
drive_results_base_path = "/content/drive/MyDrive/experiment_results/Mycert_outputs"

# 假设您在第四部分运行的实验对应的配置文件和输出子目录
# 请根据实际情况调整
experiments_configs_and_outputs = {
    "baseline_experiment": "configs/baseline_config.yaml",
    "ablation_experiment": "configs/ablation_config.yaml",
    "tune_experiment": "configs/tune_config.yaml",
    "imbalance_experiment": "configs/imbalance_config.yaml",
    # "general_train_eval": "configs/gen_config.yaml" # 如果运行了通用训练评估
}

print("\n--- 验证输出并补充保存配置文件到Google Drive ---")
if not os.path.exists('/content/drive/MyDrive'):
    print("Google Drive 未挂载。跳过此步骤。")
else:
    for exp_name, config_rel_path in experiments_configs_and_outputs.items():
        # 构造配置文件在Colab项目中的完整路径
        config_source_path = os.path.join(project_root_colab, config_rel_path)
        # 构造此实验在Google Drive上的输出子目录路径
        exp_drive_output_dir = os.path.join(drive_results_base_path, exp_name)
        
        print(f"\n处理实验: {exp_name}")
        
        # 1. 检查实验输出子目录是否存在
        if os.path.exists(exp_drive_output_dir) and os.path.isdir(exp_drive_output_dir):
            print(f"  ✅ 找到实验输出目录: {exp_drive_output_dir}")

            # 2. 复制该实验使用的配置文件到其Google Drive输出子目录中
            if os.path.exists(config_source_path):
                config_filename = os.path.basename(config_source_path)
                config_destination_path = os.path.join(exp_drive_output_dir, f"used_{config_filename}")
                try:
                    shutil.copy2(config_source_path, config_destination_path) # copy2 保留元数据
                    print(f"  ✅ 配置文件已复制: {config_source_path} -> {config_destination_path}")
                except Exception as e:
                    print(f"  ❌ 复制配置文件 {config_source_path} 失败: {e}")
            else:
                print(f"  ⚠️ 警告: 配置文件 {config_source_path} 未找到，无法复制。")
            
            # 3. (可选) 检查该目录下是否有一些预期的关键文件
            # expected_files_in_output = ["metrics.csv", "model.pth", "summary.txt"] # 示例
            # for fname in expected_files_in_output:
            #     fpath_in_drive = os.path.join(exp_drive_output_dir, fname)
            #     if os.path.exists(fpath_in_drive):
            #         print(f"    Found expected file: {fpath_in_drive}")
            #     else:
            #         print(f"    Missing expected file: {fpath_in_drive}")

        else:
            print(f"  ⚠️ 警告: 实验 {exp_name} 的输出目录 {exp_drive_output_dir} 未找到。")

print(f"\n请再次检查 Google Drive 目录: {drive_results_base_path} 及其子目录。")
```

---

## 第七部分：W&B 和 Optuna (可选)

如果您在项目中使用 `wandb` 或 `optuna`：

### 7.1 Weights & Biases (wandb)
确保您已登录 `wandb`。在 Colab 中，通常会提示您进行身份验证。
```python
# import wandb
# wandb.login() # 运行此命令会提示输入 API 密钥或通过浏览器登录

# 您的训练脚本中应该已经包含了 wandb.init() 等相关代码
# 例如:
# wandb.init(project="my_multimodal_anomaly_detection", entity="your_wandb_username", config=config_dict)
```
实验结束后，所有日志将自动同步到 `wandb` 服务器。

### 7.2 Optuna
`Optuna` 的研究结果通常保存在内存中或数据库中。如果您的脚本将 Optuna study 保存到文件 (例如 SQLite 数据库)，请确保也将其复制到 Google Drive。
```python
# 示例：如果 Optuna study 保存为 optuna_study.db
# items_to_copy["optuna_study.db"] = "optuna_study.db"
# 然后重新运行第六部分中的复制命令，或单独复制:

# study_db_path = "/content/Mycert/optuna_study.db"
# drive_study_db_path = os.path.join(drive_output_dir, "optuna_study.db")
# if os.path.exists(study_db_path) and os.path.exists('/content/drive/MyDrive'):
#    shutil.copy2(study_db_path, drive_study_db_path)
#    print(f"Optuna study DB 已复制到: {drive_study_db_path}")
```

---
祝您实验顺利！ 