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

### 4.1 执行实验
根据您的项目配置，运行主实验脚本。
请确保 `--config_path` 指向您在 Colab 环境中的配置文件路径。
同时，根据需要调整 `--run_type` (例如 `train`, `evaluate`, `train_and_evaluate`, `ablation_study`)。

```python
%cd /content/Mycert

# 示例命令，请根据您的实际情况修改
# 确保配置文件路径正确，例如 /content/Mycert/configs/your_config.json
# 如果您的数据不在默认位置，可能需要通过参数指定数据路径

!python main_experiment.py \
    --config_path "/content/Mycert/configs/gen_config.json" \
    --run_type "train_and_evaluate" \
    # --data_dir "/content/Mycert/data" # 如果需要，取消注释并设置数据目录
    # --output_dir drive_results_path # 如果希望脚本直接输出到Drive，请确保脚本支持此参数
```
**注意：**
1.  确保 `gen_config.json` (或您使用的配置文件) 中的路径是 Colab 环境中的绝对路径 (例如 `/content/Mycert/data/...`) 或者相对于 `main_experiment.py` 的正确相对路径。
2.  如果您的脚本会将结果保存在特定文件夹 (如 `results/`)，您可能需要后续步骤将这些结果复制到 Google Drive。

---

## 第五部分：结果展示与可视化

### 5.1 加载实验结果 (示例)
假设您的实验结果 (例如，一个 CSV 文件) 保存在项目的 `results/` 目录下。

```python
import pandas as pd
import matplotlib.pyplot as plt
import os

# 假设结果CSV文件路径，请根据您的脚本输出进行修改
results_csv_path = "/content/Mycert/results/experiment_metrics.csv" # 修改为实际路径
drive_results_path = "/content/drive/MyDrive/experiment_results/Mycert_outputs" # 之前定义的Drive路径

if os.path.exists(results_csv_path):
    df_results = pd.read_csv(results_csv_path)
    print("实验结果概览:")
    print(df_results.head())

    # 示例：绘制某个指标的变化 (假设CSV中有 'epoch' 和 'loss' 列)
    if 'epoch' in df_results.columns and 'loss' in df_results.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['epoch'], df_results['loss'], marker='o')
        plt.title('训练损失变化')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 保存图像到项目目录
        plot_filename = "training_loss_plot.png"
        plot_save_path = os.path.join("/content/Mycert/results/", plot_filename) # 保存到项目内部
        plt.savefig(plot_save_path)
        print(f"损失曲线图已保存到: {plot_save_path}")
        
        plt.show()
    else:
        print("未能找到 'epoch' 或 'loss' 列用于绘图。")
else:
    print(f"结果文件 {results_csv_path} 未找到。请检查脚本输出和文件路径。")

```

### 5.2 其他可视化
根据您的项目需求，添加更多的可视化代码单元。例如，混淆矩阵、ROC 曲线等。

---

## 第六部分：保存输出到 Google Drive (如果已挂载)

如果实验脚本没有直接将所有输出保存到 Google Drive，您可以使用以下命令手动复制。

```python
import shutil
import os

project_root = "/content/Mycert/"
drive_output_dir = "/content/drive/MyDrive/experiment_results/Mycert_outputs" # 确保此路径与第三部分一致

# 要复制的文件或文件夹列表 (相对于 project_root)
items_to_copy = {
    "configs/gen_config.json": "gen_config_used.json", # 复制并重命名配置文件
    "results/experiment_metrics.csv": "experiment_metrics.csv",   # 假设的指标文件
    "results/training_loss_plot.png": "training_loss_plot.png"    # 假设的图像文件
    # "results/": "all_results_backup" # 也可以复制整个文件夹
}

# 确保 Google Drive 已挂载
if os.path.exists('/content/drive/MyDrive'):
    for item_rel_path, target_name in items_to_copy.items():
        source_path = os.path.join(project_root, item_rel_path)
        destination_path = os.path.join(drive_output_dir, target_name)
        
        if os.path.exists(source_path):
            try:
                if os.path.isdir(source_path):
                    # 如果目标已存在且是目录，先删除，shutil.copytree不覆盖
                    if os.path.exists(destination_path):
                        shutil.rmtree(destination_path)
                    shutil.copytree(source_path, destination_path)
                    print(f"文件夹已复制: {source_path} -> {destination_path}")
                else:
                    shutil.copy2(source_path, destination_path) # copy2保留元数据
                    print(f"文件已复制: {source_path} -> {destination_path}")
            except Exception as e:
                print(f"复制 {source_path} 到 {destination_path} 失败: {e}")
        else:
            print(f"源文件/文件夹未找到: {source_path}")
else:
    print("Google Drive 未挂载。跳过保存到 Drive 的步骤。")

print(f"请检查 Google Drive 目录: {drive_output_dir}")
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