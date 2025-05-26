# 📁 文件架构设计说明

## 🏗️ **目录结构概览**

`feature_extraction_scenario/` 项目目录结构如下：

```
feature_extraction_scenario/
├── core_logic/                    # 核心特征提取逻辑与模型
│   ├── models/                    # 机器学习模型组件
│   │   ├── base_model/            # 基础模型 (Transformer, GNN, Fusion, Head)
│   │   ├── text_encoder/          # 文本编码模型 (BERT)
│   │   └── structure_encoder/     # 结构化数据模型 (LightGBM)
│   ├── train_pipeline/            # 模型训练流水线 (trainer.py, multimodal_model.py)
│   ├── dataset_pipeline.py        # 数据集处理和特征提取主流水线
│   ├── encoder.py                 # 事件编码器
│   ├── utils.py                   # 通用工具函数
│   ├── temporal.py                # 时间特征相关处理
│   ├── user_context.py            # 用户上下文及风险画像
│   ├── config.py                  # 配置文件 (ModelConfig等)
│   ├── email_features.py          # 邮件特征提取 (示例)
│   ├── http_features.py           # HTTP特征提取 (示例)
│   ├── ...                        # 其他特定事件类型特征提取模块
│   └── __init__.py
├── docs/                          # 项目文档
│   ├── FILE_STRUCTURE.md          # 本文件 - 项目文件结构说明
│   ├── PIPELINE_USAGE.md          # 流水线使用指南
│   ├── ARCHITECTURE.md            # 系统架构设计
│   ├── API_REFERENCE.md           # API参考（如果适用）
│   ├── DATA_SAFETY_GUIDE.md       # 数据安全指南
│   ├── SECURITY_UPDATE_SUMMARY.md # 安全更新总结
│   └── README.md                  # docs目录说明
├── output/                        # 存放所有生成的输出文件 (被 .gitignore 忽略)
│   ├── checkpoints/               # 模型训练的检查点
│   ├── exploration_results/       # 数据探索和实验结果
│   ├── tmp/                       # 临时文件
│   └── README.md                  # output目录说明
├── scripts/                       # 可执行脚本、辅助工具脚本
│   ├── data_exploration.py        # (示例) 数据探索脚本
│   ├── run_pipeline_cli.py        # (示例) 命令行运行流水线的脚本
│   └── __init__.py
├── tests/                         # 测试代码
│   ├── sample_test_data/          # 存放测试流程使用的样本数据 (如 .csv 文件)
│   │   └── answers/               # (示例) 测试用恶意用户标签
│   ├── backup_data/               # test_pipeline.py 自动创建的真实数据备份目录
│   ├── test_pipeline.py           # 测试 dataset_pipeline.py
│   ├── test_models.py             # 测试 core_logic/models/ 中的各个模型模块
│   ├── test_complete_system.py    # 测试完整的模型集成和流程
│   └── __init__.py
├── README.md                      # 项目主README文件
├── .gitignore                     # Git忽略配置文件
└── backup_YYYYMMDD_HHMMSS/        # (示例)旧的备份目录 (被 .gitignore 忽略)
```

## 🎯 **核心目录说明**

*   **`core_logic/`**: 包含项目的所有核心业务逻辑。
    *   `models/`: 存放所有机器学习模型的定义。拆分为 `base_model` (可复用组件) 和特定类型的编码器/分支。
    *   `train_pipeline/`: 包含用于训练多模态模型的代码。
    *   `dataset_pipeline.py`: 负责从原始数据到最终特征的完整处理流程。
    *   `encoder.py`, `utils.py`, 等: 提供流水线所需的辅助功能和特定特征提取逻辑。
*   **`docs/`**: 存放所有项目相关的Markdown文档，用于解释项目结构、用法、设计决策等。
*   **`output/`**: 所有由脚本运行产生的输出（如模型、结果、临时文件）都应保存到此目录。此目录通常被添加到 `.gitignore`。
*   **`scripts/`**: 包含一些用于运行特定任务（如数据探索、启动训练、运行特定分析）的独立脚本。
*   **`tests/`**: 包含所有单元测试、集成测试和系统测试。
    *   `sample_test_data/`: 为测试提供小型、可复现的数据集。
    *   `test_pipeline.py`: 专注于测试数据处理和特征提取流水线的正确性，其生成的输出会进入 `feature_extraction_scenario/test_output/` (此目录也被 `.gitignore` 忽略)。

## 🔄 **数据和模型流向 (概念性)**

1.  **原始数据**: 假设位于项目外部或特定的、受控的数据存储位置 (例如 `../../data/r4.2/`)。
2.  **测试数据**: 位于 `tests/sample_test_data/`，用于开发和测试。
3.  **`dataset_pipeline.py` (在 `core_logic/`)**: 
    *   读取原始数据或测试数据。
    *   进行预处理、特征编码 (使用 `encoder.py` 和 `core_logic/*_features.py`)。
    *   生成中间数据 (如 `DataByWeek/`, `NumDataByWeek/`) 和最终的特征级别文件 (如 `WeekLevelFeatures/`)。这些在常规运行时可能输出到项目根目录下的相应文件夹（已被git忽略），在 `test_pipeline.py` 运行时输出到 `test_output/`。
4.  **模型训练 (使用 `core_logic/train_pipeline/` 中的脚本)**:
    *   加载 `dataset_pipeline.py` 生成的特征。
    *   使用 `core_logic/models/` 中定义的模型进行训练。
    *   将训练好的模型检查点保存到 `output/checkpoints/`。
5.  **模型推理/应用**: 
    *   加载 `output/checkpoints/` 中的模型。
    *   对新数据进行特征提取和预测。
    *   结果可能保存到 `output/exploration_results/` 或其他指定位置。

## 🛠️ **脚本和测试运行**

*   **运行核心流水线 (示例)**:
    ```bash
    python feature_extraction_scenario/core_logic/dataset_pipeline.py # (可能需要调整参数或通过包装脚本调用)
    ```
*   **运行测试**:
    ```bash
    # 确保在 InProject/ 目录下，或者PYTHONPATH正确设置
    python -m unittest discover -s feature_extraction_scenario/tests 
    # 或者单独运行测试文件
    python feature_extraction_scenario/tests/test_pipeline.py
    python feature_extraction_scenario/tests/test_models.py
    python feature_extraction_scenario/tests/test_complete_system.py
    ```
*   **运行自定义脚本**:
    ```bash
    python feature_extraction_scenario/scripts/your_script_name.py
    ```

## 📦 **依赖管理**

(建议添加项目的依赖管理方式，例如 `requirements.txt` 或 `conda` 环境文件说明)

---
*此文件结构旨在提高项目的模块化、可维护性和可测试性。* 