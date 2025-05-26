#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-modal Anomaly Detection Model Configuration
多模态异常检测模型配置文件
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ModelConfig:
    """模型配置"""
    # 模块启用控制
    enable_gnn: bool = True          # 是否启用GNN用户图嵌入
    enable_bert: bool = True         # 是否启用BERT文本编码
    enable_lgbm: bool = True         # 是否启用LightGBM结构化特征
    enable_transformer: bool = True   # 是否启用Transformer序列建模
    
    # 模态启用控制 (用于消融实验)
    enabled_modalities: List[str] = field(default_factory=lambda: ["behavior", "graph", "text", "structured"])
    
    # 模型维度
    hidden_dim: int = 256            # 隐藏层维度
    sequence_length: int = 128       # 序列长度
    num_heads: int = 8               # 注意力头数
    num_layers: int = 6              # Transformer层数
    dropout: float = 0.1             # 通用dropout率
    
    # GNN配置
    gnn_hidden_dim: int = 128        # GNN隐藏维度
    gnn_num_layers: int = 3          # GNN层数
    gnn_dropout: float = 0.1         # GNN dropout
    
    # BERT配置
    bert_model_name: str = "bert-base-uncased"  # BERT模型名称
    bert_max_length: int = 512       # BERT最大序列长度
    bert_freeze_layers: int = 0      # 冻结的BERT层数（0表示不冻结）
    
    # LightGBM配置
    lgbm_num_leaves: int = 31        # LightGBM叶子数
    lgbm_max_depth: int = -1         # LightGBM最大深度
    lgbm_learning_rate: float = 0.05  # LightGBM学习率
    lgbm_feature_fraction: float = 0.9  # LightGBM特征采样率
    lgbm_n_estimators: int = 100      # 新增
    lgbm_bagging_fraction: float = 0.8  # 新增
    lgbm_bagging_freq: int = 5        # 新增
    
    # 融合配置
    fusion_type: str = "attention"   # 融合类型：attention, concat, weighted
    fusion_hidden_dim: int = 128     # 融合层隐藏维度
    
    # 分类头配置
    num_classes: int = 2             # 分类数（正常/异常）
    head_dropout: float = 0.3        # 分类头dropout

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 32             # 批大小
    learning_rate: float = 1e-4      # 学习率
    epochs: int = 3                # 训练轮数 (别名)
    num_epochs: int = 3           # 训练轮数
    warmup_steps: int = 1000         # 预热步数
    weight_decay: float = 1e-5       # 权重衰减
    device: str = "auto"             # 训练设备
    
    # 优化器配置
    optimizer: str = "adamw"         # 优化器类型
    scheduler: str = "cosine"        # 学习率调度器
    
    # 早停和保存
    patience: int = 10               # 早停patience
    save_top_k: int = 3             # 保存最好的k个模型
    monitor_metric: str = "val_f1"   # 监控指标
    
    # 验证和测试
    val_check_interval: int = 100    # 验证间隔
    test_split: float = 0.2         # 测试集比例
    val_split: float = 0.2          # 验证集比例
    
    # 数据增强
    enable_data_augmentation: bool = True  # 是否启用数据增强
    augmentation_prob: float = 0.3   # 数据增强概率

@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "../data/r4.2"  # 默认数据集路径 (CERT r4.2)
    processed_dir: str = "./processed_data"  # 处理后数据的保存目录
    max_sequence_length: int = 128  # 行为序列的最大长度
    min_events_per_user: int = 10  # 用户最少事件数，低于此值的用户将被过滤
    max_users: Optional[int] = None  # 最大用户数，None表示不限制
    
    numerical_features: List[str] = field(default_factory=lambda: ['hour', 'day_of_week', 'event_count', 'unique_files', 'unique_urls'])
    categorical_features: List[str] = field(default_factory=lambda: ['event_type', 'department', 'role'])
    text_features: List[str] = field(default_factory=lambda: ['email_content', 'url_content', 'file_name'])
    
    time_window_days: int = 7  # 时间窗口大小（天）
    overlap_ratio: float = 0.5  # 窗口重叠比例
    
    # 新增数据处理相关属性 (for run_experiment and pipeline control)
    data_version: str = "r4.2"
    source_dir: str = "../data/r4.2" # CERT r4.2的源路径, 相对于experiments目录，向上1级到InProject
    work_dir_base: str = "./pipeline_workdir" # 流水线自身的工作数据保存基目录
    start_week: int = 0
    end_week: Optional[int] = None # e.g. 5 means weeks 0,1,2,3,4
    max_weeks: int = 72  # 最大周数，默认为72周
    sequence_length: int = 128 # Should be consistent with max_sequence_length
    feature_dim: int = 256 # Default feature dimension for encoder, matches MultiModalDataPipeline constructor
    num_cores: int = 8 # Default num_cores for pipeline parallel tasks
    sample_ratio: float = 1.0 # Sampling ratio for CSV reading, 1.0 means no sampling

    # 强制重新生成标志
    force_regenerate_combined_weeks: bool = False
    force_regenerate_analysis_levels: bool = False # CSV files from step4_multi_level_analysis
    force_regenerate_user_info: bool = False     
    force_regenerate_event_features: bool = False
    force_regenerate_analysis: bool = False      
    force_regenerate_training_data: bool = False 
    force_regenerate_graphs: bool = False        
    force_regenerate_sequences: bool = False     
    force_regenerate_structured_features: bool = False
    force_regenerate_text_data: bool = False # For extracted text data

@dataclass
class GeneralizationConfig:
    """泛化能力评估实验配置"""
    num_gen_runs: int = 5  # 泛化实验运行次数
    base_seed: int = 42    # 泛化实验基础种子

@dataclass
class Config:
    """完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generalization: GeneralizationConfig = field(default_factory=GeneralizationConfig) # 新增泛化配置
    
    # 设备和环境
    device: str = "cuda"             # 设备类型
    num_workers: int = 4             # 数据加载器工作进程数
    seed: int = 42                   # 随机种子
    
    # 日志和输出
    log_dir: str = "./logs"          # 日志目录
    output_dir: str = "./outputs"    # 输出目录
    experiment_name: str = "multimodal_anomaly_detection"  # 实验名称
    
    # 调试模式
    debug: bool = False              # 是否为调试模式
    fast_dev_run: bool = False       # 快速开发模式（只运行少量数据）

def get_config() -> Config:
    """获取默认配置"""
    return Config()

def load_config_from_file(config_path: str) -> Config:
    """从文件加载配置"""
    import json
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 这里可以添加更复杂的配置加载逻辑
    config = Config()
    
    # 递归更新配置
    def update_config(config_obj, config_dict):
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                if isinstance(value, dict):
                    update_config(getattr(config_obj, key), value)
                else:
                    setattr(config_obj, key, value)
    
    update_config(config, config_dict)
    return config

def save_config_to_file(config: Config, config_path: str):
    """保存配置到文件"""
    import json
    from dataclasses import asdict
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

# 默认配置实例
default_config = get_config()

if __name__ == "__main__":
    # 测试配置
    config = get_config()
    print("🔧 配置测试:")
    print(f"  启用模块: GNN={config.model.enable_gnn}, BERT={config.model.enable_bert}, LGBM={config.model.enable_lgbm}")
    print(f"  模型维度: hidden_dim={config.model.hidden_dim}")
    print(f"  训练参数: batch_size={config.training.batch_size}, lr={config.training.learning_rate}")
    print(f"  数据路径: {config.data.data_dir}")
    
    # 保存默认配置
    save_config_to_file(config, "configs/default_config.json")
    print("✅ 默认配置保存到 configs/default_config.json") 