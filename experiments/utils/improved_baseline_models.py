#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Traditional Machine Learning Baseline Models
改进版传统机器学习基线模型
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           f1_score, precision_recall_curve, auc, average_precision_score,
                           precision_score, recall_score, accuracy_score, make_scorer)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import shap
from typing import Dict, Any, Tuple, List, Optional
import pickle
import os
import logging
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class ImprovedBaselineModelTrainer:
    """改进版传统机器学习基线模型训练器"""
    
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        """
        初始化改进版基线模型训练器
        
        Args:
            model_type: 模型类型 ("random_forest" 或 "xgboost")
            random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.best_params = None
        
        # 不在初始化时创建模型，而是在参数优化后创建
        
    def extract_rf_features(self, multimodal_data: Dict[str, Any]) -> pd.DataFrame:
        """
        为Random Forest提取特征 - 使用更多原始特征和交互特征
        
        Args:
            multimodal_data: 多模态数据字典
            
        Returns:
            特征DataFrame
        """
        features_list = []
        user_ids = multimodal_data.get('users', [])
        
        print("🌲 为Random Forest提取丰富特征...")
        
        # 1. 详细的行为序列特征
        if 'behavior_sequences' in multimodal_data:
            behavior_data = multimodal_data['behavior_sequences']
            if isinstance(behavior_data, np.ndarray) and len(behavior_data.shape) == 3:
                for i, user_id in enumerate(user_ids):
                    if i < behavior_data.shape[0]:
                        user_sequence = behavior_data[i]  # [sequence_length, feature_dim]
                        
                        # 基础统计特征
                        features = {
                            'user_id': user_id,
                            # 全局统计
                            'seq_mean': np.mean(user_sequence),
                            'seq_std': np.std(user_sequence),
                            'seq_max': np.max(user_sequence),
                            'seq_min': np.min(user_sequence),
                            'seq_median': np.median(user_sequence),
                            'seq_q25': np.percentile(user_sequence, 25),
                            'seq_q75': np.percentile(user_sequence, 75),
                            'seq_iqr': np.percentile(user_sequence, 75) - np.percentile(user_sequence, 25),
                            'seq_skew': self._calculate_skewness(user_sequence.flatten()),
                            'seq_kurtosis': self._calculate_kurtosis(user_sequence.flatten()),
                            'seq_range': np.max(user_sequence) - np.min(user_sequence),
                            
                            # 活动模式特征
                            'seq_activity_rate': np.mean(user_sequence > 0),
                            'seq_zero_ratio': np.mean(user_sequence == 0),
                            'seq_high_activity_ratio': np.mean(user_sequence > np.mean(user_sequence)),
                            
                            # 变异性特征
                            'seq_cv': np.std(user_sequence) / (np.mean(user_sequence) + 1e-8),
                            'seq_mad': np.mean(np.abs(user_sequence - np.median(user_sequence))),
                        }
                        
                        # 时间序列特征 - 更详细
                        daily_activity = np.mean(user_sequence, axis=1)
                        features.update({
                            'daily_activity_mean': np.mean(daily_activity),
                            'daily_activity_std': np.std(daily_activity),
                            'daily_activity_max': np.max(daily_activity),
                            'daily_activity_min': np.min(daily_activity),
                            'daily_activity_trend': self._calculate_trend(daily_activity),
                            'daily_activity_autocorr': self._calculate_autocorr(daily_activity),
                            'peak_activity_day': np.argmax(daily_activity),
                            'activity_consistency': 1 / (1 + np.std(daily_activity)),
                            'activity_burst_count': self._count_bursts(daily_activity),
                            'activity_plateau_count': self._count_plateaus(daily_activity),
                        })
                        
                        # 特征维度级别的统计
                        for dim in range(min(user_sequence.shape[1], 10)):  # 限制维度数量
                            dim_data = user_sequence[:, dim]
                            features.update({
                                f'dim_{dim}_mean': np.mean(dim_data),
                                f'dim_{dim}_std': np.std(dim_data),
                                f'dim_{dim}_max': np.max(dim_data),
                                f'dim_{dim}_trend': self._calculate_trend(dim_data),
                            })
                        
                        features_list.append(features)
        
        # 2. 结构化特征 - 更详细
        if 'structured_features' in multimodal_data:
            structured_data = multimodal_data['structured_features']
            if isinstance(structured_data, np.ndarray):
                for i, user_id in enumerate(user_ids):
                    if i < structured_data.shape[0]:
                        user_features = structured_data[i]
                        
                        struct_features = {
                            'struct_mean': np.mean(user_features),
                            'struct_std': np.std(user_features),
                            'struct_max': np.max(user_features),
                            'struct_min': np.min(user_features),
                            'struct_median': np.median(user_features),
                            'struct_nonzero_count': np.count_nonzero(user_features),
                            'struct_sparsity': 1 - (np.count_nonzero(user_features) / len(user_features)),
                            'struct_entropy': self._calculate_entropy(user_features),
                            'struct_energy': np.sum(user_features ** 2),
                        }
                        
                        if i < len(features_list):
                            features_list[i].update(struct_features)
                        else:
                            struct_features['user_id'] = user_id
                            features_list.append(struct_features)
        
        # 3. 图特征
        if 'node_features' in multimodal_data:
            node_features = multimodal_data['node_features']
            if isinstance(node_features, np.ndarray):
                for i, user_id in enumerate(user_ids):
                    if i < node_features.shape[0]:
                        user_node_features = node_features[i]
                        
                        graph_features = {
                            'node_feature_mean': np.mean(user_node_features),
                            'node_feature_std': np.std(user_node_features),
                            'node_centrality_proxy': np.sum(user_node_features),
                            'node_feature_max': np.max(user_node_features),
                            'node_feature_energy': np.sum(user_node_features ** 2),
                        }
                        
                        if i < len(features_list):
                            features_list[i].update(graph_features)
                        else:
                            graph_features['user_id'] = user_id
                            features_list.append(graph_features)
        
        # 4. 文本特征
        if 'text_content' in multimodal_data:
            text_data = multimodal_data['text_content']
            if isinstance(text_data, list):
                for i, user_id in enumerate(user_ids):
                    if i < len(text_data):
                        user_text = text_data[i] if text_data[i] else ""
                        
                        text_features = {
                            'text_length': len(user_text),
                            'text_word_count': len(user_text.split()) if user_text else 0,
                            'text_char_diversity': len(set(user_text.lower())) if user_text else 0,
                            'text_avg_word_length': np.mean([len(word) for word in user_text.split()]) if user_text else 0,
                            'text_sentence_count': user_text.count('.') + user_text.count('!') + user_text.count('?'),
                            'text_uppercase_ratio': sum(1 for c in user_text if c.isupper()) / (len(user_text) + 1),
                        }
                        
                        if i < len(features_list):
                            features_list[i].update(text_features)
                        else:
                            text_features['user_id'] = user_id
                            features_list.append(text_features)
        
        # 转换为DataFrame并创建交互特征
        if features_list:
            df = pd.DataFrame(features_list)
            df = df.fillna(0)
            
            # 为Random Forest添加交互特征
            print("🔗 为Random Forest创建交互特征...")
            df = self._create_interaction_features(df)
            
            return df
        else:
            return pd.DataFrame()
    
    def extract_xgb_features(self, multimodal_data: Dict[str, Any]) -> pd.DataFrame:
        """
        为XGBoost提取特征 - 利用其对缺失值和特征选择的优势
        
        Args:
            multimodal_data: 多模态数据字典
            
        Returns:
            特征DataFrame
        """
        features_list = []
        user_ids = multimodal_data.get('users', [])
        
        print("🚀 为XGBoost提取优化特征...")
        
        # 1. 保留原始特征（XGBoost能自动处理特征选择）
        if 'behavior_sequences' in multimodal_data:
            behavior_data = multimodal_data['behavior_sequences']
            if isinstance(behavior_data, np.ndarray) and len(behavior_data.shape) == 3:
                for i, user_id in enumerate(user_ids):
                    if i < behavior_data.shape[0]:
                        user_sequence = behavior_data[i]
                        
                        # 基础特征（让XGBoost自己选择重要的）
                        features = {
                            'user_id': user_id,
                            'seq_mean': np.mean(user_sequence),
                            'seq_std': np.std(user_sequence),
                            'seq_max': np.max(user_sequence),
                            'seq_min': np.min(user_sequence),
                            'seq_median': np.median(user_sequence),
                            'seq_skew': self._calculate_skewness(user_sequence.flatten()),
                            'seq_kurtosis': self._calculate_kurtosis(user_sequence.flatten()),
                            'seq_activity_rate': np.mean(user_sequence > 0),
                            'seq_zero_ratio': np.mean(user_sequence == 0),
                            'seq_range': np.max(user_sequence) - np.min(user_sequence),
                        }
                        
                        # 时间序列特征
                        daily_activity = np.mean(user_sequence, axis=1)
                        features.update({
                            'daily_activity_mean': np.mean(daily_activity),
                            'daily_activity_std': np.std(daily_activity),
                            'daily_activity_trend': self._calculate_trend(daily_activity),
                            'peak_activity_day': np.argmax(daily_activity),
                            'activity_consistency': 1 / (1 + np.std(daily_activity)),
                        })
                        
                        # 添加原始序列的展平版本（部分）- XGBoost能处理高维
                        flattened = user_sequence.flatten()
                        for j in range(min(len(flattened), 50)):  # 限制特征数量
                            features[f'raw_seq_{j}'] = flattened[j]
                        
                        # 故意引入一些缺失值（测试XGBoost的缺失值处理能力）
                        if np.random.random() < 0.1:  # 10%的概率
                            features['seq_std'] = np.nan
                        if np.random.random() < 0.05:  # 5%的概率
                            features['daily_activity_trend'] = np.nan
                        
                        features_list.append(features)
        
        # 2. 结构化特征（保持简洁，让XGBoost自动选择）
        if 'structured_features' in multimodal_data:
            structured_data = multimodal_data['structured_features']
            if isinstance(structured_data, np.ndarray):
                for i, user_id in enumerate(user_ids):
                    if i < structured_data.shape[0]:
                        user_features = structured_data[i]
                        
                        struct_features = {
                            'struct_mean': np.mean(user_features),
                            'struct_std': np.std(user_features),
                            'struct_max': np.max(user_features),
                            'struct_min': np.min(user_features),
                            'struct_nonzero_count': np.count_nonzero(user_features),
                            'struct_sparsity': 1 - (np.count_nonzero(user_features) / len(user_features)),
                        }
                        
                        # 添加原始结构化特征
                        for j, val in enumerate(user_features[:20]):  # 限制数量
                            struct_features[f'struct_raw_{j}'] = val
                        
                        if i < len(features_list):
                            features_list[i].update(struct_features)
                        else:
                            struct_features['user_id'] = user_id
                            features_list.append(struct_features)
        
        # 3. 其他特征（简化版本）
        if 'node_features' in multimodal_data:
            node_features = multimodal_data['node_features']
            if isinstance(node_features, np.ndarray):
                for i, user_id in enumerate(user_ids):
                    if i < node_features.shape[0]:
                        user_node_features = node_features[i]
                        
                        graph_features = {
                            'node_feature_mean': np.mean(user_node_features),
                            'node_feature_std': np.std(user_node_features),
                            'node_centrality_proxy': np.sum(user_node_features),
                        }
                        
                        if i < len(features_list):
                            features_list[i].update(graph_features)
                        else:
                            graph_features['user_id'] = user_id
                            features_list.append(graph_features)
        
        if 'text_content' in multimodal_data:
            text_data = multimodal_data['text_content']
            if isinstance(text_data, list):
                for i, user_id in enumerate(user_ids):
                    if i < len(text_data):
                        user_text = text_data[i] if text_data[i] else ""
                        
                        text_features = {
                            'text_length': len(user_text),
                            'text_word_count': len(user_text.split()) if user_text else 0,
                            'text_char_diversity': len(set(user_text.lower())) if user_text else 0,
                            'text_avg_word_length': np.mean([len(word) for word in user_text.split()]) if user_text else 0,
                        }
                        
                        if i < len(features_list):
                            features_list[i].update(text_features)
                        else:
                            text_features['user_id'] = user_id
                            features_list.append(text_features)
        
        # 转换为DataFrame（不创建交互特征，让XGBoost自己学习）
        if features_list:
            df = pd.DataFrame(features_list)
            # 对于XGBoost，不填充缺失值，让它自己处理
            return df
        else:
            return pd.DataFrame()
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """为Random Forest创建交互特征"""
        if 'user_id' in df.columns:
            df_features = df.drop('user_id', axis=1)
        else:
            df_features = df.copy()
        
        # 选择最重要的几个特征进行交互
        important_features = ['seq_mean', 'seq_std', 'daily_activity_mean', 'struct_mean', 'seq_activity_rate']
        available_features = [f for f in important_features if f in df_features.columns]
        
        if len(available_features) >= 2:
            # 创建二阶交互特征
            for i, feat1 in enumerate(available_features):
                for feat2 in available_features[i+1:]:
                    # 乘积交互
                    df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                    # 比值交互（避免除零）
                    df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)
        
        return df
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        if len(data) == 0 or np.std(data) == 0:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        if len(data) == 0 or np.std(data) == 0:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """计算趋势（线性回归斜率）"""
        if len(data) < 2:
            return 0
        x = np.arange(len(data))
        return np.polyfit(x, data, 1)[0]
    
    def _calculate_autocorr(self, data: np.ndarray, lag: int = 1) -> float:
        """计算自相关"""
        if len(data) <= lag:
            return 0
        return np.corrcoef(data[:-lag], data[lag:])[0, 1] if not np.isnan(np.corrcoef(data[:-lag], data[lag:])[0, 1]) else 0
    
    def _count_bursts(self, data: np.ndarray, threshold_factor: float = 1.5) -> int:
        """计算活动突发次数"""
        if len(data) == 0:
            return 0
        threshold = np.mean(data) * threshold_factor
        return np.sum(data > threshold)
    
    def _count_plateaus(self, data: np.ndarray, tolerance: float = 0.1) -> int:
        """计算平台期次数"""
        if len(data) < 3:
            return 0
        diff = np.abs(np.diff(data))
        plateau_mask = diff < tolerance
        # 计算连续平台期的数量
        plateaus = 0
        in_plateau = False
        for is_plateau in plateau_mask:
            if is_plateau and not in_plateau:
                plateaus += 1
                in_plateau = True
            elif not is_plateau:
                in_plateau = False
        return plateaus
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """计算熵"""
        if len(data) == 0:
            return 0
        # 将数据离散化
        hist, _ = np.histogram(data, bins=10)
        hist = hist[hist > 0]  # 移除零值
        if len(hist) == 0:
            return 0
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob))
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        优化超参数
        
        Args:
            X: 特征矩阵
            y: 标签
            
        Returns:
            最佳参数字典
        """
        print(f"🔧 为 {self.model_type} 优化超参数...")
        
        if self.model_type == "random_forest":
            # Random Forest参数网格
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
            
        elif self.model_type == "xgboost":
            # XGBoost参数网格 - 更激进的参数
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10, 12],  # 更深的树
                'learning_rate': [0.1, 0.2, 0.3],  # 更高的学习率
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2],
            }
            
            base_model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        # 使用分层交叉验证进行网格搜索
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # 使用F1作为评分标准（二分类）
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1',  # 二分类F1
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"✅ 最佳参数: {grid_search.best_params_}")
        print(f"✅ 最佳CV F1分数: {grid_search.best_score_:.4f}")
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def prepare_data(self, multimodal_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据 - 使用差异化特征提取
        
        Args:
            multimodal_data: 多模态数据
            
        Returns:
            (X, y, feature_names)
        """
        # 根据模型类型使用不同的特征提取方法
        if self.model_type == "random_forest":
            features_df = self.extract_rf_features(multimodal_data)
        elif self.model_type == "xgboost":
            features_df = self.extract_xgb_features(multimodal_data)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        if features_df.empty:
            raise ValueError("无法提取特征，数据可能为空或格式不正确")
        
        # 获取标签
        labels = multimodal_data.get('labels', [])
        if len(labels) == 0:
            raise ValueError("缺少标签数据")
        
        # 确保特征和标签数量匹配
        min_samples = min(len(features_df), len(labels))
        features_df = features_df.iloc[:min_samples]
        labels = labels[:min_samples]
        
        # 移除user_id列（如果存在）
        if 'user_id' in features_df.columns:
            features_df = features_df.drop('user_id', axis=1)
        
        # 获取特征名称
        self.feature_names = list(features_df.columns)
        
        # 转换为numpy数组
        X = features_df.values
        y = np.array(labels)
        
        # 对于Random Forest，进行标准化；对于XGBoost，保持原始值
        if self.model_type == "random_forest":
            X = self.scaler.fit_transform(X)
        elif self.model_type == "xgboost":
            # XGBoost不需要标准化，但需要处理无穷值
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X, y, self.feature_names
    
    def train_with_cv(self, multimodal_data: Dict[str, Any], cv_folds: int = 5) -> Dict[str, Any]:
        """
        使用交叉验证训练模型
        
        Args:
            multimodal_data: 多模态数据
            cv_folds: 交叉验证折数
            
        Returns:
            训练结果字典
        """
        # 准备数据
        X, y, feature_names = self.prepare_data(multimodal_data)
        
        print(f"📊 数据准备完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
        print(f"📊 类别分布: {np.bincount(y)}")
        
        # 检查类别平衡
        if len(np.unique(y)) < 2:
            raise ValueError("标签中只有一个类别，无法进行二分类")
        
        # 超参数优化
        best_params = self.optimize_hyperparameters(X, y)
        
        # 使用最佳参数进行交叉验证
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # 定义评估指标
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1',  # 二分类F1
            'precision': 'precision',  # 二分类精确率
            'recall': 'recall',  # 二分类召回率
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision'  # PR-AUC
        }
        
        print(f"🔄 开始 {cv_folds} 折交叉验证...")
        cv_results = cross_validate(
            self.model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # 计算交叉验证统计
        cv_stats = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            cv_stats[f'{metric}_test_mean'] = np.mean(test_scores)
            cv_stats[f'{metric}_test_std'] = np.std(test_scores)
            cv_stats[f'{metric}_train_mean'] = np.mean(train_scores)
            cv_stats[f'{metric}_train_std'] = np.std(train_scores)
        
        # 在全部数据上重新训练最终模型
        print("🚀 在全部数据上训练最终模型...")
        self.model.fit(X, y)
        
        # 获取特征重要性
        feature_importance = self._get_feature_importance()
        
        results = {
            'model_type': self.model_type,
            'best_params': best_params,
            'cv_results': cv_stats,
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'class_distribution': np.bincount(y).tolist()
        }
        
        # 打印结果
        print(f"✅ {self.model_type} 交叉验证完成")
        print(f"   测试集 F1: {cv_stats['f1_test_mean']:.4f} ± {cv_stats['f1_test_std']:.4f}")
        print(f"   测试集 AUC: {cv_stats['roc_auc_test_mean']:.4f} ± {cv_stats['roc_auc_test_std']:.4f}")
        print(f"   测试集 PR-AUC: {cv_stats['average_precision_test_mean']:.4f} ± {cv_stats['average_precision_test_std']:.4f}")
        print(f"   测试集 精确率: {cv_stats['precision_test_mean']:.4f} ± {cv_stats['precision_test_std']:.4f}")
        print(f"   测试集 召回率: {cv_stats['recall_test_mean']:.4f} ± {cv_stats['recall_test_std']:.4f}")
        
        return results
    
    def _get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return np.zeros(len(self.feature_names))
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'best_params': self.best_params
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 改进模型已保存到: {filepath}")

def run_improved_baseline_comparison(multimodal_data: Dict[str, Any], 
                                   output_dir: str,
                                   models: List[str] = None,
                                   cv_folds: int = 5) -> Dict[str, Any]:
    """
    运行改进版基线模型对比实验
    
    Args:
        multimodal_data: 多模态数据
        output_dir: 输出目录
        models: 要测试的模型列表
        cv_folds: 交叉验证折数
        
    Returns:
        对比结果
    """
    if models is None:
        models = ["random_forest", "xgboost"]
    
    results = {}
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"🔬 测试改进版基线模型: {model_type}")
        print(f"{'='*60}")
        
        try:
            # 创建训练器
            trainer = ImprovedBaselineModelTrainer(model_type=model_type)
            
            # 使用交叉验证训练模型
            model_results = trainer.train_with_cv(multimodal_data, cv_folds=cv_folds)
            
            # 保存模型
            model_path = os.path.join(output_dir, f"improved_{model_type}_model.pkl")
            trainer.save_model(model_path)
            
            # 保存结果
            results[model_type] = model_results
            
        except Exception as e:
            print(f"❌ {model_type} 改进模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = {'error': str(e)}
    
    # 打印对比结果
    print(f"\n{'='*60}")
    print("📊 改进版基线模型对比结果")
    print(f"{'='*60}")
    
    for model_type, result in results.items():
        if 'error' not in result:
            cv_results = result['cv_results']
            print(f"\n🔬 {model_type.upper()}:")
            print(f"   F1 Score: {cv_results['f1_test_mean']:.4f} ± {cv_results['f1_test_std']:.4f}")
            print(f"   ROC-AUC:  {cv_results['roc_auc_test_mean']:.4f} ± {cv_results['roc_auc_test_std']:.4f}")
            print(f"   PR-AUC:   {cv_results['average_precision_test_mean']:.4f} ± {cv_results['average_precision_test_std']:.4f}")
            print(f"   精确率:   {cv_results['precision_test_mean']:.4f} ± {cv_results['precision_test_std']:.4f}")
            print(f"   召回率:   {cv_results['recall_test_mean']:.4f} ± {cv_results['recall_test_std']:.4f}")
            print(f"   特征数:   {result['n_features']}")
        else:
            print(f"\n❌ {model_type.upper()}: {result['error']}")
    
    return results 