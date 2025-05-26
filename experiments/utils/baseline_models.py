#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traditional Machine Learning Baseline Models
传统机器学习基线模型
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import xgboost as xgb
import shap
from typing import Dict, Any, Tuple, List, Optional
import pickle
import os
import logging

class BaselineModelTrainer:
    """传统机器学习基线模型训练器"""
    
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        """
        初始化基线模型训练器
        
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
        
        # 初始化模型
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def extract_traditional_features(self, multimodal_data: Dict[str, Any]) -> pd.DataFrame:
        """
        从多模态数据中提取传统手工特征
        
        Args:
            multimodal_data: 多模态数据字典
            
        Returns:
            特征DataFrame
        """
        features_list = []
        user_ids = multimodal_data.get('users', [])
        
        # 1. 行为序列统计特征
        if 'behavior_sequences' in multimodal_data:
            behavior_data = multimodal_data['behavior_sequences']
            if isinstance(behavior_data, np.ndarray) and len(behavior_data.shape) == 3:
                # Shape: [num_users, sequence_length, feature_dim]
                for i, user_id in enumerate(user_ids):
                    if i < behavior_data.shape[0]:
                        user_sequence = behavior_data[i]  # [sequence_length, feature_dim]
                        
                        # 统计特征
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
                            'seq_range': np.max(user_sequence) - np.min(user_sequence)
                        }
                        
                        # 时间序列特征
                        daily_activity = np.mean(user_sequence, axis=1)  # 每天的平均活动
                        features.update({
                            'daily_activity_mean': np.mean(daily_activity),
                            'daily_activity_std': np.std(daily_activity),
                            'daily_activity_trend': self._calculate_trend(daily_activity),
                            'peak_activity_day': np.argmax(daily_activity),
                            'activity_consistency': 1 / (1 + np.std(daily_activity))
                        })
                        
                        features_list.append(features)
        
        # 2. 结构化特征
        if 'structured_features' in multimodal_data:
            structured_data = multimodal_data['structured_features']
            if isinstance(structured_data, np.ndarray):
                for i, user_id in enumerate(user_ids):
                    if i < structured_data.shape[0]:
                        user_features = structured_data[i]
                        
                        # 如果features_list中已有该用户，则更新；否则创建新条目
                        if i < len(features_list):
                            features_list[i].update({
                                'struct_mean': np.mean(user_features),
                                'struct_std': np.std(user_features),
                                'struct_max': np.max(user_features),
                                'struct_min': np.min(user_features),
                                'struct_nonzero_count': np.count_nonzero(user_features),
                                'struct_sparsity': 1 - (np.count_nonzero(user_features) / len(user_features))
                            })
                        else:
                            features_list.append({
                                'user_id': user_id,
                                'struct_mean': np.mean(user_features),
                                'struct_std': np.std(user_features),
                                'struct_max': np.max(user_features),
                                'struct_min': np.min(user_features),
                                'struct_nonzero_count': np.count_nonzero(user_features),
                                'struct_sparsity': 1 - (np.count_nonzero(user_features) / len(user_features))
                            })
        
        # 3. 图特征（如果有节点特征）
        if 'node_features' in multimodal_data:
            node_features = multimodal_data['node_features']
            if isinstance(node_features, np.ndarray):
                for i, user_id in enumerate(user_ids):
                    if i < node_features.shape[0]:
                        user_node_features = node_features[i]
                        
                        graph_features = {
                            'node_feature_mean': np.mean(user_node_features),
                            'node_feature_std': np.std(user_node_features),
                            'node_centrality_proxy': np.sum(user_node_features),  # 简单的中心性代理
                        }
                        
                        if i < len(features_list):
                            features_list[i].update(graph_features)
                        else:
                            graph_features['user_id'] = user_id
                            features_list.append(graph_features)
        
        # 4. 文本特征（简单统计）
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
                            'text_avg_word_length': np.mean([len(word) for word in user_text.split()]) if user_text else 0
                        }
                        
                        if i < len(features_list):
                            features_list[i].update(text_features)
                        else:
                            text_features['user_id'] = user_id
                            features_list.append(text_features)
        
        # 转换为DataFrame
        if features_list:
            df = pd.DataFrame(features_list)
            # 填充缺失值
            df = df.fillna(0)
            return df
        else:
            # 返回空DataFrame
            return pd.DataFrame()
    
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
        """计算趋势（简单线性回归斜率）"""
        if len(data) < 2:
            return 0
        x = np.arange(len(data))
        return np.polyfit(x, data, 1)[0]
    
    def prepare_data(self, multimodal_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练数据
        
        Args:
            multimodal_data: 多模态数据
            
        Returns:
            (X, y, feature_names)
        """
        # 提取特征
        features_df = self.extract_traditional_features(multimodal_data)
        
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
        
        return X, y, self.feature_names
    
    def train(self, multimodal_data: Dict[str, Any], test_size: float = 0.2) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            multimodal_data: 多模态数据
            test_size: 测试集比例
            
        Returns:
            训练结果字典
        """
        # 准备数据
        X, y, feature_names = self.prepare_data(multimodal_data)
        
        # 数据划分
        if len(np.unique(y)) < 2 or len(y) < 4:
            # 如果样本太少或类别不足，使用简单划分
            split_idx = max(1, int(len(X) * (1 - test_size)))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练模型
        print(f"🚀 开始训练 {self.model_type} 模型...")
        self.model.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # 计算指标
        train_metrics = self._calculate_metrics(y_train, y_train_pred, X_train_scaled)
        test_metrics = self._calculate_metrics(y_test, y_test_pred, X_test_scaled)
        
        # 特征重要性
        feature_importance = self._get_feature_importance()
        
        results = {
            'model_type': self.model_type,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'predictions': {
                'y_train_true': y_train.tolist(),
                'y_train_pred': y_train_pred.tolist(),
                'y_test_true': y_test.tolist(),
                'y_test_pred': y_test_pred.tolist()
            }
        }
        
        print(f"✅ {self.model_type} 模型训练完成")
        print(f"   测试集 F1: {test_metrics['f1']:.4f}")
        print(f"   测试集 AUC: {test_metrics['auc']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        logger = logging.getLogger(__name__) # 添加日志记录器
        
        # Sanity check 1: 检查 y_true中的类别数量
        unique_classes_true = np.unique(y_true)
        if len(unique_classes_true) < 2:
            logger.warning(f"警告: y_true 中只存在一个类别 ({unique_classes_true}). AUC 将无法计算或无意义。返回的指标可能不准确。")
            # 对于F1等指标，如果只有一个类别，它们的计算也可能不符合预期，这里返回0或nan，并提示用户检查数据
            return {
                'accuracy': np.mean(y_true == y_pred), # 准确率仍可计算
                'f1': 0.0,
                'auc': np.nan, # AUC 无法计算
                'precision': 0.0,
                'recall': 0.0,
                'warning': 'y_true contains only one class'
            }

        # Sanity check 2: 检查 y_pred 是否只包含一个类别
        unique_classes_pred = np.unique(y_pred)
        if len(unique_classes_pred) < 2:
            logger.warning(f"警告: 模型预测结果 y_pred 中只存在一个类别 ({unique_classes_pred}). 这可能表明模型存在问题或数据分布极端不平衡。")

        # 基础指标
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC（需要概率预测）
        auc = np.nan # 默认为 nan
        y_proba_positive_class = None
        try:
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X)
                if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                    y_proba_positive_class = y_proba[:, 1] # 通常取正类的概率
                    
                    # Sanity check 3: 检查 y_proba_positive_class 是否全部相同
                    if len(np.unique(y_proba_positive_class)) == 1:
                        logger.warning(f"警告: 模型预测的所有样本概率值 y_score 都相同 ({y_proba_positive_class[0]}). AUC 将无法计算或无意义。")
                        auc = np.nan # 或者可以设为 0.0 或 0.5，取决于具体场景的定义
                    else:
                        auc = roc_auc_score(y_true, y_proba_positive_class)
                else:
                    logger.warning(f"警告: predict_proba 返回的概率格式不符合预期 (shape: {y_proba.shape})。无法计算 AUC。")
            else:
                logger.warning(f"警告: 模型 {self.model_type} 没有 predict_proba 方法。无法计算 AUC。")
        except ValueError as e:
            # 如果 y_true 仍然导致 roc_auc_score 出错（例如，在 predict_proba 检查后，y_true 实际上只有一个类别被传入）
            logger.error(f"错误: 计算 AUC 时发生 ValueError: {e}. 这可能仍然是因为 y_true 中只有一个类别。")
            auc = np.nan
        except Exception as e:
            logger.error(f"错误: 计算 AUC 时发生未知错误: {e}")
            auc = np.nan
        
        # 准确率
        accuracy = np.mean(y_true == y_pred)
        
        # 精确率和召回率
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 如果F1为1.0，但AUC不佳，添加额外日志
        if f1 == 1.0 and (np.isnan(auc) or auc < 0.6): # 0.6 是一个示例阈值
            logger.warning(f"警告: F1 分数为 1.0，但 AUC ({auc}) 较低或无法计算。请检查数据划分、模型过拟合或评估逻辑。")
            logger.info(f"y_true unique: {np.unique(y_true, return_counts=True)}")
            logger.info(f"y_pred unique: {np.unique(y_pred, return_counts=True)}")
            if y_proba_positive_class is not None:
                 logger.info(f"y_proba_positive_class unique: {np.unique(y_proba_positive_class, return_counts=True)}")


        return {
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall
        }
    
    def _get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return np.zeros(len(self.feature_names))
    
    def get_shap_values(self, X: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """
        计算SHAP值
        
        Args:
            X: 输入特征
            max_samples: 最大样本数（用于计算效率）
            
        Returns:
            SHAP值数组
        """
        try:
            # 限制样本数量以提高计算效率
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # 标准化
            X_sample_scaled = self.scaler.transform(X_sample)
            
            # 计算SHAP值
            if self.model_type == "random_forest":
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # 取正类的SHAP值
            elif self.model_type == "xgboost":
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample_scaled)
            else:
                # 使用KernelExplainer作为后备
                explainer = shap.KernelExplainer(self.model.predict, X_sample_scaled[:10])
                shap_values = explainer.shap_values(X_sample_scaled)
            
            return shap_values
        except Exception as e:
            print(f"⚠️ SHAP值计算失败: {e}")
            return np.zeros((len(X), len(self.feature_names)))
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        print(f"📂 模型已从 {filepath} 加载")

def run_baseline_comparison(multimodal_data: Dict[str, Any], 
                          output_dir: str,
                          models: List[str] = None) -> Dict[str, Any]:
    """
    运行基线模型对比实验
    
    Args:
        multimodal_data: 多模态数据
        output_dir: 输出目录
        models: 要测试的模型列表
        
    Returns:
        对比结果
    """
    if models is None:
        models = ["random_forest", "xgboost"]
    
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"🔬 测试基线模型: {model_type}")
        print(f"{'='*50}")
        
        try:
            # 创建训练器
            trainer = BaselineModelTrainer(model_type=model_type)
            
            # 训练模型
            model_results = trainer.train(multimodal_data)
            
            # 保存模型
            model_path = os.path.join(output_dir, f"{model_type}_model.pkl")
            trainer.save_model(model_path)
            
            # 保存结果
            results[model_type] = model_results
            
        except Exception as e:
            print(f"❌ {model_type} 模型训练失败: {e}")
            results[model_type] = {'error': str(e)}
    
    return results 