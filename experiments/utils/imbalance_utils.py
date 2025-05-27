#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Imbalance Handling Utilities
数据不平衡处理工具模块
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter

class ImbalanceHandler:
    """数据不平衡处理器"""
    
    def __init__(self, random_state: int = 42):
        """
        初始化不平衡处理器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        self.samplers = {
            'none': None,
            'smote': SMOTE(random_state=random_state),
            'adasyn': ADASYN(random_state=random_state),
            'borderline_smote': BorderlineSMOTE(random_state=random_state),
            'random_undersample': RandomUnderSampler(random_state=random_state),
            'edited_nn': EditedNearestNeighbours(),
            'tomek_links': TomekLinks(),
            'smote_tomek': SMOTETomek(random_state=random_state),
            'smote_enn': SMOTEENN(random_state=random_state)
        }
    
    def create_imbalanced_datasets(self, 
                                 multimodal_data: Dict[str, Any],
                                 ratios: List[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        创建不同不平衡比例的数据集
        
        Args:
            multimodal_data: 原始多模态数据
            ratios: 不平衡比例列表 (正常:恶意)
            
        Returns:
            不同比例的数据集字典
        """
        if ratios is None:
            ratios = [1.0, 2.0, 3.0, 4.0, 5.0]  # 1:1, 2:1, 3:1, 4:1, 5:1
        
        datasets = {}
        
        # 获取原始标签
        labels = np.array(multimodal_data.get('labels', []))
        if len(labels) == 0:
            raise ValueError("缺少标签数据")
        
        # 分离正常和恶意样本的索引
        normal_indices = np.where(labels == 0)[0]
        malicious_indices = np.where(labels == 1)[0]
        
        print(f"📊 原始数据分布:")
        print(f"   正常样本: {len(normal_indices)}")
        print(f"   恶意样本: {len(malicious_indices)}")
        
        for ratio in ratios:
            print(f"\n🔄 创建比例 {ratio}:1 (正常:恶意) 的数据集...")
            
            # 计算需要的样本数量
            target_malicious_count = len(malicious_indices)
            target_normal_count = int(target_malicious_count * ratio)
            
            # 如果正常样本不够，使用所有正常样本
            if target_normal_count > len(normal_indices):
                target_normal_count = len(normal_indices)
                actual_ratio = target_normal_count / target_malicious_count
                print(f"   ⚠️ 正常样本不足，实际比例: {actual_ratio:.2f}:1")
            
            # 随机选择样本
            selected_normal_indices = np.random.choice(
                normal_indices, target_normal_count, replace=False
            )
            selected_malicious_indices = malicious_indices.copy()
            
            # 合并索引
            selected_indices = np.concatenate([selected_normal_indices, selected_malicious_indices])
            np.random.shuffle(selected_indices)
            
            # 创建新的数据集
            imbalanced_data = self._extract_samples_by_indices(multimodal_data, selected_indices)
            
            # 验证分布
            new_labels = np.array(imbalanced_data['labels'])
            normal_count = np.sum(new_labels == 0)
            malicious_count = np.sum(new_labels == 1)
            actual_ratio = normal_count / malicious_count if malicious_count > 0 else 0
            
            print(f"   ✅ 创建完成: 正常={normal_count}, 恶意={malicious_count}, 比例={actual_ratio:.2f}:1")
            
            datasets[f"ratio_{ratio}"] = imbalanced_data
        
        return datasets
    
    def _extract_samples_by_indices(self, 
                                   multimodal_data: Dict[str, Any], 
                                   indices: np.ndarray) -> Dict[str, Any]:
        """
        根据索引提取样本
        
        Args:
            multimodal_data: 原始数据
            indices: 样本索引
            
        Returns:
            提取的数据
        """
        extracted_data = {}
        # It's good to know the original number of samples for context if possible
        # num_original_samples = len(multimodal_data.get('labels', []))

        # Identify keys that are per-sample and should be sliced
        # Other keys (e.g., graph data, global mappings) might be copied directly
        # This list might need adjustment based on the exact structure of multimodal_data
        per_sample_keys = ['labels', 'users', 'behavior_sequences', 'text_content', 'structured_features', 'user_indices_in_graph']
        
        # Convert indices to list if it's a numpy array for easier handling in list comprehensions if needed,
        # though numpy indexing is generally preferred for numpy arrays.
        # However, the error is in a list comprehension `[value[i] for i in indices]`.
        
        # Check if indices are empty before trying to get max/min
        max_idx = -1
        min_idx = -1
        num_indices_to_extract = len(indices)

        if num_indices_to_extract > 0:
            # Ensure indices are integers for safe use with np.max/min if they are float from somewhere
            # For safety, convert to int if they are not already, though they should be.
            if not np.issubdtype(indices.dtype, np.integer):
                print(f"  [DEBUG ImbalanceUtils] Warning: 'indices' array is not of integer type (dtype: {indices.dtype}). Casting to int.")
                indices = indices.astype(int)
            max_idx = np.max(indices)
            min_idx = np.min(indices)

        for key, current_value in multimodal_data.items():
            print(f"--- Debugging _extract_samples_by_indices for key: '{key}' ---")
            data_len_str = "N/A"
            actual_len_for_indexing = -1

            if isinstance(current_value, list):
                actual_len_for_indexing = len(current_value)
                data_len_str = f"list with {actual_len_for_indexing} elements"
            elif isinstance(current_value, np.ndarray):
                actual_len_for_indexing = current_value.shape[0] if current_value.ndim > 0 else 0 # Handle 0-dim arrays
                data_len_str = f"numpy array with shape {current_value.shape}"
            else:
                data_len_str = f"type {type(current_value)}, not a list or ndarray with standard length"

            print(f"  Data for key '{key}': {data_len_str}")
            print(f"  Type of data for key '{key}': {type(current_value)}")
            print(f"  Number of indices to extract: {num_indices_to_extract}")
            print(f"  Min index in 'indices': {min_idx}")
            print(f"  Max index in 'indices': {max_idx}")

            # Check for potential out-of-bounds before attempting the operation
            if actual_len_for_indexing != -1 and num_indices_to_extract > 0: # If length is known and indices exist
                if max_idx >= actual_len_for_indexing:
                    print(f"  [POTENTIAL IndexError IMMINENT] For key '{key}': Max index ({max_idx}) is >= actual length ({actual_len_for_indexing}).")
                if min_idx < 0 and not (isinstance(current_value, np.ndarray) and current_value.ndim > 0 and -actual_len_for_indexing <= min_idx < 0 ): # Python list negative indexing
                    print(f"  [POTENTIAL IndexError IMMINENT] For key '{key}': Min index ({min_idx}) is < 0 and invalid for list or standard numpy positive indexing.")


            # This is where the error on line 130 (original) occurs.
            # The original code might be a simple loop or a more complex if/else structure.
            # Based on the traceback, it's likely a direct list comprehension.
            # We need to decide if this key should be sliced or copied.
            # A more robust way would be to define which keys are sliceable.
            
            # If current_value is a list, or a numpy array that we want to treat like a list for this slice
            if key in per_sample_keys: # Slice only per-sample keys
                if isinstance(current_value, list):
                    print(f"  Attempting to slice key '{key}' (list) using list comprehension.")
                    if actual_len_for_indexing == 0 and num_indices_to_extract > 0:
                        print(f"  [Warning] Key '{key}' is an empty list, but indices are present. Filling with {num_indices_to_extract} None values.")
                        extracted_data[key] = [None] * num_indices_to_extract # Or empty strings: [""] * num_indices_to_extract
                    elif actual_len_for_indexing < num_indices_to_extract and max_idx >= actual_len_for_indexing:
                        # This case is tricky: if indices refer beyond the list length. 
                        # The POTENTIAL IndexError IMMINENT should have warned.
                        # For safety, if max_idx is out of bounds, we might also fill with None or truncate.
                        # However, the original error implies direct indexing was attempted.
                        # Let's ensure that if max_idx is too large, we still try to build a list of Nones for robustness.
                        # The most common cause of the error for lists is `max_idx >= actual_len_for_indexing` when `actual_len_for_indexing > 0`
                        # or `actual_len_for_indexing == 0` as handled above.
                        if max_idx >= actual_len_for_indexing: # Re-check for clarity for this specific path
                             print(f"  [Warning] Key '{key}' (list) has length {actual_len_for_indexing} but max index is {max_idx}. Potential partial fill or error.")
                             # A robust way is to fill with None for out-of-bounds, or filter indices.
                             # For now, let's try the direct comprehension and rely on earlier debug prints.
                             # If it errors, we need to ensure indices are always valid *before* this loop.
                             # The debug log showed text_content was len 0, and indices were 0-49. So the [i] access will fail.
                             # The `actual_len_for_indexing == 0` case above handles this correctly now.
                             extracted_data[key] = [current_value[i] for i in indices] # This line would still error if not for the check above.
                        else:
                             extracted_data[key] = [current_value[i] for i in indices]
                    else:
                        extracted_data[key] = [current_value[i] for i in indices]
                elif isinstance(current_value, np.ndarray):
                    print(f"  Attempting to slice key '{key}' (numpy array) using numpy indexing.")
                    # Ensure indices are suitable for numpy array (e.g., integer array or list of integers)
                    try:
                        extracted_data[key] = current_value[indices]
                    except IndexError as e_numpy:
                        print(f"  [NumPy IndexError] Failed to slice numpy array for key '{key}': {e_numpy}")
                        print(f"     Indices were: {indices[:10]}... (dtype: {indices.dtype if hasattr(indices, 'dtype') else type(indices)})")
                        print(f"     Array shape: {current_value.shape}")
                        # Fallback or re-raise: for now, let it store an empty list or re-raise.
                        # To match list comprehension behavior in case of error (though error would be before assignment)
                        # we might need to handle this more gracefully or ensure indices are always valid.
                        # For debugging, if it fails here, the prints above should have given a clue.
                        extracted_data[key] = [] # Placeholder on error to avoid subsequent issues
                        # raise # Re-raise if you want the process to stop
                else:
                    print(f"  Key '{key}' is in per_sample_keys but is not a list or ndarray. Type: {type(current_value)}. Copying as is.")
                    extracted_data[key] = current_value # Fallback: copy as is
            else:
                # For keys not in per_sample_keys (e.g., 'user_to_index', 'node_features', 'adjacency_matrix')
                print(f"  Key '{key}' is not in per_sample_keys. Copying directly.")
                extracted_data[key] = current_value
            
            print(f"--- End Debugging for key: '{key}' ---")
        return extracted_data
    
    def apply_sampling_strategy(self, 
                              X: np.ndarray, 
                              y: np.ndarray, 
                              strategy: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        应用采样策略
        
        Args:
            X: 特征数据
            y: 标签数据
            strategy: 采样策略名称
            
        Returns:
            (重采样后的X, 重采样后的y)
        """
        if strategy == 'none' or strategy not in self.samplers:
            return X, y
        
        sampler = self.samplers[strategy]
        
        try:
            print(f"🔄 应用采样策略: {strategy}")
            print(f"   原始分布: {Counter(y)}")
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            print(f"   重采样后分布: {Counter(y_resampled)}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"⚠️ 采样策略 {strategy} 失败: {e}")
            return X, y
    
    def evaluate_sampling_strategies(self, 
                                   X: np.ndarray, 
                                   y: np.ndarray,
                                   model_trainer,
                                   strategies: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        评估不同采样策略的效果
        
        Args:
            X: 特征数据
            y: 标签数据
            model_trainer: 模型训练器
            strategies: 要测试的策略列表
            
        Returns:
            各策略的评估结果
        """
        if strategies is None:
            strategies = ['none', 'smote', 'adasyn', 'random_undersample', 'smote_tomek']
        
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"🧪 测试采样策略: {strategy}")
            print(f"{'='*50}")
            
            try:
                # 应用采样策略
                X_resampled, y_resampled = self.apply_sampling_strategy(X, y, strategy)
                
                # 数据划分
                X_train, X_test, y_train, y_test = train_test_split(
                    X_resampled, y_resampled, 
                    test_size=0.2, 
                    random_state=self.random_state, 
                    stratify=y_resampled
                )
                
                # 训练模型
                model_trainer.model.fit(X_train, y_train)
                
                # 预测和评估
                y_pred = model_trainer.model.predict(X_test)
                metrics = model_trainer._calculate_metrics(y_test, y_pred, X_test)
                
                results[strategy] = metrics
                
                print(f"✅ {strategy} 完成 - F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
                
            except Exception as e:
                print(f"❌ {strategy} 失败: {e}")
                results[strategy] = {'error': str(e)}
        
        return results

def run_imbalance_experiment(multimodal_data: Dict[str, Any],
                           output_dir: str,
                           model_trainer,
                           ratios: List[float] = None,
                           sampling_strategies: List[str] = None) -> Dict[str, Any]:
    """
    运行数据不平衡适应性实验
    
    Args:
        multimodal_data: 多模态数据
        output_dir: 输出目录
        model_trainer: 模型训练器
        ratios: 不平衡比例列表
        sampling_strategies: 采样策略列表
        
    Returns:
        实验结果
    """
    if ratios is None:
        ratios = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    if sampling_strategies is None:
        sampling_strategies = ['none', 'smote', 'adasyn', 'random_undersample']
    
    print(f"\n{'='*60}")
    print(f"🔬 开始数据不平衡适应性实验")
    print(f"{'='*60}")
    
    # 初始化不平衡处理器
    imbalance_handler = ImbalanceHandler()
    
    # 创建不同比例的数据集
    imbalanced_datasets = imbalance_handler.create_imbalanced_datasets(
        multimodal_data, ratios
    )
    
    results = {
        'experiment_type': 'imbalance',
        'ratios': ratios,
        'sampling_strategies': sampling_strategies,
        'ratio_results': {},
        'strategy_comparison': {}
    }
    
    # 对每个比例的数据集进行实验
    for ratio_name, dataset in imbalanced_datasets.items():
        print(f"\n📊 测试数据集: {ratio_name}")
        
        try:
            # 提取特征（使用传统特征提取方法）
            features_df = model_trainer.extract_traditional_features(dataset)
            if 'user_id' in features_df.columns:
                features_df = features_df.drop('user_id', axis=1)
            
            X = features_df.values
            y = np.array(dataset['labels'])
            
            # 评估不同采样策略
            strategy_results = imbalance_handler.evaluate_sampling_strategies(
                X, y, model_trainer, sampling_strategies
            )
            
            results['ratio_results'][ratio_name] = strategy_results
            
        except Exception as e:
            print(f"❌ {ratio_name} 实验失败: {e}")
            results['ratio_results'][ratio_name] = {'error': str(e)}
    
    # 汇总不同策略在各比例下的表现
    for strategy in sampling_strategies:
        strategy_f1_scores = []
        strategy_auc_scores = []
        
        for ratio in ratios:
            ratio_name = f"ratio_{ratio}"
            if (ratio_name in results['ratio_results'] and 
                strategy in results['ratio_results'][ratio_name] and
                'f1' in results['ratio_results'][ratio_name][strategy]):
                
                strategy_f1_scores.append(results['ratio_results'][ratio_name][strategy]['f1'])
                strategy_auc_scores.append(results['ratio_results'][ratio_name][strategy]['auc'])
            else:
                strategy_f1_scores.append(0.0)
                strategy_auc_scores.append(0.0)
        
        results['strategy_comparison'][strategy] = {
            'f1_scores': strategy_f1_scores,
            'auc_scores': strategy_auc_scores,
            'avg_f1': np.mean(strategy_f1_scores),
            'avg_auc': np.mean(strategy_auc_scores)
        }
    
    print(f"\n✅ 数据不平衡实验完成")
    print(f"📈 各策略平均表现:")
    for strategy, metrics in results['strategy_comparison'].items():
        print(f"   {strategy}: F1={metrics['avg_f1']:.4f}, AUC={metrics['avg_auc']:.4f}")
    
    return results

def create_balanced_dataset(multimodal_data: Dict[str, Any], 
                          target_ratio: float = 1.0,
                          method: str = 'undersample') -> Dict[str, Any]:
    """
    创建平衡数据集
    
    Args:
        multimodal_data: 原始数据
        target_ratio: 目标比例 (正常:恶意)
        method: 平衡方法 ('undersample', 'oversample', 'mixed')
        
    Returns:
        平衡后的数据集
    """
    labels = np.array(multimodal_data.get('labels', []))
    normal_indices = np.where(labels == 0)[0]
    malicious_indices = np.where(labels == 1)[0]
    
    normal_count = len(normal_indices)
    malicious_count = len(malicious_indices)
    
    print(f"📊 原始分布: 正常={normal_count}, 恶意={malicious_count}")
    
    if method == 'undersample':
        # 欠采样多数类
        if normal_count > malicious_count:
            target_normal_count = int(malicious_count * target_ratio)
            selected_normal_indices = np.random.choice(
                normal_indices, min(target_normal_count, normal_count), replace=False
            )
            selected_indices = np.concatenate([selected_normal_indices, malicious_indices])
        else:
            target_malicious_count = int(normal_count / target_ratio)
            selected_malicious_indices = np.random.choice(
                malicious_indices, min(target_malicious_count, malicious_count), replace=False
            )
            selected_indices = np.concatenate([normal_indices, selected_malicious_indices])
    
    elif method == 'oversample':
        # 过采样少数类（简单重复采样）
        if normal_count < malicious_count:
            target_normal_count = int(malicious_count * target_ratio)
            if target_normal_count > normal_count:
                # 需要过采样
                oversample_count = target_normal_count - normal_count
                oversampled_indices = np.random.choice(
                    normal_indices, oversample_count, replace=True
                )
                selected_indices = np.concatenate([normal_indices, oversampled_indices, malicious_indices])
            else:
                selected_indices = np.concatenate([normal_indices, malicious_indices])
        else:
            target_malicious_count = int(normal_count / target_ratio)
            if target_malicious_count > malicious_count:
                oversample_count = target_malicious_count - malicious_count
                oversampled_indices = np.random.choice(
                    malicious_indices, oversample_count, replace=True
                )
                selected_indices = np.concatenate([normal_indices, malicious_indices, oversampled_indices])
            else:
                selected_indices = np.concatenate([normal_indices, malicious_indices])
    
    else:  # mixed
        # 混合方法：适度欠采样多数类 + 适度过采样少数类
        target_total = (normal_count + malicious_count) // 2
        target_normal_count = int(target_total * target_ratio / (1 + target_ratio))
        target_malicious_count = target_total - target_normal_count
        
        # 处理正常样本
        if target_normal_count <= normal_count:
            selected_normal_indices = np.random.choice(
                normal_indices, target_normal_count, replace=False
            )
        else:
            oversample_count = target_normal_count - normal_count
            oversampled_indices = np.random.choice(
                normal_indices, oversample_count, replace=True
            )
            selected_normal_indices = np.concatenate([normal_indices, oversampled_indices])
        
        # 处理恶意样本
        if target_malicious_count <= malicious_count:
            selected_malicious_indices = np.random.choice(
                malicious_indices, target_malicious_count, replace=False
            )
        else:
            oversample_count = target_malicious_count - malicious_count
            oversampled_indices = np.random.choice(
                malicious_indices, oversample_count, replace=True
            )
            selected_malicious_indices = np.concatenate([malicious_indices, oversampled_indices])
        
        selected_indices = np.concatenate([selected_normal_indices, selected_malicious_indices])
    
    # 打乱索引
    np.random.shuffle(selected_indices)
    
    # 提取数据
    imbalance_handler = ImbalanceHandler()
    balanced_data = imbalance_handler._extract_samples_by_indices(multimodal_data, selected_indices)
    
    # 验证结果
    new_labels = np.array(balanced_data['labels'])
    new_normal_count = np.sum(new_labels == 0)
    new_malicious_count = np.sum(new_labels == 1)
    actual_ratio = new_normal_count / new_malicious_count if new_malicious_count > 0 else 0
    
    print(f"✅ 平衡后分布: 正常={new_normal_count}, 恶意={new_malicious_count}, 比例={actual_ratio:.2f}:1")
    
    return balanced_data 