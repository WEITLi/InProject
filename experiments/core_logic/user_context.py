#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户上下文编码模块
提取用户上下文特征，包括角色、部门、IT管理员身份、OCEAN心理特征等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from utils import FeatureEncoder

def encode_user_context(user_context: Dict[str, Any], 
                       feature_encoder: FeatureEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """
    编码用户上下文特征
    
    Args:
        user_context: 用户上下文字典，包含角色、部门、OCEAN等信息
        feature_encoder: 特征编码器
        
    Returns:
        Tuple[user_features, mask]: 用户上下文特征向量和mask
    """
    features = []
    masks = []
    
    # 1. 角色特征编码
    role_features, role_mask = encode_role_features(user_context, feature_encoder)
    features.extend(role_features)
    masks.extend(role_mask)
    
    # 2. 部门特征编码
    dept_features, dept_mask = encode_department_features(user_context, feature_encoder)
    features.extend(dept_features)
    masks.extend(dept_mask)
    
    # 3. IT管理员身份
    is_it_admin = user_context.get('ITAdmin', 0)
    features.append(float(is_it_admin))
    masks.append(True)
    
    # 4. OCEAN心理特征
    ocean_features, ocean_mask = encode_ocean_features(user_context, feature_encoder)
    features.extend(ocean_features)
    masks.extend(ocean_mask)
    
    # 5. PC使用特征
    pc_features, pc_mask = encode_pc_usage_features(user_context, feature_encoder)
    features.extend(pc_features)
    masks.extend(pc_mask)
    
    # 6. 权限级别特征
    permission_features, permission_mask = encode_permission_features(user_context)
    features.extend(permission_features)
    masks.extend(permission_mask)
    
    return np.array(features, dtype=np.float32), np.array(masks, dtype=bool)

def encode_role_features(user_context: Dict[str, Any], 
                        feature_encoder: FeatureEncoder) -> Tuple[List[float], List[bool]]:
    """
    编码用户角色特征
    
    Args:
        user_context: 用户上下文
        feature_encoder: 特征编码器
        
    Returns:
        角色特征和mask
    """
    role = user_context.get('role', 'unknown')
    
    # 角色层级编码
    role_hierarchy = {
        'Employee': 1,
        'Supervisor': 2, 
        'Manager': 3,
        'Director': 4,
        'Executive': 5,
        'unknown': 0
    }
    
    role_level = role_hierarchy.get(role, 0) / 5.0  # 标准化到0-1
    
    # 角色类型one-hot编码
    role_types = ['Employee', 'Supervisor', 'Manager', 'Director', 'Executive']
    role_onehot = [1.0 if role == rtype else 0.0 for rtype in role_types]
    
    features = [role_level] + role_onehot
    masks = [True] * len(features)
    
    return features, masks

def encode_department_features(user_context: Dict[str, Any], 
                             feature_encoder: FeatureEncoder) -> Tuple[List[float], List[bool]]:
    """
    编码部门特征
    
    Args:
        user_context: 用户上下文
        feature_encoder: 特征编码器
        
    Returns:
        部门特征和mask
    """
    dept = user_context.get('dept', 'unknown')
    
    # 部门敏感度编码（基于数据访问敏感性）
    dept_sensitivity = {
        'IT': 5,           # 最高敏感性
        'Finance': 4,      # 财务敏感
        'HR': 4,           # 人事敏感
        'Legal': 4,        # 法务敏感
        'Executive': 5,    # 高管敏感
        'Marketing': 2,    # 中等敏感
        'Sales': 2,        # 中等敏感
        'Engineering': 3,  # 技术敏感
        'Operations': 2,   # 运营敏感
        'unknown': 1       # 低敏感性
    }
    
    sensitivity_score = dept_sensitivity.get(dept, 1) / 5.0  # 标准化
    
    # 部门类型特征
    is_technical_dept = 1.0 if dept in ['IT', 'Engineering'] else 0.0
    is_business_critical = 1.0 if dept in ['Finance', 'HR', 'Legal', 'Executive'] else 0.0
    is_customer_facing = 1.0 if dept in ['Marketing', 'Sales'] else 0.0
    
    features = [sensitivity_score, is_technical_dept, is_business_critical, is_customer_facing]
    masks = [True] * len(features)
    
    return features, masks

def encode_ocean_features(user_context: Dict[str, Any], 
                         feature_encoder: FeatureEncoder) -> Tuple[List[float], List[bool]]:
    """
    编码OCEAN心理特征
    
    Args:
        user_context: 用户上下文
        feature_encoder: 特征编码器
        
    Returns:
        OCEAN特征和mask
    """
    ocean_traits = ['O', 'C', 'E', 'A', 'N']
    features = []
    masks = []
    
    for trait in ocean_traits:
        value = user_context.get(trait)
        if value is not None:
            # 假设OCEAN值已经是0-1范围，如果不是需要标准化
            try:
                normalized_value = float(value)
                if normalized_value > 1.0:
                    normalized_value = normalized_value / 5.0  # 假设原始范围是1-5
                features.append(normalized_value)
                masks.append(True)
            except (ValueError, TypeError):
                features.append(0.0)
                masks.append(False)
        else:
            features.append(0.0)
            masks.append(False)
    
    return features, masks

def encode_pc_usage_features(user_context: Dict[str, Any], 
                           feature_encoder: FeatureEncoder) -> Tuple[List[float], List[bool]]:
    """
    编码PC使用特征
    
    Args:
        user_context: 用户上下文
        feature_encoder: 特征编码器
        
    Returns:
        PC使用特征和mask
    """
    features = []
    masks = []
    
    # PC类型（个人PC vs 共享PC）
    pc_type = user_context.get('pc_type', 0)  # 0: 个人, 1: 共享, 2: 他人, 3: 主管
    pc_type_normalized = pc_type / 3.0
    features.append(pc_type_normalized)
    masks.append(True)
    
    # 是否使用共享PC
    has_shared_pc = user_context.get('sharedpc') is not None
    features.append(1.0 if has_shared_pc else 0.0)
    masks.append(True)
    
    # PC数量（如果有多台PC）
    pc_count = user_context.get('npc', 1)
    pc_count_normalized = min(pc_count / 5.0, 1.0)  # 最多5台PC
    features.append(pc_count_normalized)
    masks.append(True)
    
    return features, masks

def encode_permission_features(user_context: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码权限级别特征
    
    Args:
        user_context: 用户上下文
        
    Returns:
        权限特征和mask
    """
    features = []
    masks = []
    
    # 基于角色和部门推断权限级别
    role = user_context.get('role', 'Employee')
    dept = user_context.get('dept', 'unknown')
    is_it_admin = user_context.get('ITAdmin', 0)
    
    # 数据访问权限级别
    data_access_level = 0
    if is_it_admin:
        data_access_level = 5
    elif role in ['Executive', 'Director']:
        data_access_level = 4
    elif role == 'Manager':
        data_access_level = 3
    elif role == 'Supervisor':
        data_access_level = 2
    else:
        data_access_level = 1
    
    # 部门特殊权限调整
    if dept in ['IT', 'Finance', 'HR']:
        data_access_level = min(data_access_level + 1, 5)
    
    features.append(data_access_level / 5.0)
    masks.append(True)
    
    # 系统管理权限
    system_admin_level = 1.0 if is_it_admin else 0.0
    features.append(system_admin_level)
    masks.append(True)
    
    return features, masks

def encode_supervisor_relationship(user_context: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码上下级关系特征
    
    Args:
        user_context: 用户上下文
        
    Returns:
        关系特征和mask
    """
    features = []
    masks = []
    
    # 是否有直接下属
    has_subordinates = user_context.get('role') in ['Supervisor', 'Manager', 'Director', 'Executive']
    features.append(1.0 if has_subordinates else 0.0)
    masks.append(True)
    
    # 是否有直接上级
    has_supervisor = user_context.get('sup') is not None
    features.append(1.0 if has_supervisor else 0.0)
    masks.append(True)
    
    return features, masks

def encode_behavioral_risk_profile(user_context: Dict[str, Any]) -> Dict[str, float]:
    """
    基于用户上下文计算行为风险画像
    
    Args:
        user_context: 用户上下文
        
    Returns:
        风险画像特征字典
    """
    risk_profile = {}
    
    # 特权用户风险
    is_it_admin = user_context.get('ITAdmin', 0)
    role = user_context.get('role', 'Employee')
    dept = user_context.get('dept', 'unknown')
    
    privilege_risk = 0
    if is_it_admin:
        privilege_risk += 0.4
    if role in ['Executive', 'Director']:
        privilege_risk += 0.3
    if dept in ['IT', 'Finance', 'HR']:
        privilege_risk += 0.2
    if role == 'Manager':
        privilege_risk += 0.1
    
    risk_profile['privilege_risk'] = min(privilege_risk, 1.0)
    
    # 数据访问风险
    data_access_risk = 0
    if dept in ['Finance', 'HR', 'Legal']:
        data_access_risk += 0.3
    if role in ['Manager', 'Director', 'Executive']:
        data_access_risk += 0.2
    if is_it_admin:
        data_access_risk += 0.4
    
    risk_profile['data_access_risk'] = min(data_access_risk, 1.0)
    
    # 心理风险（基于OCEAN特征）
    psychological_risk = 0
    if 'N' in user_context:  # 神经质
        try:
            neuroticism = float(user_context['N'])
            if neuroticism > 1:
                neuroticism /= 5.0  # 标准化
            psychological_risk += neuroticism * 0.3
        except:
            pass
    
    if 'C' in user_context:  # 责任心（低责任心=高风险）
        try:
            conscientiousness = float(user_context['C'])
            if conscientiousness > 1:
                conscientiousness /= 5.0
            psychological_risk += (1.0 - conscientiousness) * 0.2
        except:
            pass
    
    risk_profile['psychological_risk'] = min(psychological_risk, 1.0)
    
    # 综合风险评分
    risk_profile['overall_risk'] = (
        risk_profile['privilege_risk'] * 0.4 +
        risk_profile['data_access_risk'] * 0.4 +
        risk_profile['psychological_risk'] * 0.2
    )
    
    return risk_profile

def get_user_context_from_dataframe(user_id: str, 
                                   user_df: pd.DataFrame) -> Dict[str, Any]:
    """
    从用户数据框中提取用户上下文
    
    Args:
        user_id: 用户ID
        user_df: 用户数据框
        
    Returns:
        用户上下文字典
    """
    if user_id not in user_df.index:
        return {'user_id': user_id}
    
    user_row = user_df.loc[user_id]
    context = {'user_id': user_id}
    
    # 复制所有可用字段
    for col in user_df.columns:
        context[col] = user_row[col]
    
    return context

def validate_user_context(user_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证和清理用户上下文数据
    
    Args:
        user_context: 原始用户上下文
        
    Returns:
        清理后的用户上下文
    """
    cleaned_context = {}
    
    # 必要字段的默认值
    defaults = {
        'role': 'Employee',
        'dept': 'unknown',
        'ITAdmin': 0,
        'O': 0.5, 'C': 0.5, 'E': 0.5, 'A': 0.5, 'N': 0.5
    }
    
    # 应用默认值
    for key, default_value in defaults.items():
        cleaned_context[key] = user_context.get(key, default_value)
    
    # 复制其他字段
    for key, value in user_context.items():
        if key not in cleaned_context:
            cleaned_context[key] = value
    
    # 数据类型转换和验证
    if 'ITAdmin' in cleaned_context:
        try:
            cleaned_context['ITAdmin'] = int(cleaned_context['ITAdmin'])
        except:
            cleaned_context['ITAdmin'] = 0
    
    # OCEAN特征验证
    for trait in ['O', 'C', 'E', 'A', 'N']:
        if trait in cleaned_context:
            try:
                value = float(cleaned_context[trait])
                # 确保在合理范围内
                if value > 5:
                    value = value / 5.0
                cleaned_context[trait] = max(0.0, min(1.0, value))
            except:
                cleaned_context[trait] = 0.5
    
    return cleaned_context 