#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备特征编码模块
提取设备连接活动特征，包括USB设备、外接存储等设备操作
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from utils import FeatureEncoder

def encode_device_features(event_dict: Dict[str, Any], 
                          feature_encoder: FeatureEncoder,
                          data_version: str = 'r4.2') -> Tuple[np.ndarray, np.ndarray]:
    """
    编码设备特征（兼容原始feature_extraction.py的逻辑）
    
    Args:
        event_dict: 设备事件字典
        feature_encoder: 特征编码器
        data_version: 数据版本
        
    Returns:
        Tuple[device_features, mask]: 设备特征向量和mask
    """
    features = []
    masks = []
    
    # 1. 设备活动类型特征
    activity_features, activity_mask = encode_device_activity_features(event_dict)
    features.extend(activity_features)
    masks.extend(activity_mask)
    
    # 2. 设备使用时长特征
    duration_features, duration_mask = encode_device_duration_features(event_dict)
    features.extend(duration_features)
    masks.extend(duration_mask)
    
    # 3. 根据数据版本添加额外特征
    if data_version in ['r5.1', 'r5.2', 'r6.1', 'r6.2']:
        # 设备内容特征
        content_features, content_mask = encode_device_content_features(event_dict)
        features.extend(content_features)
        masks.extend(content_mask)
        
        # 文件树长度特征（r5.2+）
        if data_version not in ['r4.1', 'r4.2']:
            tree_features, tree_mask = encode_file_tree_features(event_dict)
            features.extend(tree_features)
            masks.extend(tree_mask)
    
    return np.array(features, dtype=np.float32), np.array(masks, dtype=bool)

def encode_device_activity_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码设备活动类型特征
    
    Args:
        event_dict: 设备事件字典
        
    Returns:
        活动特征和mask
    """
    features = []
    masks = []
    
    activity = event_dict.get('activity', '')
    
    if activity and not pd.isna(activity):
        activity_str = str(activity).lower()
        
        # 设备活动类型分类
        activity_type = classify_device_activity(activity_str)
        
        # 活动类型one-hot编码：unknown, connect, disconnect, access, other
        activity_onehot = [0.0] * 5
        if 0 <= activity_type <= 4:
            activity_onehot[activity_type] = 1.0
        
        features.extend(activity_onehot)
        masks.extend([True] * 5)
        
        # 活动风险等级
        risk_level = get_device_activity_risk_level(activity_type)
        features.append(risk_level)
        masks.append(True)
        
        # 是否为USB设备
        is_usb = 1.0 if 'usb' in activity_str else 0.0
        features.append(is_usb)
        masks.append(True)
        
    else:
        # 无活动信息
        features.extend([0.0] * 7)  # 5个类型 + 风险等级 + USB标识
        masks.extend([False] * 7)
    
    return features, masks

def encode_device_duration_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码设备使用时长特征
    
    Args:
        event_dict: 设备事件字典
        
    Returns:
        时长特征和mask
    """
    features = []
    masks = []
    
    # 从事件中提取时长信息（可能需要根据实际数据格式调整）
    # 这里假设有duration字段或者可以从其他字段推断
    
    # 模拟USB使用时长（分钟）
    # 在实际实现中，这可能需要从会话级别数据计算
    duration = event_dict.get('usb_duration', 0)
    
    try:
        duration_minutes = float(duration) if duration else 0.0
        
        # 时长标准化（假设最大8小时）
        duration_normalized = min(duration_minutes / 480.0, 1.0)
        features.append(duration_normalized)
        masks.append(True)
        
        # 时长分类
        duration_category = get_duration_category(duration_minutes)
        features.append(duration_category)
        masks.append(True)
        
    except (ValueError, TypeError):
        features.extend([0.0, 0.0])
        masks.extend([False, False])
    
    return features, masks

def encode_device_content_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码设备内容特征（新版本数据）
    
    Args:
        event_dict: 设备事件字典
        
    Returns:
        内容特征和mask
    """
    features = []
    masks = []
    
    content = event_dict.get('content', '')
    
    if content and not pd.isna(content):
        content_str = str(content)
        
        # 内容长度
        content_length = len(content_str)
        content_length_normalized = min(content_length / 1000.0, 1.0)  # 标准化
        features.append(content_length_normalized)
        masks.append(True)
        
        # 设备信息分析
        device_info_features = analyze_device_content(content_str)
        features.extend(device_info_features)
        masks.extend([True] * len(device_info_features))
        
    else:
        # 无内容信息
        features.extend([0.0] * 4)  # 长度 + 3个设备信息特征
        masks.extend([False] * 4)
    
    return features, masks

def encode_file_tree_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码文件树长度特征（r5.2+版本）
    
    Args:
        event_dict: 设备事件字典
        
    Returns:
        文件树特征和mask
    """
    features = []
    masks = []
    
    # 文件树长度（可能表示设备上的文件数量或目录深度）
    file_tree_len = event_dict.get('file_tree_len', 0)
    
    try:
        tree_length = int(file_tree_len) if file_tree_len else 0
        
        # 文件树长度标准化
        tree_length_normalized = min(tree_length / 1000.0, 1.0)  # 假设最大1000个文件
        features.append(tree_length_normalized)
        masks.append(True)
        
    except (ValueError, TypeError):
        features.append(0.0)
        masks.append(False)
    
    return features, masks

def classify_device_activity(activity: str) -> int:
    """
    分类设备活动类型
    
    Args:
        activity: 活动字符串
        
    Returns:
        活动类型：0=unknown, 1=connect, 2=disconnect, 3=access, 4=other
    """
    activity_lower = activity.lower()
    
    if any(keyword in activity_lower for keyword in ['connect', 'plug', 'insert']):
        return 1  # 连接
    elif any(keyword in activity_lower for keyword in ['disconnect', 'unplug', 'remove']):
        return 2  # 断开连接
    elif any(keyword in activity_lower for keyword in ['access', 'read', 'write', 'copy']):
        return 3  # 访问
    elif any(keyword in activity_lower for keyword in ['device', 'usb', 'storage']):
        return 4  # 其他设备活动
    else:
        return 0  # 未知

def get_device_activity_risk_level(activity_type: int) -> float:
    """
    获取设备活动风险等级
    
    Args:
        activity_type: 活动类型
        
    Returns:
        风险等级 (0.0-1.0)
    """
    risk_mapping = {
        0: 0.1,  # unknown
        1: 0.3,  # connect - 中等风险
        2: 0.2,  # disconnect - 低风险
        3: 0.8,  # access - 高风险（数据传输）
        4: 0.4   # other - 中等风险
    }
    
    return risk_mapping.get(activity_type, 0.1)

def get_duration_category(duration_minutes: float) -> float:
    """
    获取使用时长分类
    
    Args:
        duration_minutes: 时长（分钟）
        
    Returns:
        时长分类 (0.0-1.0)
    """
    if duration_minutes < 1:  # < 1分钟
        return 0.1
    elif duration_minutes < 5:  # < 5分钟
        return 0.3
    elif duration_minutes < 30:  # < 30分钟
        return 0.5
    elif duration_minutes < 120:  # < 2小时
        return 0.7
    else:  # >= 2小时
        return 1.0

def analyze_device_content(content: str) -> List[float]:
    """
    分析设备内容特征
    
    Args:
        content: 设备内容字符串
        
    Returns:
        设备内容特征列表
    """
    features = []
    
    content_lower = content.lower()
    
    # 是否包含存储设备信息
    has_storage_info = 1.0 if any(keyword in content_lower for keyword in 
                                 ['storage', 'disk', 'drive', 'volume']) else 0.0
    features.append(has_storage_info)
    
    # 是否包含USB设备信息
    has_usb_info = 1.0 if any(keyword in content_lower for keyword in 
                             ['usb', 'flash', 'thumb', 'portable']) else 0.0
    features.append(has_usb_info)
    
    # 是否包含文件系统信息
    has_filesystem_info = 1.0 if any(keyword in content_lower for keyword in 
                                    ['ntfs', 'fat32', 'exfat', 'filesystem']) else 0.0
    features.append(has_filesystem_info)
    
    return features

def analyze_device_patterns(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    分析设备使用模式
    
    Args:
        events: 设备事件列表
        
    Returns:
        设备模式特征字典
    """
    if not events:
        return {}
    
    patterns = {}
    
    # 活动类型分布
    activity_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for event in events:
        activity = event.get('activity', '')
        if activity and not pd.isna(activity):
            activity_type = classify_device_activity(str(activity))
            activity_counts[activity_type] = activity_counts.get(activity_type, 0) + 1
    
    total_activities = sum(activity_counts.values())
    if total_activities > 0:
        for activity_type, count in activity_counts.items():
            patterns[f'activity_type_{activity_type}_ratio'] = count / total_activities
    
    # USB设备使用比例
    usb_count = sum(1 for e in events if 'usb' in str(e.get('activity', '')).lower())
    patterns['usb_usage_ratio'] = usb_count / len(events)
    
    # 平均使用时长
    durations = []
    for event in events:
        duration = event.get('usb_duration', 0)
        try:
            durations.append(float(duration) if duration else 0.0)
        except:
            pass
    
    if durations:
        patterns['avg_duration'] = np.mean(durations)
        patterns['duration_variance'] = np.var(durations)
        patterns['long_usage_ratio'] = sum(1 for d in durations if d > 60) / len(durations)  # >1小时
    else:
        patterns['avg_duration'] = 0
        patterns['duration_variance'] = 0
        patterns['long_usage_ratio'] = 0
    
    # 设备连接频率
    connect_count = sum(1 for e in events if 
                       any(keyword in str(e.get('activity', '')).lower() 
                          for keyword in ['connect', 'plug', 'insert']))
    patterns['connect_frequency'] = connect_count / len(events)
    
    return patterns

def detect_suspicious_device_patterns(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    检测可疑的设备使用模式
    
    Args:
        events: 设备事件列表
        
    Returns:
        可疑模式评分字典
    """
    if not events:
        return {}
    
    suspicious_patterns = {}
    
    # 频繁USB使用
    usb_count = sum(1 for e in events if 'usb' in str(e.get('activity', '')).lower())
    usb_ratio = usb_count / len(events)
    suspicious_patterns['high_usb_usage'] = min(usb_ratio * 2, 1.0)
    
    # 长时间设备使用
    long_usage_count = 0
    for event in events:
        duration = event.get('usb_duration', 0)
        try:
            if float(duration) > 120:  # > 2小时
                long_usage_count += 1
        except:
            pass
    
    long_usage_ratio = long_usage_count / len(events)
    suspicious_patterns['long_device_usage'] = min(long_usage_ratio * 5, 1.0)
    
    # 频繁连接/断开
    connect_disconnect_count = 0
    for event in events:
        activity = str(event.get('activity', '')).lower()
        if any(keyword in activity for keyword in 
              ['connect', 'disconnect', 'plug', 'unplug']):
            connect_disconnect_count += 1
    
    connect_disconnect_ratio = connect_disconnect_count / len(events)
    suspicious_patterns['frequent_connect_disconnect'] = min(connect_disconnect_ratio * 3, 1.0)
    
    # 数据访问活动
    access_count = sum(1 for e in events if 
                      any(keyword in str(e.get('activity', '')).lower() 
                         for keyword in ['access', 'read', 'write', 'copy']))
    access_ratio = access_count / len(events)
    suspicious_patterns['data_access_ratio'] = min(access_ratio * 2, 1.0)
    
    return suspicious_patterns

def calculate_device_risk_score(events: List[Dict[str, Any]], 
                               user_context: Dict[str, Any] = None) -> float:
    """
    计算设备使用综合风险评分
    
    Args:
        events: 设备事件列表
        user_context: 用户上下文（可选）
        
    Returns:
        综合风险评分 (0.0-1.0)
    """
    if not events:
        return 0.0
    
    risk_score = 0.0
    
    # 基于活动类型的风险
    activity_risk = 0.0
    for event in events:
        activity = event.get('activity', '')
        if activity and not pd.isna(activity):
            activity_type = classify_device_activity(str(activity))
            activity_risk += get_device_activity_risk_level(activity_type)
    
    activity_risk = activity_risk / len(events)
    risk_score += activity_risk * 0.4
    
    # 基于使用模式的风险
    patterns = analyze_device_patterns(events)
    
    # USB使用频率风险
    usb_risk = patterns.get('usb_usage_ratio', 0) * 0.3
    risk_score += usb_risk
    
    # 长时间使用风险
    long_usage_risk = patterns.get('long_usage_ratio', 0) * 0.2
    risk_score += long_usage_risk
    
    # 频繁连接风险
    connect_risk = patterns.get('connect_frequency', 0) * 0.1
    risk_score += connect_risk
    
    # 用户上下文调整
    if user_context:
        # IT管理员可能有合理的设备使用需求
        if user_context.get('ITAdmin', 0):
            risk_score *= 0.7
        
        # 高权限用户的设备使用风险更高
        role = user_context.get('role', 'Employee')
        if role in ['Director', 'Executive']:
            risk_score *= 1.2
    
    return min(risk_score, 1.0) 