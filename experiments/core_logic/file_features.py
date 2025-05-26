#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件操作特征编码模块
提取文件活动特征，包括文件类型、大小、路径、USB传输等，兼容不同CERT数据版本
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Any
from utils import FeatureEncoder

def encode_file_features(event_dict: Dict[str, Any], 
                        feature_encoder: FeatureEncoder,
                        data_version: str = 'r4.2') -> Tuple[np.ndarray, np.ndarray]:
    """
    编码文件操作特征（兼容原始feature_extraction.py的逻辑）
    
    Args:
        event_dict: 文件事件字典
        feature_encoder: 特征编码器
        data_version: 数据版本
        
    Returns:
        Tuple[file_features, mask]: 文件特征向量和mask
    """
    features = []
    masks = []
    
    # 1. 文件类型特征
    type_features, type_mask = encode_file_type_features(event_dict)
    features.extend(type_features)
    masks.extend(type_mask)
    
    # 2. 文件大小特征
    size_features, size_mask = encode_file_size_features(event_dict, feature_encoder)
    features.extend(size_features)
    masks.extend(size_mask)
    
    # 3. 文件路径特征
    path_features, path_mask = encode_file_path_features(event_dict)
    features.extend(path_features)
    masks.extend(path_mask)
    
    # 4. 磁盘位置特征
    disk_features, disk_mask = encode_disk_location_features(event_dict)
    features.extend(disk_features)
    masks.extend(disk_mask)
    
    # 5. 根据数据版本添加额外特征
    if data_version in ['r5.1', 'r5.2', 'r6.1', 'r6.2']:
        # 文件活动类型特征
        activity_features, activity_mask = encode_file_activity_features(event_dict)
        features.extend(activity_features)
        masks.extend(activity_mask)
        
        # USB传输特征
        usb_features, usb_mask = encode_usb_transfer_features(event_dict)
        features.extend(usb_features)
        masks.extend(usb_mask)
    
    return np.array(features, dtype=np.float32), np.array(masks, dtype=bool)

def encode_file_type_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码文件类型特征（基于原始file_process函数）
    
    Args:
        event_dict: 文件事件字典
        
    Returns:
        文件类型特征和mask
    """
    features = []
    masks = []
    
    file_path = event_dict.get('url/fname', '')
    
    if file_path and not pd.isna(file_path):
        # 提取文件扩展名
        try:
            if '.' in file_path:
                extension = file_path.split('.')[-1].lower()
            else:
                extension = 'unknown'
        except:
            extension = 'unknown'
        
        # 文件类型分类（基于原始逻辑）
        file_type_encoding = get_file_type_encoding(extension)
        
        # 文件类型one-hot编码：other, zip, image, doc, text, exe
        type_onehot = [0.0] * 6
        if 1 <= file_type_encoding <= 6:
            type_onehot[file_type_encoding - 1] = 1.0
        
        features.extend(type_onehot)
        masks.extend([True] * 6)
        
        # 文件风险等级
        risk_level = get_file_risk_level(extension)
        features.append(risk_level)
        masks.append(True)
        
    else:
        # 无文件路径信息
        features.extend([0.0] * 7)  # 6个类型 + 1个风险等级
        masks.extend([False] * 7)
    
    return features, masks

def encode_file_size_features(event_dict: Dict[str, Any], 
                             feature_encoder: FeatureEncoder) -> Tuple[List[float], List[bool]]:
    """
    编码文件大小特征
    
    Args:
        event_dict: 文件事件字典
        feature_encoder: 特征编码器
        
    Returns:
        文件大小特征和mask
    """
    features = []
    masks = []
    
    # 从content字段获取文件大小（内容长度）
    content = event_dict.get('content', '')
    
    if content and not pd.isna(content):
        content_length = len(str(content))
        
        # 对数标准化（文件大小变化范围很大）
        if content_length > 0:
            size_normalized = np.log10(content_length + 1) / 10.0  # 假设最大10^10字节
        else:
            size_normalized = 0.0
            
        features.append(size_normalized)
        masks.append(True)
        
        # 文件大小分类
        size_category = get_file_size_category(content_length)
        features.append(size_category)
        masks.append(True)
        
    else:
        # 无内容信息
        features.extend([0.0, 0.0])
        masks.extend([False, False])
    
    # 单词数量（如果是文本文件）
    if content and not pd.isna(content):
        word_count = len(str(content).split())
        word_count_normalized = min(word_count / 10000.0, 1.0)  # 标准化
        features.append(word_count_normalized)
        masks.append(True)
    else:
        features.append(0.0)
        masks.append(False)
    
    return features, masks

def encode_file_path_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码文件路径特征
    
    Args:
        event_dict: 文件事件字典
        
    Returns:
        路径特征和mask
    """
    features = []
    masks = []
    
    file_path = event_dict.get('url/fname', '')
    
    if file_path and not pd.isna(file_path):
        path_str = str(file_path)
        
        # 文件深度（目录层级）
        depth = path_str.count('\\') if '\\' in path_str else path_str.count('/')
        depth_normalized = min(depth / 20.0, 1.0)  # 标准化，最大20层
        features.append(depth_normalized)
        masks.append(True)
        
        # 路径长度
        path_length = len(path_str)
        path_length_normalized = min(path_length / 500.0, 1.0)  # 标准化
        features.append(path_length_normalized)
        masks.append(True)
        
        # 是否在系统目录
        is_system_path = any(sys_dir in path_str.lower() for sys_dir in 
                           ['windows', 'system32', 'program files', 'users'])
        features.append(1.0 if is_system_path else 0.0)
        masks.append(True)
        
        # 是否在用户目录
        is_user_path = any(user_dir in path_str.lower() for user_dir in 
                          ['desktop', 'documents', 'downloads', 'pictures'])
        features.append(1.0 if is_user_path else 0.0)
        masks.append(True)
        
        # 是否临时文件
        is_temp_file = any(temp_indicator in path_str.lower() for temp_indicator in 
                          ['temp', 'tmp', 'cache'])
        features.append(1.0 if is_temp_file else 0.0)
        masks.append(True)
        
    else:
        # 无路径信息
        features.extend([0.0] * 5)
        masks.extend([False] * 5)
    
    return features, masks

def encode_disk_location_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码磁盘位置特征
    
    Args:
        event_dict: 文件事件字典
        
    Returns:
        磁盘位置特征和mask
    """
    features = []
    masks = []
    
    file_path = event_dict.get('url/fname', '')
    
    if file_path and not pd.isna(file_path):
        path_str = str(file_path)
        
        # 磁盘类型判断（基于原始逻辑）
        disk_type = 0  # 默认其他
        if path_str.startswith('C:') or path_str.startswith('C\\'):
            disk_type = 1  # C盘（本地磁盘）
        elif path_str.startswith('R:') or path_str.startswith('R\\'):
            disk_type = 2  # R盘（可能是网络驱动器）
        
        # 磁盘类型one-hot编码
        disk_onehot = [0.0, 0.0, 0.0]
        if 0 <= disk_type <= 2:
            disk_onehot[disk_type] = 1.0
        
        features.extend(disk_onehot)
        masks.extend([True] * 3)
        
    else:
        # 无路径信息
        features.extend([0.0] * 3)
        masks.extend([False] * 3)
    
    return features, masks

def encode_file_activity_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码文件活动类型特征（新版本数据）
    
    Args:
        event_dict: 文件事件字典
        
    Returns:
        活动特征和mask
    """
    features = []
    masks = []
    
    activity = event_dict.get('activity', '')
    
    # 文件活动类型映射
    activity_mapping = {
        'file open': 1,
        'file copy': 2,
        'file write': 3,
        'file delete': 4
    }
    
    activity_code = activity_mapping.get(str(activity).lower(), 0)
    
    # 活动类型one-hot编码
    activity_onehot = [0.0] * 5  # unknown, open, copy, write, delete
    if 0 <= activity_code <= 4:
        activity_onehot[activity_code] = 1.0
    
    features.extend(activity_onehot)
    masks.extend([True] * 5)
    
    # 活动风险等级
    risk_level = get_activity_risk_level(activity_code)
    features.append(risk_level)
    masks.append(True)
    
    return features, masks

def encode_usb_transfer_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码USB传输特征（新版本数据）
    
    Args:
        event_dict: 文件事件字典
        
    Returns:
        USB传输特征和mask
    """
    features = []
    masks = []
    
    # 传输到USB
    to_usb = event_dict.get('to', '')
    is_to_usb = 1.0 if str(to_usb).lower() == 'true' else 0.0
    features.append(is_to_usb)
    masks.append(True)
    
    # 从USB传输
    from_usb = event_dict.get('from', '')
    is_from_usb = 1.0 if str(from_usb).lower() == 'true' else 0.0
    features.append(is_from_usb)
    masks.append(True)
    
    # USB传输风险评分
    usb_risk = 0.0
    if is_to_usb > 0:
        usb_risk += 0.7  # 向USB传输风险较高
    if is_from_usb > 0:
        usb_risk += 0.3  # 从USB传输风险较低
    
    features.append(min(usb_risk, 1.0))
    masks.append(True)
    
    return features, masks

def get_file_type_encoding(extension: str) -> int:
    """
    获取文件类型编码（基于原始逻辑）
    
    Args:
        extension: 文件扩展名
        
    Returns:
        文件类型编码：1=other, 2=zip, 3=image, 4=doc, 5=text, 6=exe
    """
    ext_lower = extension.lower()
    
    if ext_lower in ['zip', 'rar', '7z', 'tar', 'gz']:
        return 2  # 压缩文件
    elif ext_lower in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff']:
        return 3  # 图像文件
    elif ext_lower in ['doc', 'docx', 'pdf', 'xls', 'xlsx', 'ppt', 'pptx']:
        return 4  # 文档文件
    elif ext_lower in ['txt', 'cfg', 'rtf', 'log', 'xml', 'json']:
        return 5  # 文本文件
    elif ext_lower in ['exe', 'sh', 'bat', 'com', 'cmd', 'msi']:
        return 6  # 可执行文件
    else:
        return 1  # 其他文件

def get_file_risk_level(extension: str) -> float:
    """
    获取文件风险等级
    
    Args:
        extension: 文件扩展名
        
    Returns:
        风险等级 (0.0-1.0)
    """
    ext_lower = extension.lower()
    
    # 高风险文件类型
    if ext_lower in ['exe', 'bat', 'cmd', 'com', 'scr', 'pif']:
        return 1.0
    elif ext_lower in ['sh', 'msi', 'vbs', 'js']:
        return 0.8
    elif ext_lower in ['zip', 'rar', '7z']:
        return 0.6  # 压缩文件可能包含恶意内容
    elif ext_lower in ['doc', 'docx', 'xls', 'xlsx', 'pdf']:
        return 0.4  # 文档可能包含宏
    elif ext_lower in ['txt', 'cfg', 'log']:
        return 0.2  # 配置文件可能敏感
    else:
        return 0.1  # 低风险

def get_file_size_category(size_bytes: int) -> float:
    """
    获取文件大小分类
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        大小分类 (0.0-1.0)
    """
    if size_bytes < 1024:  # < 1KB
        return 0.1
    elif size_bytes < 1024 * 1024:  # < 1MB
        return 0.3
    elif size_bytes < 10 * 1024 * 1024:  # < 10MB
        return 0.5
    elif size_bytes < 100 * 1024 * 1024:  # < 100MB
        return 0.7
    else:  # >= 100MB
        return 1.0

def get_activity_risk_level(activity_code: int) -> float:
    """
    获取文件活动风险等级
    
    Args:
        activity_code: 活动代码
        
    Returns:
        风险等级 (0.0-1.0)
    """
    risk_mapping = {
        0: 0.1,  # unknown
        1: 0.2,  # file open
        2: 0.7,  # file copy (高风险)
        3: 0.5,  # file write
        4: 0.8   # file delete (高风险)
    }
    
    return risk_mapping.get(activity_code, 0.1)

def analyze_file_patterns(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    分析文件操作模式
    
    Args:
        events: 文件事件列表
        
    Returns:
        文件模式特征字典
    """
    if not events:
        return {}
    
    patterns = {}
    
    # 文件类型分布
    type_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}  # other, zip, image, doc, text, exe
    
    for event in events:
        file_path = event.get('url/fname', '')
        if file_path and not pd.isna(file_path):
            try:
                extension = file_path.split('.')[-1] if '.' in file_path else 'unknown'
                file_type = get_file_type_encoding(extension)
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
            except:
                type_counts[1] += 1
    
    total_files = sum(type_counts.values())
    if total_files > 0:
        for file_type, count in type_counts.items():
            patterns[f'type_{file_type}_ratio'] = count / total_files
    
    # 活动类型分布
    activity_counts = {'open': 0, 'copy': 0, 'write': 0, 'delete': 0}
    
    for event in events:
        activity = str(event.get('activity', '')).lower()
        if 'open' in activity:
            activity_counts['open'] += 1
        elif 'copy' in activity:
            activity_counts['copy'] += 1
        elif 'write' in activity:
            activity_counts['write'] += 1
        elif 'delete' in activity:
            activity_counts['delete'] += 1
    
    for activity, count in activity_counts.items():
        patterns[f'{activity}_ratio'] = count / len(events)
    
    # USB传输模式
    usb_to_count = sum(1 for e in events if str(e.get('to', '')).lower() == 'true')
    usb_from_count = sum(1 for e in events if str(e.get('from', '')).lower() == 'true')
    
    patterns['usb_to_ratio'] = usb_to_count / len(events)
    patterns['usb_from_ratio'] = usb_from_count / len(events)
    
    # 文件大小统计
    sizes = []
    for event in events:
        content = event.get('content', '')
        if content and not pd.isna(content):
            sizes.append(len(str(content)))
    
    if sizes:
        patterns['avg_file_size'] = np.mean(sizes)
        patterns['file_size_variance'] = np.var(sizes)
        patterns['large_file_ratio'] = sum(1 for s in sizes if s > 10000000) / len(sizes)  # >10MB
    else:
        patterns['avg_file_size'] = 0
        patterns['file_size_variance'] = 0
        patterns['large_file_ratio'] = 0
    
    return patterns

def detect_suspicious_file_patterns(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    检测可疑的文件操作模式
    
    Args:
        events: 文件事件列表
        
    Returns:
        可疑模式评分字典
    """
    if not events:
        return {}
    
    suspicious_patterns = {}
    
    # 大量可执行文件操作
    exe_count = 0
    for event in events:
        file_path = event.get('url/fname', '')
        if file_path and not pd.isna(file_path):
            extension = file_path.split('.')[-1].lower() if '.' in file_path else ''
            if extension in ['exe', 'bat', 'cmd', 'sh']:
                exe_count += 1
    
    exe_ratio = exe_count / len(events)
    suspicious_patterns['high_executable_ratio'] = min(exe_ratio * 10, 1.0)
    
    # 大量删除操作
    delete_count = sum(1 for e in events if 'delete' in str(e.get('activity', '')).lower())
    delete_ratio = delete_count / len(events)
    suspicious_patterns['high_delete_ratio'] = min(delete_ratio * 5, 1.0)
    
    # 大量USB传输
    usb_count = sum(1 for e in events if 
                   str(e.get('to', '')).lower() == 'true' or 
                   str(e.get('from', '')).lower() == 'true')
    usb_ratio = usb_count / len(events)
    suspicious_patterns['high_usb_ratio'] = min(usb_ratio * 3, 1.0)
    
    # 系统文件访问
    system_file_count = 0
    for event in events:
        file_path = event.get('url/fname', '')
        if file_path and not pd.isna(file_path):
            path_lower = str(file_path).lower()
            if any(sys_dir in path_lower for sys_dir in 
                  ['system32', 'windows', 'program files']):
                system_file_count += 1
    
    system_ratio = system_file_count / len(events)
    suspicious_patterns['system_file_access_ratio'] = min(system_ratio * 5, 1.0)
    
    # 夜间文件操作（需要时间戳信息）
    # 这里可以结合时间特征进行分析
    
    return suspicious_patterns

def get_file_sensitivity_score(file_path: str, content: str = '') -> float:
    """
    计算文件敏感度评分
    
    Args:
        file_path: 文件路径
        content: 文件内容
        
    Returns:
        敏感度评分 (0.0-1.0)
    """
    score = 0.0
    
    # 基于路径的敏感度
    if file_path:
        path_lower = file_path.lower()
        
        # 系统关键目录
        if any(sys_dir in path_lower for sys_dir in 
              ['system32', 'windows', 'program files']):
            score += 0.4
        
        # 用户敏感目录
        if any(user_dir in path_lower for user_dir in 
              ['documents', 'desktop', 'passwords']):
            score += 0.3
        
        # 敏感文件名
        if any(sensitive in path_lower for sensitive in 
              ['password', 'secret', 'confidential', 'private']):
            score += 0.5
    
    # 基于内容的敏感度
    if content:
        content_lower = content.lower()
        
        # 敏感关键词
        sensitive_keywords = [
            'password', 'secret', 'confidential', 'private',
            'credit card', 'ssn', 'social security',
            'bank account', 'financial', 'salary'
        ]
        
        keyword_count = sum(1 for keyword in sensitive_keywords 
                          if keyword in content_lower)
        score += min(keyword_count * 0.2, 0.6)
    
    return min(score, 1.0) 