#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邮件特征编码模块
提取邮件活动特征，包括收发件人、附件、大小等，兼容不同CERT数据版本
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any
from utils import FeatureEncoder

def encode_email_features(event_dict: Dict[str, Any], 
                         feature_encoder: FeatureEncoder,
                         data_version: str = 'r4.2') -> Tuple[np.ndarray, np.ndarray]:
    """
    编码邮件特征（兼容原始feature_extraction.py的逻辑）
    
    Args:
        event_dict: 邮件事件字典
        feature_encoder: 特征编码器
        data_version: 数据版本
        
    Returns:
        Tuple[email_features, mask]: 邮件特征向量和mask
    """
    features = []
    masks = []
    
    # 1. 收件人特征
    recipient_features, recipient_mask = encode_recipient_features(event_dict)
    features.extend(recipient_features)
    masks.extend(recipient_mask)
    
    # 2. 邮件大小特征
    size_features, size_mask = encode_email_size_features(event_dict, feature_encoder)
    features.extend(size_features)
    masks.extend(size_mask)
    
    # 3. 邮件内容特征
    content_features, content_mask = encode_email_content_features(event_dict, feature_encoder)
    features.extend(content_features)
    masks.extend(content_mask)
    
    # 4. 外部联系人特征
    external_features, external_mask = encode_external_contact_features(event_dict)
    features.extend(external_features)
    masks.extend(external_mask)
    
    # 5. 根据数据版本添加额外特征
    if data_version in ['r5.1', 'r5.2', 'r6.1', 'r6.2']:
        # 发送/接收活动特征
        activity_features, activity_mask = encode_email_activity_features(event_dict)
        features.extend(activity_features)
        masks.extend(activity_mask)
        
        # 附件特征
        attachment_features, attachment_mask = encode_attachment_features(event_dict, data_version)
        features.extend(attachment_features)
        masks.extend(attachment_mask)
    
    return np.array(features, dtype=np.float32), np.array(masks, dtype=bool)

def encode_recipient_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码收件人相关特征（基于原始email_process函数）
    
    Args:
        event_dict: 邮件事件字典
        
    Returns:
        收件人特征和mask
    """
    features = []
    masks = []
    
    # 解析收件人
    to_recipients = []
    cc_recipients = []
    bcc_recipients = []
    
    # 处理TO字段
    to_field = event_dict.get('to', '')
    if to_field and not pd.isna(to_field):
        to_recipients = [r.strip() for r in str(to_field).split(';') if r.strip()]
    
    # 处理CC字段
    cc_field = event_dict.get('cc', '')
    if cc_field and not pd.isna(cc_field) and str(cc_field) != 'nan':
        cc_recipients = [r.strip() for r in str(cc_field).split(';') if r.strip()]
    
    # 处理BCC字段
    bcc_field = event_dict.get('bcc', '')
    if bcc_field and not pd.isna(bcc_field) and str(bcc_field) != 'nan':
        bcc_recipients = [r.strip() for r in str(bcc_field).split(';') if r.strip()]
    
    all_recipients = to_recipients + cc_recipients + bcc_recipients
    
    # 特征1: 总收件人数量
    n_recipients = len(all_recipients)
    features.append(min(n_recipients / 20.0, 1.0))  # 标准化，最多20个收件人
    masks.append(True)
    
    # 特征2: BCC收件人数量
    n_bcc_recipients = len(bcc_recipients)
    features.append(min(n_bcc_recipients / 10.0, 1.0))  # 标准化
    masks.append(True)
    
    # 特征3: 是否有外部收件人
    has_external = any('dtaa.com' not in recipient for recipient in all_recipients)
    features.append(1.0 if has_external else 0.0)
    masks.append(True)
    
    # 特征4: 外部收件人数量
    n_external = sum(1 for recipient in all_recipients if 'dtaa.com' not in recipient)
    features.append(min(n_external / 10.0, 1.0))  # 标准化
    masks.append(True)
    
    # 特征5: 是否有外部BCC收件人
    has_external_bcc = any('dtaa.com' not in recipient for recipient in bcc_recipients)
    features.append(1.0 if has_external_bcc else 0.0)
    masks.append(True)
    
    return features, masks

def encode_email_size_features(event_dict: Dict[str, Any], 
                              feature_encoder: FeatureEncoder) -> Tuple[List[float], List[bool]]:
    """
    编码邮件大小特征
    
    Args:
        event_dict: 邮件事件字典
        feature_encoder: 特征编码器
        
    Returns:
        大小特征和mask
    """
    features = []
    masks = []
    
    # 邮件大小
    size = event_dict.get('size', 0)
    try:
        size_bytes = int(size) if size else 0
        # 对数标准化（邮件大小通常有很大的变化范围）
        if size_bytes > 0:
            size_normalized = np.log10(size_bytes + 1) / 10.0  # 假设最大10^10字节
        else:
            size_normalized = 0.0
        features.append(size_normalized)
        masks.append(True)
    except (ValueError, TypeError):
        features.append(0.0)
        masks.append(False)
    
    return features, masks

def encode_email_content_features(event_dict: Dict[str, Any], 
                                 feature_encoder: FeatureEncoder) -> Tuple[List[float], List[bool]]:
    """
    编码邮件内容特征
    
    Args:
        event_dict: 邮件事件字典
        feature_encoder: 特征编码器
        
    Returns:
        内容特征和mask
    """
    features = []
    masks = []
    
    content = event_dict.get('content', '')
    
    if content and not pd.isna(content):
        content_str = str(content)
        
        # 特征1: 内容长度
        content_length = len(content_str)
        content_length_normalized = min(content_length / 10000.0, 1.0)  # 标准化，最大10000字符
        features.append(content_length_normalized)
        masks.append(True)
        
        # 特征2: 单词数量
        word_count = len(content_str.split())
        word_count_normalized = min(word_count / 1000.0, 1.0)  # 标准化，最大1000词
        features.append(word_count_normalized)
        masks.append(True)
        
        # 特征3: 是否包含敏感关键词
        sensitive_keywords = ['password', 'confidential', 'secret', 'private', 'urgent', 'important']
        has_sensitive = any(keyword in content_str.lower() for keyword in sensitive_keywords)
        features.append(1.0 if has_sensitive else 0.0)
        masks.append(True)
        
    else:
        # 无内容
        features.extend([0.0, 0.0, 0.0])
        masks.extend([False, False, False])
    
    return features, masks

def encode_external_contact_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码外部联系人特征
    
    Args:
        event_dict: 邮件事件字典
        
    Returns:
        外部联系人特征和mask
    """
    features = []
    masks = []
    
    # 分析发件人
    from_field = event_dict.get('from', '')
    is_external_sender = 'dtaa.com' not in str(from_field) if from_field else False
    features.append(1.0 if is_external_sender else 0.0)
    masks.append(True)
    
    # 分析收件人域名多样性
    all_recipients = []
    for field in ['to', 'cc', 'bcc']:
        field_value = event_dict.get(field, '')
        if field_value and not pd.isna(field_value):
            recipients = [r.strip() for r in str(field_value).split(';') if r.strip()]
            all_recipients.extend(recipients)
    
    # 提取域名
    domains = set()
    for recipient in all_recipients:
        if '@' in recipient:
            domain = recipient.split('@')[-1].lower()
            domains.add(domain)
    
    # 域名多样性
    domain_diversity = min(len(domains) / 5.0, 1.0)  # 标准化，最多5个不同域名
    features.append(domain_diversity)
    masks.append(True)
    
    return features, masks

def encode_email_activity_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码邮件活动特征（发送/接收）
    
    Args:
        event_dict: 邮件事件字典
        
    Returns:
        活动特征和mask
    """
    features = []
    masks = []
    
    activity = event_dict.get('activity', '')
    
    # 发送邮件
    is_send = 1.0 if activity == 'Send' else 0.0
    features.append(is_send)
    masks.append(True)
    
    # 接收/查看邮件
    is_receive = 1.0 if activity in ['Receive', 'View'] else 0.0
    features.append(is_receive)
    masks.append(True)
    
    return features, masks

def encode_attachment_features(event_dict: Dict[str, Any], 
                              data_version: str) -> Tuple[List[float], List[bool]]:
    """
    编码附件特征（基于原始逻辑）
    
    Args:
        event_dict: 邮件事件字典
        data_version: 数据版本
        
    Returns:
        附件特征和mask
    """
    features = []
    masks = []
    
    if data_version in ['r4.1', 'r4.2']:
        # 旧版本只有附件数量
        n_att = event_dict.get('#att', 0)
        try:
            n_attachments = int(n_att) if n_att else 0
            features.append(min(n_attachments / 10.0, 1.0))  # 标准化
            masks.append(True)
        except:
            features.append(0.0)
            masks.append(False)
        
    else:
        # 新版本有详细附件信息
        att_field = event_dict.get('att', '')
        
        if att_field and not pd.isna(att_field):
            attachments = str(att_field).split(';')
            n_attachments = len([att for att in attachments if att.strip()])
            
            # 附件数量
            features.append(min(n_attachments / 10.0, 1.0))
            masks.append(True)
            
            # 分析附件类型和大小
            att_type_counts = [0, 0, 0, 0, 0, 0]  # other, zip, image, doc, text, exe
            att_size_totals = [0, 0, 0, 0, 0, 0]  # 对应类型的总大小
            total_att_size = 0
            
            for att in attachments:
                if att.strip() and '.' in att:
                    att_features = process_attachment(att)
                    if att_features:
                        att_type_vec, att_size_vec = att_features
                        for i in range(6):
                            att_type_counts[i] += att_type_vec[i]
                            att_size_totals[i] += att_size_vec[i]
                        total_att_size += sum(att_size_vec)
            
            # 添加附件类型特征
            features.extend([min(count / 5.0, 1.0) for count in att_type_counts])
            masks.extend([True] * 6)
            
            # 添加附件大小特征（对数标准化）
            for size_total in att_size_totals:
                if size_total > 0:
                    size_normalized = np.log10(size_total + 1) / 10.0
                else:
                    size_normalized = 0.0
                features.append(size_normalized)
                masks.append(True)
            
        else:
            # 无附件
            features.extend([0.0] * 13)  # 1个数量 + 6个类型 + 6个大小
            masks.extend([False] * 13)
    
    return features, masks

def process_attachment(att_str: str) -> Optional[Tuple[List[int], List[float]]]:
    """
    处理单个附件字符串（基于原始file_process逻辑）
    
    Args:
        att_str: 附件字符串，格式如 "filename.ext(size)"
        
    Returns:
        (type_vector, size_vector) 或 None
    """
    try:
        if '.' not in att_str:
            return None
            
        parts = att_str.split('.')
        if len(parts) < 2:
            return None
            
        ext_part = parts[1]
        if '(' not in ext_part:
            return None
            
        # 提取扩展名和大小
        ext = ext_part[:ext_part.find('(')]
        size_str = ext_part[ext_part.find("(")+1:ext_part.find(")")]
        
        try:
            att_size = float(size_str)
        except:
            att_size = 0.0
        
        # 分类附件类型
        type_vector = [0, 0, 0, 0, 0, 0]  # other, zip, image, doc, text, exe
        size_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        ext_lower = ext.lower()
        if ext_lower in ['zip', 'rar', '7z']:
            idx = 1
        elif ext_lower in ['jpg', 'png', 'bmp', 'gif']:
            idx = 2
        elif ext_lower in ['doc', 'docx', 'pdf']:
            idx = 3
        elif ext_lower in ['txt', 'cfg', 'rtf']:
            idx = 4
        elif ext_lower in ['exe', 'sh', 'bat']:
            idx = 5
        else:
            idx = 0
        
        type_vector[idx] = 1
        size_vector[idx] = att_size
        
        return type_vector, size_vector
        
    except:
        return None

def analyze_email_patterns(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    分析邮件使用模式
    
    Args:
        events: 邮件事件列表
        
    Returns:
        邮件模式特征字典
    """
    if not events:
        return {}
    
    patterns = {}
    
    # 发送vs接收比例
    send_count = sum(1 for e in events if e.get('activity') == 'Send')
    receive_count = len(events) - send_count
    
    if len(events) > 0:
        patterns['send_ratio'] = send_count / len(events)
        patterns['receive_ratio'] = receive_count / len(events)
    
    # 外部邮件比例
    external_count = 0
    for event in events:
        # 检查收件人
        for field in ['to', 'cc', 'bcc']:
            field_value = event.get(field, '')
            if field_value and 'dtaa.com' not in str(field_value):
                external_count += 1
                break
    
    patterns['external_ratio'] = external_count / len(events)
    
    # 平均邮件大小
    sizes = []
    for event in events:
        size = event.get('size', 0)
        try:
            sizes.append(int(size) if size else 0)
        except:
            pass
    
    if sizes:
        patterns['avg_size'] = np.mean(sizes)
        patterns['size_variance'] = np.var(sizes)
    else:
        patterns['avg_size'] = 0
        patterns['size_variance'] = 0
    
    # 附件使用模式
    attachment_count = 0
    for event in events:
        if 'att' in event:
            att_field = event.get('att', '')
            if att_field and not pd.isna(att_field):
                attachments = str(att_field).split(';')
                attachment_count += len([att for att in attachments if att.strip()])
        elif '#att' in event:
            try:
                attachment_count += int(event.get('#att', 0))
            except:
                pass
    
    patterns['attachment_ratio'] = attachment_count / len(events)
    
    return patterns

def detect_suspicious_email_patterns(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    检测可疑的邮件模式
    
    Args:
        events: 邮件事件列表
        
    Returns:
        可疑模式评分字典
    """
    if not events:
        return {}
    
    suspicious_patterns = {}
    
    # 大量外部邮件
    external_count = 0
    for event in events:
        for field in ['to', 'cc', 'bcc']:
            field_value = event.get(field, '')
            if field_value and 'dtaa.com' not in str(field_value):
                external_count += 1
                break
    
    external_ratio = external_count / len(events)
    suspicious_patterns['high_external_ratio'] = min(external_ratio * 2, 1.0)
    
    # 大量BCC使用
    bcc_count = sum(1 for e in events if e.get('bcc') and not pd.isna(e.get('bcc')))
    bcc_ratio = bcc_count / len(events)
    suspicious_patterns['high_bcc_usage'] = min(bcc_ratio * 5, 1.0)
    
    # 异常大的邮件
    large_email_count = 0
    for event in events:
        size = event.get('size', 0)
        try:
            if int(size) > 10000000:  # 10MB
                large_email_count += 1
        except:
            pass
    
    large_email_ratio = large_email_count / len(events)
    suspicious_patterns['large_email_ratio'] = min(large_email_ratio * 10, 1.0)
    
    # 可执行文件附件
    exe_attachment_count = 0
    for event in events:
        att_field = event.get('att', '')
        if att_field and not pd.isna(att_field):
            attachments = str(att_field).split(';')
            for att in attachments:
                if any(ext in att.lower() for ext in ['.exe', '.sh', '.bat']):
                    exe_attachment_count += 1
    
    exe_ratio = exe_attachment_count / len(events)
    suspicious_patterns['executable_attachment_ratio'] = min(exe_ratio * 20, 1.0)
    
    return suspicious_patterns 