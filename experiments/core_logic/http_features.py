#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP特征编码模块
提取HTTP/Web浏览活动特征，包括URL分析、域名分类、内容特征等
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from utils import FeatureEncoder

def encode_http_features(event_dict: Dict[str, Any], 
                        feature_encoder: FeatureEncoder,
                        data_version: str = 'r4.2') -> Tuple[np.ndarray, np.ndarray]:
    """
    编码HTTP特征（兼容原始feature_extraction.py的逻辑）
    
    Args:
        event_dict: HTTP事件字典
        feature_encoder: 特征编码器
        data_version: 数据版本
        
    Returns:
        Tuple[http_features, mask]: HTTP特征向量和mask
    """
    features = []
    masks = []
    
    # 1. URL基本特征
    url_features, url_mask = encode_url_features(event_dict)
    features.extend(url_features)
    masks.extend(url_mask)
    
    # 2. 域名分类特征
    domain_features, domain_mask = encode_domain_features(event_dict)
    features.extend(domain_features)
    masks.extend(domain_mask)
    
    # 3. 内容特征
    content_features, content_mask = encode_http_content_features(event_dict, feature_encoder)
    features.extend(content_features)
    masks.extend(content_mask)
    
    # 4. 根据数据版本添加额外特征
    if data_version in ['r6.1', 'r6.2']:
        # HTTP活动类型特征
        activity_features, activity_mask = encode_http_activity_features(event_dict)
        features.extend(activity_features)
        masks.extend(activity_mask)
    
    return np.array(features, dtype=np.float32), np.array(masks, dtype=bool)

def encode_url_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码URL基本特征
    
    Args:
        event_dict: HTTP事件字典
        
    Returns:
        URL特征和mask
    """
    features = []
    masks = []
    
    url = event_dict.get('url/fname', '')
    
    if url and not pd.isna(url):
        url_str = str(url)
        
        # URL长度
        url_length = len(url_str)
        url_length_normalized = min(url_length / 500.0, 1.0)  # 标准化
        features.append(url_length_normalized)
        masks.append(True)
        
        # URL深度（路径层级）
        if '//' in url_str:
            path_part = url_str.split('//', 1)[1]
            if '/' in path_part:
                path = path_part.split('/', 1)[1]
                url_depth = path.count('/')
            else:
                url_depth = 0
        else:
            url_depth = url_str.count('/') - 2 if url_str.count('/') >= 2 else 0
        
        url_depth_normalized = min(url_depth / 10.0, 1.0)  # 标准化
        features.append(url_depth_normalized)
        masks.append(True)
        
        # URL复杂度特征
        complexity_features = analyze_url_complexity(url_str)
        features.extend(complexity_features)
        masks.extend([True] * len(complexity_features))
        
    else:
        # 无URL信息
        features.extend([0.0] * 5)  # 长度 + 深度 + 3个复杂度特征
        masks.extend([False] * 5)
    
    return features, masks

def encode_domain_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码域名分类特征（基于原始http_process函数）
    
    Args:
        event_dict: HTTP事件字典
        
    Returns:
        域名特征和mask
    """
    features = []
    masks = []
    
    url = event_dict.get('url/fname', '')
    
    if url and not pd.isna(url):
        domain_category = get_domain_category(str(url))
        
        # 域名类别one-hot编码：other, social, cloud, job, leak, hack
        category_onehot = [0.0] * 6
        if 1 <= domain_category <= 6:
            category_onehot[domain_category - 1] = 1.0
        
        features.extend(category_onehot)
        masks.extend([True] * 6)
        
        # 域名风险评分
        risk_score = get_domain_risk_score(domain_category)
        features.append(risk_score)
        masks.append(True)
        
    else:
        # 无URL信息
        features.extend([0.0] * 7)  # 6个类别 + 1个风险评分
        masks.extend([False] * 7)
    
    return features, masks

def encode_http_content_features(event_dict: Dict[str, Any], 
                                feature_encoder: FeatureEncoder) -> Tuple[List[float], List[bool]]:
    """
    编码HTTP内容特征
    
    Args:
        event_dict: HTTP事件字典
        feature_encoder: 特征编码器
        
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
        content_length_normalized = min(content_length / 10000.0, 1.0)  # 标准化
        features.append(content_length_normalized)
        masks.append(True)
        
        # 单词数量
        word_count = len(content_str.split())
        word_count_normalized = min(word_count / 1000.0, 1.0)  # 标准化
        features.append(word_count_normalized)
        masks.append(True)
        
        # 内容类型分析
        content_type_features = analyze_content_type(content_str)
        features.extend(content_type_features)
        masks.extend([True] * len(content_type_features))
        
    else:
        # 无内容信息
        features.extend([0.0] * 5)  # 长度 + 单词数 + 3个内容类型特征
        masks.extend([False] * 5)
    
    return features, masks

def encode_http_activity_features(event_dict: Dict[str, Any]) -> Tuple[List[float], List[bool]]:
    """
    编码HTTP活动类型特征（新版本数据）
    
    Args:
        event_dict: HTTP事件字典
        
    Returns:
        活动特征和mask
    """
    features = []
    masks = []
    
    activity = event_dict.get('activity', '')
    
    # HTTP活动类型映射
    activity_mapping = {
        'www visit': 1,
        'www download': 2,
        'www upload': 3
    }
    
    activity_code = activity_mapping.get(str(activity).lower(), 0)
    
    # 活动类型one-hot编码
    activity_onehot = [0.0] * 4  # unknown, visit, download, upload
    if 0 <= activity_code <= 3:
        activity_onehot[activity_code] = 1.0
    
    features.extend(activity_onehot)
    masks.extend([True] * 4)
    
    # 活动风险等级
    risk_level = get_http_activity_risk_level(activity_code)
    features.append(risk_level)
    masks.append(True)
    
    return features, masks

def get_domain_category(url: str) -> int:
    """
    获取域名类别（基于原始http_process函数）
    
    Args:
        url: URL字符串
        
    Returns:
        域名类别：1=other, 2=social, 3=cloud, 4=job, 5=leak, 6=hack
    """
    try:
        # 提取域名
        domain_match = re.findall(r"//(.*?)/", url)
        if not domain_match:
            return 1
        
        domainname = domain_match[0]
        domainname = domainname.replace("www.", "")
        
        # 处理子域名
        dn = domainname.split(".")
        if len(dn) > 2 and not any(special in domainname for special in 
                                  ["google.com", '.co.uk', '.co.nz', 'live.com']):
            domainname = ".".join(dn[-2:])
        
        # 云存储服务
        if domainname in ['dropbox.com', 'drive.google.com', 'mega.co.nz', 'account.live.com']:
            return 3
        
        # 泄露/黑客网站
        elif domainname in ['wikileaks.org', 'freedom.press', 'theintercept.com']:
            return 5
        
        # 社交网络
        elif domainname in ['facebook.com', 'twitter.com', 'plus.google.com', 'instagr.am', 
                           'instagram.com', 'flickr.com', 'linkedin.com', 'reddit.com', 
                           'about.com', 'youtube.com', 'pinterest.com', 'tumblr.com', 
                           'quora.com', 'vine.co', 'match.com', 't.co']:
            return 2
        
        # 求职网站
        elif domainname in ['indeed.com', 'monster.com', 'careerbuilder.com', 'simplyhired.com']:
            return 4
        
        # 基于关键词的判断
        elif ('job' in domainname and ('hunt' in domainname or 'search' in domainname)) or \
             ('aol.com' in domainname and ("recruit" in url or "job" in url)):
            return 4
        
        # 黑客工具网站
        elif domainname in ['webwatchernow.com', 'actionalert.com', 'relytec.com', 
                           'refog.com', 'wellresearchedreviews.com', 'softactivity.com', 
                           'spectorsoft.com', 'best-spy-soft.com'] or 'keylog' in domainname:
            return 6
        
        else:
            return 1  # 其他
            
    except:
        return 1

def get_domain_risk_score(category: int) -> float:
    """
    获取域名风险评分
    
    Args:
        category: 域名类别
        
    Returns:
        风险评分 (0.0-1.0)
    """
    risk_mapping = {
        1: 0.1,  # other
        2: 0.3,  # social - 中等风险
        3: 0.8,  # cloud - 高风险（数据泄露）
        4: 0.6,  # job - 中高风险（内部威胁）
        5: 1.0,  # leak - 最高风险
        6: 1.0   # hack - 最高风险
    }
    
    return risk_mapping.get(category, 0.1)

def get_http_activity_risk_level(activity_code: int) -> float:
    """
    获取HTTP活动风险等级
    
    Args:
        activity_code: 活动代码
        
    Returns:
        风险等级 (0.0-1.0)
    """
    risk_mapping = {
        0: 0.1,  # unknown
        1: 0.2,  # visit
        2: 0.6,  # download - 中高风险
        3: 0.9   # upload - 高风险
    }
    
    return risk_mapping.get(activity_code, 0.1)

def analyze_url_complexity(url: str) -> List[float]:
    """
    分析URL复杂度特征
    
    Args:
        url: URL字符串
        
    Returns:
        复杂度特征列表
    """
    features = []
    
    # 特殊字符比例
    special_chars = sum(1 for c in url if c in '?&=%-+[]{}|\\:";\'<>,.!@#$^*()~`')
    special_ratio = special_chars / len(url) if len(url) > 0 else 0
    features.append(min(special_ratio * 5, 1.0))  # 放大并归一化
    
    # 数字比例
    digits = sum(1 for c in url if c.isdigit())
    digit_ratio = digits / len(url) if len(url) > 0 else 0
    features.append(min(digit_ratio * 3, 1.0))
    
    # 是否包含IP地址
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    has_ip = 1.0 if re.search(ip_pattern, url) else 0.0
    features.append(has_ip)
    
    return features

def analyze_content_type(content: str) -> List[float]:
    """
    分析HTTP内容类型特征
    
    Args:
        content: 内容字符串
        
    Returns:
        内容类型特征列表
    """
    features = []
    
    content_lower = content.lower()
    
    # 是否包含表单数据
    has_form_data = 1.0 if any(keyword in content_lower for keyword in 
                              ['<form', 'input', 'submit', 'password']) else 0.0
    features.append(has_form_data)
    
    # 是否包含脚本
    has_script = 1.0 if any(keyword in content_lower for keyword in 
                           ['<script', 'javascript', 'onclick']) else 0.0
    features.append(has_script)
    
    # 是否包含敏感信息
    has_sensitive = 1.0 if any(keyword in content_lower for keyword in 
                              ['password', 'ssn', 'credit card', 'confidential']) else 0.0
    features.append(has_sensitive)
    
    return features

def analyze_http_patterns(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    分析HTTP浏览模式
    
    Args:
        events: HTTP事件列表
        
    Returns:
        HTTP模式特征字典
    """
    if not events:
        return {}
    
    patterns = {}
    
    # 域名类别分布
    category_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    for event in events:
        url = event.get('url/fname', '')
        if url and not pd.isna(url):
            category = get_domain_category(str(url))
            category_counts[category] = category_counts.get(category, 0) + 1
    
    total_visits = sum(category_counts.values())
    if total_visits > 0:
        for category, count in category_counts.items():
            patterns[f'category_{category}_ratio'] = count / total_visits
    
    # 活动类型分布
    activity_counts = {'visit': 0, 'download': 0, 'upload': 0}
    
    for event in events:
        activity = str(event.get('activity', '')).lower()
        if 'visit' in activity:
            activity_counts['visit'] += 1
        elif 'download' in activity:
            activity_counts['download'] += 1
        elif 'upload' in activity:
            activity_counts['upload'] += 1
    
    for activity, count in activity_counts.items():
        patterns[f'{activity}_ratio'] = count / len(events)
    
    # URL长度统计
    url_lengths = []
    for event in events:
        url = event.get('url/fname', '')
        if url and not pd.isna(url):
            url_lengths.append(len(str(url)))
    
    if url_lengths:
        patterns['avg_url_length'] = np.mean(url_lengths)
        patterns['url_length_variance'] = np.var(url_lengths)
        patterns['long_url_ratio'] = sum(1 for l in url_lengths if l > 200) / len(url_lengths)
    else:
        patterns['avg_url_length'] = 0
        patterns['url_length_variance'] = 0
        patterns['long_url_ratio'] = 0
    
    # 域名多样性
    domains = set()
    for event in events:
        url = event.get('url/fname', '')
        if url and not pd.isna(url):
            try:
                domain_match = re.findall(r"//(.*?)/", str(url))
                if domain_match:
                    domain = domain_match[0].replace("www.", "")
                    domains.add(domain)
            except:
                pass
    
    patterns['domain_diversity'] = len(domains) / len(events) if len(events) > 0 else 0
    
    return patterns

def detect_suspicious_http_patterns(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    检测可疑的HTTP浏览模式
    
    Args:
        events: HTTP事件列表
        
    Returns:
        可疑模式评分字典
    """
    if not events:
        return {}
    
    suspicious_patterns = {}
    
    # 高风险域名访问
    high_risk_count = 0
    for event in events:
        url = event.get('url/fname', '')
        if url and not pd.isna(url):
            category = get_domain_category(str(url))
            if category in [3, 5, 6]:  # cloud, leak, hack
                high_risk_count += 1
    
    high_risk_ratio = high_risk_count / len(events)
    suspicious_patterns['high_risk_domain_ratio'] = min(high_risk_ratio * 5, 1.0)
    
    # 大量下载活动
    download_count = sum(1 for e in events if 'download' in str(e.get('activity', '')).lower())
    download_ratio = download_count / len(events)
    suspicious_patterns['high_download_ratio'] = min(download_ratio * 3, 1.0)
    
    # 上传活动（高风险）
    upload_count = sum(1 for e in events if 'upload' in str(e.get('activity', '')).lower())
    upload_ratio = upload_count / len(events)
    suspicious_patterns['upload_activity_ratio'] = min(upload_ratio * 10, 1.0)
    
    # 异常长URL
    long_url_count = 0
    for event in events:
        url = event.get('url/fname', '')
        if url and not pd.isna(url) and len(str(url)) > 300:
            long_url_count += 1
    
    long_url_ratio = long_url_count / len(events)
    suspicious_patterns['long_url_ratio'] = min(long_url_ratio * 5, 1.0)
    
    # IP地址访问
    ip_access_count = 0
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    for event in events:
        url = event.get('url/fname', '')
        if url and not pd.isna(url) and re.search(ip_pattern, str(url)):
            ip_access_count += 1
    
    ip_ratio = ip_access_count / len(events)
    suspicious_patterns['ip_access_ratio'] = min(ip_ratio * 10, 1.0)
    
    return suspicious_patterns 