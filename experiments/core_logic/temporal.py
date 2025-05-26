#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间特征编码模块
提取时间上下文特征，包括工作时间、会话时长、工作日/周末等时态信息
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from utils import FeatureEncoder, create_time_features

def encode_temporal_features(event_dict: Dict[str, Any], 
                           feature_encoder: FeatureEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """
    编码时间特征
    
    Args:
        event_dict: 事件字典，包含时间信息
        feature_encoder: 特征编码器
        
    Returns:
        Tuple[temporal_features, mask]: 时间特征向量和mask
    """
    features = []
    masks = []
    
    # 解析时间戳
    timestamp = parse_timestamp(event_dict.get('date'))
    
    if timestamp is None:
        # 无效时间戳，返回零向量
        return np.zeros(12), np.zeros(12, dtype=bool)
    
    # 1. 基础时间特征
    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0=周一, 6=周日
    day_of_month = timestamp.day
    month = timestamp.month
    
    # 2. 工作时间判断 (8:00-17:00)
    is_work_hour = 1 if 8 <= hour <= 17 else 0
    
    # 3. 周末判断
    is_weekend = 1 if day_of_week >= 5 else 0  # 5=周六, 6=周日
    
    # 4. 周期性编码（使用sin/cos避免边界问题）
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # 5. 组合时间类型
    # 1: 工作日工作时间, 2: 工作日非工作时间, 3: 周末工作时间, 4: 周末非工作时间
    if is_weekend:
        time_category = 3 if is_work_hour else 4
    else:
        time_category = 1 if is_work_hour else 2
    
    # 构建特征向量
    temporal_features = np.array([
        hour / 24.0,         # 标准化小时
        day_of_week / 7.0,   # 标准化星期
        is_work_hour,        # 是否工作时间
        is_weekend,          # 是否周末
        hour_sin,            # 小时周期编码
        hour_cos,
        day_sin,             # 星期周期编码
        day_cos,
        month_sin,           # 月份周期编码
        month_cos,
        time_category / 4.0, # 标准化时间类别
        0.0                  # 保留位（可用于会话时长等）
    ], dtype=np.float32)
    
    # 所有时间特征都有效
    temporal_mask = np.ones(len(temporal_features), dtype=bool)
    
    return temporal_features, temporal_mask

def encode_session_temporal_features(events: List[Dict[str, Any]], 
                                   feature_encoder: FeatureEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """
    编码会话级别的时间特征
    
    Args:
        events: 会话内的事件列表
        feature_encoder: 特征编码器
        
    Returns:
        Tuple[session_temporal_features, mask]: 会话时间特征和mask
    """
    if not events:
        return np.zeros(15), np.zeros(15, dtype=bool)
    
    # 解析所有事件的时间戳
    timestamps = []
    for event in events:
        ts = parse_timestamp(event.get('date'))
        if ts:
            timestamps.append(ts)
    
    if not timestamps:
        return np.zeros(15), np.zeros(15, dtype=bool)
    
    timestamps.sort()
    
    # 会话开始和结束时间
    session_start = timestamps[0]
    session_end = timestamps[-1]
    
    # 会话持续时间（分钟）
    session_duration = (session_end - session_start).total_seconds() / 60
    
    # 事件间平均间隔（分钟）
    if len(timestamps) > 1:
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 
                    for i in range(len(timestamps)-1)]
        avg_interval = np.mean(intervals)
        max_interval = np.max(intervals)
    else:
        avg_interval = 0.0
        max_interval = 0.0
    
    # 会话开始时间特征
    start_hour = session_start.hour
    start_is_work_hour = 1 if 8 <= start_hour <= 17 else 0
    start_is_weekend = 1 if session_start.weekday() >= 5 else 0
    
    # 工作时间内活动比例
    work_hour_events = sum(1 for ts in timestamps if 8 <= ts.hour <= 17)
    work_hour_ratio = work_hour_events / len(timestamps)
    
    # 跨天会话标识
    is_cross_day = 1 if session_start.date() != session_end.date() else 0
    
    # 夜间活动比例 (22:00-6:00)
    night_events = sum(1 for ts in timestamps if ts.hour >= 22 or ts.hour <= 6)
    night_ratio = night_events / len(timestamps)
    
    # 周末活动比例
    weekend_events = sum(1 for ts in timestamps if ts.weekday() >= 5)
    weekend_ratio = weekend_events / len(timestamps)
    
    # 构建会话时间特征
    session_features = np.array([
        session_duration / 480.0,  # 标准化（8小时为1）
        avg_interval / 60.0,       # 标准化（1小时为1）
        max_interval / 60.0,       # 标准化（1小时为1）
        start_hour / 24.0,         # 标准化开始小时
        start_is_work_hour,        # 开始时是否工作时间
        start_is_weekend,          # 开始时是否周末
        work_hour_ratio,           # 工作时间活动比例
        night_ratio,               # 夜间活动比例
        weekend_ratio,             # 周末活动比例
        is_cross_day,              # 是否跨天
        len(timestamps) / 100.0,   # 标准化事件数量
        0.0,                       # 保留位
        0.0,                       # 保留位
        0.0,                       # 保留位
        0.0                        # 保留位
    ], dtype=np.float32)
    
    session_mask = np.ones(len(session_features), dtype=bool)
    
    return session_features, session_mask

def parse_timestamp(time_str: str) -> Optional[datetime]:
    """
    解析时间戳字符串
    
    Args:
        time_str: 时间字符串，支持多种格式
        
    Returns:
        datetime对象，如果解析失败返回None
    """
    if not time_str or pd.isna(time_str):
        return None
    
    # 支持的时间格式
    formats = [
        '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%m/%d/%Y %H:%M',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%m/%d/%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(time_str), fmt)
        except ValueError:
            continue
    
    # 尝试pandas解析
    try:
        return pd.to_datetime(time_str)
    except:
        return None

def get_time_category(timestamp: datetime) -> int:
    """
    获取时间类别
    
    Args:
        timestamp: 时间戳
        
    Returns:
        时间类别：1=工作日工作时间, 2=工作日非工作时间, 3=周末工作时间, 4=周末非工作时间
    """
    is_weekend = timestamp.weekday() >= 5
    is_work_hour = 8 <= timestamp.hour <= 17
    
    if is_weekend:
        return 3 if is_work_hour else 4
    else:
        return 1 if is_work_hour else 2

def calculate_time_since_last_event(current_event: Dict[str, Any], 
                                  previous_event: Dict[str, Any] = None) -> float:
    """
    计算与上一个事件的时间间隔
    
    Args:
        current_event: 当前事件
        previous_event: 上一个事件
        
    Returns:
        时间间隔（分钟），如果无法计算返回0
    """
    if not previous_event:
        return 0.0
    
    current_time = parse_timestamp(current_event.get('date'))
    previous_time = parse_timestamp(previous_event.get('date'))
    
    if not current_time or not previous_time:
        return 0.0
    
    time_diff = (current_time - previous_time).total_seconds() / 60
    return max(0.0, time_diff)  # 确保非负值

def encode_time_patterns(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    编码时间模式特征
    
    Args:
        events: 事件列表
        
    Returns:
        时间模式特征字典
    """
    if not events:
        return {}
    
    timestamps = []
    for event in events:
        ts = parse_timestamp(event.get('date'))
        if ts:
            timestamps.append(ts)
    
    if not timestamps:
        return {}
    
    timestamps.sort()
    
    patterns = {}
    
    # 活动时间分布
    hour_counts = [0] * 24
    day_counts = [0] * 7
    
    for ts in timestamps:
        hour_counts[ts.hour] += 1
        day_counts[ts.weekday()] += 1
    
    # 峰值活动时间
    peak_hour = hour_counts.index(max(hour_counts))
    peak_day = day_counts.index(max(day_counts))
    
    patterns['peak_hour'] = peak_hour / 24.0
    patterns['peak_day'] = peak_day / 7.0
    
    # 活动规律性（标准差）
    patterns['hour_regularity'] = 1.0 - (np.std(hour_counts) / (np.mean(hour_counts) + 1e-6))
    patterns['day_regularity'] = 1.0 - (np.std(day_counts) / (np.mean(day_counts) + 1e-6))
    
    # 夜间活动比例
    night_count = sum(1 for ts in timestamps if ts.hour >= 22 or ts.hour <= 6)
    patterns['night_activity_ratio'] = night_count / len(timestamps)
    
    # 周末活动比例
    weekend_count = sum(1 for ts in timestamps if ts.weekday() >= 5)
    patterns['weekend_activity_ratio'] = weekend_count / len(timestamps)
    
    # 活动密度变化
    if len(timestamps) > 1:
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 60 
                    for i in range(len(timestamps)-1)]
        patterns['activity_density'] = len(timestamps) / (max(intervals) + 1e-6)
        patterns['interval_variance'] = np.var(intervals) / (np.mean(intervals) + 1e-6)
    else:
        patterns['activity_density'] = 0.0
        patterns['interval_variance'] = 0.0
    
    return patterns

def encode_work_schedule_compliance(events: List[Dict[str, Any]], 
                                  work_start: int = 8, 
                                  work_end: int = 17) -> Dict[str, float]:
    """
    编码工作时间合规性特征
    
    Args:
        events: 事件列表
        work_start: 工作开始时间（小时）
        work_end: 工作结束时间（小时）
        
    Returns:
        工作时间合规性特征
    """
    if not events:
        return {}
    
    timestamps = []
    for event in events:
        ts = parse_timestamp(event.get('date'))
        if ts:
            timestamps.append(ts)
    
    if not timestamps:
        return {}
    
    compliance = {}
    
    # 工作时间内活动统计
    work_day_events = [ts for ts in timestamps if ts.weekday() < 5]  # 工作日
    work_hour_events = [ts for ts in work_day_events 
                       if work_start <= ts.hour <= work_end]
    
    if work_day_events:
        compliance['work_hour_compliance'] = len(work_hour_events) / len(work_day_events)
    else:
        compliance['work_hour_compliance'] = 0.0
    
    # 早到和晚退统计
    early_events = [ts for ts in work_day_events if ts.hour < work_start]
    late_events = [ts for ts in work_day_events if ts.hour > work_end]
    
    if work_day_events:
        compliance['early_activity_ratio'] = len(early_events) / len(work_day_events)
        compliance['late_activity_ratio'] = len(late_events) / len(work_day_events)
    else:
        compliance['early_activity_ratio'] = 0.0
        compliance['late_activity_ratio'] = 0.0
    
    # 周末工作统计
    weekend_events = [ts for ts in timestamps if ts.weekday() >= 5]
    compliance['weekend_work_ratio'] = len(weekend_events) / len(timestamps)
    
    return compliance 