#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内部威胁检测特征提取系统演示脚本

展示如何使用该系统进行完整的特征提取和异常检测流程
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# 导入特征提取模块
from encoder import EventEncoder
from utils import FeatureEncoder
from temporal import encode_session_temporal_features
from user_context import encode_behavioral_risk_profile
from email import detect_suspicious_email_patterns
from file import detect_suspicious_file_patterns
from http import detect_suspicious_http_patterns
from device import detect_suspicious_device_patterns

def create_sample_data():
    """创建示例数据"""
    print("🔧 创建示例数据...")
    
    # 示例事件数据
    events_data = [
        # 正常用户活动
        {
            'type': 'logon',
            'date': '01/15/2024 08:30:00',
            'user': 'ACM2278',
            'pc': 'PC-1234'
        },
        {
            'type': 'email',
            'date': '01/15/2024 09:15:00',
            'user': 'ACM2278',
            'from': 'ACM2278@dtaa.com',
            'to': 'colleague@dtaa.com',
            'cc': '',
            'bcc': '',
            'size': '1024',
            'content': 'Meeting reminder for tomorrow',
            'activity': 'Send'
        },
        {
            'type': 'http',
            'date': '01/15/2024 10:30:00',
            'user': 'ACM2278',
            'url/fname': 'https://www.company-intranet.com/reports',
            'content': 'Internal company reports page',
            'activity': 'www visit'
        },
        {
            'type': 'file',
            'date': '01/15/2024 11:45:00',
            'user': 'ACM2278',
            'pc': 'PC-1234',
            'url/fname': 'C:\\Users\\ACM2278\\Documents\\report.docx',
            'content': 'Monthly financial report content...',
            'activity': 'file open',
            'to': 'false',
            'from': 'false'
        },
        
        # 可疑用户活动
        {
            'type': 'email',
            'date': '01/15/2024 18:30:00',  # 下班后
            'user': 'ACM2278',
            'from': 'ACM2278@dtaa.com',
            'to': 'external@gmail.com;competitor@rival.com',  # 外部邮件
            'cc': '',
            'bcc': 'personal@gmail.com',  # 密送外部
            'size': '5120000',  # 大邮件
            'content': 'Confidential company strategy document attached',
            'activity': 'Send',
            'att': 'strategy.pdf(2048000);financial_data.xlsx(1024000)'  # 大附件
        },
        {
            'type': 'http',
            'date': '01/15/2024 19:15:00',
            'user': 'ACM2278',
            'url/fname': 'https://www.wikileaks.org/documents',  # 高风险域名
            'content': 'Document leak website content',
            'activity': 'www upload'  # 上传活动
        },
        {
            'type': 'file',
            'date': '01/15/2024 19:45:00',
            'user': 'ACM2278',
            'pc': 'PC-1234',
            'url/fname': 'C:\\Users\\ACM2278\\Desktop\\sensitive_employee_data.xlsx',
            'content': 'Employee personal information, salaries, SSN...',
            'activity': 'file copy',
            'to': 'true',  # USB传输
            'from': 'false'
        },
        {
            'type': 'device',
            'date': '01/15/2024 20:00:00',
            'user': 'ACM2278',
            'activity': 'USB Connect',
            'content': 'External USB drive detected - 64GB capacity',
            'file_tree_len': '1500'
        }
    ]
    
    # 用户上下文数据
    users_data = [
        {
            'user_id': 'ACM2278',
            'role': 'Manager',
            'dept': 'Finance',
            'ITAdmin': 0,
            'O': 0.6,  # 开放性
            'C': 0.9,  # 责任心
            'E': 0.7,  # 外向性
            'A': 0.8,  # 宜人性
            'N': 0.2,  # 神经质
            'pc_type': 0,
            'sharedpc': None,
            'npc': 1
        }
    ]
    
    events_df = pd.DataFrame(events_data)
    users_df = pd.DataFrame(users_data).set_index('user_id')
    
    return events_df, users_df

def demonstrate_basic_usage():
    """演示基础使用流程"""
    print("\n📝 基础使用演示")
    print("=" * 50)
    
    # 创建数据
    events_df, users_df = create_sample_data()
    
    # 1. 初始化编码器
    print("1️⃣ 初始化编码器")
    encoder = EventEncoder(feature_dim=256, data_version='r5.2')
    
    # 2. 拟合编码器
    print("2️⃣ 拟合编码器")
    encoder.fit(events_df, users_df)
    
    # 3. 编码单个事件
    print("3️⃣ 编码单个事件")
    event_dict = events_df.iloc[1].to_dict()  # 邮件事件
    user_context = users_df.loc['ACM2278'].to_dict()
    
    features, mask = encoder.encode_event(event_dict, user_context)
    
    print(f"   📊 特征向量维度: {features.shape}")
    print(f"   ✅ 有效特征数: {mask.sum()}/{len(mask)} ({mask.mean():.1%})")
    print(f"   📈 特征值范围: [{features.min():.3f}, {features.max():.3f}]")
    
    # 4. 编码事件序列
    print("4️⃣ 编码事件序列")
    user_events = events_df[events_df['user'] == 'ACM2278'].to_dict('records')
    seq_features, seq_mask = encoder.encode_event_sequence(
        user_events, user_context, max_sequence_length=10
    )
    
    print(f"   📊 序列特征矩阵: {seq_features.shape}")
    print(f"   ✅ 序列有效率: {seq_mask.mean():.1%}")
    
    return encoder, events_df, users_df

def demonstrate_risk_analysis():
    """演示风险分析功能"""
    print("\n🚨 风险分析演示")
    print("=" * 50)
    
    events_df, users_df = create_sample_data()
    
    # 按事件类型分组
    user_events = events_df[events_df['user'] == 'ACM2278'].to_dict('records')
    user_context = users_df.loc['ACM2278'].to_dict()
    
    email_events = [e for e in user_events if e['type'] == 'email']
    http_events = [e for e in user_events if e['type'] == 'http']
    file_events = [e for e in user_events if e['type'] == 'file']
    device_events = [e for e in user_events if e['type'] == 'device']
    
    print("1️⃣ 用户上下文风险分析")
    context_risk = encode_behavioral_risk_profile(user_context)
    print(f"   🎯 总体风险评分: {context_risk['overall_risk']:.3f}")
    print(f"   🔑 权限风险: {context_risk['privilege_risk']:.3f}")
    print(f"   📊 数据访问风险: {context_risk['data_access_risk']:.3f}")
    print(f"   🧠 心理风险: {context_risk['psychological_risk']:.3f}")
    
    print("\n2️⃣ 邮件行为风险分析")
    if email_events:
        email_risks = detect_suspicious_email_patterns(email_events)
        for pattern, score in email_risks.items():
            if score > 0.1:
                print(f"   ⚠️  {pattern}: {score:.3f}")
    
    print("\n3️⃣ HTTP浏览风险分析")
    if http_events:
        http_risks = detect_suspicious_http_patterns(http_events)
        for pattern, score in http_risks.items():
            if score > 0.1:
                print(f"   ⚠️  {pattern}: {score:.3f}")
    
    print("\n4️⃣ 文件操作风险分析")
    if file_events:
        file_risks = detect_suspicious_file_patterns(file_events)
        for pattern, score in file_risks.items():
            if score > 0.1:
                print(f"   ⚠️  {pattern}: {score:.3f}")
    
    print("\n5️⃣ 设备使用风险分析")
    if device_events:
        device_risks = detect_suspicious_device_patterns(device_events)
        for pattern, score in device_risks.items():
            if score > 0.1:
                print(f"   ⚠️  {pattern}: {score:.3f}")
    
    # 综合风险评分
    all_risks = []
    if email_events:
        all_risks.extend(email_risks.values())
    if http_events:
        all_risks.extend(http_risks.values())
    if file_events:
        all_risks.extend(file_risks.values())
    if device_events:
        all_risks.extend(device_risks.values())
    
    overall_risk = (
        np.mean(list(email_risks.values()) if email_events else [0]) * 0.3 +
        np.mean(list(http_risks.values()) if http_events else [0]) * 0.3 +
        np.mean(list(file_risks.values()) if file_events else [0]) * 0.2 +
        context_risk['overall_risk'] * 0.2
    )
    
    print(f"\n🎯 用户综合风险评分: {overall_risk:.3f}")
    
    # 风险等级判断
    if overall_risk > 0.7:
        risk_level = "🔴 高风险"
    elif overall_risk > 0.4:
        risk_level = "🟡 中风险"
    else:
        risk_level = "🟢 低风险"
    
    print(f"🏷️  风险等级: {risk_level}")

def demonstrate_time_analysis():
    """演示时间特征分析"""
    print("\n⏰ 时间特征分析演示")
    print("=" * 50)
    
    events_df, users_df = create_sample_data()
    encoder = EventEncoder(feature_dim=256, data_version='r5.2')
    encoder.fit(events_df, users_df)
    
    # 会话级时间分析
    user_events = events_df[events_df['user'] == 'ACM2278'].to_dict('records')
    
    session_temporal, _ = encode_session_temporal_features(user_events, encoder.feature_encoder)
    
    print("1️⃣ 会话时间特征")
    print(f"   📅 会话持续时长: {session_temporal[0]*480:.1f} 分钟")
    print(f"   ⏱️  平均事件间隔: {session_temporal[1]*60:.1f} 分钟")
    print(f"   ⏰ 开始时间: {session_temporal[3]*24:.1f} 点")
    print(f"   🕘 工作时间活动比例: {session_temporal[6]:.1%}")
    print(f"   🌙 夜间活动比例: {session_temporal[7]:.1%}")
    print(f"   📅 周末活动比例: {session_temporal[8]:.1%}")
    print(f"   🔄 是否跨天: {'是' if session_temporal[9] > 0 else '否'}")
    
    # 检测异常时间模式
    print("\n2️⃣ 异常时间模式检测")
    
    # 夜间活动检测
    if session_temporal[7] > 0.3:  # 超过30%夜间活动
        print("   ⚠️  检测到异常夜间活动")
    
    # 周末工作检测
    if session_temporal[8] > 0.5:  # 超过50%周末活动
        print("   ⚠️  检测到异常周末工作")
    
    # 长会话检测
    if session_temporal[0] > 0.5:  # 超过4小时
        print("   ⚠️  检测到异常长时间会话")
    
    # 非工作时间检测
    if session_temporal[6] < 0.5:  # 工作时间活动少于50%
        print("   ⚠️  检测到大量非工作时间活动")

def demonstrate_feature_visualization():
    """演示特征可视化"""
    print("\n📊 特征可视化演示")
    print("=" * 50)
    
    events_df, users_df = create_sample_data()
    encoder = EventEncoder(feature_dim=256, data_version='r5.2')
    encoder.fit(events_df, users_df)
    
    # 编码所有事件
    user_context = users_df.loc['ACM2278'].to_dict()
    all_features = []
    all_masks = []
    event_types = []
    
    for _, event in events_df.iterrows():
        if event['user'] == 'ACM2278':
            event_dict = event.to_dict()
            features, mask = encoder.encode_event(event_dict, user_context)
            all_features.append(features)
            all_masks.append(mask)
            event_types.append(event['type'])
    
    all_features = np.array(all_features)
    all_masks = np.array(all_masks)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('特征提取系统可视化分析', fontsize=16, fontweight='bold')
    
    # 1. 特征分布直方图
    axes[0, 0].hist(all_features[all_masks], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('有效特征值分布')
    axes[0, 0].set_xlabel('特征值')
    axes[0, 0].set_ylabel('频数')
    
    # 2. mask模式
    mask_pattern = all_masks.mean(axis=0)
    axes[0, 1].plot(mask_pattern, color='green', linewidth=2)
    axes[0, 1].set_title('特征有效性模式')
    axes[0, 1].set_xlabel('特征索引')
    axes[0, 1].set_ylabel('有效率')
    axes[0, 1].set_ylim([0, 1])
    
    # 3. 事件类型特征热图
    feature_by_type = {}
    for i, event_type in enumerate(event_types):
        if event_type not in feature_by_type:
            feature_by_type[event_type] = []
        valid_features = all_features[i][all_masks[i]]
        if len(valid_features) > 0:
            feature_by_type[event_type].append(valid_features.mean())
    
    type_names = list(feature_by_type.keys())
    type_scores = [np.mean(feature_by_type[t]) for t in type_names]
    
    bars = axes[1, 0].bar(type_names, type_scores, color=['red', 'blue', 'green', 'orange'])
    axes[1, 0].set_title('不同事件类型的平均特征值')
    axes[1, 0].set_ylabel('平均特征值')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 时间序列特征
    time_features = []
    for _, event in events_df.iterrows():
        if event['user'] == 'ACM2278':
            timestamp = pd.to_datetime(event['date'])
            time_features.append(timestamp.hour + timestamp.minute/60.0)
    
    axes[1, 1].plot(time_features, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1, 1].set_title('事件时间分布')
    axes[1, 1].set_xlabel('事件序号')
    axes[1, 1].set_ylabel('时间 (小时)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 可视化图表已保存为 'feature_analysis.png'")
    plt.show()

def demonstrate_batch_processing():
    """演示批量处理"""
    print("\n🔄 批量处理演示")
    print("=" * 50)
    
    # 创建更大的示例数据集
    print("1️⃣ 创建大型数据集")
    events_df, users_df = create_sample_data()
    
    # 模拟多用户数据
    users = ['ACM2278', 'ACM1796', 'CMP2946', 'BTH8471', 'DYQ9624']
    expanded_events = []
    
    for user in users:
        for _, event in events_df.iterrows():
            new_event = event.copy()
            new_event['user'] = user
            # 随机调整时间
            base_time = pd.to_datetime(event['date'])
            time_offset = timedelta(hours=np.random.randint(-5, 5))
            new_event['date'] = (base_time + time_offset).strftime('%m/%d/%Y %H:%M:%S')
            expanded_events.append(new_event)
    
    large_events_df = pd.DataFrame(expanded_events)
    print(f"   📊 数据集大小: {len(large_events_df)} 条事件, {len(users)} 个用户")
    
    # 2. 批量特征提取
    print("2️⃣ 批量特征提取")
    encoder = EventEncoder(feature_dim=256, data_version='r5.2')
    encoder.fit(large_events_df, users_df)
    
    # 为每个用户创建随机用户上下文
    user_contexts = {}
    for user in users:
        user_contexts[user] = {
            'user_id': user,
            'role': np.random.choice(['Employee', 'Manager', 'Director']),
            'dept': np.random.choice(['IT', 'Finance', 'HR', 'Marketing']),
            'ITAdmin': np.random.choice([0, 1]),
            'O': np.random.uniform(0.2, 0.8),
            'C': np.random.uniform(0.2, 0.8),
            'E': np.random.uniform(0.2, 0.8),
            'A': np.random.uniform(0.2, 0.8),
            'N': np.random.uniform(0.2, 0.8)
        }
    
    # 批量处理
    batch_size = 10
    user_features = {}
    
    for user in users:
        user_events = large_events_df[large_events_df['user'] == user].to_dict('records')
        user_context = user_contexts[user]
        
        # 分批处理事件
        all_features = []
        for i in range(0, len(user_events), batch_size):
            batch_events = user_events[i:i+batch_size]
            for event in batch_events:
                features, mask = encoder.encode_event(event, user_context)
                all_features.append(features)
        
        user_features[user] = np.array(all_features)
    
    print(f"   ✅ 完成 {len(users)} 个用户的特征提取")
    
    # 3. 统计分析
    print("3️⃣ 批量统计分析")
    
    # 计算各用户的平均特征
    user_avg_features = {}
    for user, features in user_features.items():
        user_avg_features[user] = features.mean(axis=0)
    
    # 用户相似度分析
    print("   📊 用户相似度分析:")
    users_list = list(user_avg_features.keys())
    for i in range(len(users_list)):
        for j in range(i+1, len(users_list)):
            user1, user2 = users_list[i], users_list[j]
            similarity = np.corrcoef(
                user_avg_features[user1], 
                user_avg_features[user2]
            )[0, 1]
            print(f"     {user1} ↔ {user2}: {similarity:.3f}")
    
    # 4. 保存结果
    print("4️⃣ 保存处理结果")
    import pickle
    
    # 保存特征数据
    with open('batch_features.pkl', 'wb') as f:
        pickle.dump(user_features, f)
    
    # 保存编码器
    encoder.save_encoder('batch_encoder.pkl')
    
    print("   💾 结果已保存: batch_features.pkl, batch_encoder.pkl")

def main():
    """主演示函数"""
    print("🎯 内部威胁检测特征提取系统演示")
    print("=" * 60)
    print("这个演示将展示系统的主要功能和使用方法")
    
    try:
        # 基础使用演示
        encoder, events_df, users_df = demonstrate_basic_usage()
        
        # 风险分析演示
        demonstrate_risk_analysis()
        
        # 时间特征分析演示
        demonstrate_time_analysis()
        
        # 特征可视化演示
        demonstrate_feature_visualization()
        
        # 批量处理演示
        demonstrate_batch_processing()
        
        print("\n🎉 演示完成!")
        print("📚 更多详细信息请参考 README.md 和 API_REFERENCE.md")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("💡 请检查依赖是否正确安装，或查看故障排除指南")

if __name__ == "__main__":
    main() 