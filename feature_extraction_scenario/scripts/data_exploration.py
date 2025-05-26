#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CERT r4.2 数据集探索和预处理策略
用于华为多模态异常检测系统的数据分析
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import json
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

# 设置中文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class CERTDataExplorer:
    """CERT数据集探索器"""
    
    def __init__(self, data_version: str = 'r4.2'):
        """
        初始化数据探索器
        
        Args:
            data_version: 数据版本
        """
        self.data_version = data_version
        self.data_dir = f'../data/{data_version}/'
        
        # 数据文件配置
        self.data_files = {
            'device': 'device.csv',
            'email': 'email.csv', 
            'file': 'file.csv',
            'http': 'http.csv',
            'logon': 'logon.csv',
            'psychometric': 'psychometric.csv'
        }
        
        # 恶意用户标签文件
        self.insiders_file = os.path.join(self.data_dir, 'answers', 'insiders.csv')
        self.ldap_dir = os.path.join(self.data_dir, 'LDAP')
        
        # 创建输出目录
        self.output_dir = './exploration_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🔍 CERT {data_version} 数据探索器初始化完成")
        print(f"📁 数据目录: {os.path.abspath(self.data_dir)}")
        print(f"📊 输出目录: {os.path.abspath(self.output_dir)}")
    
    def check_data_availability(self) -> Dict[str, bool]:
        """检查数据文件可用性"""
        print("\n" + "="*60)
        print("📋 数据文件可用性检查")
        print("="*60)
        
        availability = {}
        total_size = 0
        
        for data_type, filename in self.data_files.items():
            file_path = os.path.join(self.data_dir, filename)
            exists = os.path.exists(file_path)
            availability[data_type] = exists
            
            if exists:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                status = f"✅ 存在 ({size_mb:.1f} MB)"
            else:
                status = "❌ 缺失"
            
            print(f"  {data_type:12s}: {status}")
        
        # 检查其他重要文件
        other_files = {
            'insiders': self.insiders_file,
            'LDAP目录': self.ldap_dir
        }
        
        for name, path in other_files.items():
            exists = os.path.exists(path)
            availability[name] = exists
            status = "✅ 存在" if exists else "❌ 缺失"
            print(f"  {name:12s}: {status}")
        
        print(f"\n📊 总数据大小: {total_size:.1f} MB")
        print(f"📈 数据完整性: {sum(availability.values())}/{len(availability)} 个文件可用")
        
        return availability
    
    def load_sample_data(self, sample_size: int = 10000) -> Dict[str, pd.DataFrame]:
        """加载样本数据进行快速探索"""
        print(f"\n📥 加载样本数据 (每个文件 {sample_size:,} 行)")
        print("-" * 40)
        
        sample_data = {}
        
        for data_type, filename in self.data_files.items():
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"⚠️  跳过 {data_type}: 文件不存在")
                continue
            
            try:
                # 读取样本数据
                df = pd.read_csv(file_path, nrows=sample_size)
                sample_data[data_type] = df
                
                print(f"✅ {data_type:8s}: {len(df):,} 行 x {len(df.columns)} 列")
                
            except Exception as e:
                print(f"❌ {data_type:8s}: 读取失败 - {e}")
        
        return sample_data
    
    def analyze_data_schema(self, sample_data: Dict[str, pd.DataFrame]) -> Dict:
        """分析数据模式和结构"""
        print(f"\n🔍 数据模式分析")
        print("="*60)
        
        schema_info = {}
        
        for data_type, df in sample_data.items():
            print(f"\n📊 {data_type.upper()} 数据结构:")
            print("-" * 30)
            
            # 基本信息
            info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'unique_counts': df.nunique().to_dict()
            }
            
            schema_info[data_type] = info
            
            # 打印列信息
            for col in df.columns:
                dtype = df[col].dtype
                missing = df[col].isnull().sum()
                unique = df[col].nunique()
                missing_pct = (missing / len(df)) * 100
                
                print(f"  {col:15s}: {str(dtype):10s} | "
                      f"缺失: {missing:4d} ({missing_pct:5.1f}%) | "
                      f"唯一值: {unique:6d}")
            
            # 显示样本数据
            print(f"\n📋 {data_type} 样本数据:")
            print(df.head(3).to_string())
        
        return schema_info
    
    def analyze_temporal_patterns(self, sample_data: Dict[str, pd.DataFrame]) -> Dict:
        """分析时间模式"""
        print(f"\n⏰ 时间模式分析")
        print("="*60)
        
        temporal_info = {}
        
        for data_type, df in sample_data.items():
            if 'date' not in df.columns:
                print(f"⚠️  {data_type}: 无日期列，跳过时间分析")
                continue
            
            print(f"\n📅 {data_type.upper()} 时间分析:")
            print("-" * 30)
            
            # 转换日期
            df['date'] = pd.to_datetime(df['date'])
            
            # 时间范围
            date_range = {
                'start_date': df['date'].min(),
                'end_date': df['date'].max(),
                'duration_days': (df['date'].max() - df['date'].min()).days
            }
            
            print(f"  时间范围: {date_range['start_date']} 到 {date_range['end_date']}")
            print(f"  持续天数: {date_range['duration_days']} 天")
            
            # 按周统计
            df['week'] = ((df['date'] - df['date'].min()).dt.days // 7)
            week_counts = df['week'].value_counts().sort_index()
            
            print(f"  周数范围: 第 {week_counts.index.min()} 周 到 第 {week_counts.index.max()} 周")
            print(f"  平均每周事件: {week_counts.mean():.1f}")
            
            # 按小时统计
            df['hour'] = df['date'].dt.hour
            hour_dist = df['hour'].value_counts().sort_index()
            
            print(f"  活跃时间: {hour_dist.index.min()}:00 - {hour_dist.index.max()}:00")
            print(f"  峰值时间: {hour_dist.idxmax()}:00 ({hour_dist.max()} 事件)")
            
            temporal_info[data_type] = {
                'date_range': date_range,
                'week_distribution': week_counts.to_dict(),
                'hour_distribution': hour_dist.to_dict()
            }
        
        return temporal_info
    
    def analyze_user_patterns(self, sample_data: Dict[str, pd.DataFrame]) -> Dict:
        """分析用户行为模式"""
        print(f"\n👥 用户行为模式分析")
        print("="*60)
        
        user_info = {}
        all_users = set()
        
        for data_type, df in sample_data.items():
            if 'user' not in df.columns:
                print(f"⚠️  {data_type}: 无用户列，跳过用户分析")
                continue
            
            print(f"\n👤 {data_type.upper()} 用户分析:")
            print("-" * 30)
            
            users = df['user'].dropna()
            user_counts = users.value_counts()
            all_users.update(users.unique())
            
            print(f"  用户总数: {len(user_counts)}")
            print(f"  事件总数: {len(users)}")
            print(f"  平均每用户事件: {len(users) / len(user_counts):.1f}")
            print(f"  最活跃用户: {user_counts.index[0]} ({user_counts.iloc[0]} 事件)")
            print(f"  事件分布:")
            print(f"    前10%用户占事件: {user_counts.head(len(user_counts)//10).sum() / len(users) * 100:.1f}%")
            
            user_info[data_type] = {
                'total_users': len(user_counts),
                'total_events': len(users),
                'top_users': user_counts.head(10).to_dict(),
                'user_distribution': user_counts.describe().to_dict()
            }
        
        print(f"\n🌐 跨数据源用户分析:")
        print(f"  总唯一用户数: {len(all_users)}")
        
        return user_info
    
    def load_insider_labels(self) -> pd.DataFrame:
        """加载内部威胁者标签"""
        print(f"\n🚨 内部威胁者标签分析")
        print("="*60)
        
        if not os.path.exists(self.insiders_file):
            print(f"❌ 内部威胁者文件不存在: {self.insiders_file}")
            return pd.DataFrame()
        
        try:
            insiders_df = pd.read_csv(self.insiders_file)
            print(f"✅ 成功加载内部威胁者数据: {len(insiders_df)} 个威胁者")
            
            print(f"\n📋 威胁者信息:")
            print("-" * 30)
            
            for _, row in insiders_df.iterrows():
                user = row['user']
                scenario = row.get('scenario', 'Unknown')
                start = row.get('start', 'Unknown')
                end = row.get('end', 'Unknown')
                
                print(f"  {user}: 场景 {scenario}, 时间 {start} - {end}")
            
            # 分析威胁场景分布
            if 'scenario' in insiders_df.columns:
                scenario_counts = insiders_df['scenario'].value_counts()
                print(f"\n📊 威胁场景分布:")
                for scenario, count in scenario_counts.items():
                    print(f"  场景 {scenario}: {count} 个用户")
            
            return insiders_df
            
        except Exception as e:
            print(f"❌ 加载内部威胁者数据失败: {e}")
            return pd.DataFrame()
    
    def analyze_data_quality(self, sample_data: Dict[str, pd.DataFrame]) -> Dict:
        """分析数据质量"""
        print(f"\n🔍 数据质量分析")
        print("="*60)
        
        quality_report = {}
        
        for data_type, df in sample_data.items():
            print(f"\n📊 {data_type.upper()} 数据质量:")
            print("-" * 30)
            
            # 缺失值分析
            missing_analysis = {}
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            
            print(f"  总单元格数: {total_cells:,}")
            print(f"  缺失单元格: {missing_cells:,} ({missing_cells/total_cells*100:.2f}%)")
            
            # 按列分析缺失值
            missing_by_col = df.isnull().sum()
            missing_cols = missing_by_col[missing_by_col > 0]
            
            if len(missing_cols) > 0:
                print(f"  缺失值列数: {len(missing_cols)}/{len(df.columns)}")
                for col, missing_count in missing_cols.items():
                    pct = missing_count / len(df) * 100
                    print(f"    {col}: {missing_count} ({pct:.1f}%)")
            else:
                print(f"  ✅ 无缺失值")
            
            # 重复值分析
            duplicates = df.duplicated().sum()
            print(f"  重复行数: {duplicates} ({duplicates/len(df)*100:.2f}%)")
            
            # 数据类型一致性
            type_issues = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # 检查是否应该是数值类型
                    try:
                        pd.to_numeric(df[col].dropna())
                        type_issues.append(f"{col} (可能应为数值型)")
                    except:
                        pass
            
            if type_issues:
                print(f"  类型问题: {', '.join(type_issues)}")
            else:
                print(f"  ✅ 数据类型一致")
            
            quality_report[data_type] = {
                'missing_percentage': missing_cells/total_cells*100,
                'duplicate_percentage': duplicates/len(df)*100,
                'missing_columns': missing_by_col.to_dict(),
                'type_issues': type_issues
            }
        
        return quality_report
    
    def create_data_summary_report(self, schema_info: Dict, temporal_info: Dict, 
                                 user_info: Dict, quality_report: Dict, 
                                 insiders_df: pd.DataFrame) -> str:
        """创建数据摘要报告"""
        print(f"\n📝 生成数据摘要报告")
        print("="*60)
        
        report = []
        report.append("# CERT r4.2 数据集探索报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 数据概览
        report.append("## 📊 数据概览")
        report.append("")
        total_events = sum(info['shape'][0] for info in schema_info.values())
        total_columns = sum(len(info['columns']) for info in schema_info.values())
        
        report.append(f"- **数据源数量**: {len(schema_info)} 个")
        report.append(f"- **总事件数**: {total_events:,} (样本)")
        report.append(f"- **总字段数**: {total_columns}")
        report.append(f"- **内部威胁者**: {len(insiders_df)} 个")
        report.append("")
        
        # 各数据源详情
        report.append("## 📋 数据源详情")
        report.append("")
        
        for data_type, info in schema_info.items():
            report.append(f"### {data_type.upper()}")
            report.append(f"- **维度**: {info['shape'][0]:,} 行 x {info['shape'][1]} 列")
            report.append(f"- **字段**: {', '.join(info['columns'])}")
            
            if data_type in quality_report:
                quality = quality_report[data_type]
                report.append(f"- **数据质量**: 缺失率 {quality['missing_percentage']:.2f}%, 重复率 {quality['duplicate_percentage']:.2f}%")
            
            if data_type in temporal_info:
                temporal = temporal_info[data_type]
                date_range = temporal['date_range']
                report.append(f"- **时间范围**: {date_range['start_date'].date()} 到 {date_range['end_date'].date()} ({date_range['duration_days']} 天)")
            
            if data_type in user_info:
                user = user_info[data_type]
                report.append(f"- **用户数**: {user['total_users']:,} 个用户, 平均每用户 {user['total_events']/user['total_users']:.1f} 事件")
            
            report.append("")
        
        # 数据质量总结
        report.append("## 🔍 数据质量总结")
        report.append("")
        
        avg_missing = np.mean([q['missing_percentage'] for q in quality_report.values()])
        avg_duplicate = np.mean([q['duplicate_percentage'] for q in quality_report.values()])
        
        report.append(f"- **平均缺失率**: {avg_missing:.2f}%")
        report.append(f"- **平均重复率**: {avg_duplicate:.2f}%")
        
        # 主要发现
        report.append("## 🎯 主要发现")
        report.append("")
        report.append("### 数据特点")
        
        # 找出最大的数据源
        largest_source = max(schema_info.items(), key=lambda x: x[1]['shape'][0])
        report.append(f"- **最大数据源**: {largest_source[0]} ({largest_source[1]['shape'][0]:,} 事件)")
        
        # 找出用户最多的数据源
        if user_info:
            most_users_source = max(user_info.items(), key=lambda x: x[1]['total_users'])
            report.append(f"- **用户最多**: {most_users_source[0]} ({most_users_source[1]['total_users']:,} 用户)")
        
        # 数据质量问题
        report.append("")
        report.append("### 数据质量问题")
        
        high_missing = [name for name, q in quality_report.items() if q['missing_percentage'] > 10]
        if high_missing:
            report.append(f"- **高缺失率数据源**: {', '.join(high_missing)}")
        
        high_duplicate = [name for name, q in quality_report.items() if q['duplicate_percentage'] > 5]
        if high_duplicate:
            report.append(f"- **高重复率数据源**: {', '.join(high_duplicate)}")
        
        # 预处理建议
        report.append("")
        report.append("## 💡 预处理建议")
        report.append("")
        report.append("### 数据清洗")
        report.append("- 处理缺失值：根据业务逻辑填充或删除")
        report.append("- 去除重复记录：保留最新或最完整的记录")
        report.append("- 数据类型转换：确保日期、数值类型正确")
        report.append("")
        
        report.append("### 特征工程")
        report.append("- 时间特征：提取小时、星期、月份等时间特征")
        report.append("- 用户特征：计算用户活动频率、模式等")
        report.append("- 行为序列：构建用户行为时间序列")
        report.append("- 异常标注：基于内部威胁者标签创建训练标签")
        report.append("")
        
        report.append("### 多模态融合")
        report.append("- 文本特征：邮件内容、文件名等文本信息")
        report.append("- 结构化特征：登录时间、设备信息等")
        report.append("- 图特征：用户关系网络、通信图等")
        report.append("- 时序特征：行为序列、时间模式等")
        
        # 保存报告
        report_text = "\n".join(report)
        report_file = os.path.join(self.output_dir, 'data_exploration_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✅ 报告已保存: {report_file}")
        
        return report_text
    
    def create_visualization_dashboard(self, sample_data: Dict[str, pd.DataFrame],
                                     temporal_info: Dict, user_info: Dict,
                                     schema_info: Dict) -> None:
        """创建可视化仪表板"""
        print(f"\n📊 创建可视化仪表板")
        print("="*60)
        
        # 设置图形大小
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 数据源大小对比
        plt.subplot(2, 3, 1)
        if schema_info: # 确保 schema_info 存在
            data_sizes = [info['shape'][0] for info in schema_info.values()]
            data_names = list(schema_info.keys())

            plt.bar(data_names, data_sizes, color='skyblue', alpha=0.7)
            plt.title('Data Source Sizes (Sample)', fontsize=14, fontweight='bold')
            plt.xlabel('Data Source')
            plt.ylabel('Number of Records')
            plt.xticks(rotation=45)
        else:
            print("⚠️ 无法生成数据源大小对比图：缺少 schema_info 数据")
        
        # 2. 时间分布（以第一个有时间数据的源为例）
        plt.subplot(2, 3, 2)
        if temporal_info:
            first_temporal = list(temporal_info.values())[0]
            weeks = list(first_temporal['week_distribution'].keys())
            counts = list(first_temporal['week_distribution'].values())
            
            plt.plot(weeks, counts, marker='o', linewidth=2, markersize=4)
            plt.title('Weekly Activity Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Week Number')
            plt.ylabel('Number of Events')
            plt.grid(True, alpha=0.3)
        
        # 3. 用户活动分布
        plt.subplot(2, 3, 3)
        if user_info:
            # 合并所有用户事件数
            all_user_events = []
            for info in user_info.values():
                all_user_events.extend(info['top_users'].values())
            
            plt.hist(all_user_events, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
            plt.title('User Activity Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Events per User')
            plt.ylabel('Number of Users')
            plt.yscale('log')
        
        # 4. 小时活动模式
        plt.subplot(2, 3, 4)
        if temporal_info:
            # 合并所有小时分布
            hour_totals = defaultdict(int)
            for info in temporal_info.values():
                for hour, count in info['hour_distribution'].items():
                    hour_totals[hour] += count
            
            hours = sorted(hour_totals.keys())
            counts = [hour_totals[h] for h in hours]
            
            plt.bar(hours, counts, color='orange', alpha=0.7)
            plt.title('Hourly Activity Pattern', fontsize=14, fontweight='bold')
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Events')
            plt.xticks(range(0, 24, 2))
        
        # 5. 数据质量热图
        plt.subplot(2, 3, 5)
        quality_data = []
        quality_labels = []
        
        for data_type, df in sample_data.items():
            missing_rates = (df.isnull().sum() / len(df) * 100).values
            quality_data.append(missing_rates)
            quality_labels.append(data_type)
        
        if quality_data:
            # 确保所有数组长度一致
            max_cols = max(len(row) for row in quality_data)
            quality_matrix = np.zeros((len(quality_data), max_cols))
            
            for i, row in enumerate(quality_data):
                quality_matrix[i, :len(row)] = row
            
            im = plt.imshow(quality_matrix, cmap='Reds', aspect='auto')
            plt.title('Missing Data Heatmap (%)', fontsize=14, fontweight='bold')
            plt.xlabel('Column Index')
            plt.ylabel('Data Source')
            plt.yticks(range(len(quality_labels)), quality_labels)
            plt.colorbar(im, shrink=0.8)
        
        # 6. 数据源关系（用户重叠）
        plt.subplot(2, 3, 6)
        if len(sample_data) >= 2:
            # 计算用户重叠
            user_sets = {}
            for data_type, df in sample_data.items():
                if 'user' in df.columns:
                    user_sets[data_type] = set(df['user'].dropna().unique())
            
            if len(user_sets) >= 2:
                # 创建重叠矩阵
                sources = list(user_sets.keys())
                overlap_matrix = np.zeros((len(sources), len(sources)))
                
                for i, source1 in enumerate(sources):
                    for j, source2 in enumerate(sources):
                        if i == j:
                            overlap_matrix[i, j] = len(user_sets[source1])
                        else:
                            overlap = len(user_sets[source1] & user_sets[source2])
                            overlap_matrix[i, j] = overlap
                
                im = plt.imshow(overlap_matrix, cmap='Blues', aspect='auto')
                plt.title('User Overlap Between Sources', fontsize=14, fontweight='bold')
                plt.xlabel('Data Source')
                plt.ylabel('Data Source')
                plt.xticks(range(len(sources)), sources, rotation=45)
                plt.yticks(range(len(sources)), sources)
                plt.colorbar(im, shrink=0.8)
                
                # 添加数值标注
                for i in range(len(sources)):
                    for j in range(len(sources)):
                        plt.text(j, i, f'{int(overlap_matrix[i, j])}', 
                               ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        dashboard_file = os.path.join(self.output_dir, 'data_exploration_dashboard.png')
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化仪表板已保存: {dashboard_file}")
    
    def run_full_exploration(self, sample_size: int = 10000) -> Dict:
        """运行完整的数据探索流程"""
        print("🚀 启动CERT数据集完整探索流程")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # 1. 检查数据可用性
            availability = self.check_data_availability()
            
            # 2. 加载样本数据
            sample_data = self.load_sample_data(sample_size)
            
            if not sample_data:
                print("❌ 无法加载任何数据，终止探索")
                return {}
            
            # 3. 分析数据模式
            schema_info = self.analyze_data_schema(sample_data)
            
            # 4. 分析时间模式
            temporal_info = self.analyze_temporal_patterns(sample_data)
            
            # 5. 分析用户模式
            user_info = self.analyze_user_patterns(sample_data)
            
            # 6. 加载威胁者标签
            insiders_df = self.load_insider_labels()
            
            # 7. 分析数据质量
            quality_report = self.analyze_data_quality(sample_data)
            
            # 8. 生成摘要报告
            report = self.create_data_summary_report(
                schema_info, temporal_info, user_info, quality_report, insiders_df
            )
            
            # 9. 创建可视化仪表板
            self.create_visualization_dashboard(sample_data, temporal_info, user_info, schema_info)
            
            # 10. 保存探索结果
            exploration_results = {
                'availability': availability,
                'schema_info': schema_info,
                'temporal_info': temporal_info,
                'user_info': user_info,
                'quality_report': quality_report,
                'insiders_count': len(insiders_df),
                'exploration_time': (datetime.now() - start_time).total_seconds()
            }
            
            results_file = os.path.join(self.output_dir, 'exploration_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                # 转换不可序列化的对象
                serializable_results = {}
                for key, value in exploration_results.items():
                    if key == 'temporal_info':
                        # 转换日期时间对象
                        serializable_value = {}
                        for data_type, info in value.items():
                            serializable_value[data_type] = {}
                            for k, v in info.items():
                                if k == 'date_range':
                                    serializable_value[data_type][k] = {
                                        'start_date': v['start_date'].isoformat(),
                                        'end_date': v['end_date'].isoformat(),
                                        'duration_days': v['duration_days']
                                    }
                                else:
                                    serializable_value[data_type][k] = v
                        serializable_results[key] = serializable_value
                    elif key == 'schema_info':
                         # 转换 dtype 对象为字符串
                        serializable_value = {}
                        for data_type, info in value.items():
                            serializable_value[data_type] = {
                                'shape': info['shape'],
                                'columns': info['columns'],
                                # 将 dtype 对象转换为字符串
                                'dtypes': {col: str(dtype) for col, dtype in info['dtypes'].items()},
                                'missing_values': info['missing_values'],
                                'unique_counts': info['unique_counts']
                            }
                        serializable_results[key] = serializable_value
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n🎉 数据探索完成!")
            print(f"⏱️  总耗时: {total_time:.1f} 秒")
            print(f"📁 结果保存在: {os.path.abspath(self.output_dir)}")
            print(f"📊 探索了 {len(sample_data)} 个数据源，共 {sum(len(df) for df in sample_data.values()):,} 条记录")
            
            return exploration_results
            
        except Exception as e:
            print(f"❌ 数据探索失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

def main():
    """主函数"""
    print("🔍 CERT r4.2 数据集探索和预处理策略")
    print("="*80)
    
    # 创建探索器
    explorer = CERTDataExplorer(data_version='r4.2')
    
    # 运行完整探索
    results = explorer.run_full_exploration(sample_size=50000)  # 增加样本大小以获得更好的统计
    
    if results:
        print("\n📋 探索结果摘要:")
        print(f"  - 数据源数量: {len(results.get('schema_info', {}))}")
        print(f"  - 数据质量平均分: {np.mean([q['missing_percentage'] for q in results.get('quality_report', {}).values()]):.2f}% 缺失率")
        print(f"  - 内部威胁者数量: {results.get('insiders_count', 0)}")
        print(f"  - 探索耗时: {results.get('exploration_time', 0):.1f} 秒")

if __name__ == "__main__":
    main() 