# 内部威胁检测 - 特征提取系统使用文档

## 📋 目录
- [系统概述](#系统概述)
- [快速开始](#快速开始)
- [API参考](#api参考)
- [配置说明](#配置说明)
- [示例代码](#示例代码)
- [故障排除](#故障排除)

## 🎯 系统概述

这是一个针对内部威胁检测的模块化特征提取系统，能够将各种用户活动事件（邮件、文件、HTTP、设备操作）转换为统一的数值向量表示，支持后续的机器学习模型训练。

### 主要特性

- ✅ **模块化设计** - 8个独立功能模块，易于维护和扩展
- ✅ **版本兼容** - 支持CERT数据集多个版本 (r4.2, r5.2, r6.2)
- ✅ **缺失数据处理** - mask机制自动处理不完整日志
- ✅ **统一输出** - 所有事件转为固定长度向量
- ✅ **风险评估** - 内置异常模式检测和风险评分

### 支持的事件类型

| 事件类型 | 模块 | 主要特征 |
|---------|------|---------|
| 邮件活动 | email.py | 收件人、附件、大小、外部联系人 |
| 文件操作 | file.py | 文件类型、路径、USB传输、活动类型 |
| 网络浏览 | http.py | URL分析、域名分类、内容特征 |
| 设备连接 | device.py | USB活动、使用时长、文件树 |
| 时间上下文 | temporal.py | 工作时间、会话时长、时间模式 |
| 用户上下文 | user_context.py | 角色、部门、OCEAN特征、权限 |

## 🚀 快速开始

### 1. 环境准备

```bash
# Python依赖
pip install numpy pandas scikit-learn torch

# 确保项目路径
cd /path/to/feature_extraction_scenario/
```

### 2. 基础使用

```python
from encoder import EventEncoder
import pandas as pd

# 1. 创建编码器
encoder = EventEncoder(
    feature_dim=256,      # 输出向量维度
    data_version='r4.2'   # 数据版本
)

# 2. 加载数据并拟合编码器
events_df = pd.read_csv('events.csv')
user_df = pd.read_csv('users.csv')

encoder.fit(events_df, user_df)

# 3. 编码单个事件
event_dict = {
    'type': 'email',
    'date': '12/01/2023 09:30:00',
    'user': 'ACM2278',
    'pc': 'PC-1234',
    'to': 'colleague@dtaa.com',
    'size': '1024',
    'content': 'Meeting reminder'
}

user_context = {
    'role': 'Employee',
    'dept': 'Engineering', 
    'ITAdmin': 0,
    'O': 0.7, 'C': 0.8, 'E': 0.6, 'A': 0.9, 'N': 0.3
}

# 获取特征向量
features, mask = encoder.encode_event(event_dict, user_context)
print(f"特征向量维度: {features.shape}")  # (256,)
print(f"有效特征数: {mask.sum()}")
```

### 3. 批量处理

```python
# 编码事件序列
events = [event_dict1, event_dict2, event_dict3]
seq_features, seq_mask = encoder.encode_event_sequence(
    events, 
    user_context, 
    max_sequence_length=100
)
print(f"序列特征维度: {seq_features.shape}")  # (100, 256)
```

## 📚 API参考

### EventEncoder 类

#### 构造函数
```python
EventEncoder(feature_dim=256, data_version='r4.2', device='cpu')
```

**参数:**
- `feature_dim` (int): 输出特征向量维度，默认256
- `data_version` (str): CERT数据版本，支持 'r4.2', 'r5.2', 'r6.2'
- `device` (str): 计算设备，默认'cpu'

#### 主要方法

##### fit(events_data, user_data=None)
拟合编码器参数

**参数:**
- `events_data` (DataFrame): 事件数据，必须包含 ['type', 'date', 'user'] 列
- `user_data` (DataFrame): 用户数据，可选

**示例:**
```python
encoder.fit(events_df, users_df)
```

##### encode_event(event_dict, user_context=None)
编码单个事件

**参数:**
- `event_dict` (dict): 事件字典
- `user_context` (dict): 用户上下文，可选

**返回:**
- `Tuple[np.ndarray, np.ndarray]`: (特征向量, mask向量)

##### encode_event_sequence(events, user_context=None, max_sequence_length=100)
编码事件序列

**参数:**
- `events` (List[dict]): 事件列表
- `user_context` (dict): 用户上下文
- `max_sequence_length` (int): 最大序列长度

**返回:**
- `Tuple[np.ndarray, np.ndarray]`: (序列特征矩阵, 序列mask矩阵)

##### save_encoder(filepath) / load_encoder(filepath)
保存/加载编码器状态

### FeatureEncoder 类

#### 主要方法

```python
from utils import FeatureEncoder

fe = FeatureEncoder(embedding_dim=64, max_vocab_size=10000)

# 数值特征处理
fe.fit_numerical_scaler(data, 'feature_name')
transformed, mask = fe.transform_numerical(data, 'feature_name')

# 分类特征处理  
fe.fit_categorical_encoder(categories, 'feature_name')
encoded, mask = fe.transform_categorical(categories, 'feature_name')

# 文本特征处理
tfidf_features, mask = fe.text_to_features(texts, 'feature_name')
```

## ⚙️ 配置说明

### 数据版本差异

不同CERT版本支持的特征有所差异：

```python
# r4.2 - 基础版本
- 邮件: 基本收发信息、附件数量
- 文件: 基本类型、大小
- HTTP: URL、域名分类
- 设备: 基本连接信息

# r5.2 - 增强版本
+ 邮件: 详细附件信息、活动类型
+ 文件: USB传输、活动类型  
+ 设备: 内容信息、文件树

# r6.2 - 完整版本
+ HTTP: 活动类型 (访问/下载/上传)
+ 设备: 扩展设备信息
```

### 特征维度自定义

```python
# 自定义特征维度
encoder = EventEncoder(feature_dim=512)  # 更大的向量

# 或者修改各模块维度
encoder.feature_dims['email'] = 30  # 增加邮件特征维度
```

### 缺失数据策略

```python
# 在utils.py中的FeatureEncoder支持多种填充策略
transform_numerical(data, 'feature', fill_missing=True, fill_value=0.0)
transform_categorical(data, 'feature', fill_missing=True, fill_value='unknown')
```

## 💡 示例代码

### 示例1: 处理邮件事件

```python
from encoder import EventEncoder

# 邮件事件示例
email_event = {
    'type': 'email',
    'date': '01/15/2024 14:30:00',
    'user': 'ACM2278',
    'from': 'user@dtaa.com',
    'to': 'external@gmail.com;colleague@dtaa.com',
    'cc': 'manager@dtaa.com', 
    'bcc': '',
    'size': '2048',
    'content': 'Confidential project report attached',
    'activity': 'Send',
    'att': 'report.pdf(1024);data.xlsx(512)'  # r5.2+版本
}

user_info = {
    'role': 'Manager',
    'dept': 'Finance',
    'ITAdmin': 0,
    'O': 0.6, 'C': 0.9, 'E': 0.7, 'A': 0.8, 'N': 0.2
}

encoder = EventEncoder(data_version='r5.2')
encoder.fit(events_df, users_df)

features, mask = encoder.encode_event(email_event, user_info)

# 分析特征
feature_names = encoder.get_feature_names()
valid_features = features[mask]
print(f"有效特征数: {len(valid_features)}")
print(f"特征范围: [{features.min():.3f}, {features.max():.3f}]")
```

### 示例2: 处理文件操作事件

```python
# 文件操作事件
file_event = {
    'type': 'file',
    'date': '01/15/2024 16:45:00',
    'user': 'ACM2278',
    'pc': 'PC-1234',
    'url/fname': 'C:\\Users\\ACM2278\\Documents\\sensitive_data.xlsx',
    'content': 'Employee salary information...',  # 文件内容
    'activity': 'file copy',  # r5.2+
    'to': 'true',  # USB传输
    'from': 'false'
}

features, mask = encoder.encode_event(file_event, user_info)

# 可以分析文件操作风险
from file import detect_suspicious_file_patterns
risk_patterns = detect_suspicious_file_patterns([file_event])
print(f"文件操作风险评分: {risk_patterns}")
```

### 示例3: 会话级分析

```python
# 用户会话分析
session_events = [
    {'type': 'logon', 'date': '01/15/2024 08:30:00', 'user': 'ACM2278'},
    {'type': 'email', 'date': '01/15/2024 09:15:00', 'user': 'ACM2278', ...},
    {'type': 'http', 'date': '01/15/2024 10:30:00', 'user': 'ACM2278', ...},
    {'type': 'file', 'date': '01/15/2024 11:45:00', 'user': 'ACM2278', ...},
    {'type': 'device', 'date': '01/15/2024 14:20:00', 'user': 'ACM2278', ...}
]

# 编码整个会话
session_features, session_mask = encoder.encode_event_sequence(
    session_events, 
    user_info,
    max_sequence_length=50
)

print(f"会话特征矩阵: {session_features.shape}")  # (50, 256)

# 会话级时间特征分析
from temporal import encode_session_temporal_features
session_temporal, _ = encode_session_temporal_features(session_events, encoder.feature_encoder)
print(f"会话时间特征: 持续时长={session_temporal[0]*480:.1f}分钟")
```

### 示例4: 批量处理和保存

```python
import pickle
import numpy as np

# 批量处理多个用户
all_users = ['ACM2278', 'ACM1796', 'CMP2946']
user_features = {}

for user_id in all_users:
    # 获取用户事件
    user_events = events_df[events_df['user'] == user_id].to_dict('records')
    user_context = get_user_context_from_dataframe(user_id, users_df)
    
    # 编码事件序列
    seq_features, seq_mask = encoder.encode_event_sequence(
        user_events, user_context, max_sequence_length=100
    )
    
    user_features[user_id] = {
        'features': seq_features,
        'mask': seq_mask,
        'context': user_context
    }

# 保存结果
with open('user_features.pkl', 'wb') as f:
    pickle.dump(user_features, f)

# 保存编码器
encoder.save_encoder('event_encoder.pkl')
```

### 示例5: 异常检测应用

```python
# 异常模式检测
from email import detect_suspicious_email_patterns
from http import detect_suspicious_http_patterns
from file import detect_suspicious_file_patterns

def analyze_user_risk(user_events, user_context):
    """分析用户风险"""
    
    # 按事件类型分组
    email_events = [e for e in user_events if e['type'] == 'email']
    http_events = [e for e in user_events if e['type'] == 'http'] 
    file_events = [e for e in user_events if e['type'] == 'file']
    
    # 检测各类异常模式
    email_risks = detect_suspicious_email_patterns(email_events)
    http_risks = detect_suspicious_http_patterns(http_events)
    file_risks = detect_suspicious_file_patterns(file_events)
    
    # 用户上下文风险
    from user_context import encode_behavioral_risk_profile
    context_risks = encode_behavioral_risk_profile(user_context)
    
    # 综合风险评分
    overall_risk = (
        np.mean(list(email_risks.values())) * 0.3 +
        np.mean(list(http_risks.values())) * 0.3 + 
        np.mean(list(file_risks.values())) * 0.2 +
        context_risks['overall_risk'] * 0.2
    )
    
    return {
        'overall_risk': overall_risk,
        'email_risks': email_risks,
        'http_risks': http_risks, 
        'file_risks': file_risks,
        'context_risks': context_risks
    }

# 使用示例
user_risk = analyze_user_risk(session_events, user_info)
print(f"用户整体风险评分: {user_risk['overall_risk']:.3f}")
```

## 🔧 故障排除

### 常见问题

**Q1: 编码器拟合失败**
```python
# 检查数据格式
print(events_df.columns)  # 确保包含必要列
print(events_df['type'].value_counts())  # 检查事件类型

# 数据预处理
events_df = events_df.dropna(subset=['type', 'date', 'user'])
```

**Q2: 特征向量维度不匹配**
```python
# 检查配置
print(f"编码器特征维度: {encoder.feature_dim}")
print(f"实际输出维度: {features.shape}")

# 重新初始化编码器
encoder = EventEncoder(feature_dim=512)  # 调整维度
```

**Q3: 缺失数据处理**
```python
# 检查mask覆盖率
mask_ratio = mask.mean()
print(f"有效特征比例: {mask_ratio:.2%}")

if mask_ratio < 0.5:
    print("警告: 超过50%的特征缺失，请检查数据质量")
```

**Q4: 性能优化**
```python
# 批处理优化
batch_size = 1000
for i in range(0, len(events), batch_size):
    batch_events = events[i:i+batch_size]
    # 处理批次...

# 内存优化
import gc
gc.collect()  # 手动垃圾回收
```

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 特征可视化
import matplotlib.pyplot as plt

# 绘制特征分布
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(features[mask], bins=50, alpha=0.7)
plt.title('有效特征分布')

plt.subplot(1, 2, 2) 
plt.plot(mask.astype(int))
plt.title('特征mask模式')
plt.show()

# 特征重要性分析
feature_names = encoder.get_feature_names()
valid_indices = np.where(mask)[0]
important_features = [(feature_names[i], features[i]) 
                     for i in valid_indices if abs(features[i]) > 0.1]
print(f"重要特征 (|value| > 0.1): {len(important_features)}")
```

## 📝 更新日志

- **v1.0** - 初始版本，支持基础特征提取
- **v1.1** - 添加r5.2/r6.2版本支持 
- **v1.2** - 增强异常检测功能
- **v1.3** - 优化性能和内存使用

## 🤝 贡献指南

如需扩展新的特征模块：

1. 创建新的模块文件 `new_feature.py`
2. 实现 `encode_xxx_features()` 函数
3. 在 `encoder.py` 中集成新模块
4. 更新特征维度配置
5. 添加对应的测试用例

---

📞 **技术支持**: 如有问题请查看FAQ或提交issue 