# API 快速参考

## 🎯 核心类

### EventEncoder - 统一事件编码器

```python
from encoder import EventEncoder

encoder = EventEncoder(feature_dim=256, data_version='r4.2', device='cpu')
```

#### 方法

| 方法 | 描述 | 输入 | 输出 |
|------|------|------|------|
| `fit(events_data, user_data)` | 拟合编码器 | DataFrame | None |
| `encode_event(event_dict, user_context)` | 编码单个事件 | dict, dict | (features, mask) |
| `encode_event_sequence(events, user_context, max_len)` | 编码事件序列 | List[dict], dict, int | (seq_features, seq_mask) |
| `get_feature_names()` | 获取特征名称 | None | List[str] |
| `save_encoder(filepath)` | 保存编码器 | str | None |
| `load_encoder(filepath)` | 加载编码器 | str | None |

### FeatureEncoder - 基础特征编码器

```python
from utils import FeatureEncoder

fe = FeatureEncoder(embedding_dim=64, max_vocab_size=10000)
```

#### 方法

| 方法 | 描述 | 输入 | 输出 |
|------|------|------|------|
| `fit_numerical_scaler(data, name, method)` | 拟合数值标准化器 | array, str, str | None |
| `transform_numerical(data, name, fill_missing, fill_value)` | 转换数值特征 | array, str, bool, float | (transformed, mask) |
| `fit_categorical_encoder(data, name)` | 拟合分类编码器 | List[str], str | None |
| `transform_categorical(data, name, fill_missing, fill_value)` | 转换分类特征 | List[str], str, bool, str | (encoded, mask) |
| `text_to_features(texts, name, max_features)` | 文本转TF-IDF | List[str], str, int | (tfidf_matrix, mask) |
| `binning_transform(data, n_bins, strategy)` | 数值分箱 | array, int, str | (binned, mask) |

## 🔧 工具函数

### 时间特征 (temporal.py)

```python
from temporal import encode_temporal_features, parse_timestamp

# 编码时间特征
features, mask = encode_temporal_features(event_dict, feature_encoder)

# 解析时间戳
dt = parse_timestamp('01/15/2024 14:30:00')
```

### 用户上下文 (user_context.py)

```python
from user_context import encode_user_context, encode_behavioral_risk_profile

# 编码用户上下文
features, mask = encode_user_context(user_context, feature_encoder)

# 风险画像
risk_profile = encode_behavioral_risk_profile(user_context)
```

### 特定事件类型编码

#### 邮件特征 (email.py)
```python
from email import encode_email_features, detect_suspicious_email_patterns

features, mask = encode_email_features(event_dict, feature_encoder, 'r4.2')
risk_patterns = detect_suspicious_email_patterns(email_events)
```

#### 文件特征 (file.py)
```python
from file import encode_file_features, detect_suspicious_file_patterns

features, mask = encode_file_features(event_dict, feature_encoder, 'r4.2')
risk_patterns = detect_suspicious_file_patterns(file_events)
```

#### HTTP特征 (http.py)
```python
from http import encode_http_features, detect_suspicious_http_patterns

features, mask = encode_http_features(event_dict, feature_encoder, 'r4.2')
risk_patterns = detect_suspicious_http_patterns(http_events)
```

#### 设备特征 (device.py)
```python
from device import encode_device_features, detect_suspicious_device_patterns

features, mask = encode_device_features(event_dict, feature_encoder, 'r4.2')
risk_patterns = detect_suspicious_device_patterns(device_events)
```

## 📊 数据格式

### 事件字典格式

```python
# 邮件事件
email_event = {
    'type': 'email',
    'date': '01/15/2024 14:30:00',
    'user': 'ACM2278',
    'from': 'sender@dtaa.com',
    'to': 'recipient@dtaa.com;external@gmail.com',
    'cc': 'cc@dtaa.com',
    'bcc': '',
    'size': '2048',
    'content': 'Email content...',
    'activity': 'Send',  # r5.2+
    'att': 'file.pdf(1024);data.xlsx(512)'  # r5.2+
}

# 文件事件
file_event = {
    'type': 'file',
    'date': '01/15/2024 16:45:00',
    'user': 'ACM2278',
    'pc': 'PC-1234',
    'url/fname': 'C:\\Users\\ACM2278\\Documents\\file.xlsx',
    'content': 'File content...',
    'activity': 'file copy',  # r5.2+
    'to': 'true',  # USB传输 r5.2+
    'from': 'false'  # USB传输 r5.2+
}

# HTTP事件
http_event = {
    'type': 'http',
    'date': '01/15/2024 10:30:00',
    'user': 'ACM2278',
    'url/fname': 'https://www.example.com/page',
    'content': 'Web page content...',
    'activity': 'www visit'  # r6.2+
}

# 设备事件
device_event = {
    'type': 'device',
    'date': '01/15/2024 14:20:00',
    'user': 'ACM2278',
    'activity': 'USB Connect',
    'content': 'Device information...',
    'file_tree_len': '150'  # r5.2+
}
```

### 用户上下文格式

```python
user_context = {
    'user_id': 'ACM2278',
    'role': 'Manager',           # Employee, Supervisor, Manager, Director, Executive
    'dept': 'Finance',           # IT, Finance, HR, Marketing, Engineering, etc.
    'ITAdmin': 0,                # 0 or 1
    'O': 0.6,                    # Openness (0.0-1.0)
    'C': 0.9,                    # Conscientiousness (0.0-1.0)
    'E': 0.7,                    # Extraversion (0.0-1.0)
    'A': 0.8,                    # Agreeableness (0.0-1.0)
    'N': 0.2,                    # Neuroticism (0.0-1.0)
    'pc_type': 0,                # 0: 个人, 1: 共享, 2: 他人, 3: 主管
    'sharedpc': None,            # 共享PC信息
    'npc': 1                     # PC数量
}
```

## ⚙️ 配置选项

### 数据版本对应特征

| 版本 | 邮件特征 | 文件特征 | HTTP特征 | 设备特征 |
|------|----------|----------|----------|----------|
| r4.2 | 基础收发信息 | 基础类型大小 | URL域名分类 | 基础连接 |
| r5.2 | +详细附件+活动类型 | +USB传输+活动类型 | 同r4.2 | +内容+文件树 |
| r6.2 | 同r5.2 | 同r5.2 | +活动类型 | 同r5.2 |

### 特征维度配置

```python
# 默认维度分配 (r4.2)
feature_dims = {
    'temporal': 12,        # 时间特征
    'user_context': 10,    # 用户上下文
    'email': 9,           # 邮件特征
    'file': 5,            # 文件特征
    'http': 5,            # HTTP特征
    'device': 2,          # 设备特征
    'event_type': 6,      # 事件类型
    'activity': 20        # 活动特征
}
```

## 🔍 返回值说明

### 特征向量和mask

所有编码函数都返回 `(features, mask)` 元组：

- `features`: numpy数组，包含编码后的数值特征
- `mask`: 布尔数组，标记哪些特征有效（True）哪些缺失（False）

### 风险评分字典

异常检测函数返回风险评分字典，值范围0.0-1.0：

```python
risk_patterns = {
    'high_external_ratio': 0.3,      # 外部联系人比例
    'large_email_ratio': 0.1,        # 大邮件比例
    'executable_attachment_ratio': 0.0  # 可执行附件比例
}
```

## 💡 最佳实践

1. **数据预处理**: 确保时间戳格式正确，缺失值用None或NaN表示
2. **版本选择**: 根据数据集选择正确的data_version参数
3. **内存管理**: 处理大量数据时使用批处理
4. **特征检查**: 定期检查mask比例，确保数据质量
5. **模型保存**: 训练完成后保存编码器状态以便复用 