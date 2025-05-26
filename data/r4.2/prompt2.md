我们正在开发一个针对内部威胁检测任务的场景识别系统，数据包括：

时态上下文（工作时间、会话时长、工作日/周末等）
用户上下文（角色、部门、是否 IT 管理员、心理特征 OCEAN）
行为特征：涵盖 Email、文件操作、Web 浏览、设备连接等多类型事件
我们希望对项目进行以下重构和拓展：

① 重构特征提取逻辑
目标是拆分并替换 feature_extraction.py：

保留原始 pipeline，防止主流程损坏；
创建新目录 feature_extraction_scenario/，并拆分为多个子模块：
feature_extraction_scenario/
├── encoder.py               # encode_event(event_dict)
├── temporal.py              # encode_temporal_features()
├── user_context.py          # encode_user_context()
├── email.py                 # encode_email_features()
├── file.py                  # encode_file_features()
├── http.py                  # encode_http_features()
├── device.py                # encode_device_features()
├── utils.py                 # 标准化、分箱、embedding 工具函数
使用 mask + 缺省填充策略支持不完整日志
所有编码保持统一向量输出格式，适配后续模型输入
② 审查原有 feature_extraction.py 的方法
我们需要：

思考原始 feature_extraction.py 中的编码逻辑是否符合我们的新特征；
逐个理解函数含义；
保留有用函数（如 get_mal_userdata），并重新封装；
特别关注那些根据 CERT 数据集版本进行“条件编码”的部分逻辑，对我们很有帮助；
所有旧逻辑不应删除，只做旁路替换。
③ 用户图建模：支持多种 GNN 模型（GCN / GAT / SAGE）
我们将构建用户相关图结构（如用户-部门-角色关系图）并嵌入用户行为建模中。具体要求：

每个用户节点包含上下文特征（角色、OCEAN 等）；
支持切换多种 GNN 模型（通过参数设置）：
GNN_TYPE = 'GCN'  # or 'GAT', 'SAGE'
模型结构如下：
if GNN_TYPE == 'GCN':
    self.gnn_layer = GCNConv(...)
elif GNN_TYPE == 'GAT':
    self.gnn_layer = GATConv(...)
elif GNN_TYPE == 'SAGE':
    self.gnn_layer = SAGEConv(...)
输出的 user embedding 将与 Transformer 事件序列融合（通过拼接或 Cross-Attention）
图结构样例包括：
用户 — 部门
用户 — 终端设备
用户 — URL 网站
用户 — 同一团队成员
④ 实现建议
每条事件输出一个定长向量
Transformer 模型按 Session 序列输入编码向量
用户图模块作为外部 user embedding 提供，作为 token 加入序列或用于异常评分

训练与评估模块：
支持小样本调试，包含伪标签、自监督损失等选项
精度、召回率、F1、AUC、混淆矩阵评估函数
训练过程可视化与日志分析接口
扩展支持：
兼容 Colab，支持 A100 训练和大序列输入
配置文件与训练日志保存机制
可选对比模型（LSTM、随机森林等）接口
请按模块编写 Python 代码（建议 PyTorch + HuggingFace 或自定义 Transformer 实现），保留适当注释，确保便于在 Colab 中部署实验和进行小样本+全量对比分析。