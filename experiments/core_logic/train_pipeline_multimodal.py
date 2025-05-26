import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter
from typing import Dict, Optional, Any # Added Any for training_data type hint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score #确保导入
import logging # 确保 logging 被导入

class MultiModalTrainer:
    # ... (init 和其他方法)

    def evaluate(self, data_loader: DataLoader, model: nn.Module, device: torch.device, phase: str = "test") -> Dict[str, float]:
        """
        在给定的数据加载器上评估模型
        Args:
            data_loader: 数据加载器 (验证或测试)
            model: 要评估的模型
            device: 设备 (cpu/cuda)
            phase: 评估阶段 ("val" 或 "test")
        Returns:
            包含评估指标的字典
        """
        logger = logging.getLogger(__name__) # 获取日志记录器
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = [] # 用于存储原始概率输出

        with torch.no_grad():
            for batch_data in data_loader:
                # ... (数据移动到设备和模型前向传播的逻辑)
                # 假设 batch_data 是一个字典，包含 'labels' 和其他模态数据
                # 例如: labels = batch_data['labels'].to(device)
                #       outputs, _, _ = model(batch_data) # 假设模型返回 (logits, preds, probs) 或类似结构
                
                # 示例：假设模型输出是 logits
                # ---- 请根据您的实际模型输出结构调整以下部分 ----
                labels = batch_data['labels'].to(device)
                
                # 从 batch_data 中提取模型需要的所有输入
                model_inputs = {k: v.to(device) for k, v in batch_data.items() if k != 'users'}
                # 注意: model 的 forward 方法需要能接受这样的字典

                # 确保模型返回的是可以直接用于计算损失和概率的原始输出 (logits)
                # 以及预测的类别 (preds) 和用于AUC的概率 (probs_for_auc)
                # 例如: raw_outputs, _ = model(model_inputs) # 假设模型直接返回logits或包含logits的元组
                try:
                    # 假设模型返回一个包含原始输出（logits）的元组或直接返回logits
                    # model_output = model(model_inputs)
                    # raw_outputs = model_output[0] if isinstance(model_output, tuple) else model_output 
                    # --- 根据您模型的实际输出进行调整 ---
                    # 假设 model(**model_inputs) 返回 (final_output, predicted_labels, attention_weights)
                    # 其中 final_output 是 logits
                    final_output, _, _ = model(**model_inputs)
                    raw_outputs = final_output

                except Exception as e:
                    logger.error(f"在 {phase} 阶段模型前向传播时出错: {e}")
                    logger.error(f"Batch data keys: {batch_data.keys()}")
                    # 可以选择跳过这个batch或抛出异常
                    continue

                # 从原始输出计算概率和预测类别
                if raw_outputs.ndim > 1 and raw_outputs.shape[1] > 1: # 分类任务的logits
                    probs = torch.softmax(raw_outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    probs_for_auc = probs[:, 1] # 取正类的概率用于AUC
                else: # 可能的回归或异常得分任务 (需要调整)
                    # logger.warning(f"模型原始输出格式未知: {raw_outputs.shape}，将尝试sigmoid")
                    probs = torch.sigmoid(raw_outputs).squeeze() # 如果是单个输出代表正类概率
                    preds = (probs > 0.5).long()
                    probs_for_auc = probs
                # ---- 上述概率和预测的转换逻辑需要根据您模型的具体输出来确定 ----

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs_for_auc.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_score = np.array(all_probs) # 用于AUC的概率分数

        # --- Sanity Checks --- 
        logger.info(f"[{phase.upper()}] 评估: y_true 类别分布: {Counter(y_true)}")
        logger.info(f"[{phase.upper()}] 评估: y_pred 类别分布: {Counter(y_pred)}")
        if len(np.unique(y_score)) < 5: # 如果概率值的独特性很低
            logger.info(f"[{phase.upper()}] 评估: y_score (概率) 的独特值 (前5个): {np.unique(y_score)[:5]}")
        else:
            logger.info(f"[{phase.upper()}] 评估: y_score (概率) 的独特值数量: {len(np.unique(y_score))}")

        metrics = {}
        unique_classes_true = np.unique(y_true)

        if len(unique_classes_true) < 2:
            logger.warning(f"警告 [{phase.upper()}]: y_true 中只存在一个类别 ({unique_classes_true}). AUC 将无法计算或无意义。F1等指标也可能不准确。")
            metrics['accuracy'] = np.mean(y_true == y_pred) if len(y_true) > 0 else 0.0
            metrics['f1'] = 0.0
            metrics['auc'] = np.nan
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['warning'] = f'{phase}_y_true_one_class'
            return metrics

        if len(np.unique(y_pred)) < 2:
            logger.warning(f"警告 [{phase.upper()}]: 模型预测结果 y_pred 中只存在一个类别 ({np.unique(y_pred)}). 模型可能存在偏见或数据问题。")

        metrics['accuracy'] = np.mean(y_true == y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        # AUC 计算
        if len(np.unique(y_score)) == 1:
            logger.warning(f"警告 [{phase.upper()}]: 所有样本的预测概率 y_score 都相同 ({y_score[0] if len(y_score)>0 else 'N/A'}). AUC 将为 nan。")
            metrics['auc'] = np.nan
        else:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_score)
            except ValueError as e:
                logger.error(f"错误 [{phase.upper()}]: 计算 AUC 时发生 ValueError: {e}. 这通常是因为 y_true 中只有一个类别 (再次检查)。")
                metrics['auc'] = np.nan
            except Exception as e:
                logger.error(f"错误 [{phase.upper()}]: 计算 AUC 时发生未知错误: {e}")
                metrics['auc'] = np.nan

        if metrics.get('f1') == 1.0 and (np.isnan(metrics.get('auc')) or metrics.get('auc', 0) < 0.6):
            logger.warning(f"警告 [{phase.upper()}]: F1 分数为 1.0，但 AUC ({metrics.get('auc')}) 较低或无法计算。请检查。")

        logger.info(f"[{phase.upper()}] 指标: {metrics}")
        return metrics

    def train(self, training_data: Dict[str, Any]) -> Optional[nn.Module]:
        # ... (训练开始部分，数据加载器准备等)
        # 在训练循环结束后，通常会有一个最终的测试集评估
        # 确保在这里调用 self.evaluate(self.test_loader, self.model, self.device, phase="test")
        # 并将结果记录到 self.train_history 或类似的字典中
        logger = logging.getLogger(__name__) # 获取日志记录器
        # 例如，在训练循环之后：
        if self.test_loader:
            logger.info("\n============================================================")
            logger.info("在测试集上评估最佳模型")
            logger.info("============================================================")
            # 加载最佳模型状态 (如果在训练中保存了的话)
            # if os.path.exists(self.best_model_path):
            #     self.model.load_state_dict(torch.load(self.best_model_path))
            # else:
            #     logger.warning(f"未找到最佳模型路径: {self.best_model_path}。将使用当前模型状态进行测试。")
            
            test_metrics = self.evaluate(self.test_loader, self.model, self.device, phase="test")
            self.train_history['test_accuracy'] = test_metrics.get('accuracy', 0.0)
            self.train_history['test_precision'] = test_metrics.get('precision', 0.0)
            self.train_history['test_recall'] = test_metrics.get('recall', 0.0)
            self.train_history['test_f1'] = test_metrics.get('f1', 0.0)
            self.train_history['test_auc'] = test_metrics.get('auc', np.nan)

            logger.info(f"测试结果:")
            for key, value in test_metrics.items():
                logger.info(f"  {key.capitalize()}: {value:.4f}" if isinstance(value, float) else f"  {key.capitalize()}: {value}")

            # ... (绘制和保存图表等)
        # ... (方法结束)
        return self.model # 或者根据您的逻辑返回

# ... (文件其余部分) 