#!/usr/bin/env python3
"""Model Trainer for Multi-modal Anomaly Detection"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm # For progress bars
import time
import os
import sys
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score
from typing import Optional, Dict, List, Tuple, Union

# Add model path - assuming trainer.py is in train_pipeline, and models are in ../models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# This might need adjustment based on your exact execution context and project structure.
# If you run from the project root, you might not need sys.path.append here if using relative imports like:
# from ..models.your_model import YourModel

# Placeholder for the actual model import
# from ..train_pipeline.multimodal_model import MultiModalAnomalyDetector 
# We will import it directly for now, assuming execution from a place where it's discoverable
# or that the sys.path.append works as intended.

class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module, # Loss function
                 device: torch.device,
                 epochs: int,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 save_dir: str = 'checkpoints',
                 model_name: str = 'multimodal_anomaly_detector'
                 ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.save_dir = save_dir
        self.model_name = model_name
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_val_pr_auc = float('-inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_pr_auc': [],
            'val_roc_auc': []
        }

    def _prepare_batch(self, batch_data: Dict[str, Union[torch.Tensor, List]]) -> Dict[str, Union[torch.Tensor, List]]:
        """Prepare a batch of data by moving tensors to the correct device."""
        prepared_batch = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            elif isinstance(value, list) and all(isinstance(item, str) for item in value): # List of texts
                prepared_batch[key] = value # Texts remain on CPU, handled by tokenizer inside model
            else:
                prepared_batch[key] = value 
        return prepared_batch

    def train_epoch(self, epoch_num: int) -> float:
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num+1}/{self.epochs} [Train]", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Assuming batch is a dictionary of inputs for the model
            # And a 'labels' key for the ground truth
            
            inputs = self._prepare_batch(batch['inputs'])
            labels = batch['labels'].to(self.device) # Assuming labels are tensors
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs) # model's forward should return a dict with 'logits'
            logits = outputs['logits'] 
            
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(self.train_loader)
        self.history['train_loss'].append(avg_loss)
        print(f"Epoch {epoch_num+1} Train Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, epoch_num: int, loader: DataLoader, phase: str = "Val") -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            epoch_num: Current epoch number (for logging).
            loader: DataLoader for the evaluation dataset.
            phase: Phase name (e.g., "Val", "Test").

        Returns:
            A dictionary containing evaluation metrics.
        """
        self.model.eval()
        total_loss = 0
        
        all_labels = []
        all_probabilities = []
        all_anomaly_scores = [] # Assuming model outputs this
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{self.epochs} [{phase}]", leave=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                inputs = self._prepare_batch(batch['inputs'])
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(inputs)
                logits = outputs['logits']
                
                # Get probabilities and anomaly scores from model outputs
                probabilities = outputs['probabilities'] # Assuming model returns probabilities
                anomaly_scores = outputs['anomaly_scores'] # Assuming model returns anomaly_scores

                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Collect labels, probabilities, and anomaly scores
                all_labels.append(labels.cpu())
                all_probabilities.append(probabilities.cpu())
                all_anomaly_scores.append(anomaly_scores.cpu())

                # You can still calculate batch accuracy if you want, but main metrics are collected
                # probabilities = torch.softmax(logits, dim=1)
                # predictions = torch.argmax(probabilities, dim=1)
                # correct_predictions += (predictions == labels).sum().item()
                # total_samples += labels.size(0)

                progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(loader)
        
        # Concatenate collected tensors
        all_labels = torch.cat(all_labels).numpy()
        all_probabilities = torch.cat(all_probabilities).numpy()
        all_anomaly_scores = torch.cat(all_anomaly_scores).numpy()
        
        # Calculate metrics
        # For binary classification metrics, we need probabilities/scores for the positive class (anomaly)
        # Assuming class 1 is anomaly
        anomaly_probabilities = all_probabilities[:, 1] if all_probabilities.shape[1] > 1 else all_probabilities.squeeze()

        # Calculate predictions for F1 score (requires a threshold)
        # For F1, we typically pick a threshold (e.g., 0.5 for binary classification)
        # In practice, you might want to find an optimal threshold based on the validation set
        # For simplicity here, we use 0.5
        predicted_labels = (anomaly_probabilities > 0.5).astype(int)

        # Calculate F1 Score
        f1 = f1_score(all_labels, predicted_labels)

        # Calculate Precision-Recall AUC
        # precision_recall_curve returns precision, recall, thresholds
        precision, recall, _ = precision_recall_curve(all_labels, anomaly_probabilities)
        pr_auc = auc(recall, precision)

        # Calculate ROC AUC
        try:
             roc_auc = roc_auc_score(all_labels, anomaly_probabilities)
        except ValueError:
             # This can happen if only one class is present in the batch/loader
             roc_auc = float('nan') # Or some other indicator

        accuracy = accuracy_score(all_labels, predicted_labels)
        precision = precision_score(all_labels, predicted_labels, zero_division=0)
        recall = recall_score(all_labels, predicted_labels, zero_division=0)

        metrics = {
            f'{phase}_loss': avg_loss,
            f'{phase}_accuracy': accuracy,
            f'{phase}_precision': precision,
            f'{phase}_recall': recall,
            f'{phase}_f1': f1,
            f'{phase}_pr_auc': pr_auc,
            f'{phase}_roc_auc': roc_auc,
        }

        # Print metrics
        print(f"Epoch {epoch_num+1} {phase} Metrics: ", end="")
        for name, value in metrics.items():
             print(f"{name}: {value:.4f} ", end="")
        print() # Newline

        if phase == "Val":
            for metric_key_suffix, metric_value in metrics.items():
                history_key = metric_key_suffix.replace(f"{phase}_", "val_", 1)
                if history_key in self.history:
                    self.history[history_key].append(metric_value)
                else:
                    print(f"Warning: Metric key {history_key} for history not pre-initialized. Adding it.")
                    self.history[history_key] = [metric_value]
            
        return metrics

    def train(self):
        print(f"Starting training for {self.epochs} epochs on device: {self.device}")
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(epoch, self.val_loader, phase="Val")
            val_loss = val_metrics[f'Val_loss']
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s")
            
            if self.lr_scheduler:
                # Special handling for ReduceLROnPlateau which needs a metric
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            
            # Save best model
            current_val_pr_auc = val_metrics[f'Val_pr_auc']
            if current_val_pr_auc > self.best_val_pr_auc:
                self.best_val_pr_auc = current_val_pr_auc
                save_path = os.path.join(self.save_dir, f"{self.model_name}_best_pr_auc.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Best model (based on PR_AUC) saved to {save_path} (Val PR_AUC: {self.best_val_pr_auc:.4f})")

            # Save checkpoint periodically (e.g., every 5 epochs)
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'best_val_pr_auc': self.best_val_pr_auc,
                    'history': self.history
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        print("Training finished.")
        # Optionally, load best model and run on a test set if available
        # Example loading best PR_AUC model and evaluating on test set:
        # best_pr_auc_model_path = os.path.join(self.save_dir, f"{self.model_name}_best_pr_auc.pth")
        # if os.path.exists(best_pr_auc_model_path) and self.test_loader:
        #     print("\nEvaluating best PR_AUC model on Test set...")
        #     self.model.load_state_dict(torch.load(best_pr_auc_model_path, map_location=self.device))
        #     self.evaluate(self.epochs - 1, self.test_loader, phase="Test")
            
        return self.history

# Example Usage (Illustrative - requires actual data, model, etc.)
if __name__ == '__main__':
    # This is a very basic example and needs to be adapted with actual components.
    
    # 0. Import necessary components (model, dataset)
    from train_pipeline_multimodal.multimodal_model import MultiModalAnomalyDetector # Adjusted import

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dummy Dataset and DataLoader
    class DummyDataset(Dataset):
        def __init__(self, num_samples=100, seq_len=32, beh_dim=128, node_feat_dim=10, num_nodes=50, struct_dim=20, is_train=True):
            self.num_samples = num_samples
            self.is_train = is_train
            # Behavior sequences: [batch_size, seq_len, feature_dim]
            self.behavior_sequences = torch.randn(num_samples, seq_len, beh_dim)
            # Node features: [num_nodes, node_feat_dim]
            self.node_features = torch.randn(num_nodes, node_feat_dim)
            # Adjacency matrix: [num_nodes, num_nodes]
            self.adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
            # Text content: List[str]
            self.text_content = [f"Sample text for item {i}" for i in range(num_samples)]
            # Structured features: [batch_size, feature_dim]
            self.structured_features = torch.randn(num_samples, struct_dim)
            # Labels
            self.labels = torch.randint(0, 2, (num_samples,)) # Binary classification (0: normal, 1: anomaly)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            inputs = {
                'behavior_sequences': self.behavior_sequences[idx],
                'node_features': self.node_features, # GNN usually takes all nodes
                'adjacency_matrix': self.adjacency_matrix, # And full adj matrix
                'text_content': self.text_content[idx], # For BERT, usually one text per sample
                'structured_features': self.structured_features[idx]
            }
            # In a real scenario, text_content might be a list for a batch, 
            # but BERTTextEncoder expects a list of strings, so if batching,
            # the collate_fn would handle preparing a list of texts for the batch.
            # For this dummy dataset, BERTTextEncoder will get a list containing one string.

            return {'inputs': inputs, 'labels': self.labels[idx]}

    # Custom collate_fn if text_content needs special handling for batching
    def custom_collate_fn(batch):
        # batch is a list of {'inputs': inputs_dict, 'labels': label_tensor}
        
        # Separate inputs and labels
        input_dicts = [item['inputs'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        
        # Batch individual input modalities
        batched_inputs = {}
        # Handle behavior_sequences (stack them)
        batched_inputs['behavior_sequences'] = torch.stack([d['behavior_sequences'] for d in input_dicts])
        
        # For GNN, node_features and adjacency_matrix are typically graph-level and don't change per item in batch
        # Or they are batched using special GNN batching libraries. Here, we assume they are passed as is.
        # For simplicity in this dummy example, we'll just take the first one.
        # In a real GNN setup, you'd use PyTorch Geometric or DGL for batching graphs.
        batched_inputs['node_features'] = input_dicts[0]['node_features'] 
        batched_inputs['adjacency_matrix'] = input_dicts[0]['adjacency_matrix']
        
        # Handle text_content (collect into a list of strings)
        batched_inputs['text_content'] = [d['text_content'] for d in input_dicts]
        
        # Handle structured_features (stack them)
        batched_inputs['structured_features'] = torch.stack([d['structured_features'] for d in input_dicts])
        
        return {'inputs': batched_inputs, 'labels': labels}

    train_dataset = DummyDataset(num_samples=200)
    val_dataset = DummyDataset(num_samples=50)
    
    # Use custom_collate_fn for DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    # 3. Model
    # Configuration for the model - ensure these match your previously tested configs
    # The output_dim of BERT in MultiModalAnomalyDetector's bert_config should match
    # the output_dim you tested BERTTextEncoder with (e.g., 128 in its own test).
    model = MultiModalAnomalyDetector(
        embed_dim=128, # General embedding dim for fusion output and head input
        transformer_config={'input_dim': 128, 'hidden_dim': 128}, # behavior_sequences feature_dim must be 128
        gnn_config={'input_dim': 10, 'output_dim': 128},        # node_feat_dim for GNN
        bert_config={'output_dim': 128}, # BERTTextEncoder output_dim
        lgbm_config={'input_dim': 20, 'output_dim': 128}, # structured_features dim
        fusion_config={'embed_dim': 128}, # Input to fusion from encoders
        head_config={'input_dim': 128, 'num_classes': 2} # Input from fusion, output 2 classes
    ).to(device)

    # 4. Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss() # Suitable for multi-class (here, binary) classification

    # 4.5. Setup Learning Rate Scheduler (Optional)
    # 当验证损失在 patience 个 epoch 内没有下降时，降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # 监控指标是损失，希望它最小化
        factor=0.1,      # 降低学习率的因子 (新学习率 = old_lr * factor)
        patience=5,      # 当监控指标连续 patience 个 epoch 没有改善时触发
        verbose=True     # 打印学习率调整信息
    )

    # 5. Trainer
    epochs = 5 # Adjust epochs as needed
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        lr_scheduler=scheduler, # 将调度器实例传递进去
        save_dir='checkpoints',
        model_name='multimodal_anomaly_detector'
    )

    print("Starting dummy training example...")
    try:
        trainer.train()
        print("Dummy training example finished.")
    except Exception as e:
        print(f"Error during dummy training: {e}")
        import traceback
        traceback.print_exc()

    # Further steps:
    # - Implement actual Dataset class for your data.
    # - Implement data preprocessing and feature engineering.
    # - Fine-tune hyperparameters.
    # - Add more detailed logging and visualization (e.g., TensorBoard). 