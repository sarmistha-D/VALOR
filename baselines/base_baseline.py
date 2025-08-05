"""
Base class for all baseline models
Provides common functionality for evaluation and metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm
import random

class BaseBaseline(ABC):
    """
    Abstract base class for all baseline models
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.classification_heads = None
        self.logger = logging.getLogger(f"Baseline_{model_name}")
        
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model and processor"""
        pass
    
    @abstractmethod
    def create_classification_heads(self) -> Dict[str, nn.Module]:
        """Create classification heads for aspect and severity"""
        pass
    
    @abstractmethod
    def process_input(self, text: str, image) -> Dict[str, torch.Tensor]:
        """Process text and image input"""
        pass
    
    @abstractmethod
    def extract_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from the model"""
        pass
    
    def predict(self, text: str, image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for a single sample
        
        Args:
            text: Input text
            image: Input image (PIL Image or tensor)
            
        Returns:
            aspect_logits, severity_logits
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Process input
        inputs = self.process_input(text, image)
        
        # Extract features
        with torch.no_grad():
            features = self.extract_features(inputs)
            
            # Ensure features have correct shape
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # Classification
            aspect_logits = self.classification_heads['aspect_classifier'](features)
            severity_logits = self.classification_heads['severity_classifier'](features)
        
        return aspect_logits, severity_logits

    def fine_tune(self, train_samples: List[Dict], val_samples: List[Dict], 
                 epochs: int = 5, learning_rate: float = 1e-4, 
                 weight_decay: float = 0.01, batch_size: int = 8) -> Dict[str, Any]:
        """
        Fine-tune the model on training data (simplified version)
        
        Args:
            train_samples: List of training samples
            val_samples: List of validation samples
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"\n🔄 Fine-tuning {self.model_name} model for {epochs} epochs...")
        
        # Set model to training mode for classification heads only
        self.model.eval()  # Keep backbone frozen
        for head in self.classification_heads.values():
            head.train()
        
        # Prepare optimizer - only train classification heads
        trainable_params = []
        for head in self.classification_heads.values():
            trainable_params.extend(list(head.parameters()))
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {'train_loss': [], 'val_accuracy': []}
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            # Shuffle training samples
            shuffled_samples = train_samples.copy()
            random.shuffle(shuffled_samples)
            
            # Process in batches
            for i in range(0, len(shuffled_samples), batch_size):
                batch = shuffled_samples[i:i+batch_size]
                batch_loss = 0.0
                batch_correct = 0
                
                optimizer.zero_grad()
                
                # Process each sample in the batch
                for sample in batch:
                    try:
                        # Get inputs and labels
                        text = sample['text']
                        image = sample['image']
                        aspect_label = torch.tensor([sample['aspect_label']], device=self.device)
                        severity_label = torch.tensor([sample['severity_label']], device=self.device)
                        
                        # Forward pass
                        aspect_logits, severity_logits = self.predict(text, image)
                        
                        # Calculate loss
                        loss_aspect = criterion(aspect_logits, aspect_label)
                        loss_severity = criterion(severity_logits, severity_label)
                        loss = loss_aspect + loss_severity
                        
                        # Backward pass
                        loss.backward()
                        
                        # Track metrics
                        batch_loss += loss.item()
                        
                        # Check predictions
                        aspect_pred = aspect_logits.argmax(dim=1)
                        severity_pred = severity_logits.argmax(dim=1)
                        
                        if aspect_pred.item() == aspect_label.item():
                            batch_correct += 0.5
                        if severity_pred.item() == severity_label.item():
                            batch_correct += 0.5
                        
                    except Exception as e:
                        print(f"Error processing sample: {e}")
                        continue
                
                # Update weights
                optimizer.step()
                
                # Update statistics
                epoch_loss += batch_loss
                correct_predictions += batch_correct
                total_predictions += len(batch)
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / max(1, len(shuffled_samples))
            train_accuracy = correct_predictions / max(1, total_predictions)
            
            # Validation
            val_metrics = self.evaluate_batch(
                [s['text'] for s in val_samples],
                [s['image'] for s in val_samples],
                [s['aspect_label'] for s in val_samples],
                [s['severity_label'] for s in val_samples]
            )
            val_accuracy = val_metrics['overall_accuracy']
            
            # Update history
            history['train_loss'].append(avg_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Train_Acc={train_accuracy:.4f}, Val_Acc={val_accuracy:.4f}")
        
        # Set back to eval mode
        for head in self.classification_heads.values():
            head.eval()
        
        print(f"✅ Fine-tuning completed for {self.model_name}")
        return history
    
    def evaluate_batch(self, texts: List[str], images: List, 
                      aspect_labels: List[int], severity_labels: List[int]) -> Dict[str, float]:
        """
        Evaluate a batch of samples with robust error handling
        
        Args:
            texts: List of text inputs
            images: List of image inputs
            aspect_labels: List of aspect labels
            severity_labels: List of severity labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        for head in self.classification_heads.values():
            head.eval()
        
        aspect_preds = []
        severity_preds = []
        valid_indices = []
        
        print(f"Evaluating {len(texts)} samples...")
        
        for i, (text, image, aspect_label, severity_label) in enumerate(zip(texts, images, aspect_labels, severity_labels)):
            try:
                aspect_logits, severity_logits = self.predict(text, image)
                
                # Get predictions
                aspect_pred = aspect_logits.argmax(dim=1).cpu().numpy()[0]
                severity_pred = severity_logits.argmax(dim=1).cpu().numpy()[0]
                
                aspect_preds.append(aspect_pred)
                severity_preds.append(severity_pred)
                valid_indices.append(i)
                
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                # Skip failed samples instead of adding dummy predictions
                continue
        
        if len(valid_indices) == 0:
            print("❌ No valid predictions made!")
            return {
                'aspect_accuracy': 0.0, 'aspect_precision': 0.0, 'aspect_recall': 0.0, 'aspect_f1': 0.0,
                'severity_accuracy': 0.0, 'severity_precision': 0.0, 'severity_recall': 0.0, 'severity_f1': 0.0,
                'overall_accuracy': 0.0, 'overall_f1': 0.0
            }
        
        # Filter labels to match valid predictions
        valid_aspect_labels = [aspect_labels[i] for i in valid_indices]
        valid_severity_labels = [severity_labels[i] for i in valid_indices]
        
        # Convert to numpy arrays
        aspect_labels_np = np.array(valid_aspect_labels)
        severity_labels_np = np.array(valid_severity_labels)
        aspect_preds_np = np.array(aspect_preds)
        severity_preds_np = np.array(severity_preds)
        
        print(f"✅ Successfully processed {len(valid_indices)}/{len(texts)} samples")
        print(f"Aspect predictions distribution: {np.bincount(aspect_preds_np)}")
        print(f"Severity predictions distribution: {np.bincount(severity_preds_np)}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(aspect_labels_np, aspect_preds_np, 
                                       severity_labels_np, severity_preds_np)
        
        return metrics
    
    def calculate_metrics(self, aspect_labels: np.ndarray, aspect_preds: np.ndarray,
                         severity_labels: np.ndarray, severity_preds: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics with proper error handling
        
        Args:
            aspect_labels: True aspect labels
            aspect_preds: Predicted aspect labels
            severity_labels: True severity labels
            severity_preds: Predicted severity labels
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        try:
            # Aspect classification metrics
            aspect_accuracy = accuracy_score(aspect_labels, aspect_preds)
            
            # Handle case where only one class is predicted
            if len(np.unique(aspect_preds)) == 1:
                print(f"⚠️ Aspect model predicting only class {aspect_preds[0]}")
                aspect_precision = aspect_accuracy
                aspect_recall = aspect_accuracy
                aspect_f1 = aspect_accuracy
            else:
                aspect_precision, aspect_recall, aspect_f1, _ = precision_recall_fscore_support(
                    aspect_labels, aspect_preds, average='weighted', zero_division=0
                )
            
            # Severity classification metrics
            severity_accuracy = accuracy_score(severity_labels, severity_preds)
            
            if len(np.unique(severity_preds)) == 1:
                print(f"⚠️ Severity model predicting only class {severity_preds[0]}")
                severity_precision = severity_accuracy
                severity_recall = severity_accuracy
                severity_f1 = severity_accuracy
            else:
                severity_precision, severity_recall, severity_f1, _ = precision_recall_fscore_support(
                    severity_labels, severity_preds, average='weighted', zero_division=0
                )
            
            # Overall metrics
            overall_accuracy = (aspect_accuracy + severity_accuracy) / 2
            overall_f1 = (aspect_f1 + severity_f1) / 2
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            # Return zero metrics on error
            return {
                'aspect_accuracy': 0.0, 'aspect_precision': 0.0, 'aspect_recall': 0.0, 'aspect_f1': 0.0,
                'severity_accuracy': 0.0, 'severity_precision': 0.0, 'severity_recall': 0.0, 'severity_f1': 0.0,
                'overall_accuracy': 0.0, 'overall_f1': 0.0
            }
        
        metrics.update({
            # Aspect metrics
            'aspect_accuracy': float(aspect_accuracy),
            'aspect_precision': float(aspect_precision),
            'aspect_recall': float(aspect_recall),
            'aspect_f1': float(aspect_f1),
            
            # Severity metrics
            'severity_accuracy': float(severity_accuracy),
            'severity_precision': float(severity_precision),
            'severity_recall': float(severity_recall),
            'severity_f1': float(severity_f1),
            
            # Overall metrics
            'overall_accuracy': float(overall_accuracy),
            'overall_f1': float(overall_f1),
        })
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print formatted evaluation metrics
        
        Args:
            metrics: Dictionary of metrics
        """
        print(f"\n📊 Evaluation Results for {self.model_name}")
        print("=" * 60)
        
        print("Aspect Classification:")
        print(f"  Accuracy:  {metrics['aspect_accuracy']:.4f}")
        print(f"  Precision: {metrics['aspect_precision']:.4f}")
        print(f"  Recall:    {metrics['aspect_recall']:.4f}")
        print(f"  F1-Score:  {metrics['aspect_f1']:.4f}")
        
        print("\nSeverity Classification:")
        print(f"  Accuracy:  {metrics['severity_accuracy']:.4f}")
        print(f"  Precision: {metrics['severity_precision']:.4f}")
        print(f"  Recall:    {metrics['severity_recall']:.4f}")
        print(f"  F1-Score:  {metrics['severity_f1']:.4f}")
        
        print(f"\nOverall Performance:")
        print(f"  Average Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Average F1-Score: {metrics['overall_f1']:.4f}")
        
        print("=" * 60)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        info = {
            'model_name': self.model_name,
            'device': str(self.device),
            'model_loaded': True
        }
        
        # Get model parameters count
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.classification_heads['aspect_classifier'].parameters()) + \
                              sum(p.numel() for p in self.classification_heads['severity_classifier'].parameters())
            
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            })
        except Exception as e:
            info['parameter_count_error'] = str(e)
        
        return info 