#!/usr/bin/env python3
"""
VALOR Training Script is here
Unified training, hyperparameter tuning, evaluation, and inference for VALOR model
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import os
import json
import sys
import gc
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.valor import VALOR
from utils.dataset import load_comp2_dataset, create_dataloaders
from utils.metrics import compute_metrics
from utils.hyperparameter_tuner import AdvancedHyperparameterTuner
from utils.data_augmentation import (
    mixup_data, mixup_criterion, cutmix_data, 
    advanced_mixup_data, advanced_cutmix_data, 
    advanced_random_erasing, advanced_color_jitter
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
import torchvision.transforms as T

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("valor_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VALORTrainer:
    """Complete VALOR trainer with hyperparameter tuning"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.best_val_accuracy = 0.0
        self.best_params = {}
        
        # Create directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Check resources
        self._check_resources()
    
    def _check_resources(self):
        """Check GPU memory and disk space before training"""
        import shutil
        # Check GPU memory
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[Resource Check] GPU total memory: {total_mem:.2f} GB")
        # Check disk space
        path = getattr(self.config, 'model_dir', '.')
        if not os.path.exists(path):
            path = '.'  # fallback to current directory
        total, used, free = shutil.disk_usage(path)
        print(f"[Resource Check] Disk space: {free // (1024 ** 3)} GB free")
    
    def _load_data(self):
        """Load dataset and create dataloaders"""
        logger.info("📊 Loading dataset...")
        
        # Setup tokenizer and transforms
        tokenizer = AutoTokenizer.from_pretrained(self.config.text_model)
        image_transforms = T.Compose([
            T.Resize((self.config.image_size, self.config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.config.image_mean, std=self.config.image_std)
        ])
        
        # Load dataset
        train_dataset, val_dataset, test_dataset = load_comp2_dataset(
            self.config, tokenizer, image_transforms
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, self.config
        )
        
        logger.info(f"✅ Data loaded successfully")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def _create_model(self, params: Dict[str, Any]):
        """Create VALOR model with given parameters"""
        # Update config with parameters
        for key, value in params.items():
            setattr(self.config, key, value)
        
        # Create model
        model = VALOR(
            hidden_dim=self.config.valor_hidden_dim,
            num_aspect_classes=len(self.config.aspect_classes),
            num_severity_classes=len(self.config.severity_classes),
            num_experts=self.config.valor_num_experts,
            freeze_encoders=self.config.valor_freeze_encoders
        )
        
        return model.to(self.device)
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}
        
        # Classification losses
        aspect_loss = F.cross_entropy(outputs['aspect_logits'], targets['aspect_labels'])
        severity_loss = F.cross_entropy(outputs['severity_logits'], targets['severity_labels'])
        
        losses['aspect_loss'] = aspect_loss
        losses['severity_loss'] = severity_loss
        
        # Load balancing loss
        if 'load_balance_loss' in outputs:
            losses['load_balance_loss'] = outputs['load_balance_loss']
        
        # SAS loss
        if 'sas_loss' in outputs:
            losses['sas_loss'] = outputs['sas_loss']
        
        # Analysis loss
        if 'analysis_loss' in outputs:
            losses['analysis_loss'] = outputs['analysis_loss']
        
        # Total loss
        total_loss = aspect_loss + severity_loss
        
        if 'load_balance_loss' in losses:
            total_loss += self.config.load_balance_weight * losses['load_balance_loss']
        
        if 'sas_loss' in losses:
            total_loss += self.config.lambda_sas * losses['sas_loss']
        
        if 'analysis_loss' in losses:
            total_loss += self.config.lambda_analysis * losses['analysis_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _hyperparameter_tuning(self, train_loader, val_loader) -> Dict[str, Any]:
        """Advanced hyperparameter tuning with Optuna"""
        logger.info("🔍 Starting advanced hyperparameter tuning with Optuna...")
        
        # Create model creator function
        def model_creator(params):
            return self._create_model(params)
        
        # Create data loaders tuple
        data_loaders = (train_loader, val_loader, None)
        
        # Initialize advanced tuner
        tuner = AdvancedHyperparameterTuner(
            config=self.config,
            model_creator=model_creator,
            data_loaders=data_loaders,
            trainer_class=self.__class__
        )
        
        # Run optimization
        best_params, tuning_results = tuner.optimize()
        
        # Generate plots and analysis
        tuner.plot_optimization_history()
        
        # Log parameter importance
        importance = tuner.get_parameter_importance()
        logger.info("📊 Parameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {param}: {imp:.4f}")
        
        return best_params
    
    def train(self, tune_hyperparams=False):
        """Main training function"""
        logger.info("🚀 Starting VALOR Training")
        logger.info("=" * 50)
        
        try:
            # Load data
            train_loader, val_loader, test_loader = self._load_data()
            
            # Hyperparameter tuning if requested
            if tune_hyperparams:
                logger.info("🔍 Running hyperparameter tuning...")
                best_params = self._hyperparameter_tuning(train_loader, val_loader)
                
                # Update config with best parameters
                for key, value in best_params.items():
                    setattr(self.config, key, value)
                
                logger.info(f"✅ Best parameters found: {best_params}")
            
            # Create model
            model = self._create_model({})
            
            # Setup optimizer and scheduler
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
            
            # Learning rate scheduler
            total_steps = len(train_loader) * self.config.num_epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
            
            # Training loop
            best_val_accuracy = 0.0
            patience_counter = 0
            
            for epoch in range(self.config.num_epochs):
                logger.info(f"📚 Epoch {epoch+1}/{self.config.num_epochs}")
                
                # Training
                train_metrics = self._train_epoch(model, train_loader, optimizer, scheduler, {})
                
                # Validation
                val_metrics = self._validate_epoch(model, val_loader)
                
                # Log metrics
                logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, Aspect Acc: {train_metrics['aspect_accuracy']:.4f}, Severity Acc: {train_metrics['severity_accuracy']:.4f}")
                logger.info(f"Val - Loss: {val_metrics['total_loss']:.4f}, Aspect Acc: {val_metrics['aspect_accuracy']:.4f}, Severity Acc: {val_metrics['severity_accuracy']:.4f}")
                
                # Save best model
                if val_metrics['aspect_accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['aspect_accuracy']
                    self._save_best_model(model, optimizer, val_metrics, {})
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break
            
            # Final evaluation on test set
            logger.info("🧪 Final evaluation on test set...")
            test_metrics = self._validate_epoch(model, test_loader)
            
            logger.info("📊 Final Results:")
            logger.info(f"Aspect Accuracy: {test_metrics['aspect_accuracy']:.4f}")
            logger.info(f"Severity Accuracy: {test_metrics['severity_accuracy']:.4f}")
            logger.info(f"Macro F1 (Aspect): {test_metrics['macro_f1_aspect']:.4f}")
            logger.info(f"Macro F1 (Severity): {test_metrics['macro_f1_severity']:.4f}")
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            raise
    
    def _train_epoch(self, model: nn.Module, train_loader, optimizer: optim.Optimizer, 
                    scheduler: Optional[optim.lr_scheduler._LRScheduler], 
                    params: Dict[str, Any]) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        aspect_correct = 0
        severity_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            input_ids = batch['text_inputs']['input_ids'].to(self.device)
            attention_mask = batch['text_inputs']['attention_mask'].to(self.device)
            images = batch['images'].to(self.device)
            aspect_labels = batch['aspect_labels'].to(self.device)
            severity_labels = batch['severity_labels'].to(self.device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, images)
            
            # Compute loss
            targets = {
                'aspect_labels': aspect_labels,
                'severity_labels': severity_labels
            }
            losses = self._compute_loss(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Update metrics
            total_loss += losses['total_loss'].item()
            
            # Calculate accuracy
            aspect_preds = outputs['aspect_logits'].argmax(dim=1)
            severity_preds = outputs['severity_logits'].argmax(dim=1)
            
            aspect_correct += (aspect_preds == aspect_labels).sum().item()
            severity_correct += (severity_preds == severity_labels).sum().item()
            total_samples += aspect_labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'aspect_acc': f"{aspect_correct/total_samples:.4f}",
                'severity_acc': f"{severity_correct/total_samples:.4f}"
            })
        
        return {
            'total_loss': total_loss / len(train_loader),
            'aspect_accuracy': aspect_correct / total_samples,
            'severity_accuracy': severity_correct / total_samples
        }
    
    def _validate_epoch(self, model: nn.Module, val_loader) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        all_aspect_preds = []
        all_aspect_labels = []
        all_severity_preds = []
        all_severity_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                input_ids = batch['text_inputs']['input_ids'].to(self.device)
                attention_mask = batch['text_inputs']['attention_mask'].to(self.device)
                images = batch['images'].to(self.device)
                aspect_labels = batch['aspect_labels'].to(self.device)
                severity_labels = batch['severity_labels'].to(self.device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask, images)
                
                # Compute loss
                targets = {
                    'aspect_labels': aspect_labels,
                    'severity_labels': severity_labels
                }
                losses = self._compute_loss(outputs, targets)
                
                total_loss += losses['total_loss'].item()
                
                # Collect predictions
                aspect_preds = outputs['aspect_logits'].argmax(dim=1)
                severity_preds = outputs['severity_logits'].argmax(dim=1)
                
                all_aspect_preds.extend(aspect_preds.cpu().numpy())
                all_aspect_labels.extend(aspect_labels.cpu().numpy())
                all_severity_preds.extend(severity_preds.cpu().numpy())
                all_severity_labels.extend(severity_labels.cpu().numpy())
        
        # Compute metrics
        aspect_metrics = compute_metrics(all_aspect_preds, all_aspect_labels)
        severity_metrics = compute_metrics(all_severity_preds, all_severity_labels)
        
        return {
            'total_loss': total_loss / len(val_loader),
            'aspect_accuracy': aspect_metrics['accuracy'],
            'severity_accuracy': severity_metrics['accuracy'],
            'macro_f1_aspect': aspect_metrics['macro_f1'],
            'macro_f1_severity': severity_metrics['macro_f1']
        }
    
    def _save_best_model(self, model: nn.Module, optimizer: optim.Optimizer, 
                        metrics: Dict[str, Any], params: Dict[str, Any]):
        """Save the best model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'params': params,
            'config': self.config,
            'timestamp': timestamp
        }
        
        save_path = f"checkpoints/valor_best_{timestamp}.pt"
        torch.save(checkpoint, save_path)
        logger.info(f"💾 Best model saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Model loaded from {checkpoint_path}")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on test set"""
        logger.info("🧪 Evaluating model on test set...")
        
        # Load data
        _, _, test_loader = self._load_data()
        
        # Create model
        model = self._create_model({})
        
        # Evaluate
        metrics = self._validate_epoch(model, test_loader)
        
        return metrics
    
    def predict_single(self, text: str, image_path: str) -> Dict[str, Any]:
        """Run inference on a single example"""
        logger.info(f"🔮 Running inference on: {text[:50]}...")
        
        # Create model
        model = self._create_model({})
        
        # Load and preprocess image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Tokenize text
        tokenizer = AutoTokenizer.from_pretrained(self.config.text_model)
        text_inputs = tokenizer(
            text,
            max_length=self.config.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Preprocess image
        image_transforms = T.Compose([
            T.Resize((self.config.image_size, self.config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.config.image_mean, std=self.config.image_std)
        ])
        image_tensor = image_transforms(image).unsqueeze(0)
        
        # Move to device
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
            
            # Get predictions
            aspect_probs = F.softmax(outputs['aspect_logits'], dim=1)
            severity_probs = F.softmax(outputs['severity_logits'], dim=1)
            
            aspect_pred = aspect_probs.argmax(dim=1).item()
            severity_pred = severity_probs.argmax(dim=1).item()
            
            aspect_confidence = aspect_probs.max(dim=1)[0].item()
            severity_confidence = severity_probs.max(dim=1)[0].item()
        
        return {
            'aspect': self.config.aspect_classes[aspect_pred],
            'severity': self.config.severity_classes[severity_pred],
            'aspect_confidence': aspect_confidence,
            'severity_confidence': severity_confidence
        }

def train_model(args):
    """Train the VALOR model"""
    logger.info("Starting VALOR training...")
    
    # Update config from command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Initialize trainer
    trainer = VALORTrainer(config)
    
    # Start training
    logger.info(f"Training with batch size: {config.batch_size}, epochs: {config.num_epochs}, lr: {config.learning_rate}")
    results = trainer.train(tune_hyperparams=args.tune)
    
    # Log final results
    logger.info(f"Training completed. Final results:")
    logger.info(f"Aspect accuracy: {results['aspect_accuracy']:.4f}")
    logger.info(f"Severity accuracy: {results['severity_accuracy']:.4f}")
    logger.info(f"Macro F1 (aspect): {results['macro_f1_aspect']:.4f}")
    logger.info(f"Macro F1 (severity): {results['macro_f1_severity']:.4f}")
    
    return results

def evaluate_model(args):
    """Evaluate the VALOR model"""
    logger.info(f"Evaluating VALOR model from checkpoint: {args.checkpoint}")
    
    # Initialize trainer
    trainer = VALORTrainer(config)
    
    # Load checkpoint
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    else:
        logger.error("No checkpoint provided for evaluation")
        return
    
    # Run evaluation
    results = trainer.evaluate()
    
    # Log evaluation results
    logger.info(f"Evaluation results:")
    logger.info(f"Aspect accuracy: {results['aspect_accuracy']:.4f}")
    logger.info(f"Severity accuracy: {results['severity_accuracy']:.4f}")
    logger.info(f"Macro F1 (aspect): {results['macro_f1_aspect']:.4f}")
    logger.info(f"Macro F1 (severity): {results['macro_f1_severity']:.4f}")
    
    return results

def predict_single(args):
    """Run inference on a specific example"""
    logger.info(f"Running inference with VALOR model")
    
    # Initialize trainer
    trainer = VALORTrainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Get prediction
    prediction = trainer.predict_single(
        text=args.text,
        image_path=args.image
    )
    
    # Log prediction
    logger.info(f"Prediction results:")
    logger.info(f"Aspect: {prediction['aspect']} (confidence: {prediction['aspect_confidence']:.4f})")
    logger.info(f"Severity: {prediction['severity']} (confidence: {prediction['severity_confidence']:.4f})")
    
    return prediction

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VALOR Training and Evaluation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--batch-size', type=int, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning before training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run inference on single example')
    predict_parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    predict_parser.add_argument('--text', type=str, required=True, help='Input text')
    predict_parser.add_argument('--image', type=str, required=True, help='Path to input image')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'predict':
        predict_single(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 