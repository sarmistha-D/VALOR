#!/usr/bin/env python3
"""
Run a single baseline model
"""

import os
import sys
import argparse
import torch
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
try:
    from .run_baselines import BaselineRunner
except ImportError:
    from run_baselines import BaselineRunner

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run a single baseline model")
    parser.add_argument(
        "model",
        type=str,
        choices=[
            "CLIP", "DeepSeek-VL", "ViLT", "VisualBERT", "ALBEF", 
            "GIT", "FLAVA", "ImageBind", "UNITER", "Flash-Gemini", 
            "Gemma3", "Paligemma", "SMOL-VLM"
        ],
        help="Model to run"
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune the model before evaluation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for fine-tuning"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for fine-tuning"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for fine-tuning"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="baseline_checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="Load existing checkpoint instead of fine-tuning"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to specific checkpoint file to load"
    )
    return parser.parse_args()

def find_best_checkpoint(checkpoint_dir: str, model_name: str) -> str:
    """Find the best checkpoint for a model"""
    import glob
    
    # Find all checkpoints for this model
    pattern = os.path.join(checkpoint_dir, f"{model_name}_*.pt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Extract F1 scores from filenames
    best_checkpoint = None
    best_f1 = -1
    
    for checkpoint in checkpoints:
        try:
            # Extract F1 score from filename (format: model_epochX_f1_Y.YYYY.pt)
            f1_str = checkpoint.split('_f1_')[1].split('.pt')[0]
            f1 = float(f1_str)
            
            if f1 > best_f1:
                best_f1 = f1
                best_checkpoint = checkpoint
        except:
            continue
    
    return best_checkpoint

def run_single_baseline(model_name: str, args):
    """Run a single baseline model"""
    # Initialize config
    config = Config()
    config.fine_tune_baselines = args.fine_tune
    config.fine_tune_epochs = args.epochs
    config.fine_tune_lr = args.lr
    config.fine_tune_batch_size = args.batch_size
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize runner
    runner = BaselineRunner(config)
    
    # Get model class
    if model_name not in runner.baseline_models:
        print(f"❌ Model {model_name} not found")
        return
    
    model_class = runner.baseline_models[model_name]
    
    # Load dataset
    samples = runner.load_dataset()
    
    # Split dataset
    train_samples, val_samples, test_samples = runner.split_dataset(samples)
    
    # Initialize model
    model = model_class(device=str(runner.device))
    
    # Load model
    if not model.load_model():
        print(f"❌ Failed to load {model_name} model")
        return
    
    # Check if we should load checkpoint
    checkpoint_path = None
    if args.load_checkpoint:
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_path = find_best_checkpoint(args.checkpoint_dir, model_name)
            
        if checkpoint_path:
            print(f"📥 Loading checkpoint for {model_name} from {checkpoint_path}")
            if not model.load_checkpoint(checkpoint_path):
                print(f"⚠️ Failed to load checkpoint for {model_name}. Will fine-tune instead.")
                checkpoint_path = None
        else:
            print(f"⚠️ No checkpoint found for {model_name}")
    
    # Fine-tune if needed
    if args.fine_tune and not (args.load_checkpoint and checkpoint_path):
        try:
            print(f"🔄 Fine-tuning {model_name} model...")
            history = model.fine_tune(
                train_samples=train_samples,
                val_samples=val_samples,
                epochs=args.epochs,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir
            )
            print(f"✅ Fine-tuning completed for {model_name}")
        except Exception as e:
            print(f"⚠️ Fine-tuning failed for {model_name}: {e}")
            print("⚠️ Proceeding with evaluation using pre-trained model...")
    
    # Evaluate on test set
    print(f"📊 Evaluating {model_name} on test set...")
    metrics = model.evaluate_batch(
        [s['text'] for s in test_samples],
        [s['image'] for s in test_samples],
        [s['aspect_label'] for s in test_samples],
        [s['severity_label'] for s in test_samples]
    )
    
    # Print metrics
    model.print_metrics(metrics)
    
    # Get model info
    model_info = model.get_model_info()
    
    # Return results
    return {
        "model_name": model_name,
        "success": True,
        "metrics": metrics,
        "model_info": model_info,
        "checkpoint_path": checkpoint_path
    }

def main():
    """Main function"""
    args = parse_args()
    run_single_baseline(args.model, args)

if __name__ == "__main__":
    main() 