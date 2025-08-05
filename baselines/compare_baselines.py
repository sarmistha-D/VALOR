#!/usr/bin/env python3
"""
Compare all baseline models with fine-tuning and visualization
"""

import os
import sys
import argparse
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from tabulate import tabulate
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to path for direct execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
try:
    from .run_baselines import BaselineRunner
except ImportError:
    from run_baselines import BaselineRunner

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare baseline models")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune the models before evaluation"
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
        "--output",
        type=str,
        default="baseline_comparison",
        help="Output file prefix"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="baseline_checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--load-checkpoints",
        action="store_true",
        help="Load existing checkpoints instead of fine-tuning"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of models to run (default: all models)"
    )
    return parser.parse_args()

def create_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison table from results"""
    data = []
    
    for model_name, result in results.items():
        if result.get("success", False):
            metrics = result["metrics"]
            data.append({
                "Model": model_name,
                "Aspect Acc": metrics["aspect_accuracy"],
                "Aspect F1": metrics["aspect_f1"],
                "Severity Acc": metrics["severity_accuracy"],
                "Severity F1": metrics["severity_f1"],
                "Overall Acc": metrics["overall_accuracy"],
                "Overall F1": metrics["overall_f1"]
            })
    
    # Create DataFrame and sort by overall F1 score
    df = pd.DataFrame(data)
    df = df.sort_values(by="Overall F1", ascending=False)
    
    return df

def plot_comparison(df: pd.DataFrame, output_prefix: str):
    """Create comparison plots"""
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    # Plot overall metrics
    ax = sns.barplot(
        data=df,
        x="Model",
        y="Overall F1",
        palette="viridis"
    )
    ax.set_title("Overall F1 Score Comparison", fontsize=16)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add value labels
    for i, v in enumerate(df["Overall F1"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_overall_f1.png", dpi=300)
    
    # Plot aspect vs severity metrics
    plt.figure(figsize=(16, 8))
    
    # Reshape data for grouped bar plot
    plot_data = pd.melt(
        df,
        id_vars=["Model"],
        value_vars=["Aspect F1", "Severity F1"],
        var_name="Metric",
        value_name="F1 Score"
    )
    
    ax = sns.barplot(
        data=plot_data,
        x="Model",
        y="F1 Score",
        hue="Metric",
        palette="Set2"
    )
    ax.set_title("Aspect vs Severity F1 Score Comparison", fontsize=16)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_aspect_severity_f1.png", dpi=300)
    
    # Create heatmap of all metrics
    plt.figure(figsize=(12, 8))
    metrics_df = df.set_index("Model")
    metrics_df = metrics_df.drop(columns=["Overall Acc"])  # Remove one column to make heatmap cleaner
    
    ax = sns.heatmap(
        metrics_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5
    )
    ax.set_title("Baseline Models Performance Metrics", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_metrics_heatmap.png", dpi=300)

def save_results(results: Dict[str, Dict[str, Any]], output_prefix: str):
    """Save results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_prefix}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return filename

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

def run_baseline_with_checkpoint(runner, model_name, model_class, samples, args):
    """Run a baseline model with checkpoint handling"""
    # Split dataset
    train_samples, val_samples, test_samples = runner.split_dataset(samples)
    
    # Initialize model
    model = model_class(device=str(runner.device))
    
    # Load model
    if not model.load_model():
        print(f"❌ Failed to load {model_name} model. Skipping...")
        return {
            "model_name": model_name,
            "success": False,
            "error": "Failed to load model"
        }
    
    # Check if we should load checkpoint
    checkpoint_path = None
    if args.load_checkpoints:
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir, model_name)
        if checkpoint_path:
            print(f"📥 Loading checkpoint for {model_name} from {checkpoint_path}")
            if not model.load_checkpoint(checkpoint_path):
                print(f"⚠️ Failed to load checkpoint for {model_name}. Will fine-tune instead.")
                checkpoint_path = None
    
    # Fine-tune if needed
    if args.fine_tune and not (args.load_checkpoints and checkpoint_path):
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
    results = {
        "model_name": model_name,
        "success": True,
        "metrics": metrics,
        "model_info": model_info,
        "checkpoint_path": checkpoint_path
    }
    
    return results

def main():
    """Main function"""
    args = parse_args()
    
    # Update config with command line arguments
    config.fine_tune_baselines = args.fine_tune
    config.fine_tune_epochs = args.epochs
    config.fine_tune_lr = args.lr
    config.fine_tune_batch_size = args.batch_size
    
    # Create output directory
    os.makedirs("baseline_results", exist_ok=True)
    output_prefix = f"baseline_results/{args.output}"
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize baseline runner
    runner = BaselineRunner(config)
    
    # Load dataset
    samples = runner.load_dataset()
    
    # Get models to run
    if args.models:
        models_to_run = {
            model_name: model_class 
            for model_name, model_class in runner.baseline_models.items()
            if model_name in args.models
        }
    else:
        models_to_run = runner.baseline_models
    
    # Run selected baselines
    print(f"Running {len(models_to_run)} baseline models (fine-tune={config.fine_tune_baselines})...")
    
    results = {}
    for model_name, model_class in models_to_run.items():
        print(f"\n{'='*50}")
        print(f"Running {model_name} baseline...")
        print(f"{'='*50}")
        
        result = run_baseline_with_checkpoint(runner, model_name, model_class, samples, args)
        results[model_name] = result
    
    # Save results
    results_file = save_results(results, output_prefix)
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Print table
    print("\nBaseline Models Comparison:")
    print(tabulate(df, headers="keys", tablefmt="pipe", showindex=False))
    
    # Save table as CSV
    df.to_csv(f"{output_prefix}_comparison.csv", index=False)
    
    # Plot comparison
    plot_comparison(df, output_prefix)
    
    print(f"\nComparison plots saved to {output_prefix}_*.png")
    print(f"Comparison table saved to {output_prefix}_comparison.csv")

if __name__ == "__main__":
    main() 