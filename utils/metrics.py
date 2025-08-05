import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os

def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   class_names: List[str] = None) -> Dict[str, float]:
    """Compute comprehensive metrics including accuracy, F1, precision, recall"""
    
    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Ensure targets are integers
    target_np = target_np.astype(np.int64)
    
    # Get predicted classes
    pred_classes = np.argmax(pred_np, axis=1)
    
    # Compute comprehensive metrics
    accuracy = accuracy_score(target_np, pred_classes)
    f1_macro = f1_score(target_np, pred_classes, average='macro', zero_division=0)
    f1_weighted = f1_score(target_np, pred_classes, average='weighted', zero_division=0)
    f1_micro = f1_score(target_np, pred_classes, average='micro', zero_division=0)
    
    precision_macro = precision_score(target_np, pred_classes, average='macro', zero_division=0)
    precision_weighted = precision_score(target_np, pred_classes, average='weighted', zero_division=0)
    precision_micro = precision_score(target_np, pred_classes, average='micro', zero_division=0)
    
    recall_macro = recall_score(target_np, pred_classes, average='macro', zero_division=0)
    recall_weighted = recall_score(target_np, pred_classes, average='weighted', zero_division=0)
    recall_micro = recall_score(target_np, pred_classes, average='micro', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(target_np, pred_classes, average=None, zero_division=0)
    precision_per_class = precision_score(target_np, pred_classes, average=None, zero_division=0)
    recall_per_class = recall_score(target_np, pred_classes, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'precision_micro': precision_micro,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'recall_micro': recall_micro,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class
    }
    
    # Add per-class metrics if class names provided
    if class_names:
        for i, class_name in enumerate(class_names):
            if i < len(f1_per_class):  # Ensure we don't exceed array bounds
                metrics[f'f1_{class_name.lower().replace(" ", "_")}'] = f1_per_class[i]
                metrics[f'precision_{class_name.lower().replace(" ", "_")}'] = precision_per_class[i]
                metrics[f'recall_{class_name.lower().replace(" ", "_")}'] = recall_per_class[i]
    
    return metrics

def compute_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor,
                           class_names: List[str]) -> np.ndarray:
    """Compute confusion matrix"""
    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
    target_np = targets.cpu().numpy()
    
    return confusion_matrix(target_np, pred_classes, labels=range(len(class_names)))

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         title: str, save_path: str = None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def compute_expert_usage_stats(expert_info: Dict[str, Any]) -> Dict[str, float]:
    """Compute expert usage statistics"""
    expert_usage = expert_info['expert_usage']  # (batch_size, num_experts)
    
    # Average usage per expert
    avg_usage = expert_usage.mean(dim=0)  # (num_experts,)
    
    # Usage distribution
    total_usage = expert_usage.sum(dim=0)  # (num_experts,)
    usage_percentage = total_usage / total_usage.sum() * 100
    
    # Load balancing (coefficient of variation)
    std_usage = expert_usage.std(dim=0)
    mean_usage = expert_usage.mean(dim=0)
    cv_usage = std_usage / (mean_usage + 1e-8)  # Add small epsilon to avoid division by zero
    
    stats = {
        'avg_usage_per_expert': avg_usage.cpu().numpy(),
        'usage_percentage': usage_percentage.cpu().numpy(),
        'load_balancing_cv': cv_usage.cpu().numpy(),
        'total_usage': total_usage.cpu().numpy()
    }
    
    return stats

def compute_agreement_rate(pred_logits: torch.Tensor, val_logits: torch.Tensor) -> float:
    """Compute agreement rate between prediction and validation MoEs"""
    pred_classes = torch.argmax(pred_logits, dim=1)
    val_classes = torch.argmax(val_logits, dim=1)
    
    agreement = (pred_classes == val_classes).float().mean().item()
    return agreement

def generate_classification_report(predictions: torch.Tensor, targets: torch.Tensor,
                                 class_names: List[str]) -> str:
    """Generate detailed classification report"""
    # Handle both tensor and numpy inputs
    if isinstance(predictions, torch.Tensor):
        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
    else:
        pred_classes = np.argmax(predictions, axis=1)
    
    if isinstance(targets, torch.Tensor):
        target_np = targets.cpu().numpy()
    else:
        target_np = targets
    
    report = classification_report(target_np, pred_classes, 
                                 target_names=class_names, 
                                 output_dict=False)
    return report

def save_predictions(predictions: torch.Tensor, targets: torch.Tensor,
                    texts: List[str], image_paths: List[str],
                    class_names: List[str], save_path: str):
    """Save predictions to CSV file"""
    
    pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
    pred_probs = torch.softmax(predictions, dim=1).cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Create DataFrame
    data = {
        'text': texts,
        'image_path': image_paths,
        'true_label': target_np,
        'predicted_label': pred_classes,
        'true_class': [class_names[i] for i in target_np],
        'predicted_class': [class_names[i] for i in pred_classes],
        'confidence': np.max(pred_probs, axis=1)
    }
    
    # Add confidence for each class
    for i, class_name in enumerate(class_names):
        data[f'conf_{class_name.lower().replace(" ", "_")}'] = pred_probs[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

def create_expert_usage_plot(expert_stats: Dict[str, np.ndarray], 
                           expert_names: List[str], 
                           title: str, save_path: str = None):
    """Create bar plot of expert usage"""
    
    plt.figure(figsize=(12, 6))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Usage percentage
    x = range(len(expert_names))
    ax1.bar(x, expert_stats['usage_percentage'])
    ax1.set_xlabel('Expert')
    ax1.set_ylabel('Usage Percentage (%)')
    ax1.set_title('Expert Usage Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(expert_names, rotation=45)
    
    # Add percentage labels on bars
    for i, v in enumerate(expert_stats['usage_percentage']):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Load balancing (coefficient of variation)
    ax2.bar(x, expert_stats['load_balancing_cv'])
    ax2.set_xlabel('Expert')
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('Load Balancing (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(expert_names, rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Expert usage plot saved to {save_path}")
    
    plt.show()

def compute_kl_divergence(pred_logits: torch.Tensor, val_logits: torch.Tensor) -> float:
    """Compute KL divergence between prediction and validation logits"""
    pred_probs = torch.softmax(pred_logits, dim=1)
    val_probs = torch.softmax(val_logits, dim=1)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    pred_probs = pred_probs + eps
    val_probs = val_probs + eps
    
    # Normalize
    pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)
    val_probs = val_probs / val_probs.sum(dim=1, keepdim=True)
    
    # Compute KL divergence
    kl_div = torch.sum(val_probs * torch.log(val_probs / pred_probs), dim=1)
    return kl_div.mean().item()

def print_comprehensive_metrics(aspect_metrics: Dict[str, float], 
                              severity_metrics: Dict[str, float],
                              aspect_classes: List[str], 
                              severity_classes: List[str]):
    """Print comprehensive evaluation metrics for both tasks"""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE EVALUATION METRICS")
    print("="*80)
    
    # Aspect Classification Metrics
    print("\n🎯 ASPECT CLASSIFICATION METRICS:")
    print("-" * 50)
    print(f"Accuracy:     {aspect_metrics['accuracy']:.4f}")
    print(f"F1 Macro:     {aspect_metrics['f1_macro']:.4f}")
    print(f"F1 Weighted:  {aspect_metrics['f1_weighted']:.4f}")
    print(f"Precision:    {aspect_metrics['precision_macro']:.4f}")
    print(f"Recall:       {aspect_metrics['recall_macro']:.4f}")
    
    print("\n📈 Per-Class Aspect Metrics:")
    for i, class_name in enumerate(aspect_classes):
        if i < len(aspect_metrics['f1_per_class']):  # Ensure we don't exceed array bounds
            f1 = aspect_metrics['f1_per_class'][i]
            precision = aspect_metrics['precision_per_class'][i]
            recall = aspect_metrics['recall_per_class'][i]
            print(f"  {class_name:12}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
    
    # Severity Classification Metrics
    print("\n🚨 SEVERITY CLASSIFICATION METRICS:")
    print("-" * 50)
    print(f"Accuracy:     {severity_metrics['accuracy']:.4f}")
    print(f"F1 Macro:     {severity_metrics['f1_macro']:.4f}")
    print(f"F1 Weighted:  {severity_metrics['f1_weighted']:.4f}")
    print(f"Precision:    {severity_metrics['precision_macro']:.4f}")
    print(f"Recall:       {severity_metrics['recall_macro']:.4f}")
    
    print("\n📈 Per-Class Severity Metrics:")
    for i, class_name in enumerate(severity_classes):
        if i < len(severity_metrics['f1_per_class']):  # Ensure we don't exceed array bounds
            f1 = severity_metrics['f1_per_class'][i]
            precision = severity_metrics['precision_per_class'][i]
            recall = severity_metrics['recall_per_class'][i]
            print(f"  {class_name:20}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
    
    # Summary Statistics
    print("\n📊 SUMMARY STATISTICS:")
    print("-" * 50)
    avg_aspect_f1 = aspect_metrics['f1_macro']
    avg_severity_f1 = severity_metrics['f1_macro']
    avg_aspect_acc = aspect_metrics['accuracy']
    avg_severity_acc = severity_metrics['accuracy']
    
    print(f"Average Aspect F1:     {avg_aspect_f1:.4f}")
    print(f"Average Severity F1:   {avg_severity_f1:.4f}")
    print(f"Average Aspect Acc:    {avg_aspect_acc:.4f}")
    print(f"Average Severity Acc:  {avg_severity_acc:.4f}")
    print(f"Overall F1 Score:      {(avg_aspect_f1 + avg_severity_f1) / 2:.4f}")
    print(f"Overall Accuracy:      {(avg_aspect_acc + avg_severity_acc) / 2:.4f}")
    
    print("\n" + "="*80)

def save_comprehensive_results(aspect_metrics: Dict[str, float], 
                             severity_metrics: Dict[str, float],
                             aspect_classes: List[str], 
                             severity_classes: List[str],
                             save_path: str):
    """Save comprehensive results to JSON file"""
    
    results = {
        'aspect_classification': {
            'overall_metrics': {
                'accuracy': aspect_metrics['accuracy'],
                'f1_macro': aspect_metrics['f1_macro'],
                'f1_weighted': aspect_metrics['f1_weighted'],
                'precision_macro': aspect_metrics['precision_macro'],
                'recall_macro': aspect_metrics['recall_macro']
            },
            'per_class_metrics': {}
        },
        'severity_classification': {
            'overall_metrics': {
                'accuracy': severity_metrics['accuracy'],
                'f1_macro': severity_metrics['f1_macro'],
                'f1_weighted': severity_metrics['f1_weighted'],
                'precision_macro': severity_metrics['precision_macro'],
                'recall_macro': severity_metrics['recall_macro']
            },
            'per_class_metrics': {}
        },
        'summary': {
            'overall_f1': (aspect_metrics['f1_macro'] + severity_metrics['f1_macro']) / 2,
            'overall_accuracy': (aspect_metrics['accuracy'] + severity_metrics['accuracy']) / 2
        }
    }
    
    # Add per-class metrics for aspect
    for i, class_name in enumerate(aspect_classes):
        if i < len(aspect_metrics['f1_per_class']):  # Ensure we don't exceed array bounds
            results['aspect_classification']['per_class_metrics'][class_name] = {
                'f1': aspect_metrics['f1_per_class'][i],
                'precision': aspect_metrics['precision_per_class'][i],
                'recall': aspect_metrics['recall_per_class'][i]
            }
    
    # Add per-class metrics for severity
    for i, class_name in enumerate(severity_classes):
        if i < len(severity_metrics['f1_per_class']):  # Ensure we don't exceed array bounds
            results['severity_classification']['per_class_metrics'][class_name] = {
                'f1': severity_metrics['f1_per_class'][i],
                'precision': severity_metrics['precision_per_class'][i],
                'recall': severity_metrics['recall_per_class'][i]
            }
    
    import json
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Comprehensive results saved to {save_path}") 