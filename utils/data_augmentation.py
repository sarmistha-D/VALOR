"""
Data augmentation techniques for VALOR model
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as T
import math
from typing import Dict, Tuple

def mixup_data(batch: Dict[str, torch.Tensor], alpha: float = 0.2) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Applies MixUp augmentation to a batch of data
    
    Args:
        batch: Dictionary containing batch data
        alpha: MixUp interpolation strength parameter
        
    Returns:
        mixed_batch: Dictionary with mixed data
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    # Handle both 'images' and 'image' keys
    image_key = 'images' if 'images' in batch else 'image'
    batch_size = batch[image_key].size(0)
    
    # Generate permutation indices
    index = torch.randperm(batch_size).to(batch[image_key].device)
    
    # Mix images
    mixed_images = lam * batch[image_key] + (1 - lam) * batch[image_key][index, :]
    
    # Create mixed batch
    mixed_batch = {
        'text_inputs': batch['text_inputs'],  # Keep text inputs as is
        image_key: mixed_images,
        'aspect_labels': batch['aspect_labels'],
        'severity_labels': batch['severity_labels'],
        'aspect_labels_mixed': batch['aspect_labels'][index],
        'severity_labels_mixed': batch['severity_labels'][index],
    }
    
    return mixed_batch, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Applies MixUp to the loss calculation
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels (permuted)
        lam: Mixing coefficient
        
    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(batch: Dict[str, torch.Tensor], alpha: float = 1.0) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Applies CutMix augmentation to a batch of images
    
    Args:
        batch: Dictionary containing batch data
        alpha: CutMix interpolation strength parameter
        
    Returns:
        mixed_batch: Dictionary with mixed data
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    # Handle both 'images' and 'image' keys
    image_key = 'images' if 'images' in batch else 'image'
    batch_size, _, H, W = batch[image_key].shape
    
    # Generate permutation indices
    index = torch.randperm(batch_size).to(batch[image_key].device)
    
    # Get random box coordinates
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Create mixed images
    mixed_images = batch[image_key].clone()
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = batch[image_key][index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    # Create mixed batch
    mixed_batch = {
        'text_inputs': batch['text_inputs'],  # Keep text inputs as is
        image_key: mixed_images,
        'aspect_labels': batch['aspect_labels'],
        'severity_labels': batch['severity_labels'],
        'aspect_labels_mixed': batch['aspect_labels'][index],
        'severity_labels_mixed': batch['severity_labels'][index],
    }
    
    return mixed_batch, lam 

def advanced_mixup_data(batch, alpha=0.4):
    """Advanced MixUp with stronger mixing"""
    if alpha <= 0:
        return batch, 1.0
    
    lam = np.random.beta(alpha, alpha)
    # Handle both 'images' and 'image' keys
    image_key = 'images' if 'images' in batch else 'image'
    batch_size = batch[image_key].size(0)
    index = torch.randperm(batch_size)
    
    mixed_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            mixed_batch[key] = lam * value + (1 - lam) * value[index]
        else:
            mixed_batch[key] = value
    
    # Handle mixed labels - ensure proper types
    label_keys = ['aspect_labels', 'severity_labels']
    for label_key in label_keys:
        if label_key in batch:
            mixed_batch[f'{label_key}_mixed'] = batch[label_key][index].long()
    
    return mixed_batch, lam

def advanced_cutmix_data(batch, prob=0.5):
    """Advanced CutMix with better mixing"""
    if random.random() > prob:
        return batch, 1.0
    
    # Handle both 'images' and 'image' keys
    image_key = 'images' if 'images' in batch else 'image'
    batch_size = batch[image_key].size(0)
    index = torch.randperm(batch_size)
    
    # Generate random bounding box
    lam = np.random.beta(1.0, 1.0)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(batch[image_key].size(2) * cut_rat)
    cut_h = int(batch[image_key].size(3) * cut_rat)
    
    # Random center
    cx = np.random.randint(batch[image_key].size(2))
    cy = np.random.randint(batch[image_key].size(3))
    
    bbx1 = np.clip(cx - cut_w // 2, 0, batch[image_key].size(2))
    bby1 = np.clip(cy - cut_h // 2, 0, batch[image_key].size(3))
    bbx2 = np.clip(cx + cut_w // 2, 0, batch[image_key].size(2))
    bby2 = np.clip(cy + cut_h // 2, 0, batch[image_key].size(3))
    
    mixed_batch = {}
    for key, value in batch.items():
        if key == image_key:
            mixed_images = value.clone()
            mixed_images[:, :, bbx1:bbx2, bby1:bby2] = value[index, :, bbx1:bbx2, bby1:bby2]
            mixed_batch[key] = mixed_images
        else:
            mixed_batch[key] = value
    
    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch[image_key].size(2) * batch[image_key].size(3)))
    
    # Handle mixed labels
    label_keys = ['aspect_labels', 'severity_labels']
    for label_key in label_keys:
        if label_key in batch:
            mixed_batch[f'{label_key}_mixed'] = batch[label_key][index].long()
    
    return mixed_batch, lam

def advanced_random_erasing(batch, prob=0.3):
    """Advanced Random Erasing for better regularization"""
    if random.random() > prob:
        return batch
    
    # Handle both 'images' and 'image' keys
    image_key = 'images' if 'images' in batch else 'image'
    batch_size = batch[image_key].size(0)
    for i in range(batch_size):
        if random.random() < 0.5:
            # Random erasing
            img = batch[image_key][i]
            h, w = img.size(1), img.size(2)
            
            # Random area
            area = random.uniform(0.02, 0.4) * h * w
            aspect_ratio = random.uniform(0.3, 3.33)
            
            h_erase = int(round(math.sqrt(area * aspect_ratio)))
            w_erase = int(round(math.sqrt(area / aspect_ratio)))
            
            if h_erase < h and w_erase < w:
                x1 = random.randint(0, h - h_erase)
                y1 = random.randint(0, w - w_erase)
                batch[image_key][i, :, x1:x1 + h_erase, y1:y1 + w_erase] = random.uniform(0, 1)
    
    return batch

def advanced_color_jitter(batch, prob=0.3):
    """Advanced color jittering for robustness"""
    if random.random() > prob:
        return batch
    
    # Color jitter parameters
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    saturation = random.uniform(0.8, 1.2)
    hue = random.uniform(-0.1, 0.1)
    
    # Apply color jitter
    for i in range(batch['images'].size(0)):
        img = batch['images'][i]
        img = img * brightness
        img = torch.clamp(img, 0, 1)
        batch['images'][i] = img
    
    return batch 