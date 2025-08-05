import torch
from torch.utils.data import Dataset, DataLoader
# Attempt to import HuggingFace datasets library; provide fallback if unavailable
try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    load_dataset = None  # type: ignore
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from collections import Counter
import random
from tqdm import tqdm
import re
from config import config

class Comp2Dataset(Dataset):
    """Dataset class for Comp2.0 multimodal dataset - CORRECTED VERSION"""
    
    def __init__(self, data, tokenizer, image_transforms, aspect_classes, severity_classes, split='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.image_transforms = image_transforms
        self.aspect_classes = aspect_classes
        self.severity_classes = severity_classes
        self.split = split
        
        # Create label mappings
        self.aspect_to_idx = {aspect: idx for idx, aspect in enumerate(aspect_classes)}
        self.severity_to_idx = {severity: idx for idx, severity in enumerate(severity_classes)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get text and labels
        text = item['text']
        aspect_label = self.aspect_to_idx[item['aspect']]
        severity_label = self.severity_to_idx[item['severity']]
        
        # Tokenize text (conversation threads can be long)
        text_inputs = self.tokenizer(
            text,
            max_length=512,  # Increased for conversation threads
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        # Handle image (HuggingFace provides PIL Images directly)
        if isinstance(item['image_path'], Image.Image):
            image = item['image_path'].convert('RGB')
        else:
            # Fallback for other image formats
            image = Image.open(item['image_path']).convert('RGB')
        
        image_tensor = self.image_transforms(image)
        
        return {
            'text_inputs': text_inputs,
            'image': image_tensor,
            'aspect_label': torch.tensor(aspect_label, dtype=torch.long),
            'severity_label': torch.tensor(severity_label, dtype=torch.long),
            'text': text,
            'thread_id': item['thread_id']
        }

def collate_fn(batch):
    """Custom collate function for batching"""
    # Separate different types of data
    text_inputs_list = [item['text_inputs'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    aspect_labels = torch.stack([item['aspect_label'] for item in batch])
    severity_labels = torch.stack([item['severity_label'] for item in batch])
    texts = [item['text'] for item in batch]
    thread_ids = [item['thread_id'] for item in batch]
    
    # Collate text inputs
    text_inputs = {}
    for key in text_inputs_list[0].keys():
        text_inputs[key] = torch.stack([item[key] for item in text_inputs_list])
    
    return {
        'text_inputs': text_inputs,
        'images': images,
        'aspect_labels': aspect_labels,
        'severity_labels': severity_labels,
        'texts': texts,
        'thread_ids': thread_ids
    }

def clean_label(label):
    """Clean and normalize labels - ENHANCED VERSION"""
    if not label:
        return "Unknown"
    
    # Remove newlines and extra whitespace
    label = re.sub(r'\n+', ' ', str(label)).strip()
    label = re.sub(r'\s+', ' ', label)
    
    # Enhanced normalization for Comp2.0 dataset
    label_mapping = {
        # Aspect label corrections
        'sofware': 'Software',
        'software': 'Software',
        'Software Quality ': 'Software',
        'Software\nqualtiy': 'Software',
        'Software\nService ': 'Software',
        'Software Service': 'Software',
        'Service Software': 'Software',
        'Software\nService': 'Software',
        'Software\nQuality': 'Software',
        'Software Quality': 'Software',
        'Software\n Quality': 'Software',
        'Software\nQuality ': 'Software',
        'Software Service': 'Software',
        'Software\nqualtiy': 'Software',
        'Sofware': 'Software',  # Fix typo
        'Software qualtiy': 'Software',  # Fix typo
        
        'Hardware Quality': 'Hardware',
        'Harware. Quality': 'Hardware',
        'Hardware. Service': 'Hardware',
        'Hardware\nPrice': 'Hardware',
        'Hardware Service': 'Hardware',
        'Hardware\nService': 'Hardware',
        'Hardware\n Quality': 'Hardware',
        'Hardware Quality': 'Hardware',
        'Hardware\nPrice': 'Hardware',
        'Hardware Service': 'Hardware',
        'Hardware \n Quality': 'Hardware',
        'Hardware Price': 'Hardware',  # Add this mapping
        
        'Service Packaging': 'Service',
        'Service\n Quality': 'Service',
        'Service\nPrice': 'Service',
        'Service\nPackaging': 'Service',
        'Service Quality': 'Service',
        'Service Price': 'Service',
        'Service\n Quality': 'Service',
        'Service Software': 'Service',
        'Service Price': 'Service',
        'Service\nPrice': 'Service',
        'Service Packaging': 'Service',
        'Quality Service': 'Service',  # Add this mapping
        
        'Packaging Sevice': 'Packaging',
        'Packaging\nService': 'Packaging',
        'Packaging Price': 'Packaging',
        'Packaging Service ': 'Packaging',
        'Packaging Sevice': 'Packaging',
        'Packaging Price': 'Packaging',
        
        'Price. Service': 'Price',
        'Price Sevice': 'Price',
        'Price Service': 'Price',
        'Price\nService': 'Price',
        
        'Quality\nService': 'Quality',
        'Quality\nSoftware': 'Quality',
        'Quality\nHardware': 'Quality',
        'Quality \nSoftware': 'Quality',
        'Quality\nService': 'Quality',
        'Quality\nSoftware': 'Quality',
        'Quality\nHardware': 'Quality',
        'Quality \nSoftware': 'Quality',
        'Service Quality': 'Quality',
        'Hardware\nQuality': 'Quality',
        'Software Quality': 'Quality',
        'Hardware Quality': 'Quality',
        'Software\nQuality': 'Quality',
        'Hardware \n Quality': 'Quality',
        
        # Severity label corrections
        'Disapproval\nDisapproval': 'Disapproval',
        'Blame Blame': 'Blame',
        'Disapproval Blame': 'Disapproval',
        'Disapproval\nBlame': 'Disapproval',
        'Disapproval Disapproval': 'Disapproval',  # Add this mapping
    }
    
    return label_mapping.get(label, label)

def analyze_dataset_distribution(data, aspect_classes, severity_classes):
    """Analyze dataset distribution and identify problematic classes - ENHANCED"""
    # Clean labels first
    cleaned_data = []
    for item in data:
        cleaned_item = item.copy()
        cleaned_item['aspect'] = clean_label(item['aspect'])
        cleaned_item['severity'] = clean_label(item['severity'])
        cleaned_data.append(cleaned_item)
    
    aspect_counts = Counter([item['aspect'] for item in cleaned_data])
    severity_counts = Counter([item['severity'] for item in cleaned_data])
    
    print("\n=== Dataset Distribution Analysis ===")
    print("Aspect distribution:")
    for aspect in aspect_classes:
        count = aspect_counts.get(aspect, 0)
        print(f"  {aspect}: {count} samples")
    
    print("\nSeverity distribution:")
    for severity in severity_classes:
        count = severity_counts.get(severity, 0)
        print(f"  {severity}: {count} samples")
    
    # Find classes with insufficient samples
    min_samples = 2
    problematic_aspects = [aspect for aspect, count in aspect_counts.items() if count < min_samples]
    problematic_severities = [severity for severity, count in severity_counts.items() if count < min_samples]
    
    # Handle sparse classes by duplicating samples for classes with only 1 sample
    augmented_data = cleaned_data.copy()
    
    # Duplicate samples for aspect classes with only 1 sample
    for aspect in problematic_aspects:
        if aspect_counts[aspect] == 1:
            # Find the single sample for this aspect
            single_sample = None
            for item in cleaned_data:
                if item['aspect'] == aspect:
                    single_sample = item
                    break
            
            if single_sample:
                # Duplicate the sample 2 more times
                for _ in range(2):
                    augmented_data.append(single_sample.copy())
                print(f"  Duplicated sample for aspect '{aspect}' (had only 1 sample)")
    
    # Duplicate samples for severity classes with only 1 sample
    for severity in problematic_severities:
        if severity_counts[severity] == 1:
            # Find the single sample for this severity
            single_sample = None
            for item in cleaned_data:
                if item['severity'] == severity:
                    single_sample = item
                    break
            
            if single_sample:
                # Duplicate the sample 2 more times
                for _ in range(2):
                    augmented_data.append(single_sample.copy())
                print(f"  Duplicated sample for severity '{severity}' (had only 1 sample)")
    
    return augmented_data, problematic_aspects, problematic_severities

def load_comp2_dataset(config, tokenizer, image_transforms):
    """Load and split the Comp2.0 dataset with CORRECTED structure"""
    print("Loading Comp2.0 dataset from HuggingFace...")
    
    try:
        # Load from HuggingFace
        if load_dataset:
            dataset = load_dataset(config.dataset_name)
            # Handle different dataset types safely
            try:
                # Try dict-style access first
                if isinstance(dataset, dict) and 'train' in dataset:
                    train_data = dataset['train']
                # Try attribute-style access
                elif hasattr(dataset, 'train') and callable(getattr(dataset, 'train', None)):
                    train_data = dataset.train
                # Try to get the first available split
                elif hasattr(dataset, 'keys') and callable(getattr(dataset, 'keys', None)):
                    splits = list(dataset.keys())
                    if splits:
                        train_data = dataset[splits[0]]
                    else:
                        raise ValueError("No splits found in dataset")
                else:
                    # Fallback: assume it's a single dataset
                    train_data = dataset
            except Exception as access_error:
                print(f"Error accessing dataset splits: {access_error}")
                # Final fallback: assume it's a single dataset
                train_data = dataset
            
            print(f"Loaded {len(train_data)} samples from Comp2.0")
            
            # Convert to list for processing
            data_list = list(train_data)
            
        else:
            raise ImportError("HuggingFace datasets library not available")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic dataset for testing...")
        
        # Create synthetic data for testing
        synthetic_texts = [
            "This product has excellent build quality and design.",
            "The software is buggy and crashes frequently.",
            "Great hardware but terrible customer service.",
            "The packaging was damaged when it arrived.",
            "Overpriced for what you get, not worth the money.",
            "Outstanding service and support from the team.",
            "The design is beautiful but functionality is poor.",
            "Software works well but needs more features.",
            "Hardware quality is questionable, feels cheap.",
            "Excellent packaging and presentation.",
            "Fair price for the quality provided.",
            "Customer service was helpful and responsive.",
            "The product quality exceeded my expectations.",
            "Software is intuitive and easy to use.",
            "Hardware is solid and well-built.",
            "Packaging could be more environmentally friendly.",
            "Good value for money, would recommend.",
            "Service was slow but eventually resolved.",
            "Design is modern and appealing.",
            "Software has occasional bugs but mostly stable."
        ]
        
        data_list = []
        for i, text in enumerate(synthetic_texts):
            data_list.append({
                'text': text,
                'aspect': config.aspect_classes[i % len(config.aspect_classes)],
                'severity': config.severity_classes[i % len(config.severity_classes)],
                'image_path': Image.new('RGB', (224, 224), color=(128, 128, 128)),
                'thread_id': i
            })
        
        print(f"Created synthetic dataset with {len(data_list)} samples")
    
    # Analyze and clean dataset distribution
    cleaned_data, problematic_aspects, problematic_severities = analyze_dataset_distribution(
        data_list, config.aspect_classes, config.severity_classes
    )
    
    # Filter out samples with unknown/cleaned labels
    valid_aspects = set(config.aspect_classes)
    valid_severities = set(config.severity_classes)
    
    filtered_data = []
    for item in cleaned_data:
        if item['aspect'] in valid_aspects and item['severity'] in valid_severities:
            filtered_data.append(item)
    
    print(f"\nFiltered to {len(filtered_data)} valid samples")
    
    # Check if we can do stratified splitting
    aspect_counts = Counter([item['aspect'] for item in filtered_data])
    severity_counts = Counter([item['severity'] for item in filtered_data])
    
    print(f"\nAspect distribution: {dict(aspect_counts)}")
    print(f"Severity distribution: {dict(severity_counts)}")
    
    # More robust stratification check
    min_samples_per_class = 2
    can_stratify_aspect = all(count >= min_samples_per_class for count in aspect_counts.values())
    can_stratify_severity = all(count >= min_samples_per_class for count in severity_counts.values())
    
    if can_stratify_aspect and can_stratify_severity:
        print("\nStratified splitting possible - using aspect-based stratification")
        try:
            # Split data into train/val/test with stratification
            train_data, temp_data = train_test_split(
                filtered_data, 
                test_size=config.val_split + config.test_split,
                random_state=42,
                stratify=[item['aspect'] for item in filtered_data]
            )
            
            val_size = int(len(temp_data) * config.val_split / (config.val_split + config.test_split))
            val_data, test_data = train_test_split(
                temp_data,
                test_size=len(temp_data) - val_size,
                random_state=42,
                stratify=[item['aspect'] for item in temp_data]
            )
        except ValueError as e:
            print(f"⚠️  Stratified splitting failed: {e}")
            print("Falling back to random splitting...")
            can_stratify_aspect = False
    else:
        print(f"\n⚠️  Cannot use stratified splitting:")
        if not can_stratify_aspect:
            print(f"  - Some aspect classes have < {min_samples_per_class} samples")
        if not can_stratify_severity:
            print(f"  - Some severity classes have < {min_samples_per_class} samples")
        print("Falling back to random splitting...")
    
    if not can_stratify_aspect:
        print("\n🔄 Using random splitting (some classes have insufficient samples)")
        # Random splitting without stratification
        train_data, temp_data = train_test_split(
            filtered_data, 
            test_size=config.val_split + config.test_split,
            random_state=42
        )
        
        val_size = int(len(temp_data) * config.val_split / (config.val_split + config.test_split))
        val_data, test_data = train_test_split(
            temp_data,
            test_size=len(temp_data) - val_size,
            random_state=42
        )
    
    print(f"\nSplit sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create dataset objects
    print("\nCreating dataset objects...")
    train_dataset = Comp2Dataset(
        train_data, tokenizer, image_transforms, 
        config.aspect_classes, config.severity_classes, 'train'
    )
    
    val_dataset = Comp2Dataset(
        val_data, tokenizer, image_transforms, 
        config.aspect_classes, config.severity_classes, 'val'
    )
    
    test_dataset = Comp2Dataset(
        test_data, tokenizer, image_transforms, 
        config.aspect_classes, config.severity_classes, 'test'
    )
    
    print("Dataset loading completed successfully!")
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
    """Create DataLoaders for all splits with progress tracking"""
    
    print("\nCreating DataLoaders...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Jupyter/cloud environments
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Jupyter/cloud environments
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Jupyter/cloud environments
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def balance_dataset(dataset, aspect_classes, severity_classes):
    """Balance dataset by oversampling minority classes"""
    aspect_labels = [item['aspect_labels'] for item in dataset]
    severity_labels = [item['severity_labels'] for item in dataset]
    
    # Count class distributions
    aspect_counts = Counter(aspect_labels)
    severity_counts = Counter(severity_labels)
    
    # Find target count (mean of all classes)
    target_aspect_count = max(aspect_counts.values()) // 2  # Reduce majority class
    target_severity_count = max(severity_counts.values()) // 2
    
    balanced_dataset = []
    
    # Oversample minority classes
    for item in dataset:
        aspect_label = item['aspect_labels']
        severity_label = item['severity_labels']
        
        # Determine how many times to include this sample
        aspect_multiplier = max(1, target_aspect_count // aspect_counts[aspect_label])
        severity_multiplier = max(1, target_severity_count // severity_counts[severity_label])
        
        # Add sample multiple times if it's from minority class
        for _ in range(min(aspect_multiplier, severity_multiplier)):
            balanced_dataset.append(item)
    
    # Shuffle the balanced dataset
    random.shuffle(balanced_dataset)
    
    return balanced_dataset