"""
Gemma3 Baseline Model
Google's Gemma3 model for multimodal classification
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from typing import Dict, Any
from .base_baseline import BaseBaseline

class Gemma3Baseline(BaseBaseline):
    """
    Gemma3 baseline
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__("Gemma3", device)
        self.hidden_dim = 4096  # Gemma3 hidden dimension
        
    def load_model(self) -> bool:
        """Load Gemma3 model and processor"""
        try:
            print(f"🔄 Loading Gemma3 model...")
            self.model = AutoModel.from_pretrained("google/gemma-2-9b-it")
            self.processor = AutoProcessor.from_pretrained("google/gemma-2-9b-it")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Create classification heads
            self.classification_heads = self.create_classification_heads()
            
            print(f"✅ Gemma3 model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Gemma3 model: {e}")
            return False
    
    def create_classification_heads(self) -> Dict[str, nn.Module]:
        """Create classification heads for aspect and severity"""
        heads = {
            'aspect_classifier': nn.Linear(self.hidden_dim, 6),
            'severity_classifier': nn.Linear(self.hidden_dim, 4)
        }
        
        # Move to device
        for head in heads.values():
            head.to(self.device)
        
        return heads
    
    def process_input(self, text: str, image) -> Dict[str, torch.Tensor]:
        """Process text and image input for Gemma3"""
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)
        
        return inputs
    
    def extract_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from Gemma3 model"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use pooled output for classification
            features = outputs.pooler_output
            return features 