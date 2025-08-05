"""
SMOL-VLM Baseline Model
Microsoft's SMOL-VLM model for multimodal classification
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from typing import Dict, Any
from .base_baseline import BaseBaseline

class SMOLVLMBaseline(BaseBaseline):
    """
    SMOL-VLM baseline
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__("SMOL-VLM", device)
        self.hidden_dim = 1024  # SMOL-VLM hidden dimension
        
    def load_model(self) -> bool:
        """Load SMOL-VLM model and processor"""
        try:
            print(f"🔄 Loading SMOL-VLM model...")
            self.model = AutoModel.from_pretrained("microsoft/SMOL-VLM-1.5B")
            self.processor = AutoProcessor.from_pretrained("microsoft/SMOL-VLM-1.5B")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Create classification heads
            self.classification_heads = self.create_classification_heads()
            
            print(f"✅ SMOL-VLM model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load SMOL-VLM model: {e}")
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
        """Process text and image input for SMOL-VLM"""
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
        """Extract features from SMOL-VLM model"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use pooled output for classification
            features = outputs.pooler_output
            return features 