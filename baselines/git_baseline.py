"""
GIT Baseline Model
GenerativeImage2Text model for multimodal classification
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from typing import Dict, Any
from .base_baseline import BaseBaseline

class GITBaseline(BaseBaseline):
    """
    GIT (GenerativeImage2Text) baseline
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__("GIT", device)
        self.hidden_dim = 768  # GIT base model hidden dimension
        
    def load_model(self) -> bool:
        """Load GIT model and processor"""
        try:
            print(f"🔄 Loading GIT model...")
            self.model = AutoModel.from_pretrained("microsoft/git-base")
            self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Create classification heads
            self.classification_heads = self.create_classification_heads()
            
            print(f"✅ GIT model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load GIT model: {e}")
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
        """Process text and image input for GIT"""
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
        """Extract features from GIT model"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            # GIT models return different output structures
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                features = outputs.hidden_states[-1][:, 0, :]  # CLS token
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            else:
                # Fallback to mean pooling of last hidden state
                features = outputs.last_hidden_state.mean(dim=1)
            return features 