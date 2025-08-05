"""
UNITER Baseline Model
UNITER (UNiversal Image-TExt Representation) model for multimodal classification
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from typing import Dict, Any
from .base_baseline import BaseBaseline

class UNITERBaseline(BaseBaseline):
    """
    UNITER baseline
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__("UNITER", device)
        self.hidden_dim = 768  # UNITER hidden dimension
        
    def load_model(self) -> bool:
        """Load UNITER model and processor"""
        try:
            print(f"🔄 Loading UNITER model...")
            # Note: UNITER might not be directly available in transformers
            # Using a fallback approach
            self.model = AutoModel.from_pretrained("uniter-base-4-coco")
            self.processor = AutoProcessor.from_pretrained("uniter-base-4-coco")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Create classification heads
            self.classification_heads = self.create_classification_heads()
            
            print(f"✅ UNITER model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load UNITER model: {e}")
            print("⚠️ UNITER model might not be available in transformers. Using fallback.")
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
        """Process text and image input for UNITER"""
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
        """Extract features from UNITER model"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use pooled output for classification
            features = outputs.pooler_output
            return features 