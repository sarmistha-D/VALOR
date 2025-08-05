"""
CLIP Baseline Model
Vision-Language model for multimodal classification
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any
from .base_baseline import BaseBaseline

class CLIPBaseline(BaseBaseline):
    """
    CLIP (Contrastive Language-Image Pre-training) baseline
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__("CLIP", device)
        self.hidden_dim = 1024  # CLIP base model hidden dimension (512*2 for concatenated features)
        self.model_dtype = torch.float16 if device == "cuda" else torch.float32
        
    def load_model(self) -> bool:
        """Load CLIP model and processor"""
        try:
            print(f"🔄 Loading CLIP model...")
            
            # Load CLIP model with proper error handling
            try:
                self.model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32", 
                    torch_dtype=self.model_dtype,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            except Exception as e:
                print(f"Failed to load with device_map, trying standard loading: {e}")
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.model.to(self.device)
                # Convert model to the desired dtype
                self.model = self.model.to(dtype=self.model_dtype)
            
            self.model.eval()
            
            # Create classification heads AFTER model loading so we know the dtype
            self.classification_heads = self.create_classification_heads()
            
            print(f"✅ CLIP model loaded successfully with dtype {self.model_dtype}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load CLIP model: {e}")
            return False
    
    def create_classification_heads(self) -> Dict[str, nn.Module]:
        """Create robust classification heads for aspect and severity"""
        # Improved classification heads with better architecture
        aspect_classifier = nn.Sequential(
            # First layer with dropout for regularization
            nn.Linear(self.hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(128, 6)  # 6 aspect classes
        )
        
        severity_classifier = nn.Sequential(
            # First layer with dropout for regularization
            nn.Linear(self.hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(128, 4)  # 4 severity classes
        )
        
        heads = {
            'aspect_classifier': aspect_classifier,
            'severity_classifier': severity_classifier
        }
        
        # Move to device and initialize weights WITH CORRECT DTYPE
        for head in heads.values():
            head.to(self.device, dtype=self.model_dtype)  # Match model dtype
            # Initialize weights properly
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
        
        print(f"✅ Classification heads created with dtype {self.model_dtype}")
        return heads
    
    def process_input(self, text: str, image) -> Dict[str, torch.Tensor]:
        """Process text and image input for CLIP"""
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device with correct dtype
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device, dtype=self.model_dtype if value.dtype.is_floating_point else value.dtype)
        
        return inputs
    
    def extract_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from CLIP model"""
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Combine image and text features for better performance
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            # Ensure features are properly normalized and in correct dtype
            image_features = nn.functional.normalize(image_features, p=2, dim=1).to(dtype=self.model_dtype)
            text_features = nn.functional.normalize(text_features, p=2, dim=1).to(dtype=self.model_dtype)
            
            # Use concatenated features for better classification
            combined_features = torch.cat([image_features, text_features], dim=-1)
            return combined_features 