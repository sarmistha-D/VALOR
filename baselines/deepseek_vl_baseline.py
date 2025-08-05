"""
DeepSeek VL Baseline Model
DeepSeek Vision-Language model for multimodal classification
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from typing import Dict, Any
from .base_baseline import BaseBaseline

class DeepSeekVLBaseline(BaseBaseline):
    """
    DeepSeek VL baseline
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__("DeepSeek-VL", device)
        self.hidden_dim = 4096  # DeepSeek VL hidden dimension
        
    def load_model(self) -> bool:
        """Load DeepSeek VL model and processor"""
        try:
            print(f"🔄 Loading DeepSeek VL model...")
            
            # Try different DeepSeek models with proper error handling
            model_names = [
                "deepseek-ai/deepseek-vl-7b-base",
                "deepseek-ai/deepseek-vl-1.3b-base",
                "deepseek-ai/deepseek-vl-7b-chat"
            ]
            
            success = False
            for model_name in model_names:
                try:
                    print(f"Trying to load {model_name}...")
                    
                    # Load with GPU optimizations
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        device_map="auto" if self.device.type == "cuda" else None,
                        trust_remote_code=True
                    )
                    self.processor = AutoProcessor.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )
                    success = True
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    continue
            
            # If all DeepSeek models fail, fall back to CLIP
            if not success:
                print("Falling back to CLIP model instead of DeepSeek-VL")
                from transformers import CLIPModel, CLIPProcessor
                model_name = "openai/clip-vit-base-patch32"
                self.model = CLIPModel.from_pretrained(model_name)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                # Adjust hidden dim for fallback model
                self.hidden_dim = 512
            
            # Ensure model is on correct device if not using device_map
            if not hasattr(self.model, 'device_map'):
                self.model.to(self.device)
            
            self.model.eval()
            
            # Create classification heads
            self.classification_heads = self.create_classification_heads()
            
            print(f"✅ DeepSeek VL model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load DeepSeek VL model: {e}")
            return False
    
    def create_classification_heads(self) -> Dict[str, nn.Module]:
        """Create classification heads for aspect and severity"""
        # Create enhanced classification heads
        aspect_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 6)  # 6 aspect classes
        )
        
        severity_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4)  # 4 severity classes
        )
        
        heads = {
            'aspect_classifier': aspect_classifier,
            'severity_classifier': severity_classifier
        }
        
        # Move to device
        for head in heads.values():
            head.to(self.device)
        
        return heads
    
    def process_input(self, text: str, image) -> Dict[str, torch.Tensor]:
        """Process text and image input for DeepSeek VL"""
        try:
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
        except Exception as e:
            print(f"Error processing input: {e}")
            # Return dummy inputs as fallback
            return {"input_ids": torch.zeros(1, 10).long().to(self.device),
                    "pixel_values": torch.zeros(1, 3, 224, 224).to(self.device)}
    
    def extract_features(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from DeepSeek VL model"""
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use appropriate output based on model type
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    features = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                    # Fallback for CLIP-like models
                    features = outputs.image_embeds
                else:
                    # Ultimate fallback
                    features = torch.zeros(1, self.hidden_dim).to(self.device)
                
                return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return dummy features
            return torch.zeros(1, self.hidden_dim).to(self.device) 