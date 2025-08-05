"""
ImageEncoder Module for VALOR Framework
Encodes images using HuggingFace ViT (vit-base-patch16-224-in21k) with modular design
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple

# Standard import for transformers (works in practice)
from transformers import AutoModel  # type: ignore

class ImageEncoder(nn.Module):
    """
    ImageEncoder module for encoding images using pretrained ViT
    
    INPUT:
    - pixel_values: [batch_size, 3, 224, 224]
    
    OUTPUT:
    - patch_embeddings: [batch_size, num_patches + 1, hidden_dim]
    - cls_embedding: [batch_size, hidden_dim]
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", freeze_vit: bool = True, hidden_size: int = 768):
        super().__init__()
        self.model_name = model_name
        self.freeze_vit = freeze_vit
        self.hidden_size = hidden_size
        try:
            self.vit = AutoModel.from_pretrained(model_name)
            print(f"✅ Loaded ViT model: {model_name}")
            if freeze_vit:
                for param in self.vit.parameters():
                    param.requires_grad = False
                print(f"🔒 ViT parameters frozen")
            else:
                print(f"🔓 ViT parameters trainable")
        except Exception as e:
            logging.warning(f"Could not load transformers ViT model: {e}")
            self.vit = FallbackImageEncoder(hidden_size)
            print(f"⚠️ Using fallback image encoder")

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: Tensor of shape [batch_size, 3, 224, 224]
        Returns:
            patch_embeddings: [batch_size, num_patches + 1, hidden_dim]
            cls_embedding: [batch_size, hidden_dim]
        """
        if hasattr(self.vit, 'config'):
            if self.freeze_vit:
                with torch.no_grad():
                    outputs = self.vit(pixel_values=pixel_values)
            else:
                outputs = self.vit(pixel_values=pixel_values)
            patch_embeddings = outputs.last_hidden_state  # [B, N+1, H]
            cls_embedding = patch_embeddings[:, 0, :]     # [B, H]
        else:
            patch_embeddings, cls_embedding = self.vit(pixel_values)
        return patch_embeddings, cls_embedding

    def get_hidden_size(self) -> int:
        return self.hidden_size

    def get_model_name(self) -> str:
        return self.model_name

    def is_frozen(self) -> bool:
        if hasattr(self.vit, 'parameters'):
            return not any(p.requires_grad for p in self.vit.parameters())
        return True

    def unfreeze(self):
        if hasattr(self.vit, 'parameters'):
            for param in self.vit.parameters():
                param.requires_grad = True
            self.freeze_vit = False
            print(f"🔓 ViT parameters unfrozen for fine-tuning")

    def freeze(self):
        if hasattr(self.vit, 'parameters'):
            for param in self.vit.parameters():
                param.requires_grad = False
            self.freeze_vit = True
            print(f"🔒 ViT parameters frozen")

class FallbackImageEncoder(nn.Module):
    """
    Fallback image encoder when transformers is not available
    Simple CNN-based encoder for compatibility
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convolutional layers
        x = self.conv_layers(x)
        x = x.flatten(1)
        # Fully connected layer
        cls_embedding = self.fc(x)
        # For compatibility, create a dummy patch_embeddings tensor
        batch_size = x.size(0)
        patch_embeddings = cls_embedding.unsqueeze(1)  # [B, 1, H]
        return patch_embeddings, cls_embedding

def create_image_encoder(config) -> ImageEncoder:
    """
    Factory function to create ImageEncoder with configuration
    """
    return ImageEncoder(
        model_name=getattr(config, 'image_model', 'google/vit-base-patch16-224-in21k'),
        freeze_vit=getattr(config, 'freeze_vit', True),
        hidden_size=getattr(config, 'hidden_size', 768)
    ) 