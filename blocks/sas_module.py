"""
Learnable Semantic Alignment Score (SAS) Module for VALOR Framework
Computes learnable alignment scores between text and image representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class LearnableSAS(nn.Module):
    """
    Learnable Semantic Alignment Score (SAS) Module
    
    Projects text and image representations into alignment space and computes
    similarity using an MLP to measure semantic alignment between modalities.
    
    INPUTS:
    - t_vec: [batch_size, hidden_dim] - Text representation (BERT CLS)
    - v_vec: [batch_size, hidden_dim] - Image representation (ViT CLS)
    
    OUTPUT:
    - sas_score: [batch_size] - Alignment score in range [-1, 1]
    """
    
    def __init__(self, hidden_dim: int = 768, projection_dim: int = 512,
                 use_layer_norm: bool = True, dropout: float = 0.1):
        super(LearnableSAS, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.use_layer_norm = use_layer_norm
        
        # Text projection network
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.projection_dim)
        )
        
        # Image projection network
        self.image_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.projection_dim)
        )
        
        # Layer normalization for projections
        if use_layer_norm:
            self.text_ln = nn.LayerNorm(self.projection_dim)
            self.image_ln = nn.LayerNorm(self.projection_dim)
        
        # MLP-based scoring function
        self.score_fn = nn.Sequential(
            nn.Linear(2 * self.projection_dim, self.projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.projection_dim, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize SAS network weights"""
        for module in [self.text_proj, self.image_proj]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        # Initialize scoring function
        for layer in self.score_fn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, t_vec: torch.Tensor, v_vec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through learnable SAS module
        
        Args:
            t_vec: Text representation [batch_size, hidden_dim]
            v_vec: Image representation [batch_size, hidden_dim]
            
        Returns:
            sas_score: Alignment score [batch_size] in range [-1, 1]
        """
        # Project text and image to alignment space
        t_proj = self.text_proj(t_vec)  # [batch_size, projection_dim]
        v_proj = self.image_proj(v_vec)  # [batch_size, projection_dim]
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            t_proj = self.text_ln(t_proj)
            v_proj = self.image_ln(v_proj)
        
        # Concatenate and pass through MLP
        combined = torch.cat([t_proj, v_proj], dim=-1)  # [batch_size, 2*projection_dim]
        sas_score = self.score_fn(combined).squeeze(-1)  # [batch_size]
        
        return sas_score
    
    def get_sas_info(self) -> Dict[str, Any]:
        """Get information about the SAS module"""
        return {
            "hidden_dim": self.hidden_dim,
            "projection_dim": self.projection_dim,
            "use_layer_norm": self.use_layer_norm,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

class SASLoss(nn.Module):
    """
    SAS-based loss function for encouraging semantic alignment
    """
    
    def __init__(self, margin: float = 0.3):
        super(SASLoss, self).__init__()
        self.margin = margin
    
    def forward(self, sas_score: torch.Tensor) -> torch.Tensor:
        """
        Compute SAS loss
        
        Args:
            sas_score: SAS scores [batch_size]
            
        Returns:
            sas_loss: Scalar loss value
        """
        # Margin loss - encourage alignment above margin
        sas_loss = F.relu(self.margin - sas_score).mean()
        return sas_loss

def create_sas_module(config) -> LearnableSAS:
    """
    Factory function to create SAS module based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        LearnableSAS instance
    """
    hidden_dim = getattr(config, 'valor_hidden_dim', 768)
    use_layer_norm = getattr(config, 'sas_use_layer_norm', True)
    dropout = getattr(config, 'sas_dropout', 0.1)
    
    return LearnableSAS(
        hidden_dim=hidden_dim,
        use_layer_norm=use_layer_norm,
        dropout=dropout
    ) 