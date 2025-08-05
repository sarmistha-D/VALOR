import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CrossAttention(nn.Module):
    """Cross-attention module for multimodal fusion between text and image features"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Cross-attention: text attends to image
        self.q_proj = nn.Linear(hidden_size, hidden_size)  # Query from text
        self.k_proj = nn.Linear(hidden_size, hidden_size)  # Key from image
        self.v_proj = nn.Linear(hidden_size, hidden_size)  # Value from image
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor, 
                text_mask: Optional[torch.Tensor] = None, image_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            text_features: Text features (batch_size, text_seq_len, hidden_size)
            image_features: Image features (batch_size, image_seq_len, hidden_size)
            text_mask: Text attention mask (batch_size, text_seq_len)
            image_mask: Image attention mask (batch_size, image_seq_len)
            
        Returns:
            Fused features (batch_size, text_seq_len, hidden_size)
        """
        batch_size, text_seq_len, _ = text_features.shape
        _, image_seq_len, _ = image_features.shape
        
        # Normalize inputs
        text_features_norm = self.layer_norm1(text_features)
        image_features_norm = self.layer_norm1(image_features)
        
        # Project to Q, K, V
        q = self.q_proj(text_features_norm).view(batch_size, text_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(image_features_norm).view(batch_size, image_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(image_features_norm).view(batch_size, image_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply masks if provided
        if text_mask is not None and image_mask is not None:
            # Create attention mask: (batch_size, 1, text_seq_len, image_seq_len)
            text_mask_expanded = text_mask.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, text_seq_len, 1)
            image_mask_expanded = image_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, image_seq_len)
            attention_mask = text_mask_expanded * image_mask_expanded
            
            # Apply mask to scores
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, text_seq_len, self.hidden_size
        )
        
        # Apply output projection and residual connection
        attn_output = self.out_proj(attn_output)
        fused_features = text_features + self.dropout(attn_output)
        
        # Apply feed-forward network with residual connection
        ff_output = self.ffn(self.layer_norm2(fused_features))
        fused_features = fused_features + self.dropout(ff_output)
        
        return fused_features 