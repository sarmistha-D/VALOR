"""
TextEncoder Module for VALOR Framework
Dedicated text encoding using pretrained BERT with modular design
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional

# Standard import for transformers (works in practice)
from transformers import AutoModel, AutoTokenizer  # type: ignore

class TextEncoder(nn.Module):
    """
    TextEncoder module for encoding tweet text using pretrained BERT
    
    INPUT:
    - input_ids: [batch_size, seq_len]
    - attention_mask: [batch_size, seq_len]
    
    OUTPUT:
    - token_embeddings: [batch_size, seq_len, hidden_dim]
    - cls_embedding: [batch_size, hidden_dim]
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 freeze_bert: bool = True, 
                 hidden_size: int = 768):
        super(TextEncoder, self).__init__()
        self.model_name = model_name
        self.freeze_bert = freeze_bert
        self.hidden_size = hidden_size
        try:
            self.bert = AutoModel.from_pretrained(model_name)
            print(f"✅ Loaded BERT model: {model_name}")
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
                print(f"🔒 BERT parameters frozen")
            else:
                print(f"🔓 BERT parameters trainable")
        except Exception as e:
            logging.warning(f"Could not load transformers model: {e}")
            self.bert = FallbackTextEncoder(hidden_size)
            print(f"⚠️ Using fallback text encoder")
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the text encoder
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
            attention_mask: Tensor of shape [batch_size, seq_len]
            
        Returns:
            token_embeddings: [batch_size, seq_len, hidden_dim]
            cls_embedding: [batch_size, hidden_dim]
        """
        # Handle different BERT model types
        if hasattr(self.bert, 'config'):
            # Use torch.no_grad() if BERT is frozen for efficiency
            if self.freeze_bert:
                with torch.no_grad():
                    outputs = self.bert(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            else:
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Extract token embeddings and CLS embedding
            token_embeddings = outputs.last_hidden_state  # [B, L, H]
            cls_embedding = token_embeddings[:, 0, :]     # [B, H] — CLS token
            
        else:
            # Fallback encoder
            outputs = self.bert(input_ids, attention_mask)
            token_embeddings = outputs.last_hidden_state
            cls_embedding = token_embeddings[:, 0, :]  # Use first token as CLS
        
        return token_embeddings, cls_embedding
    
    def get_hidden_size(self) -> int:
        """Get the hidden size of the encoder"""
        return self.hidden_size
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name
    
    def is_frozen(self) -> bool:
        """Check if BERT parameters are frozen"""
        if hasattr(self.bert, 'parameters'):
            return not any(p.requires_grad for p in self.bert.parameters())
        return True
    
    def unfreeze(self):
        """Unfreeze BERT parameters for fine-tuning"""
        if hasattr(self.bert, 'parameters'):
            for param in self.bert.parameters():
                param.requires_grad = True
            self.freeze_bert = False
            print(f"🔓 BERT parameters unfrozen for fine-tuning")
    
    def freeze(self):
        """Freeze BERT parameters"""
        if hasattr(self.bert, 'parameters'):
            for param in self.bert.parameters():
                param.requires_grad = False
            self.freeze_bert = True
            print(f"🔒 BERT parameters frozen")
    
    def get_tokenizer(self):
        """Get the tokenizer for this model"""
        try:
            return AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logging.warning(f"Could not load tokenizer: {e}")
            return None

class FallbackTextEncoder(nn.Module):
    """
    Fallback text encoder when transformers is not available
    Simple LSTM-based encoder for compatibility
    """
    
    def __init__(self, hidden_size: int, vocab_size: int = 50000):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Simple embedding + LSTM architecture
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(
            hidden_size, 
            hidden_size, 
            batch_first=True, 
            bidirectional=True,
            num_layers=2
        )
        self.pooler = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.pooler.weight)
        nn.init.zeros_(self.pooler.bias)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through fallback encoder
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            token_embeddings: [batch_size, seq_len, hidden_dim]
            cls_embedding: [batch_size, hidden_dim]
        """
        # Embed tokens
        embedded = self.embedding(input_ids)  # [B, L, H]
        
        # Apply attention mask
        if attention_mask is not None:
            # Create mask for LSTM (1 for valid tokens, 0 for padding)
            mask = attention_mask.bool()
            embedded = embedded * mask.unsqueeze(-1)
        
        # LSTM encoding
        lstm_out, _ = self.encoder(embedded)  # [B, L, H*2]
        
        # Apply pooling layer
        pooled = self.pooler(lstm_out)  # [B, L, H]
        pooled = self.dropout(pooled)
        
        token_embeddings = pooled  # [B, L, H]
        cls_embedding = token_embeddings[:, 0, :]  # [B, H]
        return token_embeddings, cls_embedding

def create_text_encoder(config) -> TextEncoder:
    """
    Factory function to create TextEncoder with configuration
    
    Args:
        config: Configuration object with text_model and other settings
        
    Returns:
        TextEncoder instance
    """
    return TextEncoder(
        model_name=getattr(config, 'text_model', 'bert-base-uncased'),
        freeze_bert=getattr(config, 'freeze_bert', True),
        hidden_size=getattr(config, 'hidden_size', 768)
    ) 