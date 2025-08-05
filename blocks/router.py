import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

class Router(nn.Module):
    """
    Router with hard_top1 routing strategy for VALOR framework
    Routes each input to exactly one expert using hard selection
    """
    
    def __init__(self, input_size: int, num_experts: int, top_k: int = 1, 
                 noise_std: float = 0.05, load_balance_weight: float = 0.05,
                 routing_strategy: str = "hard_top1"):
        super().__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = 1  # Fixed for hard_top1 strategy
        self.noise_std = noise_std
        self.load_balance_weight = load_balance_weight
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_size // 2, num_experts)
        )
        
        # Initialize router weights
        self._init_router_weights()
        
    def _init_router_weights(self):
        """Initialize router weights with proper scaling"""
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        
        # Initialize final layer with smaller weights for stability
        final_layer = self.router[-1]
        nn.init.normal_(final_layer.weight, std=0.01)
        nn.init.zeros_(final_layer.bias)
    
    def add_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """Add noise to router logits for exploration"""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            return logits + noise
        return logits
    
    def compute_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss"""
        # Get the probability of selecting each expert across the batch
        routing_prob = routing_weights.mean(dim=0)
        
        # Ideal uniform distribution
        uniform_prob = torch.ones_like(routing_prob) / self.num_experts
        
        # KL divergence as load balancing loss
        epsilon = 1e-10
        load_balance_loss = F.kl_div(
            (routing_prob + epsilon).log(),
            uniform_prob,
            reduction='batchmean'
        )
        
        return load_balance_loss * self.load_balance_weight
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hard_top1 routing
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            routing_weights: One-hot expert selection (batch_size, num_experts)
            load_balance_loss: Load balancing loss
        """
        # Get router logits
        router_logits = self.router(x)  # (batch_size, num_experts)
        
        # Add noise for exploration
        noisy_logits = self.add_noise(router_logits)
        
        # Get expert index with highest score
        _, indices = noisy_logits.max(dim=1)
        
        # Create one-hot vectors for each input
        routing_weights = F.one_hot(indices, num_classes=self.num_experts).float()
        
        # Compute load balance loss
        load_balance_loss = self.compute_load_balance_loss(routing_weights)
        
        return routing_weights, load_balance_loss 