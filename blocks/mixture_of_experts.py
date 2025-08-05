"""
MixtureOfExperts Module for VALOR Framework
Implementation of CoT-based mixture of experts using DeepSeek model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from blocks.router import Router
from blocks.cot_expert import CoTExpert

class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts for VALOR framework using Chain-of-Thought reasoning
    with DeepSeek coder model as the expert backbone.
    
    INPUTS:
    - x: [B, H] — Input features
    
    OUTPUTS:
    - logits: [B, C] — Classification logits
    - load_balance_loss: scalar — Load balancing loss from router
    - routing_weights: [B, E] — Expert routing weights
    """
    
    def __init__(self, hidden_dim: int, num_classes: int, num_experts: int = 4,
                 top_k: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router for expert selection (using hard_top1 routing)
        self.router = Router(
            input_size=hidden_dim,
            num_experts=num_experts,
            top_k=1,  # Using hard_top1 routing strategy
            noise_std=0.05,
            load_balance_weight=0.05,
            routing_strategy="hard_top1"
        )
        
        # Create CoT experts using DeepSeek model
        self.experts = nn.ModuleList([
            CoTExpert(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                model_name="deepseek-ai/deepseek-coder-6.7b",
                max_tokens=24,
                temperature=0.5,
                top_k=30,
                top_p=0.9,
                expert_id=i
            )
            for i in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE
        
        Args:
            x: Input features [batch_size, hidden_dim]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
            load_balance_loss: Load balancing loss
            routing_weights: Expert routing weights [batch_size, num_experts]
        """
        batch_size = x.size(0)
        
        # Route input to experts
        routing_weights, load_balance_loss = self.router(x)
        
        # Using hard_top1 routing - take the highest weight expert for each sample
        _, indices = routing_weights.max(dim=1)
        
        # Initialize output tensor
        logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        # Send each sample to its assigned expert
        for i in range(self.num_experts):
            # Find samples routed to this expert
            mask = (indices == i)
            if mask.sum() > 0:
                # Select samples for this expert
                expert_inputs = x[mask]
                
                # Pass through expert
                expert_outputs = self.experts[i](expert_inputs)
                
                # Place outputs in the correct positions
                logits[mask] = expert_outputs
        
        return logits, load_balance_loss, routing_weights
    
    def get_expert_info(self) -> Dict[str, Any]:
        """Get information about the MoE"""
        expert_info = [expert.get_expert_info() for expert in self.experts]
        return {
            "num_experts": self.num_experts,
            "expert_type": "cot",
            "model_name": "deepseek-ai/deepseek-coder-6.7b",
            "routing": "hard_top1",
            "experts": expert_info
        } 