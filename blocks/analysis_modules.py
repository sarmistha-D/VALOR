"""
Analysis Modules for VALOR Framework
Alignment, Dominance, and Complementarity analysis for validation experts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class AlignmentAnalyzer(nn.Module):
    """
    Analyzes alignment between validation experts
    Input: l_v^(1), l_v^(2) - [batch_size, num_classes]
    Output: R_avg - [1] scalar in range [0, 1]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, l_v_1: torch.Tensor, l_v_2: torch.Tensor) -> torch.Tensor:
        # Normalize logits to unit vectors
        l_v_1_norm = F.normalize(l_v_1, dim=-1)
        l_v_2_norm = F.normalize(l_v_2, dim=-1)
        
        # Compute cosine similarity
        alignment = F.cosine_similarity(l_v_1_norm, l_v_2_norm, dim=-1)
        
        # Average across batch
        R_avg = alignment.mean()
        
        return R_avg

class DominanceAnalyzer(nn.Module):
    """
    Analyzes dominance between main MoE and validation MoE
    Input: l_p^(a), l_v^(a) - [batch_size, num_classes]
    Output: Dominance - [1] scalar in range [-1, 1]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, l_p_a: torch.Tensor, l_v_a: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities
        p_p = F.softmax(l_p_a, dim=-1)
        p_v = F.softmax(l_v_a, dim=-1)
        
        # Compute correlation across batch dimension
        p_p_mean = p_p.mean(dim=0, keepdim=True)
        p_v_mean = p_v.mean(dim=0, keepdim=True)
        
        p_p_centered = p_p - p_p_mean
        p_v_centered = p_v - p_v_mean
        
        # Pearson correlation
        numerator = (p_p_centered * p_v_centered).sum(dim=1)
        p_p_var = (p_p_centered ** 2).sum(dim=1)
        p_v_var = (p_v_centered ** 2).sum(dim=1)
        
        denominator = torch.sqrt(p_p_var * p_v_var + 1e-8)
        correlation = numerator / denominator
        
        # Average across batch
        dominance = correlation.mean()
        
        return dominance

class ComplementarityAnalyzer(nn.Module):
    """
    Analyzes complementarity of validation expert predictions
    Input: l_v^(1), l_v^(2) - [batch_size, num_classes]
    Output: U_avg - [1] scalar in range [0, log(num_classes)]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, l_v_1: torch.Tensor, l_v_2: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities for each expert
        p_v_1 = F.softmax(l_v_1, dim=-1)
        p_v_2 = F.softmax(l_v_2, dim=-1)
        
        # Compute entropy for each expert
        entropy_1 = -(p_v_1 * torch.log(p_v_1 + 1e-8)).sum(dim=-1)
        entropy_2 = -(p_v_2 * torch.log(p_v_2 + 1e-8)).sum(dim=-1)
        
        # Average entropy across experts and batch
        U_avg = torch.stack([entropy_1, entropy_2]).mean()
        
        return U_avg

class ValidationAnalysisModule(nn.Module):
    """
    Combined analysis module for validation experts
    """
    
    def __init__(self):
        super().__init__()
        self.alignment_analyzer = AlignmentAnalyzer()
        self.dominance_analyzer = DominanceAnalyzer()
        self.complementarity_analyzer = ComplementarityAnalyzer()
    
    def forward(self, l_v_1: torch.Tensor, l_v_2: torch.Tensor, 
                l_p_a: torch.Tensor, l_v_a: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all analysis metrics
        
        Args:
            l_v_1: [batch_size, num_classes] - Validation Expert 1 logits
            l_v_2: [batch_size, num_classes] - Validation Expert 2 logits
            l_p_a: [batch_size, num_classes] - Main MoE aspect logits
            l_v_a: [batch_size, num_classes] - Validation MoE aspect logits
            
        Returns:
            Dictionary with R_avg, Dominance, U_avg
        """
        R_avg = self.alignment_analyzer(l_v_1, l_v_2)
        Dominance = self.dominance_analyzer(l_p_a, l_v_a)
        U_avg = self.complementarity_analyzer(l_v_1, l_v_2)
    
        return {
            'R_avg': R_avg,
            'Dominance': Dominance,
            'U_avg': U_avg
        } 