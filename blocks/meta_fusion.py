"""
Meta-Fusion Module for VALOR Framework
Fusion of prediction logits, validation logits, SAS scores, and analysis metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

class MetaFusionHead(nn.Module):
    """
    Meta-Fusion Head for adaptive logit fusion
    
    Combines:
    - Prediction MoE logits from CoT experts
    - Validation MoE logits from transformer experts
    - Semantic Alignment Score (SAS)
    - Expert routing information
    - Analysis metrics (Alignment, Dominance, Complementarity)
    
    Into final logits for classification (aspect or severity).
    
    INPUTS:
    - prediction_logits: [batch_size, num_classes] - Main MoE predictions
    - validation_logits: [batch_size, num_classes] - Validation MoE predictions
    - sas_score: [batch_size] - Semantic alignment scores
    - routing_entropy: [batch_size] - Expert routing entropy
    - analysis_metrics: Dict - Analysis metrics (R_avg, Dominance, U_avg)
    
    OUTPUT:
    - final_logits: [batch_size, num_classes] - Refined classification logits
    - fusion_weights: Dict - Fusion weights for interpretability
    """
    
    def __init__(self, num_classes: int, hidden_dim: int = 768, dropout: float = 0.1):
        super(MetaFusionHead, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Enhanced input dimensions for fusion
        # pred_logits: [B, C], val_logits: [B, C], sas_score: [B, 1], entropy: [B, 1]
        # analysis: [B, 3] (R_avg, Dominance, U_avg)
        input_dim = 2 * num_classes + 5  # 2*C + 2 (sas + entropy) + 3 (analysis)
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Learnable fusion weights for interpretability
        self.prediction_weight = nn.Parameter(torch.ones(1))
        self.validation_weight = nn.Parameter(torch.ones(1))
        self.sas_weight = nn.Parameter(torch.ones(1))
        self.entropy_weight = nn.Parameter(torch.ones(1))
        self.alignment_weight = nn.Parameter(torch.ones(1))
        self.dominance_weight = nn.Parameter(torch.ones(1))
        self.complementarity_weight = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize meta-fusion weights"""
        for module in self.fusion_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize fusion weights
        nn.init.ones_(self.prediction_weight)
        nn.init.ones_(self.validation_weight)
        nn.init.ones_(self.sas_weight)
        nn.init.ones_(self.entropy_weight)
        nn.init.ones_(self.alignment_weight)
        nn.init.ones_(self.dominance_weight)
        nn.init.ones_(self.complementarity_weight)
    
    def forward(self, prediction_logits: torch.Tensor, 
                validation_logits: torch.Tensor,
                sas_score: torch.Tensor,
                routing_entropy: torch.Tensor,
                analysis_metrics: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced forward pass with analysis metrics
        
        Args:
            prediction_logits: [batch_size, num_classes] - Main CoT MoE predictions
            validation_logits: [batch_size, num_classes] - Transformer validation predictions
            sas_score: [batch_size] - Semantic alignment scores
            routing_entropy: [batch_size] - Expert routing entropy
            analysis_metrics: Dict - Analysis metrics (R_avg, Dominance, U_avg)
            
        Returns:
            final_logits: [batch_size, num_classes] - Refined classification logits
            fusion_weights: Dictionary with fusion information
        """
        batch_size = prediction_logits.size(0)
        
        # Ensure sas_score is [B, 1]
        sas_score = sas_score.view(batch_size, 1)
        
        # Ensure routing_entropy is [B, 1]
        routing_entropy = routing_entropy.view(batch_size, 1)
        
        # Extract analysis metrics
        R_avg = analysis_metrics.get('R_avg', torch.tensor(0.0, device=prediction_logits.device))
        Dominance = analysis_metrics.get('Dominance', torch.tensor(0.0, device=prediction_logits.device))
        U_avg = analysis_metrics.get('U_avg', torch.tensor(0.0, device=prediction_logits.device))
        
        # Ensure analysis metrics are [B, 1]
        R_avg = R_avg.view(1, 1).expand(batch_size, 1)
        Dominance = Dominance.view(1, 1).expand(batch_size, 1)
        U_avg = U_avg.view(1, 1).expand(batch_size, 1)
        
        # Concatenate all inputs: [pred_logits, val_logits, sas_score, entropy, R_avg, Dominance, U_avg]
        fused_input = torch.cat([
            prediction_logits, validation_logits, sas_score, routing_entropy,
            R_avg, Dominance, U_avg
        ], dim=-1)
        
        # Apply fusion network
        output_logits = self.fusion_net(fused_input)
        
        # Prepare fusion weights for interpretability
        fusion_weights = {
            'prediction_weight': self.prediction_weight.item(),
            'validation_weight': self.validation_weight.item(),
            'sas_weight': self.sas_weight.item(),
            'entropy_weight': self.entropy_weight.item(),
            'alignment_weight': R_avg.mean().item(),
            'dominance_weight': Dominance.mean().item(),
            'complementarity_weight': U_avg.mean().item(),
            'sas_mean': sas_score.mean().item(),
            'entropy_mean': routing_entropy.mean().item()
        }
        
        return output_logits, fusion_weights
    
    def get_fusion_info(self) -> Dict[str, Any]:
        """Get information about the meta-fusion head"""
        return {
            "input_dim": 2 * self.num_classes + 5,  # Enhanced with analysis metrics
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "analysis_features": ["alignment", "dominance", "complementarity"]
        }

def create_meta_fusion_head(config, num_classes: int, task_type: str = "aspect") -> MetaFusionHead:
    """Enhanced factory function for meta-fusion heads with analysis support"""
    
    return MetaFusionHead(
        num_classes=num_classes,
        hidden_dim=config.meta_fusion_hidden_dim,
        dropout=config.meta_fusion_dropout
    ) 