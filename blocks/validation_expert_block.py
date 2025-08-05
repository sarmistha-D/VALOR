"""
ValidationExpertBlock Module for VALOR Framework
Transformer-based validation experts for secondary opinion
Enhanced with Alignment, Dominance, and Complementarity analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from transformers import AutoModel
from blocks.analysis_modules import ValidationAnalysisModule

class TransformerValidationExpert(nn.Module):
    """
    Transformer-based validation expert using DeepSeek model backbone
    Provides a "second opinion" on predictions to improve robustness
    
    INPUT:
    - x: [batch_size, hidden_dim] - fused representation
    
    OUTPUT:
    - logits: [batch_size, num_classes] - validation logits
    """
    
    def __init__(self, hidden_dim: int = 768, num_classes: int = 5, 
                 expert_id: int = 0,
                 model_name: str = "deepseek-ai/deepseek-coder-6.7b"):
        super(TransformerValidationExpert, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.expert_id = expert_id
        self.model_name = model_name
        
        # Load DeepSeek model for validation
        self.transformer = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Freeze most of transformer for efficiency
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Only fine-tune the last few layers
        for i, layer in enumerate(self.transformer.layers):
            if i >= len(self.transformer.layers) - 2:  # Only fine-tune last 2 layers
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Get transformer hidden size
        self.transformer_dim = self.transformer.config.hidden_size
        
        # Projection layers
        self.input_projection = nn.Linear(hidden_dim, self.transformer_dim)
        self.output_projection = nn.Linear(self.transformer_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer validation expert
        
        Args:
            x: Input features [batch_size, hidden_dim]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Project input to transformer dimension
        projected_input = self.input_projection(x)
        
        # Create attention mask (all 1s for no masking)
        attention_mask = torch.ones(
            projected_input.shape[0], 1, 
            device=projected_input.device
        )
        
        # Process through transformer (using only the features, not a sequence)
        with torch.no_grad():  # Avoid backprop through most of the transformer
            transformer_outputs = self.transformer(
                inputs_embeds=projected_input.unsqueeze(1),  # Add sequence dimension
                attention_mask=attention_mask
            )
        
        # Get transformer output (last hidden state)
        transformer_hidden = transformer_outputs.last_hidden_state.squeeze(1)
        
        # Project back to hidden dimension
        output = self.output_projection(transformer_hidden)
        
        # Classify
        logits = self.classifier(output)
        
        return logits
    
    def get_expert_info(self) -> Dict[str, Any]:
        """Get information about this expert"""
        return {
            "expert_id": self.expert_id,
            "expert_type": "transformer_validation",
            "model_name": self.model_name,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

class ValidationMoE(nn.Module):
    """
    Validation Mixture of Experts using transformer-based validation
    Enhanced with Alignment, Dominance, and Complementarity analysis
    
    This module provides a second opinion on predictions using transformer-based
    validation experts that complement the main prediction MoE.
    
    NEW FEATURES:
    - Alignment Analysis: Measures similarity between validation experts
    - Dominance Analysis: Measures complementarity with main MoE predictions
    - Complementarity Analysis: Measures diversity of validation expert predictions
    """
    
    def __init__(self, hidden_dim: int, num_classes: int, num_experts: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_experts = num_experts
        
        # Create transformer validation experts
        self.experts = nn.ModuleList([
            TransformerValidationExpert(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                expert_id=i,
                model_name="deepseek-ai/deepseek-coder-6.7b"
            )
            for i in range(num_experts)
        ])
        
        # Expert selection gate
        self.gate = nn.Linear(hidden_dim, num_experts)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize gate weights
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        
        # Analysis module
        self.analysis_module = ValidationAnalysisModule()
        
        # Analysis loss weights
        self.alignment_weight = 0.1
        self.dominance_weight = 0.1
        self.complementarity_weight = 0.1
        
        # Analysis thresholds
        self.tau_R = 0.3
        self.tau_S = 0.5
        self.tau_U = 1.5
    
    def forward(self, x: torch.Tensor, main_logits: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced forward pass with analysis
        
        Args:
            x: Input features [batch_size, hidden_dim]
            main_logits: [batch_size, num_classes] - Main MoE predictions (optional)
            
        Returns:
            validation_logits: [batch_size, num_classes] - validation logits
            gate_weights: [batch_size, num_experts] - expert selection weights
            analysis_metrics: Dictionary with R_avg, Dominance, U_avg
        """
        batch_size = x.size(0)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Compute expert selection gates
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Get all expert outputs for analysis
        all_expert_logits = []
        for expert in self.experts:
            expert_output = expert(x)
            all_expert_logits.append(expert_output)
        
        # Stack all expert outputs: [B, E, C]
        expert_logits_tensor = torch.stack(all_expert_logits, dim=1)
        
        # Initialize output logits
        logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        # Hard routing - select highest weight expert for each sample
        _, indices = gate_weights.max(dim=1)
        
        # Send each sample to its assigned expert
        for i in range(self.num_experts):
            mask = (indices == i)
            if mask.sum() > 0:
                expert_inputs = x[mask]
                expert_outputs = self.experts[i](expert_inputs)
                logits[mask] = expert_outputs
        
        # Compute analysis metrics
        analysis_metrics = {}
        
        if len(all_expert_logits) >= 2:
            # Alignment analysis
            alignment = self.analysis_module.alignment_analyzer(
                all_expert_logits[0], all_expert_logits[1]
            )
            analysis_metrics['R_avg'] = alignment
            
            # Complementarity analysis
            complementarity = self.analysis_module.complementarity_analyzer(
                all_expert_logits[0], all_expert_logits[1]
            )
            analysis_metrics['U_avg'] = complementarity
            
            # Dominance analysis (if main logits provided)
            if main_logits is not None:
                dominance = self.analysis_module.dominance_analyzer(main_logits, logits)
                analysis_metrics['Dominance'] = dominance
            else:
                analysis_metrics['Dominance'] = torch.tensor(0.0, device=x.device)
        
        return logits, gate_weights, analysis_metrics
    
    def compute_analysis_losses(self, analysis_metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute analysis-based loss terms
        """
        losses = {}
        
        if 'R_avg' in analysis_metrics:
            R_avg = analysis_metrics['R_avg']
            losses['alignment_loss'] = torch.relu(R_avg - self.tau_R) * self.alignment_weight
        
        if 'Dominance' in analysis_metrics:
            Dominance = analysis_metrics['Dominance']
            losses['dominance_loss'] = torch.relu(self.tau_S - Dominance) * self.dominance_weight
        
        if 'U_avg' in analysis_metrics:
            U_avg = analysis_metrics['U_avg']
            losses['complementarity_loss'] = torch.relu(self.tau_U - U_avg) * self.complementarity_weight
        
        return losses
    
    def get_validation_info(self) -> Dict[str, Any]:
        """Get information about the validation MoE"""
        expert_infos = [expert.get_expert_info() for expert in self.experts]
        return {
            "num_experts": self.num_experts,
            "expert_type": "transformer_validation",
            "model_name": "deepseek-ai/deepseek-coder-6.7b",
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "experts": expert_infos,
            "analysis_features": ["alignment", "dominance", "complementarity"]
        } 