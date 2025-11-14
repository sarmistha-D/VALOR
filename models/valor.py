"""
VALOR (Validation-Aware Learner with Expert Routing) Module
Multimodal Mixture-of-Experts model for tweet-image classification
using Chain-of-Thought reasoning experts with enhanced analysis
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Importing VALOR components
from blocks.text_encoder import TextEncoder
from blocks.image_encoder import ImageEncoder
from blocks.cross_attention import CrossAttentionFusion
from blocks.mixture_of_experts import MixtureOfExperts
from blocks.validation_expert_block import ValidationMoE
from blocks.sas_module import LearnableSAS, SASLoss, create_sas_module
from blocks.meta_fusion import MetaFusionHead, create_meta_fusion_head

class VALOR(nn.Module):
    """
    VALOR: Validation-Aware Learner with Expert Routing
    
    Enhanced architecture with analysis features:
    - expert_type: "cot"
    - valor_num_experts: 4
    - top_k: 2
    - use_validation_moe: true
    - validation_expert_type: "transformer"
    - use_meta_fusion: true
    - use_film: false
    - use_learnable_sas: true
    - sas_type: "learnable"
    - fusion_strategy: "meta"
    - routing_strategy: "hard_top1"
    - loss_type: "focal"
    - expert_backbone: "deepseek-ai/deepseek-coder-6.7b"
    - cot_use_prompt: true
    - analysis_features: ["alignment", "dominance", "complementarity"]
    
    TASKS:
    - Aspect classification (6 classes)
    - Severity classification (4 classes)
    
    INPUT:
    - input_ids: [B, L]
    - attention_mask: [B, L]
    - image: [B, 3, 224, 224]
    
    OUTPUT:
    - aspect_logits: [B, 6]
    - severity_logits: [B, 4]
    - load_balance_loss: scalar
    - analysis_loss: scalar
    """
    
    def __init__(self, hidden_dim: int = 768, num_aspect_classes: int = 6, 
                 num_severity_classes: int = 4, num_experts: int = 4,
                 freeze_encoders: bool = True):
        super(VALOR, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_aspect_classes = num_aspect_classes
        self.num_severity_classes = num_severity_classes
        self.num_experts = num_experts
        
        # Encoder dimension (BERT/ViT default)
        encoder_dim = 768
        
        # Encoders
        self.text_encoder = TextEncoder(
            model_name="bert-base-uncased",
            freeze_bert=freeze_encoders,
            hidden_size=encoder_dim
        )
        
        self.image_encoder = ImageEncoder(
            model_name="google/vit-base-patch16-224",
            freeze_vit=freeze_encoders,
            hidden_size=encoder_dim
        )
        
        # Cross-attention fusion
        self.cross_fusion = CrossAttentionFusion(
            hidden_dim=encoder_dim,
            num_heads=8
        )
        
        # Projection layer if encoder_dim != hidden_dim
        if encoder_dim != hidden_dim:
            self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        else:
            self.encoder_proj = nn.Identity()
        
        # Learnable SAS module
        self.sas_module = create_sas_module(config=None)
        self.sas_loss = SASLoss(margin=0.3)
        
        # Meta-fusion heads
        self.aspect_meta_fusion = create_meta_fusion_head(
            config=None, num_classes=num_aspect_classes, task_type="aspect"
        )
        self.severity_meta_fusion = create_meta_fusion_head(
            config=None, num_classes=num_severity_classes, task_type="severity"
        )
        
        # Mixture of Experts for aspect classification - CoT with DeepSeek
        self.aspect_moe = MixtureOfExperts(
            hidden_dim=hidden_dim,
            num_classes=num_aspect_classes,
            num_experts=num_experts,
            top_k=2
        )
        
        # Mixture of Experts for severity classification - CoT with DeepSeek
        self.severity_moe = MixtureOfExperts(
            hidden_dim=hidden_dim,
            num_classes=num_severity_classes,
            num_experts=num_experts,
            top_k=2
        )
        
        # Validation MoE for aspect classification - Transformer with analysis
        self.aspect_validation_moe = ValidationMoE(
            hidden_dim=hidden_dim,
            num_classes=num_aspect_classes,
            num_experts=2
        )
        
        # Validation MoE for severity classification - Transformer with analysis
        self.severity_validation_moe = ValidationMoE(
            hidden_dim=hidden_dim,
            num_classes=num_severity_classes,
            num_experts=2
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass through VALOR architecture with analysis
        
        Args:
            input_ids: [B, L] — Token indices
            attention_mask: [B, L] — Attention mask
            image: [B, 3, 224, 224] — Image tensor
            
        Returns:
            Dictionary with aspect_logits, severity_logits, load_balance_loss, analysis_loss
        """
        batch_size = input_ids.size(0)
        
        # Encode text and image
        text_tokens, cls_text = self.text_encoder(input_ids, attention_mask)
        patch_tokens, cls_image = self.image_encoder(image)
        
        # Create image mask (all patches are valid for ViT)
        image_mask = torch.ones(batch_size, patch_tokens.size(1), device=image.device)
        
        # Cross-attention fusion
        fused = self.cross_fusion(
            text_embeddings=text_tokens,
            text_mask=attention_mask,
            image_embeddings=patch_tokens,
            image_mask=image_mask
        )
        
        # Project to target hidden dimension if needed
        fused = self.encoder_proj(fused)
        
        # Compute SAS score
        sas_score = self.sas_module(cls_text, cls_image)  # [B]
        
        # Get CLS representation for classification
        cls_representation = fused  # [B, hidden_dim]
        
        # Pass through Mixture of Experts (CoT)
        aspect_logits, aspect_lb_loss, aspect_gates = self.aspect_moe(cls_representation)
        severity_logits, severity_lb_loss, severity_gates = self.severity_moe(cls_representation)
        
        # Create per-sample entropy values for meta-fusion
        # Since we use hard routing, entropy would be 0, so we use a small constant
        routing_entropy = torch.ones(batch_size, device=cls_representation.device) * 0.1
        
        # Pass through Validation MoE (Transformer) with analysis
        aspect_validation_logits, aspect_validation_gates, aspect_analysis = self.aspect_validation_moe(
            cls_representation, main_logits=aspect_logits
        )
        severity_validation_logits, severity_validation_gates, severity_analysis = self.severity_validation_moe(
            cls_representation, main_logits=severity_logits
        )
        
        # Apply Enhanced Meta-Fusion with analysis metrics
        aspect_logits, aspect_fusion_weights = self.aspect_meta_fusion(
            prediction_logits=aspect_logits,
            validation_logits=aspect_validation_logits,
            sas_score=sas_score,
            routing_entropy=routing_entropy,
            analysis_metrics=aspect_analysis
        )
        
        severity_logits, severity_fusion_weights = self.severity_meta_fusion(
            prediction_logits=severity_logits,
            validation_logits=severity_validation_logits,
            sas_score=sas_score,
            routing_entropy=routing_entropy,
            analysis_metrics=severity_analysis
        )
        
        # Average load balance loss from both experts
        load_balance_loss = (aspect_lb_loss + severity_lb_loss) / 2
        
        # Compute SAS loss
        sas_loss = self.sas_loss(sas_score)
        
        # Compute analysis losses
        aspect_analysis_losses = self.aspect_validation_moe.compute_analysis_losses(aspect_analysis)
        severity_analysis_losses = self.severity_validation_moe.compute_analysis_losses(severity_analysis)
        
        # Combine analysis losses
        analysis_loss = (
            aspect_analysis_losses.get('alignment_loss', torch.tensor(0.0, device=cls_representation.device)) +
            aspect_analysis_losses.get('dominance_loss', torch.tensor(0.0, device=cls_representation.device)) +
            aspect_analysis_losses.get('complementarity_loss', torch.tensor(0.0, device=cls_representation.device)) +
            severity_analysis_losses.get('alignment_loss', torch.tensor(0.0, device=cls_representation.device)) +
            severity_analysis_losses.get('dominance_loss', torch.tensor(0.0, device=cls_representation.device)) +
            severity_analysis_losses.get('complementarity_loss', torch.tensor(0.0, device=cls_representation.device))
        )
        
        return {
            'aspect_logits': aspect_logits,
            'severity_logits': severity_logits,
            'aspect_validation_logits': aspect_validation_logits,
            'severity_validation_logits': severity_validation_logits,
            'load_balance_loss': load_balance_loss,
            'sas_loss': sas_loss,
            'analysis_loss': analysis_loss,
            'sas_score': sas_score,
            'aspect_gates': aspect_gates,
            'severity_gates': severity_gates,
            'aspect_fusion_weights': aspect_fusion_weights,
            'severity_fusion_weights': severity_fusion_weights,
            'aspect_analysis': aspect_analysis,
            'severity_analysis': severity_analysis
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_info = {
            'model_name': 'VALOR',
            'architecture': 'CoT-DeepSeek with transformer validation and analysis',
            'hidden_dim': self.hidden_dim,
            'num_aspect_classes': self.num_aspect_classes,
            'num_severity_classes': self.num_severity_classes,
            'num_experts': self.num_experts,
            'expert_type': 'cot',
            'expert_backbone': 'deepseek-ai/deepseek-coder-6.7b',
            'routing_strategy': 'hard_top1',
            'use_validation_moe': True,
            'validation_expert_type': 'transformer',
            'fusion_strategy': 'meta',
            'sas_type': 'learnable',
            'loss_type': 'focal',
            'analysis_features': ['alignment', 'dominance', 'complementarity'],
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params
        }
        
        return model_info 