import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Config:
    """
    VALOR Configurations - Best architecture from ablation study
    
    Architecture specifications:
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
    """
    
    # Dataset Configuration
    dataset_name: str = "CIVil.csv" # local path to the dataset or huggingface dataset name
    train_split: float = 0.8
    val_split: float = 0.1    
    test_split: float = 0.1       
    
    # Model Configuration
    text_model: str = "bert-base-uncased"
    image_model: str = "google/vit-base-patch16-224"
    hidden_size: int = 768
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # Text and Image Processing
    max_text_length: int = 512
    image_size: int = 224
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # VALOR Architecture - BEST CONFIGURATION (FIXED)
    valor_hidden_dim: int = 768
    valor_num_experts: int = 4
    valor_freeze_encoders: bool = True
    
    # Expert Configuration (FIXED)
    expert_type: str = "cot"
    top_k: int = 2
    
    # CoT Expert Configuration (FIXED)
    cot_model_name: str = "deepseek-ai/deepseek-coder-6.7b"
    cot_temperature: float = 0.5
    cot_top_k: int = 30
    cot_top_p: float = 0.9
    cot_max_tokens: int = 24
    cot_use_prompt: bool = True  # Use prompt-based CoT
    
    # Router Configuration (FIXED)
    router_noise_std: float = 0.05
    load_balance_weight: float = 0.05
    routing_strategy: str = "hard_top1"
    
    # Cross-attention Configuration
    cross_attention_heads: int = 12
    cross_attention_dropout: float = 0.15
    
    # FiLM Configuration (DISABLED)
    use_film: bool = False
    
    # SAS Configuration (FIXED)
    use_learnable_sas: bool = True
    sas_type: str = "learnable"
    sas_use_layer_norm: bool = True
    sas_dropout: float = 0.1
    lambda_sas: float = 0.1
    
    # Meta-Fusion Configuration (FIXED)
    use_meta_fusion: bool = True
    meta_fusion_type: str = "standard"
    fusion_strategy: str = "meta"
    meta_fusion_dropout: float = 0.1
    meta_fusion_hidden_dim: int = 768
    
    # Validation MoE Configuration (FIXED)
    use_validation_moe: bool = True
    validation_expert_type: str = "transformer"
    n_validation_experts: int = 2
    validation_dropout: float = 0.1
    
    # Analysis Configuration (NEW)
    use_analysis_features: bool = True
    analysis_features: List[str] = field(default_factory=lambda: ["alignment", "dominance", "complementarity"])
    lambda_analysis: float = 0.1
    alignment_threshold: float = 0.3
    dominance_threshold: float = 0.5
    complementarity_threshold: float = 1.5
    alignment_weight: float = 0.1
    dominance_weight: float = 0.1
    complementarity_weight: float = 0.1
    
    # Baseline Models Configuration (NEW)
    fine_tune_baselines: bool = True  # Whether to fine-tune baseline models
    fine_tune_epochs: int = 10  # Number of epochs for fine-tuning
    fine_tune_lr: float = 5e-5  # Learning rate for fine-tuning
    fine_tune_batch_size: int = 8  # Batch size for fine-tuning
    fine_tune_weight_decay: float = 0.01  # Weight decay for fine-tuning
    fine_tune_freeze_backbone: bool = True  # Whether to freeze the backbone model during fine-tuning
    baseline_specific_heads: bool = True  # Whether to use model-specific classification heads
    baseline_custom_pooling: bool = True  # Whether to use custom pooling for feature extraction
    
    # Training Configuration
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Optimizer Configuration
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler Configuration
    scheduler: str = "cosine_with_restarts"
    t_mult: float = 2.0
    eta_min: float = 1e-6
    
    # Evaluation Configuration
    eval_interval: int = 1
    save_interval: int = 5
    patience: int = 15
    
    # Loss Configuration
    loss_type: str = "focal"  # Fixed to focal loss
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    aux_loss_weight: float = 0.3
    label_smoothing: float = 0.15
    
    # Hyperparameter Tuning (keeping for non-architecture parameters)
    use_hyperparameter_tuning: bool = True
    tuning_trials: int = 50  # Reduced from 100
    tuning_timeout: int = 7200
    tuning_parallel_trials: int = 4
    
    # Optuna-specific settings
    optuna_study_name: str = "valor_optimization"
    optuna_storage: str = "sqlite:///tuning_results/optuna_study.db"
    optuna_sampler: str = "tpe"
    optuna_pruner: str = "median"
    
    # Parameter search spaces for Optuna (architecture parameters removed)
    param_search_spaces: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': {'type': 'float', 'low': 1e-6, 'high': 1e-3, 'log': True},
        'batch_size': {'type': 'categorical', 'choices': [2, 4, 8, 16]},
        'weight_decay': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
        'aux_loss_weight': {'type': 'float', 'low': 0.01, 'high': 0.5},
    })
    
    # Resource management
    use_amp: bool = True  # Automatic Mixed Precision
    max_gpu_memory_usage: float = 0.8
    min_disk_space_gb: float = 1.0
    
    # Output Configuration
    output_dir: str = "outputs"
    model_dir: str = "models"
    tuning_dir: str = "tuning_results"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Class Definitions
    aspect_classes: List[str] = field(default_factory=lambda: [
        "Software", "Hardware", "Packaging", "Price", "Service", "Quality"
    ])
    severity_classes: List[str] = field(default_factory=lambda: [
        "No Explicit Reproach", "Disapproval", "Blame", "Accusation"
    ])
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tuning_dir, exist_ok=True)

# Global config instance
config = Config()