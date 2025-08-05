"""
CoTExpert Module for VALOR Framework
Chain-of-Thought reasoning expert using DeepSeek coder model
Performs prompt-based reasoning before classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

class CoTExpert(nn.Module):
    """
    Chain-of-Thought Expert for task-specific classification
    
    Uses DeepSeek model to perform reasoning before classification.
    Implements prompt-based reasoning approach.
    
    INPUT:
    - x: [batch_size, hidden_dim] - fused representation
    
    OUTPUT:
    - logits: [batch_size, num_classes] - classification logits
    """
    
    def __init__(
        self, 
        hidden_dim: int = 768, 
        num_classes: int = 5,
        model_name: str = "deepseek-ai/deepseek-coder-6.7b",
        max_tokens: int = 24,
        temperature: float = 0.5,
        top_k: int = 30,
        top_p: float = 0.9,
        expert_id: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(CoTExpert, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.expert_id = expert_id
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load DeepSeek coder model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        self.hidden_size = self.model.config.hidden_size
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projection layers
        self.input_projection = nn.Linear(hidden_dim, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Expert-specific parameters
        self.expert_scale = nn.Parameter(torch.ones(1))
        self.expert_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # CoT prompt template
        self.cot_prompt = """Analyze this multimodal input about a customer complaint from text and image. Use Chain of Thought reasoning:
Step 1: Identify the key features in the input.
Step 2: Consider what aspect the complaint is about (Software, Hardware, Packaging, Price, Service, or Quality).
Step 3: Determine the severity level (No Explicit Reproach, Disapproval, Blame, or Accusation).
Step 4: Make a classification decision based on your reasoning.
Reasoning:"""
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the CoT expert
        
        Args:
            x: Input features [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            mask: Attention mask (optional)
            
        Returns:
            logits: [batch_size, num_classes] or [batch_size, seq_len, num_classes]
        """
        # Handle different input shapes
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, hidden_dim = original_shape
            # Process each sequence position
            x = x.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
            reshape_output = True
        else:
            batch_size = x.size(0)
            reshape_output = False
        
        # Apply expert-specific transformation
        x = x * self.expert_scale + self.expert_bias
        
        # Project to model's hidden size
        projected = self.input_projection(x)
        
        # Prepare prompts for each sample
        prompts = [self.cot_prompt] * x.size(0)
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate reasoning chains
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            
            # Extract hidden states from the model
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Get last layer, last token hidden states
                hidden_states = outputs.hidden_states[-1][-1]  # [batch_size, hidden_size]
            else:
                # Fallback: run forward pass to get hidden states
                model_outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = model_outputs.hidden_states[-1][:, -1, :]
        
        # Combine projected input with generated hidden states
        combined = projected + hidden_states[:x.size(0)]  # Ensure batch size matches
        
        # Classify
        logits = self.classifier(combined)
        
        # Reshape output if needed
        if reshape_output:
            logits = logits.view(batch_size, seq_len, -1)
        
        return logits
    
    def get_expert_info(self) -> Dict[str, Any]:
        """Get information about this expert"""
        return {
            "expert_id": self.expert_id,
            "expert_type": "cot",
            "model_name": self.model_name,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "uses_prompt": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        } 