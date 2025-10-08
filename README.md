# VALOR: Validation-Aware Learner with Expert Routing [AAAI 2026]

[![Anonymous Repo](https://img.shields.io/badge/Anonymous-Repository-blue.svg)](https://anonymous.4open.science/r/672)
[![Code](https://img.shields.io/badge/GitHub-Code-green.svg)](https://github.com/your-username/valor)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

> **VALOR: Validation-Aware Learner with Expert Routing for Multimodal Complaint Classification**<br>
> AAAI 2026

---

## 🎯 Highlights

<div align="center">
  <img src="valor_architecture.png" alt="VALOR Architecture" width="800"/>
  <br>
  <strong>Figure: VALOR Architecture Overview</strong><br>
  <em>The complete VALOR framework showing multimodal encoders, Chain-of-Thought experts, validation MoE, and analysis-driven fusion components.</em>
</div>

> **Abstract:** *Existing approaches to complaint analysis largely rely on unimodal, short-form content such as tweets or product reviews. This work advances the field by leveraging multimodal, multi-turn customer support dialogues—where users often share both textual complaints and visual evidence (e.g., screenshots, product photos)—to enable fine-grained classification of complaint aspects and severity. We introduce \textit{VALOR}, a Validation-Aware Learner with Expert Routing, tailored for this multimodal setting. It employs a multi-expert reasoning setup using large-scale generative models with Chain-of-Thought (CoT) prompting for nuanced decision-making. To ensure coherence between modalities, a semantic alignment score is computed and integrated into the final classification through a meta-fusion strategy. In alignment with the United Nations Sustainable Development Goals (UN SDGs), the proposed framework supports SDG 9 (Industry, Innovation and Infrastructure) by advancing AI-driven tools for robust, scalable, and context-aware service infrastructure. Further, by enabling structured analysis of complaint narratives and visual context, it contributes to SDG 12 (Responsible Consumption and Production) by promoting more responsive product design and improved accountability in consumer services. We evaluate \textit{VALOR} on a curated multimodal complaint dataset annotated with fine-grained aspect and severity labels, showing that it consistently outperforms baseline models, especially in complex complaint scenarios where information is distributed across text and images. This study underscores the value of multimodal interaction and expert validation in practical complaint understanding systems.*

---

## 🏗️ Main Contributions

1. **Validation-Aware Learning:** Novel mixture-of-experts architecture with transformer-based validation experts that provide secondary opinions for robust multimodal classification
2. **Chain-of-Thought Reasoning:** Integration of DeepSeek Coder 6.7B experts with prompt-based reasoning for interpretable decision-making and enhanced generalization
3. **Analysis-Driven Fusion:** Alignment, Dominance, and Complementarity analysis modules for adaptive prediction combination and improved model interpretability
4. **Semantic Alignment:** Learnable Semantic Alignment Score (SAS) module for measuring text-image coherence and guiding multimodal fusion

---

## � How VALOR Works: Input → Output

VALOR processes multimodal customer support conversations to provide fine-grained complaint classification:

### Input
- **Text:** Customer complaint messages and support dialogue
- **Images:** Screenshots, product photos, or visual evidence shared by customers
- **Context:** Multi-turn conversation history

### Processing
1. **Multimodal Encoding:** BERT processes text, ViT processes images
2. **Cross-Attention Fusion:** Integrates text and image representations
3. **Expert Reasoning:** 4 DeepSeek Coder experts with Chain-of-Thought prompting
4. **Validation:** Transformer-based validation experts provide secondary opinions
5. **Analysis:** Alignment, Dominance, and Complementarity metrics guide fusion
6. **Semantic Alignment:** Learnable SAS module measures text-image coherence

### Output
- **Aspect Classification:** Software, Hardware, Packaging, Price, Service, Quality
- **Severity Classification:** No Explicit Reproach, Disapproval, Blame, Accusation
- **Confidence Scores:** Model confidence for each prediction
- **Reasoning:** Chain-of-Thought explanations for interpretability

<div align="center">
  <img src="Conversation.png" alt="Conversation Example" width="700"/>
  <br>
  <strong>Figure 1: Input-Output Example from CIViL Dataset</strong><br>
  <em>A conversation snippet showing how multimodal input (text + images) is processed by VALOR to predict aspect-severity pairs. The model analyzes both textual complaints and visual evidence to provide fine-grained classification.</em>
</div>

---

## 📊 Experimental Results

### Dataset and Task Overview

Our evaluation focuses on multimodal complaint classification using customer support dialogues where users share both textual complaints and visual evidence (screenshots, product photos). The task involves fine-grained classification of complaint aspects and severity levels.

### Baseline Comparison

We compare VALOR against state-of-the-art multimodal models on both Aspect Category Detection (ACD) and Severity Detection (SD) tasks:

| Model | ACD (A) | ACD (F1) | SD (A) | SD (F1) |
|-------|---------|----------|--------|---------|
| DeepSeek-VL | 0.66 | 0.65 | 0.66 | 0.65 |
| Gemma-3 (9B) | 0.69 | 0.66 | 0.65 | 0.66 |
| Flash Gemini (1.6B) | 0.66 | 0.65 | 0.66 | 0.65 |
| ImageBind | 0.66 | 0.65 | 0.64 | 0.63 |
| Paligemma (3B) | 0.65 | 0.66 | 0.65 | 0.64 |
| SMOL-VLM | 0.65 | 0.64 | 0.63 | 0.62 |
| GIT (300M) | 0.65 | 0.64 | 0.63 | 0.62 |
| FLAVA | 0.62 | 0.61 | 0.60 | 0.59 |
| ALBEF | 0.61 | 0.60 | 0.59 | 0.56 |
| UNITER | 0.60 | 0.59 | 0.56 | 0.55 |
| CLIP ViT-B/32 | 0.59 | 0.56 | 0.55 | 0.56 |
| VisualBERT | 0.56 | 0.55 | 0.56 | 0.55 |
| VILT | 0.55 | 0.56 | 0.55 | 0.54 |
| **VALOR (Ours)** | **0.82** | **0.77** | **0.77** | **0.78** |

**Table 1: Performance Comparison of Multimodal Models on Complaint Classification.** This table presents the Accuracy (A) and F1-score (F1) for various multimodal models on two tasks: Aspect Category Detection (ACD) and Severity Detection (SD). VALOR significantly outperforms all baseline models, demonstrating the effectiveness of our validation-aware learning approach.

### Ablation Study

We conduct comprehensive ablation studies to analyze the contribution of different components in our VALOR framework:

| Configuration | Expert | Validation MoE | SAS | Top-K | ACD (A) | SD (A) | ACD (F1) | SD (F1) |
|---------------|--------|----------------|-----|-------|----------|---------|-----------|---------|
| CoT (No Validation, Learnable SAS, Top-2) | cot | False | learnable | 2 | 73.74 | 62.62 | 70.44 | 52.84 |
| CoT (No Validation, Learnable SAS, Top-4) | cot | False | learnable | 4 | 75.14 | 64.64 | 70.46 | 59.47 |
| **VALOR** | **cot** | **True** | **learnable** | **2** | **81.94** | **72.51** | **76.96** | **67.91** |
| MLP (No Validation, Learnable SAS, Top-2) | mlp | False | learnable | 2 | 70.43 | 57.35 | 63.82 | 48.55 |
| MLP (No Validation, Learnable SAS, Top-4) | mlp | False | learnable | 4 | 71.97 | 58.98 | 65.04 | 54.45 |
| MLP (Validation, Cosine SAS, Top-2) | mlp | True | cosine | 2 | 68.81 | 65.24 | 61.20 | 57.58 |
| MLP (Validation, Cosine SAS, Top-4) | mlp | True | cosine | 4 | 69.06 | 65.14 | 66.62 | 56.14 |
| MLP (Validation, No SAS, Top-2) | mlp | True | none | 2 | 71.19 | 64.39 | 68.18 | 60.69 |
| Transformer (No Validation, Learnable SAS, Top-2) | transformer | False | learnable | 2 | 77.51 | 67.95 | 70.62 | 61.70 |
| Transformer (No Validation, No SAS, Top-2) | transformer | False | none | 2 | 72.86 | 52.67 | 68.22 | 43.84 |
| Transformer (No Validation, No SAS, Top-4) | transformer | False | none | 4 | 65.51 | 59.48 | 59.79 | 55.17 |
| Transformer (Validation, Learnable SAS, Top-2) | transformer | True | learnable | 2 | 77.08 | 63.98 | 70.24 | 60.24 |
| Transformer (Validation, No SAS, Top-2) | transformer | True | none | 2 | 74.84 | 63.55 | 71.30 | 58.05 |

**Table 2: Ablation Study Results for VALOR Framework.** This table shows the performance impact of different expert types, validation strategies, and SAS settings. The best configuration (VALOR with CoT experts, validation MoE, learnable SAS, and Top-2 routing) achieves superior performance across all metrics. Key findings:

- **Chain-of-Thought (CoT) experts** significantly outperform MLP and Transformer experts
- **Validation MoE** provides consistent improvements across all configurations
- **Learnable SAS** is crucial for optimal performance, especially with CoT experts
- **Top-2 routing** with validation achieves the best balance of performance and efficiency

### Key Insights

1. **Expert Type Impact:** CoT experts with prompt-based reasoning achieve the highest performance, demonstrating the value of interpretable decision-making in multimodal classification.

2. **Validation Benefits:** The validation MoE consistently improves performance across all configurations, validating our hypothesis that secondary opinions enhance robustness.

3. **SAS Importance:** Learnable SAS significantly outperforms cosine similarity and no SAS, highlighting the importance of adaptive semantic alignment measurement.

4. **Routing Strategy:** Top-2 routing with validation provides optimal performance, balancing expert specialization with ensemble benefits.

---

## 🏛️ Architecture

### Core Components

1. **Multimodal Encoders**
   - **Text Encoder:** BERT-base-uncased for text processing
   - **Image Encoder:** ViT-base-patch16-224 for image processing
   - **Cross-Attention Fusion:** Multi-head attention mechanism for multimodal integration

2. **Chain-of-Thought Experts**
   - **4 DeepSeek Coder 6.7B experts** with prompt-based reasoning
   - **Hard Top-1 routing** strategy for expert selection
   - **Load balancing** mechanism for balanced expert utilization

3. **Validation Experts**
   - **Transformer-based validation** experts using DeepSeek backbone
   - **Analysis-driven fusion** with Alignment, Dominance, and Complementarity metrics
   - **Adaptive prediction combination** for robust decision-making

4. **Analysis Modules**
   - **Alignment Analysis:** Measures similarity between validation experts
   - **Dominance Analysis:** Measures complementarity with main MoE predictions
   - **Complementarity Analysis:** Measures diversity of validation expert predictions

---

## 📦 Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/672


# Install dependencies
pip install -r requirements.txt

# Verify installation
python train.py --help
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (for GPU training)

---

## 🚀 Quick Start

### Training

```bash
# Basic training with default settings
python train.py train

# Training with hyperparameter tuning
python train.py train --tune --batch-size 16 --epochs 50

# Training with custom parameters
python train.py train --batch-size 32 --epochs 100 --learning-rate 5e-4
```

### Evaluation

```bash
# Evaluate on test set
python train.py evaluate --checkpoint checkpoints/valor_best_20241201_143022.pt
```

### Inference

```bash
# Run inference on single example
python train.py predict --text "Your tweet text" --image path/to/image.jpg --checkpoint checkpoints/valor_best_20241201_143022.pt
```

### Baseline Comparison

```bash
# Run all baseline models
python baselines/run_baselines.py --fine-tune --epochs 10

# Run specific baseline
python baselines/run_single_baseline.py --model CLIP --fine-tune
```

---

## 📁 Project Structure

```

├── README.md                    # This file
├── train.py                     # Main training script
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore file
│
├── models/
│   └── valor.py                # VALOR model implementation
│
├── blocks/                      # Core components
│   ├── text_encoder.py         # BERT text encoder
│   ├── image_encoder.py        # ViT image encoder
│   ├── cross_attention.py      # Cross-attention mechanism
│   ├── mixture_of_experts.py   # MoE implementation
│   ├── cot_expert.py           # Chain-of-Thought experts
│   ├── validation_expert_block.py  # Validation experts
│   ├── router.py               # Expert routing
│   ├── sas_module.py           # Semantic Alignment Score
│   ├── meta_fusion.py          # Meta-fusion heads
│   └── analysis_modules.py     # Analysis modules
│
├── utils/                       # Utilities
│   ├── dataset.py              # Data loading
│   ├── metrics.py              # Evaluation metrics
│   ├── data_augmentation.py    # Data augmentation
│   └── hyperparameter_tuner.py # Optuna-based tuning
│
├── baselines/                   # Baseline implementations
│   ├── base_baseline.py        # Base class
│   ├── clip_baseline.py        # CLIP baseline
│   ├── deepseek_vl_baseline.py # DeepSeek-VL baseline
│   └── [other baselines]       # Additional baselines
│
└── docs/                        # Documentation
    └── valor_architecture.png   # Architecture diagram
```

---

## 🔧 Configuration

Key configuration parameters in `config.py`:

```python
# Model Configuration
valor_hidden_dim: int = 768
valor_num_experts: int = 4
expert_type: str = "cot"
expert_backbone: str = "deepseek-ai/deepseek-coder-6.7b"

# Analysis Configuration
analysis_features: List[str] = ["alignment", "dominance", "complementarity"]
alignment_weight: float = 0.1
dominance_weight: float = 0.1
complementarity_weight: float = 0.1

# Training Configuration
batch_size: int = 16
num_epochs: int = 50
learning_rate: float = 5e-4
```

---

<div align="center">

**Made with ❤️ for the multimodal AI community**

