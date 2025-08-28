# MultiNLI Bias Research - Comprehensive Bias Mitigation Methods

This repository contains implementations of multiple bias mitigation methods for the MultiNLI dataset, including the ERM baseline and three advanced debiasing techniques.

## Overview

The implementation focuses on identifying and mitigating spurious correlations in natural language inference tasks, specifically using negation words as bias indicators in the MultiNLI dataset.

## Methods Implemented

### 1. ERM Baseline (Empirical Risk Minimization)
Standard training with cross-entropy loss - serves as the baseline for comparison.

### 2. Last Layer Retraining (LLR)
- Phase 1: Train full model with ERM
- Phase 2: Freeze BERT encoder, retrain only classifier with group-balanced sampling

### 3. Just Train Twice (JTT)
- Phase 1: Train with ERM to identify worst-performing examples
- Phase 2: Retrain from scratch with upweighting of worst examples

### 4. SELF (Self-adaptive Training)
- Adaptive loss weighting during training
- Dynamic reweighting based on loss trends and group performance
- Focus on improving worst-performing groups

## Features

- **BERT-base-uncased** foundation model
- **Spurious correlation detection** based on negation words
- **Group-wise evaluation** for bias analysis
- **Apple Silicon (MPS) GPU support** for faster training
- **Comprehensive comparison** of all methods
- **Checkpoint saving** for worst-case model preservation
- **Resume training** from interrupted sessions

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Individual Methods

**ERM Baseline:**
```bash
python multinli_erm_baseline.py --epochs 5
```

**ERM with checkpoint saving:**
```bash
python multinli_erm_baseline.py --epochs 5 --save_checkpoints
```

**Resume from checkpoint:**
```bash
python multinli_erm_baseline.py --resume_from model_checkpoints/erm_model_best_checkpoint.pt --save_checkpoints
```

**Last Layer Retraining:**
```bash
python last_layer_retraining.py --erm_epochs 3 --llr_epochs 5
```

**Just Train Twice:**
```bash
python just_train_twice.py --phase1_epochs 3 --phase2_epochs 5 --worst_fraction 0.2
```

**SELF (Self-adaptive Training):**
```bash
python self_adaptive_training.py --epochs 5
```

**SELF with checkpoints:**
```bash
python self_adaptive_training.py --epochs 5 --save_checkpoints
```

### Compare All Methods

**Run all methods and compare (takes several hours):**
```bash
python compare_methods.py --run_all --epochs 3
```

**Compare existing results only:**
```bash
python compare_methods.py --compare_only
```

## Checkpoint Functionality

### Automatic Best Model Saving
All methods support checkpoint saving that automatically preserves the model with the best worst-group accuracy:

```bash
# Enable checkpoint saving (saves best model automatically)
python multinli_erm_baseline.py --save_checkpoints --epochs 10

# Resume from best checkpoint
python multinli_erm_baseline.py --resume_from multinli_erm_model/checkpoints/erm_model_best_checkpoint.pt --save_checkpoints
```

### What Gets Saved
- **Model state**: Complete model weights
- **Optimizer state**: For exact training resume
- **Scheduler state**: Learning rate schedule
- **Training metrics**: Loss and accuracy history
- **Best performance**: Tracks best worst-group accuracy
- **Group statistics**: Per-group performance over time

### Checkpoint Files
```
model_name/checkpoints/
├── erm_model_best_checkpoint.pt       # Best model (highest worst-group accuracy)
├── erm_model_training_history.json    # Training metrics and progress
└── erm_model_epoch_N_checkpoint.pt    # Periodic checkpoints (if enabled)
```

### Quick Demo
```bash
# Run with checkpoint saving
python run_erm_with_checkpoints.py

# Demo resume functionality  
python run_erm_with_checkpoints.py --demo_resume
```

## Method Details

### Spurious Groups
All methods identify 6 spurious correlation groups based on label + negation presence:
- **contradiction+no_negation**: Contradiction without negation words
- **contradiction+negation**: Contradiction with negation words  
- **entailment+no_negation**: Entailment without negation words
- **entailment+negation**: Entailment with negation words
- **neutral+no_negation**: Neutral without negation words
- **neutral+negation**: Neutral with negation words

### Hyperparameters
- **Learning rate**: 1e-5 (10x higher for LLR classifier phase)
- **Batch size**: 16
- **Weight decay**: 1e-4
- **Optimizer**: AdamW with linear learning rate schedule

## Expected Performance

| Method | Overall Accuracy | Worst Group Accuracy | Improvement over ERM |
|--------|------------------|---------------------|---------------------|
| ERM Baseline | ~80-85% | ~65-75% | baseline |
| Last Layer Retraining | ~80-85% | ~70-80% | +5-10% |
| Just Train Twice | ~75-85% | ~70-80% | +5-10% |
| SELF | ~80-85% | ~70-85% | +5-15% |

*Note: Actual results may vary based on random initialization and exact hyperparameters*

## Hardware Requirements

- **Recommended**: Apple Silicon Mac (M1/M2) with MPS support
- **Alternative**: NVIDIA GPU with CUDA support  
- **Minimum**: CPU (significantly slower)

## File Structure

```
├── multinli_erm_baseline.py      # ERM baseline implementation
├── last_layer_retraining.py      # Last Layer Retraining method
├── just_train_twice.py           # Just Train Twice method
├── self_adaptive_training.py     # SELF adaptive training method
├── compare_methods.py            # Compare all methods
├── checkpoint_utils.py           # Checkpoint management utilities
├── run_erm_baseline.py          # ERM execution script
├── run_erm_with_checkpoints.py  # ERM with checkpoint demo
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Dependencies

- torch >= 2.0.0 (with MPS support for Apple Silicon)
- transformers >= 4.20.0
- datasets >= 2.0.0
- scikit-learn >= 1.0.0
- numpy >= 1.20.0
- pandas >= 1.3.0
- tqdm >= 4.60.0

## Usage Examples

### Custom Training Parameters

```bash
# ERM with custom parameters
python multinli_erm_baseline.py --batch_size 32 --lr 2e-5 --epochs 3

# LLR with longer retraining phase
python last_layer_retraining.py --erm_epochs 2 --llr_epochs 8

# JTT with higher upweighting
python just_train_twice.py --upweight_factor 20.0 --worst_fraction 0.3

# SELF with custom parameters
python self_adaptive_training.py --epochs 8 --lr 1e-5
```

### Batch Comparison

```bash
# Quick comparison with reduced epochs
python compare_methods.py --run_all --epochs 2 --batch_size 32
```

## Results Analysis

Each method saves detailed results including:
- Overall accuracy
- Worst group accuracy  
- Per-group accuracies
- Training hyperparameters
- Method-specific metrics

Results are saved as JSON files and can be compared using the comparison script.

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{kirichenko2022deep,
  title={Deep Feature Reweighting},
  author={Kirichenko, Polina and others},
  journal={arXiv preprint arXiv:2204.02937v2},
  year={2022}
}
```

- **GPU**: Recommended (CUDA-compatible)
- **RAM**: 8GB+ recommended
- **Storage**: ~2GB for model and dataset cache
- **Training time**: ~2-4 hours on GPU, longer on CPU

## Notes

- This implements the baseline method only (ERM)
- For the DFR method, additional implementation would be needed
- The spurious correlation identification follows the paper's methodology
- Results may vary slightly due to random initialization

## Citation

```bibtex
@inproceedings{kirichenko2023deep,
  title={Deep Feature Reweighting},
  author={Kirichenko, Polina and Izmailov, Pavel and Wilson, Andrew Gordon},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
