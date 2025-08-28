# ERM Baseline for MultiNLI Dataset

This repository reproduces the **ERM (Empirical Risk Minimization) baseline** from the paper "Deep Feature Reweighting" (Section 6) on the MultiNLI dataset.

## Paper Details

- **Paper**: Deep Feature Reweighting (arXiv:2204.02937v2)
- **Section**: 6 (Feature Reweighting Improves Robustness)
- **Dataset**: MultiNLI (Multi-Genre Natural Language Inference)
- **Baseline Model**: BERT-base-uncased with standard ERM training

## What This Reproduces

The ERM baseline represents **conventional training without any procedures for improving worst-group accuracies**. This is the standard BERT-based approach that the paper uses as a baseline to compare against their DFR method.

### Exact Paper Specifications:

- **Model**: `BertForSequenceClassification.from_pretrained('bert-base-uncased')`
- **Pre-training**: Book Corpus and English Wikipedia data
- **Optimizer**: AdamW with linear learning rate annealing
- **Learning rate**: 1e-5
- **Batch size**: 16
- **Weight decay**: 1e-4
- **Epochs**: 5
- **No early stopping**

### Spurious Correlations in MultiNLI:

The MultiNLI dataset has spurious correlations where negation words ("no", "never", etc.) are correlated with the contradiction class. The model learns to rely on these spurious features, leading to poor worst-group accuracy.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

The script automatically downloads the MultiNLI dataset from HuggingFace Datasets:
- **No manual download required** - the dataset will be downloaded automatically
- Training set: ~392K examples
- Validation set: ~20K examples (validation_matched)

## Usage

### Quick Start (Recommended)

```bash
python run_erm_baseline.py
```

This runs the ERM baseline with the exact hyperparameters from the paper.

### Manual Run with Custom Parameters

```bash
python multinli_erm_baseline.py --batch_size 16 --lr 1e-5 --epochs 5 --weight_decay 1e-4
```

### Available Arguments

- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-5)
- `--epochs`: Number of epochs (default: 5)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--save_model`: Path to save model (default: 'multinli_erm_model')

## Expected Results

Based on the paper, the ERM baseline should achieve:
- **Overall accuracy**: ~80-85%
- **Worst group accuracy**: ~65-75% (significantly lower due to spurious correlations)

The model will show poor performance on minority groups where the spurious correlation doesn't hold (e.g., contradictions without negation words).

## Group Definitions

The code identifies 6 groups based on label and negation presence:
- **Group 0**: Contradiction without negation (minority group)
- **Group 1**: Contradiction with negation (majority group)
- **Group 2**: Entailment without negation
- **Group 3**: Entailment with negation
- **Group 4**: Neutral without negation
- **Group 5**: Neutral with negation

## Output Files

After training, the following files will be created:
- `multinli_erm_baseline_model/`: Trained BERT model and tokenizer
- `multinli_erm_baseline_model_results.json`: Detailed results including group-wise accuracies

## Hardware Requirements

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
