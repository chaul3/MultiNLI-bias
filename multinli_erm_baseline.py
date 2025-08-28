"""
ERM Baseline for MultiNLI Dataset
Reproduces the baseline method from "Deep Feature Reweighting" paper (Section 6)

This implements standard Empirical Risk Minimization (ERM) using BERT-base-uncased
on the MultiNLI dataset with spurious correlations based on negation words.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import argparse
import os
import json
from checkpoint_utils import CheckpointManager, find_latest_checkpoint

class MultiNLIDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Reduced from 512 for speed
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        premise, hypothesis = self.texts[idx]
        
        # Tokenize the premise and hypothesis pair
        encoding = self.tokenizer(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def identify_spurious_groups(examples):
    """
    Identify spurious correlation groups based on negation words
    Following the paper's methodology for MultiNLI
    """
    # Negation words that are spuriously correlated with contradiction
    negation_words = [
        'no', 'not', 'never', 'none', 'nothing', 'nowhere', 'nobody', 
        'neither', 'nor', 'without', "n't", "don't", "won't", "can't",
        "couldn't", "shouldn't", "wouldn't", "isn't", "aren't", "wasn't", "weren't"
    ]
    
    groups = []
    
    for premise, hypothesis, label in zip(examples['premise'], examples['hypothesis'], examples['label']):
        # Check if hypothesis contains negation words
        hypothesis_lower = hypothesis.lower()
        has_negation = any(neg_word in hypothesis_lower for neg_word in negation_words)
        
        # Group definition based on label and negation presence
        if label == 0 and not has_negation:  # contradiction without negation
            group = "contradiction+no_negation"
        elif label == 0 and has_negation:    # contradiction with negation  
            group = "contradiction+negation"
        elif label == 1 and not has_negation:  # entailment without negation
            group = "entailment+no_negation"
        elif label == 1 and has_negation:    # entailment with negation
            group = "entailment+negation"
        else:  # neutral cases
            group = "neutral+no_negation" if not has_negation else "neutral+negation"
            
        groups.append(group)
    
    return groups

def load_multinli_data():
    """Load and preprocess MultiNLI dataset"""
    print("Loading MultiNLI dataset...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("multi_nli")
    
    # Process training data
    train_texts = list(zip(dataset['train']['premise'], dataset['train']['hypothesis']))
    train_labels = dataset['train']['label']
    train_groups = identify_spurious_groups(dataset['train'])
    
    # Process validation data  
    val_texts = list(zip(dataset['validation_matched']['premise'], dataset['validation_matched']['hypothesis']))
    val_labels = dataset['validation_matched']['label']
    val_groups = identify_spurious_groups(dataset['validation_matched'])
    
    return (train_texts, train_labels, train_groups), (val_texts, val_labels, val_groups)

class ERMTrainer:
    def __init__(self, model, tokenizer, device, lr=1e-5, batch_size=16, weight_decay=1e-4, save_dir=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # Initialize checkpoint manager if save_dir provided
        self.checkpoint_manager = None
        if save_dir:
            self.checkpoint_manager = CheckpointManager(
                save_dir=os.path.join(save_dir, 'checkpoints'),
                model_name='erm_model',
                save_best_only=True  # Only save when worst-group accuracy improves
            )
        
    def train(self, train_data, val_data, num_epochs=5):
        """
        Train the BERT model using ERM (standard training)
        Following the paper's hyperparameters:
        - AdamW optimizer
        - Learning rate: 1e-5
        - Batch size: 16 
        - Weight decay: 1e-4
        - 5 epochs
        - Linear learning rate annealing
        """
        train_texts, train_labels, train_groups = train_data
        val_texts, val_labels, val_groups = val_data
        
        # Create datasets
        train_dataset = MultiNLIDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = MultiNLIDataset(val_texts, val_labels, self.tokenizer)
        
        # Create data loaders with optimized settings
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,  # Parallel data loading
            pin_memory=True  # Faster GPU transfer
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            print(f"Average training loss: {avg_loss:.4f}")
            
            # Validation
            val_metrics = self.evaluate(val_data)
            print(f"Validation - Overall Acc: {val_metrics['overall_accuracy']:.4f}, "
                  f"Worst Group Acc: {val_metrics['worst_group_accuracy']:.4f}")
            
            # Save checkpoint if checkpoint manager is available
            if self.checkpoint_manager:
                saved_path = self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss=avg_loss,
                    val_metrics=val_metrics,
                    extra_info={'method': 'ERM_Baseline'}
                )
                if saved_path:
                    print(f"✅ Checkpoint saved: {os.path.basename(saved_path)}")
        
        # Print training summary
        if self.checkpoint_manager:
            self.checkpoint_manager.print_training_summary()
    
    def evaluate(self, eval_data):
        """Evaluate model and compute group-wise metrics"""
        eval_texts, eval_labels, eval_groups = eval_data
        
        eval_dataset = MultiNLIDataset(eval_texts, eval_labels, self.tokenizer)
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute overall accuracy
        overall_accuracy = accuracy_score(all_labels, all_predictions)
        
        # Compute group-wise accuracies
        group_accuracies = {}
        for group_id in set(eval_groups):
            group_mask = [g == group_id for g in eval_groups]
            group_labels = [all_labels[i] for i in range(len(all_labels)) if group_mask[i]]
            group_preds = [all_predictions[i] for i in range(len(all_predictions)) if group_mask[i]]
            
            if len(group_labels) > 0:
                group_acc = accuracy_score(group_labels, group_preds)
                group_accuracies[group_id] = group_acc
        
        worst_group_accuracy = min(group_accuracies.values()) if group_accuracies else 0.0
        
        return {
            'overall_accuracy': overall_accuracy,
            'worst_group_accuracy': worst_group_accuracy,
            'group_accuracies': group_accuracies,
            'predictions': all_predictions,
            'labels': all_labels
        }

def main():
    parser = argparse.ArgumentParser(description='ERM Baseline for MultiNLI')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_model', type=str, default='multinli_erm_model', help='Path to save model')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--save_checkpoints', action='store_true', help='Enable checkpoint saving')
    
    args = parser.parse_args()
    
    # Setup device - prioritize MPS (Apple Silicon) over CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CPU only)")
    print(f"PyTorch version: {torch.__version__}")
    
    # Load data
    train_data, val_data = load_multinli_data()
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    
    # Print group statistics
    train_groups = train_data[2]
    val_groups = val_data[2]
    
    print("\nTraining group distribution:")
    for group_id in sorted(set(train_groups)):
        count = train_groups.count(group_id)
        percentage = 100 * count / len(train_groups)
        print(f"Group {group_id}: {count} samples ({percentage:.1f}%)")
    
    # Initialize model and tokenizer
    print("\nInitializing BERT-base-uncased model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3  # contradiction, entailment, neutral
    )
    model.to(device)
    
    # Initialize trainer with checkpoint support if enabled
    save_dir = args.save_model if args.save_checkpoints else None
    trainer = ERMTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_dir=save_dir
    )
    
    # Handle resume from checkpoint
    start_epoch = 0
    if args.resume_from:
        print(f"\n📂 Resuming from checkpoint: {args.resume_from}")
        # Create dummy optimizer and scheduler for loading
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = len(train_data[0]) // args.batch_size * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
        
        checkpoint_info = trainer.checkpoint_manager.load_checkpoint(
            args.resume_from, model, optimizer, scheduler
        )
        start_epoch = checkpoint_info['epoch'] + 1
        print(f"✅ Resuming from epoch {start_epoch}")
    
    # Train model
    if start_epoch < args.epochs:
        print(f"\nStarting ERM training from epoch {start_epoch}...")
        trainer.train(train_data, val_data, num_epochs=args.epochs)
    else:
        print("Model already trained to specified epochs!")
        # Just evaluate
        final_metrics = trainer.evaluate(val_data)
        print(f"Current performance - Overall Acc: {final_metrics['overall_accuracy']:.4f}, "
              f"Worst Group Acc: {final_metrics['worst_group_accuracy']:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    
    # If we have checkpoints, load the best one for final evaluation
    if trainer.checkpoint_manager:
        best_checkpoint_path = trainer.checkpoint_manager.get_best_checkpoint_path()
        if best_checkpoint_path:
            print(f"📂 Loading best checkpoint for final evaluation...")
            # Create fresh optimizer/scheduler for loading (not used for evaluation)
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            total_steps = len(train_data[0]) // args.batch_size * args.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
            
            trainer.checkpoint_manager.load_checkpoint(best_checkpoint_path, model, optimizer, scheduler)
    
    final_metrics = trainer.evaluate(val_data)
    
    print(f"Overall Accuracy: {final_metrics['overall_accuracy']:.4f}")
    print(f"Worst Group Accuracy: {final_metrics['worst_group_accuracy']:.4f}")
    
    print("\nGroup-wise accuracies:")
    for group_id, acc in final_metrics['group_accuracies'].items():
        print(f"Group {group_id}: {acc:.4f}")
    
    # Save model (final state)
    model.save_pretrained(args.save_model)
    tokenizer.save_pretrained(args.save_model)
    
    # Save results with checkpoint info
    results = {
        'method': 'ERM_Baseline',
        'overall_accuracy': final_metrics['overall_accuracy'],
        'worst_group_accuracy': final_metrics['worst_group_accuracy'], 
        'group_accuracies': final_metrics['group_accuracies'],
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'weight_decay': args.weight_decay,
            'checkpointing_enabled': args.save_checkpoints
        }
    }
    
    # Add checkpoint info if available
    if trainer.checkpoint_manager:
        results['best_epoch'] = trainer.checkpoint_manager.best_epoch
        results['best_worst_group_acc'] = trainer.checkpoint_manager.best_worst_group_acc
        results['checkpoint_dir'] = trainer.checkpoint_manager.save_dir
    
    with open(f'{args.save_model}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel and results saved to {args.save_model}/")
    
    if trainer.checkpoint_manager:
        print(f"💾 Checkpoints saved to: {trainer.checkpoint_manager.save_dir}")
        print(f"🏆 Best model achieved {trainer.checkpoint_manager.best_worst_group_acc:.4f} worst-group accuracy at epoch {trainer.checkpoint_manager.best_epoch}")

if __name__ == "__main__":
    main()
