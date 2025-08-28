"""
Last Layer Retraining (LLR) for MultiNLI Dataset
A simple bias mitigation method that retrains only the classification head
after standard ERM training, using group-balanced sampling.

This method:
1. First trains the full model with ERM
2. Then freezes the BERT encoder and retrains only the classifier
3. Uses group-balanced sampling during retraining phase
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import json
from collections import Counter

# Import our baseline components
from multinli_erm_baseline import MultiNLIDataset, identify_spurious_groups, load_multinli_data

class LLRTrainer:
    def __init__(self, model, tokenizer, device, lr=1e-5, batch_size=16, weight_decay=1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
    def train_erm_phase(self, train_data, val_data, num_epochs=3):
        """Phase 1: Standard ERM training"""
        print("Phase 1: ERM Training...")
        
        train_texts, train_labels, train_groups = train_data
        
        # Create datasets and loaders
        train_dataset = MultiNLIDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Setup optimizer for full model
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"ERM Epoch {epoch + 1}/{num_epochs}")
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"ERM Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            print(f"ERM Average training loss: {avg_loss:.4f}")
        
        print("ERM training completed!")
    
    def create_group_balanced_sampler(self, train_groups):
        """Create a sampler that balances groups"""
        # Count samples per group
        group_counts = Counter(train_groups)
        print(f"Group distribution: {dict(group_counts)}")
        
        # Calculate weights for balanced sampling
        total_samples = len(train_groups)
        num_groups = len(group_counts)
        
        # Weight inversely proportional to group size
        group_weights = {}
        for group_id, count in group_counts.items():
            group_weights[group_id] = total_samples / (num_groups * count)
        
        # Create sample weights
        sample_weights = [group_weights[group] for group in train_groups]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def train_llr_phase(self, train_data, val_data, num_epochs=5):
        """Phase 2: Last Layer Retraining with group balancing"""
        print("\nPhase 2: Last Layer Retraining...")
        
        train_texts, train_labels, train_groups = train_data
        
        # Freeze BERT encoder, only train classifier
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f"Training parameter: {name}")
        
        # Create dataset with group-balanced sampling
        train_dataset = MultiNLIDataset(train_texts, train_labels, self.tokenizer)
        group_sampler = self.create_group_balanced_sampler(train_groups)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=group_sampler,  # Use group-balanced sampler
            num_workers=2,
            pin_memory=True
        )
        
        # Setup optimizer only for classifier parameters
        classifier_params = [p for name, p in self.model.named_parameters() if 'classifier' in name and p.requires_grad]
        optimizer = AdamW(classifier_params, lr=self.lr * 10, weight_decay=self.weight_decay)  # Higher LR for classifier
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"LLR Epoch {epoch + 1}/{num_epochs}")
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"LLR Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            print(f"LLR Average training loss: {avg_loss:.4f}")
            
            # Validation after each epoch
            val_metrics = self.evaluate(val_data)
            print(f"LLR Validation - Overall Acc: {val_metrics['overall_accuracy']:.4f}, "
                  f"Worst Group Acc: {val_metrics['worst_group_accuracy']:.4f}")
        
        # Unfreeze all parameters for final evaluation
        for param in self.model.parameters():
            param.requires_grad = True
            
        print("Last Layer Retraining completed!")
    
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
    parser = argparse.ArgumentParser(description='Last Layer Retraining for MultiNLI')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--erm_epochs', type=int, default=3, help='ERM training epochs')
    parser.add_argument('--llr_epochs', type=int, default=5, help='LLR training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_model', type=str, default='multinli_llr_model', help='Path to save model')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CPU only)")
    
    # Load data
    train_data, val_data = load_multinli_data()
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    
    # Initialize model and tokenizer
    print("\nInitializing BERT-base-uncased model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )
    model.to(device)
    
    # Initialize trainer
    trainer = LLRTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay
    )
    
    # Phase 1: ERM Training
    trainer.train_erm_phase(train_data, val_data, num_epochs=args.erm_epochs)
    
    # Phase 2: Last Layer Retraining
    trainer.train_llr_phase(train_data, val_data, num_epochs=args.llr_epochs)
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    final_metrics = trainer.evaluate(val_data)
    
    print(f"Overall Accuracy: {final_metrics['overall_accuracy']:.4f}")
    print(f"Worst Group Accuracy: {final_metrics['worst_group_accuracy']:.4f}")
    
    print("\nGroup-wise accuracies:")
    for group_id, acc in final_metrics['group_accuracies'].items():
        print(f"Group {group_id}: {acc:.4f}")
    
    # Save model and results
    model.save_pretrained(args.save_model)
    tokenizer.save_pretrained(args.save_model)
    
    results = {
        'method': 'Last Layer Retraining',
        'overall_accuracy': final_metrics['overall_accuracy'],
        'worst_group_accuracy': final_metrics['worst_group_accuracy'], 
        'group_accuracies': final_metrics['group_accuracies'],
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'erm_epochs': args.erm_epochs,
            'llr_epochs': args.llr_epochs,
            'weight_decay': args.weight_decay
        }
    }
    
    with open(f'{args.save_model}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel and results saved to {args.save_model}/")

if __name__ == "__main__":
    main()
