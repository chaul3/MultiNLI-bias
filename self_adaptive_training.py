"""
SELF (Self-adaptive Training) for MultiNLI Dataset
A bias mitigation method that adaptively reweights examples during training
based on their difficulty and group membership to improve worst-group performance.

This method:
1. Trains with adaptive loss weighting
2. Dynamically reweights examples based on their loss trends
3. Focuses on improving worst-performing groups
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
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import json
import os
from collections import defaultdict, deque
from checkpoint_utils import CheckpointManager, find_latest_checkpoint

# Import our baseline components
from multinli_erm_baseline import MultiNLIDataset, identify_spurious_groups, load_multinli_data

class SELFTrainer:
    def __init__(self, model, tokenizer, device, lr=1e-5, batch_size=16, weight_decay=1e-4, save_dir=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        
        # SELF-specific parameters
        self.loss_history = defaultdict(lambda: deque(maxlen=10))  # Track loss history per example
        self.group_weights = {}  # Adaptive weights per group
        self.example_weights = {}  # Adaptive weights per example
        
        # Initialize checkpoint manager if save_dir provided
        self.checkpoint_manager = None
        if save_dir:
            self.checkpoint_manager = CheckpointManager(
                save_dir=os.path.join(save_dir, 'checkpoints'),
                model_name='self_model',
                save_best_only=True  # Only save when worst-group accuracy improves
            )
        
    def compute_adaptive_weights(self, indices, losses, groups, epoch):
        """Compute adaptive weights based on loss history and group performance"""
        
        # Update loss history for each example
        for idx, loss, group in zip(indices, losses, groups):
            self.loss_history[idx].append(loss)
        
        # Compute group-level statistics
        group_losses = defaultdict(list)
        for idx, loss, group in zip(indices, losses, groups):
            group_losses[group].append(loss)
        
        # Update group weights (inverse of group performance)
        for group, losses_list in group_losses.items():
            avg_loss = np.mean(losses_list)
            self.group_weights[group] = avg_loss  # Higher loss = higher weight
        
        # Normalize group weights
        if self.group_weights:
            max_weight = max(self.group_weights.values())
            self.group_weights = {k: v / max_weight for k, v in self.group_weights.items()}
        
        # Compute example-level weights
        example_weights = []
        for idx, group in zip(indices, groups):
            # Base weight from group
            group_weight = self.group_weights.get(group, 1.0)
            
            # Example-specific weight based on loss trend
            if len(self.loss_history[idx]) > 1:
                # If loss is increasing (difficult example), increase weight
                recent_loss = np.mean(list(self.loss_history[idx])[-3:])
                early_loss = np.mean(list(self.loss_history[idx])[:3])
                
                if recent_loss > early_loss:
                    example_weight = 1.5  # Upweight difficult examples
                else:
                    example_weight = 1.0
            else:
                example_weight = 1.0
            
            # Combine group and example weights
            final_weight = group_weight * example_weight
            example_weights.append(final_weight)
            self.example_weights[idx] = final_weight
        
        return torch.tensor(example_weights, device=self.device)
    
    def weighted_loss(self, logits, labels, weights):
        """Compute weighted cross-entropy loss"""
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(logits, labels)
        
        # Apply weights
        weighted_losses = losses * weights
        return weighted_losses.mean()
    
    def train(self, train_data, val_data, num_epochs=5):
        """Train with SELF adaptive weighting"""
        print("Starting SELF Training with adaptive weighting...")
        
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
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # Warmup for 10% of steps
            num_training_steps=total_steps
        )
        
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"SELF Epoch {epoch + 1}/{num_epochs}")
            
            total_loss = 0
            total_weighted_loss = 0
            progress_bar = tqdm(train_loader, desc=f"SELF Training Epoch {epoch + 1}")
            
            batch_start_idx = 0
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get batch indices and groups
                batch_size = input_ids.size(0)
                batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
                batch_groups = [train_groups[i] for i in batch_indices]
                batch_start_idx += batch_size
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Compute individual losses
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                individual_losses = loss_fct(outputs.logits, labels)
                
                # Compute adaptive weights
                if epoch > 0:  # Start adaptive weighting from epoch 1
                    weights = self.compute_adaptive_weights(
                        batch_indices, 
                        individual_losses.detach().cpu().numpy(),
                        batch_groups,
                        epoch
                    )
                    
                    # Use weighted loss
                    loss = self.weighted_loss(outputs.logits, labels, weights)
                    total_weighted_loss += loss.item()
                else:
                    # Standard loss for first epoch
                    loss = individual_losses.mean()
                    # Still update loss history
                    for idx, l in zip(batch_indices, individual_losses.detach().cpu().numpy()):
                        self.loss_history[idx].append(l)
                
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': scheduler.get_last_lr()[0]
                })
            
            avg_loss = total_loss / len(train_loader)
            print(f"SELF Average training loss: {avg_loss:.4f}")
            
            if epoch > 0:
                avg_weighted_loss = total_weighted_loss / len(train_loader)
                print(f"SELF Average weighted loss: {avg_weighted_loss:.4f}")
            
            # Print group weights
            if self.group_weights:
                print("Current group weights:")
                for group, weight in sorted(self.group_weights.items()):
                    print(f"  {group}: {weight:.3f}")
            
            # Validation
            val_metrics = self.evaluate(val_data)
            print(f"SELF Validation - Overall Acc: {val_metrics['overall_accuracy']:.4f}, "
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
                    extra_info={
                        'method': 'SELF', 
                        'group_weights': dict(self.group_weights),
                        'num_examples_with_history': len(self.loss_history)
                    }
                )
                if saved_path:
                    print(f"✅ Checkpoint saved: {os.path.basename(saved_path)}")
            
        # Print training summary
        if self.checkpoint_manager:
            self.checkpoint_manager.print_training_summary()
            
        print("SELF training completed!")
    
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
    parser = argparse.ArgumentParser(description='SELF (Self-adaptive Training) for MultiNLI')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_model', type=str, default='multinli_self_model', help='Path to save model')
    parser.add_argument('--save_checkpoints', action='store_true', help='Enable checkpoint saving')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from checkpoint')
    
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
    
    # Print group statistics
    train_groups = train_data[2]
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
        num_labels=3
    )
    model.to(device)
    
    # Initialize trainer with checkpoint support if enabled
    save_dir = args.save_model if args.save_checkpoints else None
    trainer = SELFTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_dir=save_dir
    )
    
    # Train with SELF
    trainer.train(train_data, val_data, num_epochs=args.epochs)
    
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
        'method': 'SELF (Self-adaptive Training)',
        'overall_accuracy': final_metrics['overall_accuracy'],
        'worst_group_accuracy': final_metrics['worst_group_accuracy'], 
        'group_accuracies': final_metrics['group_accuracies'],
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'weight_decay': args.weight_decay
        }
    }
    
    with open(f'{args.save_model}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel and results saved to {args.save_model}/")

if __name__ == "__main__":
    main()
