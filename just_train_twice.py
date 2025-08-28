"""
Just Train Twice (JTT) for MultiNLI Dataset
A bias mitigation method that identifies worst-performing examples 
in the first training phase and upweights them in the second phase.

This method:
1. First trains with ERM to identify hard/misclassified examples
2. Identifies examples with high loss (worst examples)
3. Retrains with upweighting of these worst examples
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
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import json

# Import our baseline components
from multinli_erm_baseline import MultiNLIDataset, identify_spurious_groups, load_multinli_data

class JTTTrainer:
    def __init__(self, model, tokenizer, device, lr=1e-5, batch_size=16, weight_decay=1e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.worst_examples_indices = None
        
    def train_first_phase(self, train_data, val_data, num_epochs=3):
        """Phase 1: Standard ERM training to identify worst examples"""
        print("Phase 1: Initial ERM Training to identify worst examples...")
        
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
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"Phase 1 Epoch {epoch + 1}/{num_epochs}")
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Phase 1 Training Epoch {epoch + 1}")
            
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
            print(f"Phase 1 Average training loss: {avg_loss:.4f}")
        
        print("Phase 1 training completed!")
    
    def identify_worst_examples(self, train_data, worst_fraction=0.2):
        """Identify worst examples based on loss after first training phase"""
        print(f"\nIdentifying worst {worst_fraction*100:.0f}% of training examples...")
        
        train_texts, train_labels, train_groups = train_data
        
        train_dataset = MultiNLIDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,  # Important: don't shuffle to maintain indices
            num_workers=2,
            pin_memory=True
        )
        
        self.model.eval()
        all_losses = []
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing losses"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Get per-example losses
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(outputs.logits, labels)
                all_losses.extend(losses.cpu().numpy())
        
        # Convert to numpy array
        all_losses = np.array(all_losses)
        
        # Find worst examples (highest loss)
        num_worst = int(len(all_losses) * worst_fraction)
        worst_indices = np.argsort(all_losses)[-num_worst:]  # Indices of highest losses
        
        self.worst_examples_indices = set(worst_indices)
        
        print(f"Identified {len(self.worst_examples_indices)} worst examples")
        print(f"Average loss of worst examples: {all_losses[worst_indices].mean():.4f}")
        print(f"Average loss of all examples: {all_losses.mean():.4f}")
        
        return worst_indices
    
    def create_jtt_sampler(self, train_data, upweight_factor=10.0):
        """Create sampler that upweights worst examples"""
        train_texts, train_labels, train_groups = train_data
        
        # Create weights for each sample
        sample_weights = []
        worst_count = 0
        
        for i in range(len(train_texts)):
            if i in self.worst_examples_indices:
                sample_weights.append(upweight_factor)
                worst_count += 1
            else:
                sample_weights.append(1.0)
        
        print(f"Upweighting {worst_count} examples by factor {upweight_factor}")
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def train_second_phase(self, train_data, val_data, num_epochs=5, upweight_factor=10.0):
        """Phase 2: Retrain with upweighting of worst examples"""
        print(f"\nPhase 2: Retraining with upweighting (factor={upweight_factor})...")
        
        train_texts, train_labels, train_groups = train_data
        
        # Reset model parameters for fresh training
        # Re-initialize the model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        )
        model.to(self.device)
        self.model = model
        
        # Create dataset with JTT sampling
        train_dataset = MultiNLIDataset(train_texts, train_labels, self.tokenizer)
        jtt_sampler = self.create_jtt_sampler(train_data, upweight_factor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=jtt_sampler,  # Use JTT sampler
            num_workers=2,
            pin_memory=True
        )
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        self.model.train()
        
        for epoch in range(num_epochs):
            print(f"Phase 2 Epoch {epoch + 1}/{num_epochs}")
            
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Phase 2 Training Epoch {epoch + 1}")
            
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
            print(f"Phase 2 Average training loss: {avg_loss:.4f}")
            
            # Validation after each epoch
            val_metrics = self.evaluate(val_data)
            print(f"Phase 2 Validation - Overall Acc: {val_metrics['overall_accuracy']:.4f}, "
                  f"Worst Group Acc: {val_metrics['worst_group_accuracy']:.4f}")
        
        print("Phase 2 training completed!")
    
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
    parser = argparse.ArgumentParser(description='Just Train Twice (JTT) for MultiNLI')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--phase1_epochs', type=int, default=3, help='Phase 1 training epochs')
    parser.add_argument('--phase2_epochs', type=int, default=5, help='Phase 2 training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--worst_fraction', type=float, default=0.2, help='Fraction of worst examples to upweight')
    parser.add_argument('--upweight_factor', type=float, default=10.0, help='Upweighting factor for worst examples')
    parser.add_argument('--save_model', type=str, default='multinli_jtt_model', help='Path to save model')
    
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
    trainer = JTTTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay
    )
    
    # Phase 1: Initial ERM Training
    trainer.train_first_phase(train_data, val_data, num_epochs=args.phase1_epochs)
    
    # Identify worst examples
    trainer.identify_worst_examples(train_data, worst_fraction=args.worst_fraction)
    
    # Phase 2: JTT Retraining
    trainer.train_second_phase(train_data, val_data, num_epochs=args.phase2_epochs, 
                             upweight_factor=args.upweight_factor)
    
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
        'method': 'Just Train Twice',
        'overall_accuracy': final_metrics['overall_accuracy'],
        'worst_group_accuracy': final_metrics['worst_group_accuracy'], 
        'group_accuracies': final_metrics['group_accuracies'],
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'phase1_epochs': args.phase1_epochs,
            'phase2_epochs': args.phase2_epochs,
            'weight_decay': args.weight_decay,
            'worst_fraction': args.worst_fraction,
            'upweight_factor': args.upweight_factor
        }
    }
    
    with open(f'{args.save_model}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel and results saved to {args.save_model}/")

if __name__ == "__main__":
    main()
