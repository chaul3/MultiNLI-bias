"""
Checkpoint utilities for bias mitigation training
Handles saving and loading model checkpoints with focus on worst-group accuracy
"""

import torch
import os
import json
from datetime import datetime

class CheckpointManager:
    def __init__(self, save_dir, model_name="model", save_best_only=True, save_every_n_epochs=1):
        """
        Initialize checkpoint manager
        
        Args:
            save_dir: Directory to save checkpoints
            model_name: Base name for checkpoint files
            save_best_only: If True, only save when worst-group accuracy improves
            save_every_n_epochs: Save checkpoint every N epochs (if not save_best_only)
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs
        
        # Track best performance
        self.best_worst_group_acc = -1.0
        self.best_overall_acc = -1.0
        self.best_epoch = -1
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.training_history = {
            'epochs': [],
            'train_losses': [],
            'val_overall_accs': [],
            'val_worst_group_accs': [],
            'val_group_accs': []
        }
    
    def should_save_checkpoint(self, epoch, worst_group_acc):
        """Determine if checkpoint should be saved"""
        if self.save_best_only:
            return worst_group_acc > self.best_worst_group_acc
        else:
            return (epoch + 1) % self.save_every_n_epochs == 0
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, train_loss, 
                       val_metrics, args=None, extra_info=None):
        """
        Save model checkpoint with training state
        
        Args:
            epoch: Current epoch number
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state  
            train_loss: Training loss for this epoch
            val_metrics: Validation metrics dict
            args: Training arguments
            extra_info: Additional info to save
        """
        worst_group_acc = val_metrics['worst_group_accuracy']
        overall_acc = val_metrics['overall_accuracy']
        
        # Update training history
        self.training_history['epochs'].append(epoch)
        self.training_history['train_losses'].append(train_loss)
        self.training_history['val_overall_accs'].append(overall_acc)
        self.training_history['val_worst_group_accs'].append(worst_group_acc)
        self.training_history['val_group_accs'].append(val_metrics['group_accuracies'])
        
        # Check if we should save
        should_save = self.should_save_checkpoint(epoch, worst_group_acc)
        
        if should_save:
            # Update best metrics
            is_best = worst_group_acc > self.best_worst_group_acc
            if is_best:
                self.best_worst_group_acc = worst_group_acc
                self.best_overall_acc = overall_acc
                self.best_epoch = epoch
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'best_worst_group_acc': self.best_worst_group_acc,
                'best_overall_acc': self.best_overall_acc,
                'best_epoch': self.best_epoch,
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat(),
                'args': vars(args) if args else None,
                'extra_info': extra_info
            }
            
            # Save checkpoint
            if is_best:
                checkpoint_path = os.path.join(self.save_dir, f'{self.model_name}_best_checkpoint.pt')
                print(f"💾 Saving BEST checkpoint (worst-group acc: {worst_group_acc:.4f}) to {checkpoint_path}")
            else:
                checkpoint_path = os.path.join(self.save_dir, f'{self.model_name}_epoch_{epoch}_checkpoint.pt')
                print(f"💾 Saving checkpoint (epoch {epoch}) to {checkpoint_path}")
            
            torch.save(checkpoint, checkpoint_path)
            
            # Also save training history as JSON for easy analysis
            history_path = os.path.join(self.save_dir, f'{self.model_name}_training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            return checkpoint_path
        
        return None
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None):
        """
        Load checkpoint and restore training state
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            
        Returns:
            Dict with loaded checkpoint info
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore checkpoint manager state
        self.best_worst_group_acc = checkpoint.get('best_worst_group_acc', -1.0)
        self.best_overall_acc = checkpoint.get('best_overall_acc', -1.0)
        self.best_epoch = checkpoint.get('best_epoch', -1)
        self.training_history = checkpoint.get('training_history', {
            'epochs': [], 'train_losses': [], 'val_overall_accs': [], 
            'val_worst_group_accs': [], 'val_group_accs': []
        })
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"   Best worst-group accuracy: {self.best_worst_group_acc:.4f} (epoch {self.best_epoch})")
        
        return {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint.get('train_loss'),
            'val_metrics': checkpoint.get('val_metrics'),
            'args': checkpoint.get('args'),
            'extra_info': checkpoint.get('extra_info')
        }
    
    def get_best_checkpoint_path(self):
        """Get path to best checkpoint"""
        best_path = os.path.join(self.save_dir, f'{self.model_name}_best_checkpoint.pt')
        return best_path if os.path.exists(best_path) else None
    
    def print_training_summary(self):
        """Print summary of training progress"""
        if not self.training_history['epochs']:
            print("No training history available")
            return
        
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total epochs: {len(self.training_history['epochs'])}")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best worst-group accuracy: {self.best_worst_group_acc:.4f}")
        print(f"Best overall accuracy: {self.best_overall_acc:.4f}")
        
        if len(self.training_history['train_losses']) > 0:
            print(f"Final training loss: {self.training_history['train_losses'][-1]:.4f}")
        
        print(f"{'='*60}")

def find_latest_checkpoint(checkpoint_dir, model_name="model"):
    """Find the most recent checkpoint in a directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for best checkpoint first
    best_path = os.path.join(checkpoint_dir, f'{model_name}_best_checkpoint.pt')
    if os.path.exists(best_path):
        return best_path
    
    # Otherwise find latest epoch checkpoint
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith(f'{model_name}_epoch_') and file.endswith('_checkpoint.pt'):
            try:
                epoch = int(file.split('_epoch_')[1].split('_checkpoint.pt')[0])
                checkpoints.append((epoch, os.path.join(checkpoint_dir, file)))
            except:
                continue
    
    if checkpoints:
        # Return path of latest epoch
        return max(checkpoints, key=lambda x: x[0])[1]
    
    return None
