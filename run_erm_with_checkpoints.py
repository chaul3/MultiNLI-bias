#!/usr/bin/env python3
"""
Run ERM baseline with checkpoint saving enabled
This script demonstrates the checkpoint functionality for worst-case model saving
"""

import subprocess
import sys
import os

def run_erm_with_checkpoints():
    """Run ERM baseline with checkpoint saving"""
    
    print("🚀 Running ERM Baseline with Checkpoint Saving")
    print("="*60)
    print("This will:")
    print("- Save checkpoints when worst-group accuracy improves")
    print("- Track training progress and best model")
    print("- Allow resuming from interruptions")
    print("="*60)
    
    # Command to run ERM with checkpoints
    cmd = [
        sys.executable, "multinli_erm_baseline.py",
        "--epochs", "5",
        "--batch_size", "16", 
        "--lr", "1e-5",
        "--save_model", "erm_checkpoint_model",
        "--save_checkpoints"  # Enable checkpoint saving
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        
        # Show checkpoint directory
        checkpoint_dir = "erm_checkpoint_model/checkpoints"
        if os.path.exists(checkpoint_dir):
            print(f"\n💾 Checkpoints saved in: {checkpoint_dir}")
            files = os.listdir(checkpoint_dir)
            for file in files:
                if file.endswith('.pt') or file.endswith('.json'):
                    print(f"  - {file}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏸️  Training interrupted by user")
        print("You can resume training with:")
        print(f"python multinli_erm_baseline.py --resume_from erm_checkpoint_model/checkpoints/erm_model_best_checkpoint.pt --save_checkpoints")
        return False
    
    return True

def demo_resume():
    """Demonstrate resuming from checkpoint"""
    checkpoint_path = "erm_checkpoint_model/checkpoints/erm_model_best_checkpoint.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("Run training first with --save_checkpoints")
        return
    
    print(f"\n🔄 Demonstrating resume from checkpoint")
    print("="*60)
    
    cmd = [
        sys.executable, "multinli_erm_baseline.py",
        "--epochs", "7",  # Continue for 2 more epochs
        "--batch_size", "16",
        "--lr", "1e-5", 
        "--save_model", "erm_checkpoint_model",
        "--save_checkpoints",
        "--resume_from", checkpoint_path
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Resume training completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Resume training failed: {e}")
    except KeyboardInterrupt:
        print(f"\n⏸️  Resume training interrupted")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ERM with checkpoint demo")
    parser.add_argument("--demo_resume", action="store_true", 
                       help="Demo resuming from existing checkpoint")
    
    args = parser.parse_args()
    
    if args.demo_resume:
        demo_resume()
    else:
        run_erm_with_checkpoints()
