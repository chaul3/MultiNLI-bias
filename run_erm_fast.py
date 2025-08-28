"""
Fast version of ERM baseline with Apple Silicon GPU acceleration
"""

import subprocess
import sys

def main():
    """Run ERM baseline optimized for Apple Silicon with faster parameters"""
    
    cmd = [
        sys.executable, 'multinli_erm_baseline.py',
        '--batch_size', '32',          # Increased batch size for GPU
        '--lr', '1e-5',                # Paper's learning rate
        '--epochs', '3',               # Reduced epochs for faster demo 
        '--weight_decay', '1e-4',      # Paper's weight decay
        '--save_model', 'multinli_erm_fast_model'
    ]
    
    print("🚀 Running FAST ERM baseline with Apple Silicon GPU:")
    print("- BERT-base-uncased (pre-trained on Book Corpus and English Wikipedia)")
    print("- Batch size: 32 (increased for GPU)")
    print("- Learning rate: 1e-5 (10⁻⁵)")
    print("- Epochs: 3 (reduced for demo)")
    print("- Weight decay: 1e-4 (10⁻⁴)")
    print("- AdamW optimizer with linear learning rate annealing")
    print("- Using Apple Silicon GPU (MPS)")
    print("- Max sequence length: 256 (reduced for speed)")
    print("- Parallel data loading enabled")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
