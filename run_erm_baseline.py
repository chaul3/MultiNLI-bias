"""
Simple script to run the ERM baseline with default parameters
"""

import subprocess
import sys

def main():
    """Run the ERM baseline with paper's exact hyperparameters"""
    
    cmd = [
        sys.executable, 'multinli_erm_baseline.py',
        '--batch_size', '16',          # Paper's exact batch size
        '--lr', '1e-5',                # Paper's exact learning rate (10⁻⁵)
        '--epochs', '5',               # Paper's exact epochs
        '--weight_decay', '1e-4',      # Paper's exact weight decay (10⁻⁴)
        '--save_model', 'multinli_erm_baseline_model'
    ]
    
    print("Running ERM baseline with paper's EXACT hyperparameters:")
    print("- BERT-base-uncased (pre-trained on Book Corpus and English Wikipedia)")
    print("- Batch size: 16") 
    print("- Learning rate: 1e-5 (10⁻⁵)")
    print("- Epochs: 5")
    print("- Weight decay: 1e-4 (10⁻⁴)")
    print("- AdamW optimizer with linear learning rate annealing")
    print("- No early stopping")
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
