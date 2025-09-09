"""
Compare All Bias Mitigation Methods
Run all four methods (ERM, LLR, JTT, SELF) and compare their performance
Now includes checkpoint saving for each method to preserve best models

Cluster-friendly features:
- Environment variable support for offline mode
- Configurable device selection and multiprocessing
- Robust error handling for network issues
- Progress tracking and recovery options
"""

import torch
import json
import os
from datetime import datetime
import argparse
import shutil
import warnings

# Set environment variables for cluster compatibility
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Suppress warnings that can clutter cluster logs
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning)

def create_comparison_checkpoint_dir():
    """Create a timestamped directory for comparison checkpoints"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"comparison_checkpoints_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"📁 Created comparison checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir

def copy_best_checkpoints(method_name, model_save_path, comparison_checkpoint_dir):
    """Copy the best checkpoint from a method to the comparison directory"""
    checkpoint_source = os.path.join(model_save_path, 'checkpoints')
    
    if os.path.exists(checkpoint_source):
        # Create method-specific directory in comparison checkpoints
        method_dir = os.path.join(comparison_checkpoint_dir, method_name.replace(' ', '_').lower())
        os.makedirs(method_dir, exist_ok=True)
        
        # Look for best checkpoint files
        best_checkpoint = None
        training_history = None
        
        for file in os.listdir(checkpoint_source):
            if 'best_checkpoint.pt' in file:
                best_checkpoint = os.path.join(checkpoint_source, file)
            elif 'training_history.json' in file:
                training_history = os.path.join(checkpoint_source, file)
        
        copied_files = []
        
        # Copy best checkpoint
        if best_checkpoint and os.path.exists(best_checkpoint):
            dest_checkpoint = os.path.join(method_dir, f"{method_name.replace(' ', '_').lower()}_best_checkpoint.pt")
            shutil.copy2(best_checkpoint, dest_checkpoint)
            copied_files.append(os.path.basename(dest_checkpoint))
            print(f"  📋 Copied best checkpoint: {os.path.basename(dest_checkpoint)}")
        
        # Copy training history
        if training_history and os.path.exists(training_history):
            dest_history = os.path.join(method_dir, f"{method_name.replace(' ', '_').lower()}_training_history.json")
            shutil.copy2(training_history, dest_history)
            copied_files.append(os.path.basename(dest_history))
            print(f"  📊 Copied training history: {os.path.basename(dest_history)}")
        
        # Copy entire checkpoint directory for completeness
        if copied_files:
            full_backup_dir = os.path.join(method_dir, 'full_checkpoints')
            shutil.copytree(checkpoint_source, full_backup_dir, dirs_exist_ok=True)
            print(f"  💾 Backed up full checkpoint directory to: {os.path.basename(full_backup_dir)}")
        
        return len(copied_files) > 0
    
    else:
        print(f"  ⚠️  No checkpoints found for {method_name}")
        return False

def run_method(method_name, script_path, args_dict, enable_checkpoints=True):
    """Run a specific method with given arguments"""
    print(f"\n{'='*60}")
    print(f"Running {method_name}")
    if enable_checkpoints:
        print("💾 Checkpoints ENABLED - will save best models")
    print(f"{'='*60}")
    
    # Add checkpoint saving to args if enabled
    if enable_checkpoints:
        args_dict = args_dict.copy()  # Don't modify original
        args_dict['save_checkpoints'] = True
    
    # Import and run the method
    if method_name == "ERM Baseline":
        from multinli_erm_baseline import main as erm_main
        import sys
        
        # Temporarily modify sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ['multinli_erm_baseline.py']
        for key, value in args_dict.items():
            if value is True:  # Handle boolean flags
                sys.argv.append(f'--{key}')
            else:
                sys.argv.extend([f'--{key}', str(value)])
        
        try:
            erm_main()
        finally:
            sys.argv = original_argv
            
    elif method_name == "Last Layer Retraining":
        from last_layer_retraining import main as llr_main
        import sys
        
        original_argv = sys.argv.copy()
        sys.argv = ['last_layer_retraining.py']
        for key, value in args_dict.items():
            if value is True:
                sys.argv.append(f'--{key}')
            else:
                sys.argv.extend([f'--{key}', str(value)])
        
        try:
            llr_main()
        finally:
            sys.argv = original_argv
            
    elif method_name == "Just Train Twice":
        from just_train_twice import main as jtt_main
        import sys
        
        original_argv = sys.argv.copy()
        sys.argv = ['just_train_twice.py']
        for key, value in args_dict.items():
            if value is True:
                sys.argv.append(f'--{key}')
            else:
                sys.argv.extend([f'--{key}', str(value)])
        
        try:
            jtt_main()
        finally:
            sys.argv = original_argv
            
    elif method_name == "SELF":
        from self_adaptive_training import main as self_main
        import sys
        
        original_argv = sys.argv.copy()
        sys.argv = ['self_adaptive_training.py']
        for key, value in args_dict.items():
            if value is True:
                sys.argv.append(f'--{key}')
            else:
                sys.argv.extend([f'--{key}', str(value)])
        
        try:
            self_main()
        finally:
            sys.argv = original_argv

def load_results(results_file):
    """Load results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def compare_results(results_dict):
    """Compare and display results from all methods"""
    print(f"\n{'='*80}")
    print("COMPARISON OF ALL BIAS MITIGATION METHODS")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"{'Method':<25} {'Overall Acc':<12} {'Worst Group Acc':<15} {'Improvement':<12}")
    print(f"{'-'*25} {'-'*12} {'-'*15} {'-'*12}")
    
    # Get baseline performance for comparison
    baseline_worst = None
    if "ERM Baseline" in results_dict:
        baseline_worst = results_dict["ERM Baseline"]["worst_group_accuracy"]
    
    for method, results in results_dict.items():
        if results is None:
            print(f"{method:<25} {'FAILED':<12} {'FAILED':<15} {'N/A':<12}")
            continue
            
        overall_acc = results["overall_accuracy"]
        worst_acc = results["worst_group_accuracy"]
        
        if baseline_worst is not None and method != "ERM Baseline":
            improvement = worst_acc - baseline_worst
            improvement_str = f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}"
        else:
            improvement_str = "baseline" if method == "ERM Baseline" else "N/A"
        
        print(f"{method:<25} {overall_acc:.4f}{'':>6} {worst_acc:.4f}{'':>10} {improvement_str:<12}")
    
    print(f"\n{'='*80}")
    
    # Detailed group-wise comparison
    print("DETAILED GROUP-WISE ACCURACY COMPARISON")
    print(f"{'='*80}")
    
    # Get all groups
    all_groups = set()
    for method, results in results_dict.items():
        if results and "group_accuracies" in results:
            all_groups.update(results["group_accuracies"].keys())
    
    if all_groups:
        # Header
        header = f"{'Group':<30}"
        for method in results_dict.keys():
            header += f"{method:<12}"
        print(header)
        print("-" * len(header))
        
        # Group results
        for group in sorted(all_groups):
            row = f"{group:<30}"
            for method, results in results_dict.items():
                if results and "group_accuracies" in results and group in results["group_accuracies"]:
                    acc = results["group_accuracies"][group]
                    row += f"{acc:.4f}{'':>7}"
                else:
                    row += f"{'N/A':<12}"
            print(row)

def main():
    parser = argparse.ArgumentParser(description='Compare all bias mitigation methods')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for all methods')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (reduced for comparison)')
    parser.add_argument('--run_all', action='store_true', help='Run all methods (takes a long time)')
    parser.add_argument('--compare_only', action='store_true', help='Only compare existing results')
    parser.add_argument('--save_checkpoints', action='store_true', default=True, help='Save checkpoints for each method (default: True)')
    parser.add_argument('--no_checkpoints', action='store_true', help='Disable checkpoint saving')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], 
                       help='Device to use (auto=automatic detection)')
    parser.add_argument('--num_workers', type=int, default=0, 
                       help='Number of DataLoader workers (0=disable multiprocessing, safer for clusters)')
    
    args = parser.parse_args()
    
    # Print environment information
    print("🔍 Environment Information:")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"⚙️  DataLoader workers: {args.num_workers}")
    
    # Check cluster-friendly environment variables
    if os.environ.get('TRANSFORMERS_OFFLINE'):
        print("🌐 Running in OFFLINE mode (TRANSFORMERS_OFFLINE=1)")
    if os.environ.get('TOKENIZERS_PARALLELISM') == 'false':
        print("🔇 Tokenizer parallelism disabled (cluster-safe)")
    
    # Setup device with better cluster compatibility
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"🖥️  Using device: {device} (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🖥️  Using device: {device} (NVIDIA GPU)")
        else:
            device = torch.device('cpu')
            print(f"🖥️  Using device: {device} (CPU only)")
    else:
        device = torch.device(args.device)
        print(f"🖥️  Using device: {device} (manually specified)")
    
    # Determine checkpoint saving preference
    enable_checkpoints = args.save_checkpoints and not args.no_checkpoints
    
    # Create comparison checkpoint directory if checkpoints are enabled
    comparison_checkpoint_dir = None
    if enable_checkpoints and args.run_all and not args.compare_only:
        comparison_checkpoint_dir = create_comparison_checkpoint_dir()
    
    # Setup common parameters (reduced for faster comparison)
    common_args = {
        'batch_size': args.batch_size,
        'lr': 1e-5,
        'weight_decay': 1e-4,
        'device': args.device,
        'num_workers': args.num_workers
    }
    
    # Method-specific configurations
    methods_config = {
        "ERM Baseline": {
            'script': 'multinli_erm_baseline.py',
            'args': {**common_args, 'epochs': args.epochs, 'save_model': 'comparison_erm_model'},
            'results_file': 'comparison_erm_model_results.json'
        },
        "Last Layer Retraining": {
            'script': 'last_layer_retraining.py', 
            'args': {**common_args, 'erm_epochs': 2, 'llr_epochs': args.epochs, 'save_model': 'comparison_llr_model'},
            'results_file': 'comparison_llr_model_results.json'
        },
        "Just Train Twice": {
            'script': 'just_train_twice.py',
            'args': {**common_args, 'phase1_epochs': 2, 'phase2_epochs': args.epochs, 
                    'worst_fraction': 0.2, 'upweight_factor': 10.0, 'save_model': 'comparison_jtt_model'},
            'results_file': 'comparison_jtt_model_results.json'
        },
        "SELF": {
            'script': 'self_adaptive_training.py',
            'args': {**common_args, 'epochs': args.epochs, 'save_model': 'comparison_self_model'},
            'results_file': 'comparison_self_model_results.json'
        }
    }
    
    results_dict = {}
    
    if args.run_all and not args.compare_only:
        print(f"\n🚀 Starting comparison of all bias mitigation methods")
        if enable_checkpoints:
            print(f"💾 Checkpoint saving: ENABLED")
            print(f"📁 Checkpoints will be saved to: {comparison_checkpoint_dir}")
        else:
            print(f"💾 Checkpoint saving: DISABLED")
        print(f"⏱️  Epochs per method: {args.epochs}")
        print(f"{'='*80}")
        
        # Run all methods
        for method_name, config in methods_config.items():
            try:
                print(f"\nStarting {method_name}...")
                run_method(method_name, config['script'], config['args'], enable_checkpoints)
                
                # Copy checkpoints if enabled and method completed successfully
                if enable_checkpoints and comparison_checkpoint_dir:
                    print(f"\n📋 Copying checkpoints for {method_name}...")
                    checkpoint_copied = copy_best_checkpoints(
                        method_name, 
                        config['args']['save_model'], 
                        comparison_checkpoint_dir
                    )
                    if checkpoint_copied:
                        print(f"✅ {method_name} checkpoints saved to comparison directory")
                    else:
                        print(f"⚠️  No checkpoints found for {method_name}")
                
                # Load results
                results = load_results(config['results_file'])
                results_dict[method_name] = results
                
                if results:
                    print(f"{method_name} completed successfully!")
                    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
                    print(f"Worst Group Accuracy: {results['worst_group_accuracy']:.4f}")
                    
                    # Add checkpoint info to results if available
                    if enable_checkpoints and 'best_epoch' in results:
                        print(f"Best model saved at epoch {results['best_epoch']}")
                else:
                    print(f"Failed to load results for {method_name}")
                    
            except Exception as e:
                print(f"Error running {method_name}: {str(e)}")
                results_dict[method_name] = None
        
        # Print checkpoint summary
        if enable_checkpoints and comparison_checkpoint_dir:
            print(f"\n💾 CHECKPOINT SUMMARY")
            print(f"{'='*60}")
            print(f"All method checkpoints saved to: {comparison_checkpoint_dir}")
            
            # List what was saved
            if os.path.exists(comparison_checkpoint_dir):
                subdirs = [d for d in os.listdir(comparison_checkpoint_dir) 
                          if os.path.isdir(os.path.join(comparison_checkpoint_dir, d))]
                if subdirs:
                    print("Saved checkpoints for methods:")
                    for subdir in sorted(subdirs):
                        method_dir = os.path.join(comparison_checkpoint_dir, subdir)
                        files = os.listdir(method_dir)
                        checkpoint_files = [f for f in files if f.endswith('.pt')]
                        print(f"  📁 {subdir.replace('_', ' ').title()}: {len(checkpoint_files)} checkpoint files")
                else:
                    print("  ⚠️  No checkpoint directories found")
            print(f"{'='*60}")
    
    else:
        # Load existing results
        print("Loading existing results...")
        for method_name, config in methods_config.items():
            results = load_results(config['results_file'])
            results_dict[method_name] = results
            if results:
                print(f"Loaded results for {method_name}")
            else:
                print(f"No results found for {method_name}")
    
    # Compare results
    compare_results(results_dict)
    
    # Save comparison summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"bias_methods_comparison_{timestamp}.json"
    
    comparison_data = {
        'timestamp': timestamp,
        'parameters': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'checkpoints_enabled': enable_checkpoints
        },
        'checkpoint_directory': comparison_checkpoint_dir if enable_checkpoints else None,
        'results': results_dict
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nComparison results saved to: {comparison_file}")
    
    if enable_checkpoints and comparison_checkpoint_dir:
        print(f"💾 All method checkpoints saved to: {comparison_checkpoint_dir}")
    
    # Determine best method
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if valid_results:
        best_method = max(valid_results.keys(), 
                         key=lambda x: valid_results[x]["worst_group_accuracy"])
        print(f"\n🏆 Best method by worst-group accuracy: {best_method}")
        print(f"Worst Group Accuracy: {valid_results[best_method]['worst_group_accuracy']:.4f}")
        
        if enable_checkpoints and comparison_checkpoint_dir:
            best_method_dir = best_method.replace(' ', '_').lower()
            best_checkpoint_path = os.path.join(comparison_checkpoint_dir, best_method_dir)
            if os.path.exists(best_checkpoint_path):
                print(f"💾 Best method checkpoint available at: {best_checkpoint_path}")

if __name__ == "__main__":
    main()
