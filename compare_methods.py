"""
Compare All Bias Mitigation Methods
Run all four methods (ERM, LLR, JTT, SELF) and compare their performance
"""

import torch
import json
import os
from datetime import datetime
import argparse

def run_method(method_name, script_path, args_dict):
    """Run a specific method with given arguments"""
    print(f"\n{'='*60}")
    print(f"Running {method_name}")
    print(f"{'='*60}")
    
    # Import and run the method
    if method_name == "ERM Baseline":
        from multinli_erm_baseline import main as erm_main
        import sys
        
        # Temporarily modify sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ['multinli_erm_baseline.py']
        for key, value in args_dict.items():
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
    
    args = parser.parse_args()
    
    # Setup common parameters (reduced for faster comparison)
    common_args = {
        'batch_size': args.batch_size,
        'lr': 1e-5,
        'weight_decay': 1e-4
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
        # Run all methods
        for method_name, config in methods_config.items():
            try:
                print(f"\nStarting {method_name}...")
                run_method(method_name, config['script'], config['args'])
                
                # Load results
                results = load_results(config['results_file'])
                results_dict[method_name] = results
                
                if results:
                    print(f"{method_name} completed successfully!")
                    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
                    print(f"Worst Group Accuracy: {results['worst_group_accuracy']:.4f}")
                else:
                    print(f"Failed to load results for {method_name}")
                    
            except Exception as e:
                print(f"Error running {method_name}: {str(e)}")
                results_dict[method_name] = None
    
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
            'epochs': args.epochs
        },
        'results': results_dict
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nComparison results saved to: {comparison_file}")
    
    # Determine best method
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    if valid_results:
        best_method = max(valid_results.keys(), 
                         key=lambda x: valid_results[x]["worst_group_accuracy"])
        print(f"\nBest method by worst-group accuracy: {best_method}")
        print(f"Worst Group Accuracy: {valid_results[best_method]['worst_group_accuracy']:.4f}")

if __name__ == "__main__":
    main()
