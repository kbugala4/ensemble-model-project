#!/usr/bin/env python3
"""
Main script to run baseline experiments with sklearn ensemble methods.

Usage:
    python run_baseline_experiments.py                # Run full experiment
    python run_baseline_experiments.py --quick        # Quick test
    python run_baseline_experiments.py --single flights  # Test single dataset
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipelines.baseline_runner import BaselineExperimentRunner


def main():
    """Main function to run baseline experiments."""
    
    print("🔬 Baseline Ensemble Experiments")
    print("Testing BaggingClassifier and RandomForestClassifier")
    print("on all available datasets")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print(__doc__)
            return
        else:
            print("Invalid argument")
            return
    
    # Run full experiment
    try:
        runner = BaselineExperimentRunner()
        results = runner.run_full_baseline_experiments()  # Note: different method name
        
        print("\n🎉 All baseline experiments completed!")
        print("Check 'results/baseline/' directory for detailed results")
        
        # Show quick summary
        successful_experiments = sum(1 for dataset_results in results.values() 
                                   if 'error' not in dataset_results)
        total_experiments = len(results)
        
        print(f"\nExperiment Summary:")
        print(f"  Successfully processed: {successful_experiments}/{total_experiments} datasets")
        
        if successful_experiments > 0:
            print(f"  Results saved to: results/baseline/")
            print(f"  View detailed results in the JSON and CSV files")
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 