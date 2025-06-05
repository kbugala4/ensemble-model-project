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

from src.pipelines.runner import BaselineExperimentRunner


def quick_test():
    """Quick test function to verify the implementation works."""
    print("🔧 Running quick test...")
    
    runner = BaselineExperimentRunner()
    
    # Test with first available dataset
    datasets = runner.data_loader.list_available_datasets()
    if datasets:
        test_dataset = datasets[0]
        results = runner.run_single_dataset_experiment(test_dataset)
        
        if 'error' not in results:
            print("✅ Quick test passed!")
            return True
        else:
            print(f"❌ Quick test failed: {results['error']}")
            return False
    else:
        print("❌ No datasets available for testing")
        return False


def main():
    """Main function to run baseline experiments."""
    
    print("🔬 Baseline Ensemble Experiments")
    print("Testing BaggingClassifier and RandomForestClassifier")
    print("on all available datasets")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            print("Running quick test...")
            success = quick_test()
            if success:
                print("\n🎉 Quick test completed successfully!")
            else:
                print("\n❌ Quick test failed!")
                sys.exit(1)
            return
        
        elif sys.argv[1] == "--single":
            dataset_name = sys.argv[2] if len(sys.argv) > 2 else "flights"
            print(f"Running single dataset experiment: {dataset_name}")
            runner = BaselineExperimentRunner()
            results = runner.run_single_dataset_experiment(dataset_name)
            
            if 'error' not in results:
                print(f"\n🎉 Single dataset experiment completed!")
            else:
                print(f"\n❌ Single dataset experiment failed: {results['error']}")
                sys.exit(1)
            return
        
        elif sys.argv[1] in ["-h", "--help"]:
            print(__doc__)
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