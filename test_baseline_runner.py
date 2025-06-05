#!/usr/bin/env python3
"""
Launch test script for BaselineExperimentRunner.

This script performs various tests to verify the BaselineExperimentRunner implementation:
1. Tests data loading
2. Tests model creation
3. Tests single dataset experiment
4. Tests basic functionality
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pipelines.baseline_runner import BaselineExperimentRunner
    from src.utils.dataloader import DatasetLoader
    from src.models.sklearn_wrappers import create_bagging_classifier, create_random_forest_classifier
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_data_loader():
    """Test if DatasetLoader works correctly."""
    print("\n🔧 Testing DatasetLoader...")
    
    try:
        loader = DatasetLoader()
        datasets = loader.list_available_datasets()
        print(f"  Available datasets: {datasets}")
        
        if not datasets:
            print("  ⚠️  Warning: No datasets found")
            return False
        
        # Test loading first dataset
        dataset_name = datasets[0]
        print(f"  Testing dataset: {dataset_name}")
        
        X_train, X_test, y_train, y_test, feature_names = loader.load_dataset(dataset_name)
        
        print(f"  Dataset loaded successfully:")
        print(f"    Train shape: {X_train.shape}")
        print(f"    Test shape: {X_test.shape}")
        print(f"    Features: {len(feature_names)}")
        print(f"    Classes: {len(set(y_train))}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ DatasetLoader test failed: {e}")
        return False


def test_model_creation():
    """Test if model wrappers work correctly."""
    print("\n🔧 Testing Model Creation...")
    
    try:
        # Test BaggingClassifier
        bagging_model = create_bagging_classifier({'n_estimators': 5, 'random_state': 42})
        print(f"  ✅ BaggingClassifier created: {bagging_model.name}")
        
        # Test RandomForestClassifier  
        rf_model = create_random_forest_classifier({'n_estimators': 5, 'random_state': 42})
        print(f"  ✅ RandomForestClassifier created: {rf_model.name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation test failed: {e}")
        return False


def test_model_training():
    """Test if models can be trained on sample data."""
    print("\n🔧 Testing Model Training...")
    
    try:
        # Load a small dataset
        loader = DatasetLoader()
        datasets = loader.list_available_datasets()
        
        if not datasets:
            print("  ⚠️  No datasets available for training test")
            return False
        
        dataset_name = datasets[0]
        X_train, X_test, y_train, y_test, feature_names = loader.load_dataset(dataset_name)
        
        # Use smaller subset for quick testing
        X_train_small = X_train[:min(100, len(X_train))]
        y_train_small = y_train[:min(100, len(y_train))]
        X_test_small = X_test[:min(50, len(X_test))]
        y_test_small = y_test[:min(50, len(y_test))]
        
        # Test BaggingClassifier
        print("  Testing BaggingClassifier training...")
        bagging_model = create_bagging_classifier({'n_estimators': 3, 'random_state': 42})
        bagging_model.fit(X_train_small, y_train_small)
        
        predictions = bagging_model.predict(X_test_small)
        accuracy = sum(predictions == y_test_small) / len(y_test_small)
        print(f"    BaggingClassifier accuracy: {accuracy:.4f}")
        
        # Test RandomForestClassifier
        print("  Testing RandomForestClassifier training...")
        rf_model = create_random_forest_classifier({'n_estimators': 3, 'random_state': 42})
        rf_model.fit(X_train_small, y_train_small)
        
        predictions = rf_model.predict(X_test_small)
        accuracy = sum(predictions == y_test_small) / len(y_test_small)
        print(f"    RandomForestClassifier accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_runner_initialization():
    """Test BaselineExperimentRunner initialization."""
    print("\n🔧 Testing BaselineExperimentRunner Initialization...")
    
    try:
        runner = BaselineExperimentRunner(results_dir="results/test_baseline")
        print(f"  ✅ Runner created successfully")
        print(f"  Model configurations: {list(runner.model_configs.keys())}")
        print(f"  Results directory: {runner.results_dir}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Runner initialization failed: {e}")
        return False


def test_single_experiment():
    """Test running a single experiment."""
    print("\n🔧 Testing Single Dataset Experiment...")
    
    try:
        runner = BaselineExperimentRunner(results_dir="results/test_baseline")
        datasets = runner.data_loader.list_available_datasets()
        
        if not datasets:
            print("  ⚠️  No datasets available for experiment test")
            return False
        
        dataset_name = datasets[0]
        print(f"  Running experiment on: {dataset_name}")
        
        # Run experiment on first dataset
        results = runner.run_single_dataset_experiment(dataset_name)
        
        if 'error' in results:
            print(f"  ❌ Experiment failed: {results['error']}")
            return False
        
        print(f"  ✅ Experiment completed successfully!")
        print(f"  Models tested: {list(results.keys())}")
        
        # Show some results
        for model_name, model_results in results.items():
            if 'accuracy' in model_results:
                print(f"    {model_name}: Accuracy={model_results['accuracy']:.4f}, F1={model_results['f1_weighted']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Single experiment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests in sequence."""
    print("🚀 Starting BaselineExperimentRunner Launch Tests")
    print("=" * 60)
    
    tests = [
        ("Data Loader", test_data_loader),
        ("Model Creation", test_model_creation),
        ("Model Training", test_model_training),
        ("Runner Initialization", test_baseline_runner_initialization),
        ("Single Experiment", test_single_experiment)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! BaselineExperimentRunner is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


def quick_functionality_test():
    """Quick test to verify basic functionality."""
    print("⚡ Quick Functionality Test")
    print("-" * 30)
    
    try:
        # Test imports
        runner = BaselineExperimentRunner(results_dir="results/test_quick")
        datasets = runner.data_loader.list_available_datasets()
        
        if datasets:
            print(f"✅ Found {len(datasets)} datasets: {datasets}")
            print("✅ BaselineExperimentRunner is functional!")
            return True
        else:
            print("⚠️  No datasets found, but runner is functional")
            return True
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            success = quick_functionality_test()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(__doc__)
            print("\nUsage:")
            print("  python test_baseline_runner.py           # Run all tests")
            print("  python test_baseline_runner.py --quick   # Quick functionality test")
            return
        else:
            print("Unknown argument. Use --help for usage.")
            return
    else:
        success = run_all_tests()
    
    if success:
        print("\n🎉 Tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 