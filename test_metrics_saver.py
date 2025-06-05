#!/usr/bin/env python3
"""
Test script to demonstrate usage of MetricsSaver class.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.metrics_saver import MetricsSaver


def test_baseline_saver():
    """Test MetricsSaver with BASELINE experiment type."""
    print("🔬 Testing MetricsSaver with BASELINE experiment type")
    print("=" * 50)
    
    # Initialize saver for baseline experiments
    saver = MetricsSaver(experiment_type="BASELINE")
    
    # Sample metrics data (similar to what would come from sklearn models)
    sample_metrics_1 = {
        'accuracy': 0.9207,
        'precision_macro': 0.9248,
        'precision_weighted': 0.9223,
        'recall_macro': 0.9151,
        'recall_weighted': 0.9207,
        'f1_macro': 0.9187,
        'f1_weighted': 0.9203,
        'confusion_matrix': [[14008, 565], [1494, 9909]],
        'n_samples_train': 103904,
        'n_samples_test': 25976,
        'n_features': 22,
        'n_classes': 2,
        'model_params': {
            'n_estimators': 10,
            'random_state': 42,
            'bootstrap': True
        }
    }
    
    sample_metrics_2 = {
        'accuracy': 0.9403,
        'precision_macro': 0.9452,
        'precision_weighted': 0.9421,
        'recall_macro': 0.9349,
        'recall_weighted': 0.9403,
        'f1_macro': 0.9388,
        'f1_weighted': 0.9399,
        'confusion_matrix': [[14263, 310], [1241, 10162]],
        'n_samples_train': 103904,
        'n_samples_test': 25976,
        'n_features': 22,
        'n_classes': 2,
        'model_params': {
            'n_estimators': 50,
            'random_state': 42,
            'bootstrap': True
        }
    }
    
    # Add individual experiment results
    saver.add_experiment_results("flights", "bagging_10", sample_metrics_1)
    saver.add_experiment_results("flights", "random_forest_50", sample_metrics_2)
    
    # Add another dataset with batch results
    credit_results = {
        "bagging_10": {
            'accuracy': 0.9996,
            'f1_weighted': 0.9996,
            'f1_macro': 0.9296,
            'n_samples_test': 45570,
            'n_features': 30,
            'n_classes': 2,
            'model_params': {'n_estimators': 10, 'random_state': 42}
        },
        "random_forest_100": {
            'accuracy': 0.9996,
            'f1_weighted': 0.9996,
            'f1_macro': 0.9251,
            'n_samples_test': 45570,
            'n_features': 30,
            'n_classes': 2,
            'model_params': {'n_estimators': 100, 'random_state': 42}
        }
    }
    
    saver.add_dataset_results("credit_cards", credit_results)
    
    # Print summary
    saver.print_summary()
    
    # Get best models
    best_models = saver.get_best_models_per_dataset()
    print(f"\n🏆 Best models per dataset:")
    for dataset, best in best_models.items():
        print(f"  {dataset}: {best['model_name']} (F1: {best['score']:.4f})")
    
    # Save results
    json_path, csv_path = saver.save_results()
    
    print(f"\n✅ Baseline test completed!")
    return saver


def test_custom_saver():
    """Test MetricsSaver with CUSTOM experiment type."""
    print("\n🔬 Testing MetricsSaver with CUSTOM experiment type")
    print("=" * 50)
    
    # Initialize saver for custom experiments
    saver = MetricsSaver(experiment_type="CUSTOM")
    
    # Sample custom ensemble results
    custom_metrics = {
        'accuracy': 0.9450,
        'f1_weighted': 0.9448,
        'f1_macro': 0.9401,
        'precision_weighted': 0.9455,
        'recall_weighted': 0.9450,
        'n_samples_test': 25976,
        'n_features': 22,
        'n_classes': 2,
        'ensemble_params': {
            'n_estimators': 25,
            'bootstrap_samples': 1000,
            'feature_selection': 'sqrt-selection',
            'random_state': 42
        }
    }
    
    saver.add_experiment_results("flights", "custom_ensemble_v1", custom_metrics)
    
    # Print summary
    saver.print_summary()
    
    # Save results with custom filename
    json_path, csv_path = saver.save_results(filename_prefix="custom_ensemble_v1")
    
    print(f"\n✅ Custom test completed!")
    return saver


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🔬 Testing edge cases")
    print("=" * 30)
    
    # Test empty saver
    empty_saver = MetricsSaver("BASELINE")
    empty_saver.print_summary()
    empty_saver.save_results()
    
    # Test with error results
    error_saver = MetricsSaver("CUSTOM")
    error_saver.add_experiment_results("test_dataset", "failed_model", {"error": "Model training failed"})
    error_saver.print_summary()
    
    print("✅ Edge cases test completed!")


if __name__ == "__main__":
    print("🧪 Testing MetricsSaver Class")
    print("=" * 60)
    
    # Test baseline functionality
    baseline_saver = test_baseline_saver()
    
    # Test custom functionality
    custom_saver = test_custom_saver()
    
    # Test edge cases
    test_edge_cases()
    
    print("\n🎉 All tests completed!")
    print("Check the 'results/' directory for generated files.") 