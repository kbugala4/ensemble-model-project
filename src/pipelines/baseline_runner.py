import sys
import os
import datetime
from typing import Any, Dict
import json
import pandas as pd
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.dataloader import DatasetLoader
from utils.metrics_saver import MetricsSaver
from models.sklearn_wrappers import create_bagging_classifier, create_random_forest_classifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, log_loss
)

class BaselineExperimentRunner:
    
    def __init__(self, results_dir: str = "results/baseline"):
        self.results_dir = results_dir
        self.data_loader = DatasetLoader()
        
        # Initialize MetricsSaver for baseline experiments
        self.metrics_saver = MetricsSaver(
            experiment_type="BASELINE", 
            results_dir=results_dir
        )
        
        # Define model configurations
        # TO BE EXTRACTED TO BASELINE CONFIG
        self.model_configs = {
            'bagging_5': {
                'n_estimators': 10,
                'random_state': 42
            },
            'bagging_10': {
                'n_estimators': 10,
                'random_state': 42
            },
            'bagging_20': {
                'n_estimators': 20, 
                'random_state': 42
            },
            # 'bagging_100': {
            #     'n_estimators': 100,
            #     'random_state': 42
            # },
            'random_forest_5': {
                'n_estimators': 10,
                'random_state': 42
            },
            'random_forest_10': {
                'n_estimators': 10,
                'random_state': 42
            },
            'random_forest_20': {
                'n_estimators': 20,
                'random_state': 42
            },
            # 'random_forest_100': {
            #     'n_estimators': 100,
            #     'random_state': 42
            # }
        }
    
    def run_full_baseline_experiments(self) -> Dict[str, Any]:
        """
        Run full baseline experiments for all datasets.
        
        Returns:
            Dict[str, Any]: Dictionary containing results for each dataset and model configuration
        """
        print("🔬 Starting Full Baseline Experiments")
        print("=" * 50)
        
        for dataset_name in self.data_loader.list_available_datasets():
            print(f"\n📊 Processing dataset: {dataset_name}")
            
            try:
                # Load dataset
                X_train, X_test, y_train, y_test, feature_columns = self.data_loader.load_dataset(dataset_name)
                
                for model_name, model_config in self.model_configs.items():
                    print(f"  Training {model_name}...")
                    
                    try:
                        # Create model
                        if 'bagging' in model_name:
                            model = create_bagging_classifier(model_config)
                        elif 'random_forest' in model_name:
                            model = create_random_forest_classifier(model_config)
                        else:
                            raise ValueError(f"Invalid model name: {model_name}")
                        
                        # Train and evaluate model
                        model_results = self._train_and_evaluate_model(
                            model, X_train, X_test, y_train, y_test, feature_columns
                        )
                        
                        # Add results to metrics saver
                        self.metrics_saver.add_experiment_results(
                            dataset_name, model_name, model_results
                        )
                        
                        # Print quick results
                        print(f"    ✅ Accuracy (Test): {model_results.get('accuracy', 0):.4f}, "
                              f"F1 (Test): {model_results.get('f1_weighted', 0):.4f}")
                        print(f"    ✅ Accuracy (Train): {model_results.get('accuracy_train', 0):.4f}, "
                              f"F1 (Train): {model_results.get('f1_weighted_train', 0):.4f}")
                        
                    
                    except Exception as e:
                        print(f"    ❌ Error training {model_name}: {str(e)}")
                        error_result = {
                            'error': str(e),
                            'model_name': model_name,
                            'dataset': dataset_name
                        }
                        self.metrics_saver.add_experiment_results(
                            dataset_name, model_name, error_result
                        )
            
            except Exception as e:
                print(f"❌ Error processing dataset {dataset_name}: {str(e)}")
                dataset_error = {
                    'error': f"Dataset loading failed: {str(e)}",
                    'dataset': dataset_name
                }
                self.metrics_saver.add_dataset_results(dataset_name, dataset_error)
        
        # Print summary and save results
        self.metrics_saver.print_summary()
        json_path, csv_path = self.metrics_saver.save_results()
        
        print(f"\n✅ Full baseline experiments completed!")
        print(f"📄 Detailed results: {json_path}")
        print(f"📊 Summary: {csv_path}")
        
        return self.metrics_saver.get_results()
    
    def _train_and_evaluate_model(self, model, X_train, X_test, y_train, y_test, 
                                    feature_names) -> Dict[str, Any]:
        """Train model and calculate comprehensive metrics."""
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Get probability predictions (both RandomForest and Bagging support this)
        y_pred_proba = model.predict_proba(X_test)
        y_pred_proba_train = model.predict_proba(X_train)
        
        # Calculate test metrics (using simplified names for MetricsSaver compatibility)
        results = {
            # Standard metrics expected by MetricsSaver
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            
            # Training metrics (with _train suffix)
            'accuracy_train': float(accuracy_score(y_train, y_pred_train)),
            'precision_macro_train': float(precision_score(y_train, y_pred_train, average='macro', zero_division=0)),
            'precision_weighted_train': float(precision_score(y_train, y_pred_train, average='weighted', zero_division=0)),
            'recall_macro_train': float(recall_score(y_train, y_pred_train, average='macro', zero_division=0)),
            'recall_weighted_train': float(recall_score(y_train, y_pred_train, average='weighted', zero_division=0)),
            'f1_macro_train': float(f1_score(y_train, y_pred_train, average='macro', zero_division=0)),
            'f1_weighted_train': float(f1_score(y_train, y_pred_train, average='weighted', zero_division=0)),
            'confusion_matrix_train': confusion_matrix(y_train, y_pred_train).tolist(),
            'classification_report_train': classification_report(y_train, y_pred_train, output_dict=True),
            
            # Dataset info
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': len(feature_names),
            'n_classes': len(np.unique(y_train)),
            'model_params': model.get_params()
        }
        
        # Add AUC and log_loss metrics
        try:
            n_classes = len(np.unique(y_train))
            
            # For binary classification
            if n_classes == 2:
                results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
                results['roc_auc_train'] = float(roc_auc_score(y_train, y_pred_proba_train[:, 1]))
            # For multiclass classification
            else:
                results['roc_auc_macro'] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro'))
                results['roc_auc_weighted'] = float(roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted'))
                results['roc_auc_macro_train'] = float(roc_auc_score(y_train, y_pred_proba_train, multi_class='ovr', average='macro'))
                results['roc_auc_weighted_train'] = float(roc_auc_score(y_train, y_pred_proba_train, multi_class='ovr', average='weighted'))
            
            # Log loss (works for both binary and multiclass)
            results['log_loss'] = float(log_loss(y_test, y_pred_proba))
            results['log_loss_train'] = float(log_loss(y_train, y_pred_proba_train))
                
        except Exception as e:
            print(f"    Warning: Could not calculate AUC/log_loss metrics: {str(e)}")
        
        # Add feature importance if available
        if hasattr(model.model, 'feature_importances_'):
            importance_dict = {
                feature_names[i]: float(importance) 
                for i, importance in enumerate(model.model.feature_importances_)
            }
            results['feature_importance'] = importance_dict
            
            # Also add top 10 most important features
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            results['top_10_features'] = sorted_importance[:10]
        
        return results
    
    def get_best_models_per_dataset(self, metric: str = 'f1_weighted') -> Dict[str, Dict[str, Any]]:
        """
        Get the best performing model for each dataset.
        
        Args:
            metric: Metric to use for comparison (default: 'f1_weighted')
            
        Returns:
            Dictionary with dataset names as keys and best model info as values
        """
        return self.metrics_saver.get_best_models_per_dataset(metric)
    
    def clear_results(self) -> None:
        """Clear all collected results."""
        self.metrics_saver.clear_results()
    
    def get_results(self) -> Dict[str, Any]:
        """Get all collected results."""
        return self.metrics_saver.get_results()
    
    def save_current_results(self, filename_prefix: str = None) -> tuple[str, str]:
        """
        Save current results to files.
        
        Args:
            filename_prefix: Custom prefix for filenames
            
        Returns:
            Tuple of (json_filepath, csv_filepath)
        """
        return self.metrics_saver.save_results(filename_prefix)