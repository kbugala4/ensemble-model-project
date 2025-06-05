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
from models.sklearn_wrappers import create_bagging_classifier, create_random_forest_classifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

class BaselineExperimentRunner:
    
    def __init__(self, results_dir: str = "results/baseline"):
        self.results_dir = results_dir
        self.data_loader = DatasetLoader()  # Changed from dataset_loader to data_loader
        
        os.makedirs(self.results_dir, exist_ok=True)

         # Define model configurations
         # TO BE EXTRACED TO BASELINE CONFIG
        self.model_configs = {
            'bagging_10': {
                'n_estimators': 10,
                'random_state': 42
            },
            'bagging_50': {
                'n_estimators': 50, 
                'random_state': 42
            },
            'bagging_100': {
                'n_estimators': 100,
                'random_state': 42
            },
            'random_forest_10': {
                'n_estimators': 10,
                'random_state': 42
            },
            'random_forest_50': {
                'n_estimators': 50,
                'random_state': 42
            },
            'random_forest_100': {
                'n_estimators': 100,
                'random_state': 42
            }
        }
    
    def run_full_baseline_experiments(self) -> Dict[str, Any]:
        """
        Run full baseline experiments for all datasets.
        
        Returns:
            Dict[str, Any]: Dictionary containing results for each dataset and model configuration
        """
        results = {}
        experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for dataset_name in self.data_loader.list_available_datasets():
            print(f"Running experiments for dataset: {dataset_name}")
            
            #load dataset
            X_train, X_test, y_train, y_test, feature_columns = self.data_loader.load_dataset(dataset_name)
            
            dataset_results = {}
            
            for model_name, model_config in self.model_configs.items():
                print(f"Training {model_name} for {dataset_name}")
                
                if 'bagging' in model_name:
                    model = create_bagging_classifier(model_config)
                elif 'random_forest' in model_name:
                    model = create_random_forest_classifier(model_config)
                else:
                    raise ValueError(f"Invalid model name: {model_name}")
                
                model_results = self._train_and_evaluate_model(model, X_train, X_test, y_train, y_test, feature_columns)
                
                dataset_results[model_name] = model_results
            
            results[dataset_name] = dataset_results
        
          # Save results
        results_file = os.path.join(self.results_dir, f"baseline_results_{experiment_timestamp}.json")
        self._save_results(results, results_file)
        
        # Generate summary
        self._generate_summary_report(results, experiment_timestamp)
        
        print(f"\n✅ Experiment completed! Results saved to {self.results_dir}")
        return results

    def run_single_dataset_experiment(self, dataset_name: str) -> Dict[str, Any]:
        """
        Run experiment on a single dataset for testing purposes.
        
        Args:
            dataset_name (str): Name of the dataset to test
            
        Returns:
            Dict[str, Any]: Results for the specific dataset
        """
        print(f"🧪 Running single dataset experiment: {dataset_name}")
        
        try:
            # Load dataset
            X_train, X_test, y_train, y_test, feature_names = self.data_loader.load_dataset(dataset_name)
            
            results = {}
            
            # Test only a subset of models for quick testing
            test_configs = {
                'bagging_10': self.model_configs['bagging_10'],
                'random_forest_10': self.model_configs['random_forest_10']
            }
            
            for model_name, model_params in test_configs.items():
                print(f"  Testing {model_name}...")
                
                if 'bagging' in model_name:
                    model = create_bagging_classifier(model_params)
                else:
                    model = create_random_forest_classifier(model_params)
                
                model_results = self._train_and_evaluate_model(
                    model, X_train, X_test, y_train, y_test, feature_names
                )
                
                results[model_name] = model_results
                print(f"    Accuracy: {model_results['accuracy']:.4f}, F1: {model_results['f1_weighted']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Error in single dataset experiment: {e}")
            return {'error': str(e)}
    
    def _train_and_evaluate_model(self, model, X_train, X_test, y_train, y_test, 
                                    feature_names) -> Dict[str, Any]:
            """Train model and calculate comprehensive metrics."""
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                pass  # Some models might not support predict_proba
            
            # Calculate metrics
            results = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
                'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
                'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
                'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'n_samples_train': len(X_train),
                'n_samples_test': len(X_test),
                'n_features': len(feature_names),
                'n_classes': len(np.unique(y_train)),
                'model_params': model.get_params()
            }
            
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
    
    def _save_results(self, results: Dict, filepath: str) -> None:
        """Save results to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _generate_summary_report(self, results: Dict, timestamp: str) -> None:
        """Generate summary report with key findings."""
        
        print("\n📈 Generating Summary Report...")
        
        summary_data = []
        
        for dataset_name, dataset_results in results.items():
            if 'error' in dataset_results:
                continue
                
            for model_name, model_results in dataset_results.items():
                if 'error' not in model_results:
                    summary_data.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'accuracy': model_results['accuracy'],
                        'f1_weighted': model_results['f1_weighted'],
                        'f1_macro': model_results['f1_macro'],
                        'precision_weighted': model_results['precision_weighted'],
                        'recall_weighted': model_results['recall_weighted'],
                        'n_samples_test': model_results['n_samples_test'],
                        'n_features': model_results['n_features'],
                        'n_classes': model_results['n_classes']
                    })
        
        if not summary_data:
            print("No valid results to summarize.")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary CSV
        summary_file = os.path.join(self.results_dir, f"baseline_summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary CSV saved to: {summary_file}")
        
        # Print summary
        print("\n📈 BASELINE EXPERIMENT SUMMARY")
        print("=" * 60)
        
        for dataset in summary_df['dataset'].unique():
            print(f"\n🎯 Dataset: {dataset}")
            dataset_data = summary_df[summary_df['dataset'] == dataset]
            
            # Sort by F1 score for better readability
            dataset_data = dataset_data.sort_values('f1_weighted', ascending=False)
            
            print(dataset_data[['model', 'accuracy', 'f1_weighted', 'f1_macro']].to_string(index=False, float_format='%.4f'))
            
            # Find best model for this dataset
            if len(dataset_data) > 0:
                best_model = dataset_data.iloc[0]
                print(f"  🏆 Best model: {best_model['model']} (F1_weighted: {best_model['f1_weighted']:.4f})")
        
        # Overall statistics
        print("\n📊 OVERALL STATISTICS")
        print("=" * 30)
        
        # Best models across all datasets
        print("\n🏆 Best F1-weighted scores per dataset:")
        for dataset in summary_df['dataset'].unique():
            dataset_data = summary_df[summary_df['dataset'] == dataset]
            best = dataset_data.loc[dataset_data['f1_weighted'].idxmax()]
            print(f"  {dataset}: {best['model']} ({best['f1_weighted']:.4f})")
        
        # Model type performance
        print("\n🔍 Average performance by model type:")
        bagging_results = summary_df[summary_df['model'].str.contains('bagging')]
        rf_results = summary_df[summary_df['model'].str.contains('random_forest')]
        
        if len(bagging_results) > 0:
            print(f"  Bagging average F1: {bagging_results['f1_weighted'].mean():.4f}")
        if len(rf_results) > 0:
            print(f"  Random Forest average F1: {rf_results['f1_weighted'].mean():.4f}")