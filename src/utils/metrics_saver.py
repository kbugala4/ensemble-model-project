import os
import json
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional, Literal


class MetricsSaver:
    """
    A reusable class for saving experiment metrics to files.
    
    This class collects metrics from experiments and saves them in both
    JSON (detailed) and CSV (summary) formats. It supports different experiment
    types for proper file naming and organization.
    """
    
    def __init__(self, 
                 experiment_type: Literal["BASELINE", "CUSTOM"] = "BASELINE",
                 results_dir: Optional[str] = None):
        """
        Initialize the MetricsSaver.
        
        Args:
            experiment_type: Type of experiment for file naming ("BASELINE" or "CUSTOM")
            results_dir: Directory to save results. If None, uses default based on experiment type
        """
        self.experiment_type = experiment_type.upper()
        self.results_dir = results_dir or f"results/{experiment_type.lower()}"
        self.results = {}
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def add_experiment_results(self, 
                             dataset_name: str, 
                             model_name: str, 
                             metrics: Dict[str, Any]) -> None:
        """
        Add results from a single experiment.
        
        Args:
            dataset_name: Name of the dataset used
            model_name: Name/identifier of the model
            metrics: Dictionary containing all metrics and evaluation results
        """
        if dataset_name not in self.results:
            self.results[dataset_name] = {}
        
        self.results[dataset_name][model_name] = metrics
    
    def add_dataset_results(self, 
                          dataset_name: str, 
                          dataset_results: Dict[str, Any]) -> None:
        """
        Add results for an entire dataset (multiple models).
        
        Args:
            dataset_name: Name of the dataset
            dataset_results: Dictionary with model names as keys and metrics as values
        """
        self.results[dataset_name] = dataset_results
    
    def save_results(self, 
                    filename_prefix: Optional[str] = None,
                    include_timestamp: bool = True) -> tuple[str, str]:
        """
        Save all collected results to JSON and CSV files.
        
        Args:
            filename_prefix: Custom prefix for filenames. If None, uses experiment type
            include_timestamp: Whether to include timestamp in filenames
            
        Returns:
            Tuple of (json_filepath, csv_filepath)
        """
        if not self.results:
            print("⚠️  No results to save!")
            return "", ""
        
        # Create filenames
        prefix = filename_prefix or f"{self.experiment_type.lower()}_results"
        timestamp_suffix = f"_{self.timestamp}" if include_timestamp else ""
        
        json_filename = f"{prefix}{timestamp_suffix}.json"
        csv_filename = f"{prefix.replace('_results', '_summary')}{timestamp_suffix}.csv"
        
        json_filepath = os.path.join(self.results_dir, json_filename)
        csv_filepath = os.path.join(self.results_dir, csv_filename)
        
        # Save detailed JSON results
        self._save_json_results(json_filepath)
        
        # Save summary CSV results
        self._save_csv_summary(csv_filepath)
        
        print(f"✅ Results saved:")
        print(f"  📄 JSON (detailed): {json_filepath}")
        print(f"  📊 CSV (summary): {csv_filepath}")
        
        return json_filepath, csv_filepath
    
    def _save_json_results(self, filepath: str) -> None:
        """Save detailed results to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        except Exception as e:
            print(f"❌ Error saving JSON results: {e}")
    
    def _save_csv_summary(self, filepath: str) -> None:
        """Generate and save summary CSV with key metrics."""
        try:
            summary_data = []
            
            for dataset_name, dataset_results in self.results.items():
                if isinstance(dataset_results, dict) and 'error' not in dataset_results:
                    for model_name, model_results in dataset_results.items():
                        if isinstance(model_results, dict) and 'error' not in model_results:
                            # Extract key metrics
                            row = {
                                'dataset': dataset_name,
                                'model': model_name,
                                'experiment_type': self.experiment_type,
                                'timestamp': self.timestamp
                            }
                            
                            # Add common metrics (handle missing values gracefully)
                            metric_keys = [
                                'accuracy', 'f1_weighted', 'f1_macro',
                                'precision_weighted', 'precision_macro',
                                'recall_weighted', 'recall_macro',
                                'n_samples_test', 'n_samples_train',
                                'n_features', 'n_classes'
                            ]
                            
                            for key in metric_keys:
                                row[key] = model_results.get(key, None)
                            
                            # Add model-specific parameters if available
                            if 'model_params' in model_results:
                                params = model_results['model_params']
                                if isinstance(params, dict):
                                    row['n_estimators'] = params.get('n_estimators', None)
                                    row['random_state'] = params.get('random_state', None)
                            
                            summary_data.append(row)
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                df.to_csv(filepath, index=False)
            else:
                print("⚠️  No valid results to save in CSV format")
                
        except Exception as e:
            print(f"❌ Error saving CSV summary: {e}")
    
    def print_summary(self) -> None:
        """Print a summary of collected results."""
        if not self.results:
            print("📭 No results collected yet.")
            return
        
        print(f"\n📈 {self.experiment_type} EXPERIMENT SUMMARY")
        print("=" * 60)
        
        total_datasets = len(self.results)
        total_experiments = 0
        successful_experiments = 0
        
        for dataset_name, dataset_results in self.results.items():
            if isinstance(dataset_results, dict):
                if 'error' in dataset_results:
                    print(f"\n❌ Dataset: {dataset_name} - Error: {dataset_results['error']}")
                else:
                    print(f"\n🎯 Dataset: {dataset_name}")
                    
                    # Convert to list for sorting
                    model_results = []
                    for model_name, metrics in dataset_results.items():
                        total_experiments += 1
                        if isinstance(metrics, dict) and 'error' not in metrics:
                            successful_experiments += 1
                            f1_score = metrics.get('f1_weighted', 0)
                            accuracy = metrics.get('accuracy', 0)
                            model_results.append({
                                'model': model_name,
                                'accuracy': accuracy,
                                'f1_weighted': f1_score,
                                'f1_macro': metrics.get('f1_macro', 0)
                            })
                        else:
                            print(f"  ❌ {model_name}: Error in results")
                    
                    # Sort by F1 score and display
                    model_results.sort(key=lambda x: x['f1_weighted'], reverse=True)
                    
                    if model_results:
                        for result in model_results:
                            print(f"  {result['model']:20} | "
                                  f"Acc: {result['accuracy']:.4f} | "
                                  f"F1_w: {result['f1_weighted']:.4f} | "
                                  f"F1_m: {result['f1_macro']:.4f}")
                        
                        best_model = model_results[0]
                        print(f"  🏆 Best: {best_model['model']} (F1_weighted: {best_model['f1_weighted']:.4f})")
        
        print(f"\n📊 OVERALL STATISTICS")
        print("=" * 30)
        print(f"  Total datasets: {total_datasets}")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Successful experiments: {successful_experiments}")
        if total_experiments > 0:
            success_rate = (successful_experiments / total_experiments) * 100
            print(f"  Success rate: {success_rate:.1f}%")
    
    def get_results(self) -> Dict[str, Any]:
        """Get all collected results."""
        return self.results.copy()
    
    def clear_results(self) -> None:
        """Clear all collected results."""
        self.results.clear()
        print("🧹 Results cleared.")
    
    def get_best_models_per_dataset(self, metric: str = 'f1_weighted') -> Dict[str, Dict[str, Any]]:
        """
        Get the best performing model for each dataset based on specified metric.
        
        Args:
            metric: Metric to use for comparison (default: 'f1_weighted')
            
        Returns:
            Dictionary with dataset names as keys and best model info as values
        """
        best_models = {}
        
        for dataset_name, dataset_results in self.results.items():
            if isinstance(dataset_results, dict) and 'error' not in dataset_results:
                best_score = -1
                best_model = None
                
                for model_name, metrics in dataset_results.items():
                    if isinstance(metrics, dict) and 'error' not in metrics:
                        score = metrics.get(metric, -1)
                        if score > best_score:
                            best_score = score
                            best_model = {
                                'model_name': model_name,
                                'score': score,
                                'metric': metric,
                                'full_metrics': metrics
                            }
                
                if best_model:
                    best_models[dataset_name] = best_model
        
        return best_models 