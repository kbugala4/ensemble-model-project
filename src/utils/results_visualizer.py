#!/usr/bin/env python3
"""
Results Visualizer for Ensemble Methods

This class creates horizontal range plots showing min/max performance ranges
for RandomForestClassifier and BaggingClassifier across different metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Optional


class EnsembleResultsVisualizer:
    """
    Visualizer for ensemble method results showing min/max ranges on horizontal axis.
    
    Groups results by ensemble type (RandomForest vs Bagging) and shows performance
    ranges for each metric across different ensemble sizes.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing the results CSV files
        """
        self.results_dir = Path(results_dir)
        self.data = None
        self.metrics = [
            ('accuracy', 'Accuracy'),
            ('f1_weighted', 'F1 Weighted'),
            ('f1_macro', 'F1 Macro'),
            ('precision_weighted', 'Precision Weighted'),
            ('recall_weighted', 'Recall Weighted')
            # ('roc_auc', 'AUC')
        ]
        
    def load_data(self) -> pd.DataFrame:
        """Load and combine data from baseline and custom CSV files."""
        
        baseline_dir = self.results_dir / "baseline"
        custom_dir = self.results_dir / "custom"
        
        combined_data = []
        
        # Load baseline data
        if baseline_dir.exists():
            baseline_csv = self._find_latest_csv_files(baseline_dir)
            if baseline_csv:
                print(f"📊 Loading baseline data from: {baseline_csv.name}")
                baseline_df = pd.read_csv(baseline_csv)
                baseline_df['experiment_type'] = 'BASELINE'
                combined_data.append(baseline_df)
        
        # Load custom data
        if custom_dir.exists():
            custom_csv = self._find_latest_csv_files(custom_dir)
            if custom_csv:
                print(f"📊 Loading custom data from: {custom_csv.name}")
                custom_df = pd.read_csv(custom_csv)
                if 'experiment_type' not in custom_df.columns:
                    custom_df['experiment_type'] = 'CUSTOM'
                combined_data.append(custom_df)
        
        if not combined_data:
            raise ValueError("❌ No CSV data found in either baseline or custom directories")
        
        # Combine all data
        self.data = pd.concat(combined_data, ignore_index=True, sort=False)
        return self.data
    
    def _find_latest_csv_files(self, results_dir: Path) -> Optional[Path]:
        """Find the most recent CSV files in the results directory."""
        csv_files = glob.glob(str(results_dir / "*.csv"))
        if not csv_files:
            return None
        
        # Return the most recent file based on modification time
        latest_file = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
        return Path(latest_file)
    
    def _categorize_models(self, model_name: str) -> str:
        """Categorize model by ensemble type."""
        if 'random_forest' in model_name.lower():
            return 'Random Forest'
        elif 'bagging' in model_name.lower():
            return 'Bagging'
        else:
            return 'Other'
    
    def _get_ensemble_ranges(self, dataset_df: pd.DataFrame, metric: str, use_test: bool = True) -> Dict:
        """
        Calculate min/max ranges for each ensemble type for a given metric.
        
        Args:
            dataset_df: DataFrame filtered for specific dataset
            metric: Metric name to analyze
            use_test: Whether to use test scores (True) or train scores (False)
        
        Returns:
            Dictionary with ensemble types as keys and (min, max) tuples as values
        """
        metric_col = metric if use_test else f"{metric}_train"
        
        if metric_col not in dataset_df.columns:
            return {}
        
        ranges = {}
        
        # Group by ensemble type
        for model in dataset_df['model'].unique():
            ensemble_type = self._categorize_models(model)
            
            if ensemble_type in ['Random Forest', 'Bagging']:
                model_data = dataset_df[dataset_df['model'] == model]
                values = model_data[metric_col].dropna()
                
                if not values.empty:
                    if ensemble_type not in ranges:
                        ranges[ensemble_type] = {'values': [], 'models': []}
                    
                    ranges[ensemble_type]['values'].extend(values.tolist())
                    ranges[ensemble_type]['models'].append(model)
        
        # Calculate min/max for each ensemble type
        final_ranges = {}
        for ensemble_type, data in ranges.items():
            if data['values']:
                final_ranges[ensemble_type] = {
                    'min': min(data['values']),
                    'max': max(data['values']),
                    'models': data['models']
                }
        
        return final_ranges
    
    def create_custom_experiment_plots(self, save_plots: bool = False) -> Dict[str, Dict[str, plt.Figure]]:
        """
        Create individual plots for each custom experiment showing comparison with baseline ranges.
        
        Args:
            save_plots: Whether to save plots to files
        
        Returns:
            Nested dictionary: {dataset: {experiment: figure}}
        """
        if self.data is None:
            self.load_data()
        
        # Separate baseline and custom data
        baseline_data = self.data[self.data['experiment_type'] == 'BASELINE'].copy()
        custom_data = self.data[self.data['experiment_type'] == 'CUSTOM'].copy()
        
        if baseline_data.empty:
            print("⚠️  Warning: No baseline data found")
            return {}
        
        if custom_data.empty:
            print("⚠️  Warning: No custom experiment data found")
            return {}
        
        available_datasets = sorted(self.data['dataset'].unique())
        all_figures = {}
        
        for dataset in available_datasets:
            print(f"\n📊 Creating plots for {dataset} dataset...")
            
            dataset_baseline = baseline_data[baseline_data['dataset'] == dataset]
            dataset_custom = custom_data[custom_data['dataset'] == dataset]
            
            if dataset_baseline.empty:
                print(f"⚠️  No baseline data for {dataset}")
                continue
                
            if dataset_custom.empty:
                print(f"⚠️  No custom experiments for {dataset}")
                continue
            
            dataset_figures = {}
            
            # Create a plot for each custom experiment
            for custom_experiment in dataset_custom['model'].unique():
                print(f"  📈 Creating plot for {custom_experiment}...")
                
                fig = self._create_single_custom_experiment_plot(
                    dataset, dataset_baseline, dataset_custom, custom_experiment
                )
                
                if fig is not None:
                    dataset_figures[custom_experiment] = fig
                    
                    if save_plots:
                        output_dir = Path("plots/custom_comparisons")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        filename = f"{dataset}_{custom_experiment}_vs_baseline.png"
                        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
                        print(f"    💾 Saved: {output_dir / filename}")
            
            if dataset_figures:
                all_figures[dataset] = dataset_figures
        
        total_plots = sum(len(figs) for figs in all_figures.values())
        print(f"\n✅ Created {total_plots} custom experiment comparison plots")
        
        return all_figures
    
    def show_summary_stats(self):
        """Print summary statistics about the loaded data."""
        if self.data is None:
            self.load_data()
        
        print("\n📈 ENSEMBLE METHODS SUMMARY")
        print("=" * 50)
        
        for dataset in sorted(self.data['dataset'].unique()):
            dataset_df = self.data[self.data['dataset'] == dataset]
            print(f"\n{dataset.upper()}:")
            
            # Group models by ensemble type
            rf_models = [m for m in dataset_df['model'].unique() if 'random_forest' in m.lower()]
            bagging_models = [m for m in dataset_df['model'].unique() if 'bagging' in m.lower()]
            
            print(f"  Random Forest models: {len(rf_models)} ({', '.join(rf_models)})")
            print(f"  Bagging models: {len(bagging_models)} ({', '.join(bagging_models)})")
            
            # Show best performers
            if 'accuracy' in dataset_df.columns:
                best_acc_idx = dataset_df['accuracy'].idxmax()
                if not pd.isna(best_acc_idx):
                    best_model = dataset_df.loc[best_acc_idx, 'model']
                    best_acc = dataset_df.loc[best_acc_idx, 'accuracy']
                    print(f"  Best test accuracy: {best_acc:.4f} ({best_model})")
    
    def _create_single_custom_experiment_plot(self, dataset: str, baseline_df: pd.DataFrame, 
                                            custom_df: pd.DataFrame, custom_experiment: str,
                                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a single plot comparing one custom experiment to baseline ranges.
        
        Args:
            dataset: Dataset name
            baseline_df: Baseline data for this dataset
            custom_df: Custom data for this dataset  
            custom_experiment: Name of the custom experiment
            figsize: Figure size
        
        Returns:
            matplotlib Figure object
        """
        # Get custom experiment data
        custom_exp_data = custom_df[custom_df['model'] == custom_experiment]
        
        if custom_exp_data.empty:
            return None
        
        # Create subplots - one for each metric
        available_metrics = []
        for metric_code, metric_title in self.metrics:
            if (metric_code in baseline_df.columns and not baseline_df[metric_code].isna().all() and
                metric_code in custom_exp_data.columns and not custom_exp_data[metric_code].isna().all()):
                available_metrics.append((metric_code, metric_title))
        
        if not available_metrics:
            print(f"    ⚠️  No available metrics for {custom_experiment}")
            return None
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        colors = {
            'Random Forest': '#2E8B57', 
            'Bagging': '#4169E1',
            'Custom': '#FF6B35'
        }
        
        for i, (metric_code, metric_title) in enumerate(available_metrics):
            ax = axes[i]
            
            # Get baseline ranges
            baseline_ranges = self._get_ensemble_ranges(baseline_df, metric_code, use_test=True)
            
            # Get custom experiment value
            custom_values = custom_exp_data[metric_code].dropna()
            if custom_values.empty:
                continue
                
            custom_value = custom_values.mean()  # Average if multiple runs
            
            y_pos = 0
            
            # Plot baseline ranges
            for ensemble_type in ['Bagging', 'Random Forest']:
                if ensemble_type in baseline_ranges:
                    range_data = baseline_ranges[ensemble_type]
                    min_val = range_data['min']
                    max_val = range_data['max']
                    
                    # Plot the baseline range line
                    ax.plot([min_val, max_val], [y_pos, y_pos], 
                           color=colors[ensemble_type], 
                           linewidth=8, alpha=0.7, label=f'Baseline {ensemble_type}')
                    
                    # Plot min and max points
                    ax.scatter([min_val, max_val], [y_pos, y_pos], 
                              color=colors[ensemble_type], 
                              s=100, zorder=5)
                    
                    # Add value annotations
                    ax.annotate(f'{min_val:.4f}', (min_val, y_pos), 
                               xytext=(0, -15), textcoords='offset points',
                               ha='center', va='top', fontsize=9)
                    ax.annotate(f'{max_val:.4f}', (max_val, y_pos), 
                               xytext=(0, -15), textcoords='offset points',
                               ha='center', va='top', fontsize=9)
                    
                    y_pos += 1
            
            # Plot custom experiment point
            ax.scatter([custom_value], [y_pos], 
                      color=colors['Custom'], s=150, zorder=10, 
                      marker='D', label=f'{custom_experiment}')
            
            # Add custom value annotation
            ax.annotate(f'{custom_value:.4f}', (custom_value, y_pos), 
                       xytext=(0, -15), textcoords='offset points',
                       ha='center', va='top', fontsize=9, weight='bold')
            
            # Formatting
            ax.set_ylim(-0.5, y_pos + 0.5)
            ax.set_yticks(range(y_pos + 1))
            
            y_labels = []
            for ensemble_type in ['Bagging', 'Random Forest']:
                if ensemble_type in baseline_ranges:
                    y_labels.append(f'Baseline {ensemble_type}')
            y_labels.append(custom_experiment)
            
            ax.set_yticklabels(y_labels)
            ax.set_xlabel(f'{metric_title} Score')
            ax.set_title(f'{metric_title} - {custom_experiment} vs Baseline')
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_axisbelow(True)
            # Set fixed X-axis range from 0 to 1
            ax.set_xlim(0.7, 1)
            ax.set_xticks([0.7, 0.8, 0.9, 1])
            ax.set_xticklabels(['0.7', '0.8', '0.9', '1'])
        
        # Add single legend for the entire figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), framealpha=0.9)
        
        # Overall title
        fig.suptitle(f'{dataset.upper()} - {custom_experiment} vs Baseline Ensemble Methods', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return fig


# Example usage function
def main():
    """Example usage of the EnsembleResultsVisualizer."""
    
    # Create visualizer instance
    visualizer = EnsembleResultsVisualizer()
    
    # Load data and show summary
    visualizer.load_data()
    visualizer.show_summary_stats()
    
    # Create custom experiment comparison plots
    custom_figures = visualizer.create_custom_experiment_plots(save_plots=True)
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main() 