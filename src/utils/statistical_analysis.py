import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings


class StatisticalAnalyzer:
    """
    A comprehensive class for statistical analysis of machine learning experiment results.
    
    This class can compare baseline and ensemble methods using various statistical tests,
    calculate effect sizes, generate confidence intervals, and create visualizations.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical analyzer.
        
        Args:
            alpha: Significance level for statistical tests (default: 0.05)
        """
        self.alpha = alpha
        self.results = {}
        self.comparison_data = None
        
    def load_experiment_results(self, 
                              baseline_dir: str = "results/baseline",
                              ensemble_dir: str = "results/custom") -> None:
        """
        Load results from baseline and ensemble experiments.
        
        Args:
            baseline_dir: Directory containing baseline experiment results
            ensemble_dir: Directory containing ensemble experiment results
        """
        print("📊 Loading experiment results...")
        
        # Load baseline results
        baseline_results = self._load_latest_results(baseline_dir, "baseline")
        if baseline_results:
            self.results['baseline'] = baseline_results
            print(f"✅ Loaded baseline results: {len(baseline_results)} datasets")
        else:
            print("⚠️  No baseline results found")
            
        # Load ensemble results
        ensemble_results = self._load_latest_results(ensemble_dir, "custom")
        if ensemble_results:
            self.results['ensemble'] = ensemble_results
            print(f"✅ Loaded ensemble results: {len(ensemble_results)} datasets")
        else:
            print("⚠️  No ensemble results found")
            
        # Prepare comparison data
        if 'baseline' in self.results and 'ensemble' in self.results:
            self._prepare_comparison_data()
            print(f"📈 Prepared comparison data for analysis")
        else:
            print("❌ Cannot perform comparison - missing baseline or ensemble results")
    
    def _load_latest_results(self, results_dir: str, exp_type: str) -> Optional[Dict]:
        """Load the most recent results from a directory."""
        try:
            results_path = Path(results_dir)
            if not results_path.exists():
                return None
                
            # Find the most recent JSON file
            json_files = list(results_path.glob(f"{exp_type}_results_*.json"))
            if not json_files:
                return None
                
            latest_file = max(json_files, key=os.path.getctime)
            
            with open(latest_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error loading results from {results_dir}: {e}")
            return None
    
    def _prepare_comparison_data(self) -> None:
        """Prepare data for statistical comparison."""
        comparison_data = []
        
        baseline_data = self.results['baseline']
        ensemble_data = self.results['ensemble']
        
        # Find common datasets
        common_datasets = set(baseline_data.keys()) & set(ensemble_data.keys())
        
        for dataset in common_datasets:
            if 'error' in baseline_data[dataset] or 'error' in ensemble_data[dataset]:
                continue
                
            # Extract baseline metrics
            for model_name, metrics in baseline_data[dataset].items():
                if 'error' not in metrics:
                    comparison_data.append({
                        'dataset': dataset,
                        'method_type': 'baseline',
                        'model_name': model_name,
                        'accuracy': metrics.get('accuracy', np.nan),
                        'f1_weighted': metrics.get('f1_weighted', np.nan),
                        'f1_macro': metrics.get('f1_macro', np.nan),
                        'precision_weighted': metrics.get('precision_weighted', np.nan),
                        'recall_weighted': metrics.get('recall_weighted', np.nan),
                        'n_samples_test': metrics.get('n_samples_test', np.nan),
                        'n_features': metrics.get('n_features', np.nan)
                    })
            
            # Extract ensemble metrics
            for model_name, metrics in ensemble_data[dataset].items():
                if 'error' not in metrics:
                    comparison_data.append({
                        'dataset': dataset,
                        'method_type': 'ensemble',
                        'model_name': model_name,
                        'accuracy': metrics.get('accuracy', np.nan),
                        'f1_weighted': metrics.get('f1_weighted', np.nan),
                        'f1_macro': metrics.get('f1_macro', np.nan),
                        'precision_weighted': metrics.get('precision_weighted', np.nan),
                        'recall_weighted': metrics.get('recall_weighted', np.nan),
                        'n_samples_test': metrics.get('n_samples_test', np.nan),
                        'n_features': metrics.get('n_features', np.nan)
                    })
        
        self.comparison_data = pd.DataFrame(comparison_data)
    
    def perform_paired_comparison(self, 
                                metric: str = 'f1_weighted',
                                test_type: str = 'auto') -> Dict[str, Any]:
        """
        Perform paired statistical comparison between baseline and ensemble methods.
        
        Args:
            metric: Metric to compare (default: 'f1_weighted')
            test_type: Statistical test to use ('auto', 'ttest', 'wilcoxon')
            
        Returns:
            Dictionary with statistical test results
        """
        if self.comparison_data is None:
            raise ValueError("No comparison data available. Load experiment results first.")
        
        print(f"🔬 Performing paired comparison on {metric}")
        
        # Get best performing model for each dataset and method type
        best_results = self.comparison_data.groupby(['dataset', 'method_type'])[metric].max().reset_index()
        
        # Pivot to get baseline vs ensemble
        pivot_data = best_results.pivot(index='dataset', columns='method_type', values=metric)
        
        if 'baseline' not in pivot_data.columns or 'ensemble' not in pivot_data.columns:
            raise ValueError("Missing baseline or ensemble data for comparison")
        
        # Remove datasets with missing data
        clean_data = pivot_data.dropna()
        
        if len(clean_data) < 2:
            print("⚠️  Insufficient data for statistical testing (need at least 2 datasets)")
            return {'error': 'Insufficient data'}
        
        baseline_scores = clean_data['baseline'].values
        ensemble_scores = clean_data['ensemble'].values
        
        # Calculate basic statistics
        baseline_mean = np.mean(baseline_scores)
        ensemble_mean = np.mean(ensemble_scores)
        difference = ensemble_scores - baseline_scores
        mean_difference = np.mean(difference)
        
        # Choose statistical test
        if test_type == 'auto':
            # Use normality test to decide
            _, p_normal = stats.shapiro(difference)
            test_type = 'ttest' if p_normal > 0.05 else 'wilcoxon'
        
        # Perform statistical test
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_rel(ensemble_scores, baseline_scores)
            test_name = "Paired t-test"
        else:
            # Wilcoxon signed-rank test
            statistic, p_value = wilcoxon(ensemble_scores, baseline_scores, alternative='two-sided')
            test_name = "Wilcoxon signed-rank test"
        
        # Calculate effect size (Cohen's d for paired data)
        effect_size = mean_difference / np.std(difference, ddof=1)
        
        # Calculate confidence interval for mean difference
        se_diff = stats.sem(difference)
        ci_lower, ci_upper = stats.t.interval(
            1 - self.alpha, 
            len(difference) - 1, 
            loc=mean_difference, 
            scale=se_diff
        )
        
        # Determine practical significance
        if abs(effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        results = {
            'metric': metric,
            'test_name': test_name,
            'n_datasets': len(clean_data),
            'baseline_mean': baseline_mean,
            'ensemble_mean': ensemble_mean,
            'mean_difference': mean_difference,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'effect_interpretation': effect_interpretation,
            'confidence_interval': (ci_lower, ci_upper),
            'datasets': clean_data.index.tolist(),
            'baseline_scores': baseline_scores.tolist(),
            'ensemble_scores': ensemble_scores.tolist()
        }
        
        return results
    
    def perform_multiple_metric_comparison(self, 
                                         metrics: List[str] = None) -> Dict[str, Any]:
        """
        Perform comparison across multiple metrics with multiple comparison correction.
        
        Args:
            metrics: List of metrics to compare (default: common metrics)
            
        Returns:
            Dictionary with results for all metrics
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted']
        
        print(f"🔬 Performing multiple metric comparison: {metrics}")
        
        results = {}
        p_values = []
        
        # Perform tests for each metric
        for metric in metrics:
            try:
                metric_results = self.perform_paired_comparison(metric, test_type='auto')
                if 'error' not in metric_results:
                    results[metric] = metric_results
                    p_values.append(metric_results['p_value'])
                else:
                    print(f"⚠️  Skipping {metric}: {metric_results['error']}")
            except Exception as e:
                print(f"⚠️  Error analyzing {metric}: {e}")
        
        # Apply Bonferroni correction
        if p_values:
            corrected_alpha = self.alpha / len(p_values)
            bonferroni_significant = [p < corrected_alpha for p in p_values]
            
            # Update significance with correction
            for i, metric in enumerate(results.keys()):
                results[metric]['bonferroni_significant'] = bonferroni_significant[i]
                results[metric]['corrected_alpha'] = corrected_alpha
        
        return {
            'individual_results': results,
            'bonferroni_correction': {
                'original_alpha': self.alpha,
                'corrected_alpha': corrected_alpha if p_values else None,
                'n_comparisons': len(p_values)
            }
        }
    
    def perform_friedman_test(self, metric: str = 'f1_weighted') -> Dict[str, Any]:
        """
        Perform Friedman test for comparing multiple algorithms across datasets.
        
        Args:
            metric: Metric to analyze
            
        Returns:
            Dictionary with Friedman test results
        """
        if self.comparison_data is None:
            raise ValueError("No comparison data available. Load experiment results first.")
        
        print(f"🔬 Performing Friedman test on {metric}")
        
        # Prepare data for Friedman test
        pivot_data = self.comparison_data.pivot_table(
            index='dataset', 
            columns=['method_type', 'model_name'], 
            values=metric,
            aggfunc='first'
        )
        
        # Remove rows with any missing values
        clean_data = pivot_data.dropna()
        
        if len(clean_data) < 3:
            return {'error': 'Insufficient data for Friedman test (need at least 3 datasets)'}
        
        # Perform Friedman test
        statistic, p_value = friedmanchisquare(*[clean_data[col] for col in clean_data.columns])
        
        # Calculate mean ranks
        ranks = clean_data.rank(axis=1, ascending=False)
        mean_ranks = ranks.mean()
        
        results = {
            'metric': metric,
            'test_name': 'Friedman test',
            'n_datasets': len(clean_data),
            'n_algorithms': len(clean_data.columns),
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean_ranks': mean_ranks.to_dict(),
            'algorithms': clean_data.columns.tolist()
        }
        
        return results
    
    def calculate_win_loss_ties(self, metric: str = 'f1_weighted') -> Dict[str, Any]:
        """
        Calculate win/loss/tie statistics between baseline and ensemble methods.
        
        Args:
            metric: Metric to analyze
            
        Returns:
            Dictionary with win/loss/tie statistics
        """
        if self.comparison_data is None:
            raise ValueError("No comparison data available. Load experiment results first.")
        
        print(f"📊 Calculating win/loss/ties for {metric}")
        
        # Get best performing model for each dataset and method type
        best_results = self.comparison_data.groupby(['dataset', 'method_type'])[metric].max().reset_index()
        pivot_data = best_results.pivot(index='dataset', columns='method_type', values=metric)
        
        # Remove datasets with missing data
        clean_data = pivot_data.dropna()
        
        if 'baseline' not in clean_data.columns or 'ensemble' not in clean_data.columns:
            return {'error': 'Missing baseline or ensemble data'}
        
        # Calculate wins, losses, ties
        ensemble_wins = (clean_data['ensemble'] > clean_data['baseline']).sum()
        baseline_wins = (clean_data['baseline'] > clean_data['ensemble']).sum()
        ties = (clean_data['ensemble'] == clean_data['baseline']).sum()
        
        total = len(clean_data)
        
        results = {
            'metric': metric,
            'total_comparisons': total,
            'ensemble_wins': int(ensemble_wins),
            'baseline_wins': int(baseline_wins),
            'ties': int(ties),
            'ensemble_win_rate': ensemble_wins / total,
            'baseline_win_rate': baseline_wins / total,
            'tie_rate': ties / total,
            'datasets': clean_data.index.tolist()
        }
        
        return results
    
    def generate_statistical_report(self, 
                                  output_dir: str = "results/statistical_analysis") -> str:
        """
        Generate a comprehensive statistical analysis report.
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        print("📝 Generating statistical analysis report...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform all analyses
        metrics_comparison = self.perform_multiple_metric_comparison()
        f1_detailed = self.perform_paired_comparison('f1_weighted')
        accuracy_detailed = self.perform_paired_comparison('accuracy')
        friedman_f1 = self.perform_friedman_test('f1_weighted')
        win_loss_f1 = self.calculate_win_loss_ties('f1_weighted')
        
        # Generate report
        report_lines = []
        report_lines.append("# Statistical Analysis Report: Baseline vs Ensemble Methods")
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Significance level (α): {self.alpha}")
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("## Summary Statistics")
        if 'individual_results' in metrics_comparison:
            for metric, results in metrics_comparison['individual_results'].items():
                if 'error' not in results:
                    report_lines.append(f"### {metric.replace('_', ' ').title()}")
                    report_lines.append(f"- Baseline mean: {results['baseline_mean']:.4f}")
                    report_lines.append(f"- Ensemble mean: {results['ensemble_mean']:.4f}")
                    report_lines.append(f"- Mean difference: {results['mean_difference']:.4f}")
                    report_lines.append(f"- Effect size (Cohen's d): {results['effect_size']:.4f} ({results['effect_interpretation']})")
                    report_lines.append("")
        
        # Statistical tests
        report_lines.append("## Statistical Test Results")
        
        if 'error' not in f1_detailed:
            report_lines.append("### F1-weighted Score (Primary Metric)")
            report_lines.append(f"- Test: {f1_detailed['test_name']}")
            report_lines.append(f"- Statistic: {f1_detailed['statistic']:.4f}")
            report_lines.append(f"- P-value: {f1_detailed['p_value']:.6f}")
            report_lines.append(f"- Significant: {'Yes' if f1_detailed['significant'] else 'No'}")
            report_lines.append(f"- 95% CI for difference: [{f1_detailed['confidence_interval'][0]:.4f}, {f1_detailed['confidence_interval'][1]:.4f}]")
            report_lines.append("")
        
        # Multiple comparison correction
        if 'bonferroni_correction' in metrics_comparison:
            bc = metrics_comparison['bonferroni_correction']
            report_lines.append("### Multiple Comparison Correction (Bonferroni)")
            report_lines.append(f"- Original α: {bc['original_alpha']}")
            report_lines.append(f"- Corrected α: {bc['corrected_alpha']:.6f}")
            report_lines.append(f"- Number of comparisons: {bc['n_comparisons']}")
            report_lines.append("")
            
            report_lines.append("### Significance After Bonferroni Correction:")
            for metric, results in metrics_comparison['individual_results'].items():
                if 'bonferroni_significant' in results:
                    status = "Significant" if results['bonferroni_significant'] else "Not significant"
                    report_lines.append(f"- {metric}: {status}")
            report_lines.append("")
        
        # Win/loss analysis
        if 'error' not in win_loss_f1:
            report_lines.append("### Win/Loss/Tie Analysis (F1-weighted)")
            report_lines.append(f"- Total datasets: {win_loss_f1['total_comparisons']}")
            report_lines.append(f"- Ensemble wins: {win_loss_f1['ensemble_wins']} ({win_loss_f1['ensemble_win_rate']:.2%})")
            report_lines.append(f"- Baseline wins: {win_loss_f1['baseline_wins']} ({win_loss_f1['baseline_win_rate']:.2%})")
            report_lines.append(f"- Ties: {win_loss_f1['ties']} ({win_loss_f1['tie_rate']:.2%})")
            report_lines.append("")
        
        # Friedman test
        if 'error' not in friedman_f1:
            report_lines.append("### Friedman Test (F1-weighted)")
            report_lines.append(f"- Test statistic: {friedman_f1['statistic']:.4f}")
            report_lines.append(f"- P-value: {friedman_f1['p_value']:.6f}")
            report_lines.append(f"- Significant: {'Yes' if friedman_f1['significant'] else 'No'}")
            report_lines.append("")
        
        # Conclusions
        report_lines.append("## Conclusions")
        if 'error' not in f1_detailed:
            if f1_detailed['significant']:
                direction = "better" if f1_detailed['mean_difference'] > 0 else "worse"
                report_lines.append(f"- Ensemble methods perform significantly {direction} than baseline methods")
                report_lines.append(f"- The effect size is {f1_detailed['effect_interpretation']} (Cohen's d = {f1_detailed['effect_size']:.4f})")
            else:
                report_lines.append("- No statistically significant difference between ensemble and baseline methods")
        
        # Save report
        report_path = os.path.join(output_dir, f"statistical_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Statistical analysis report saved to: {report_path}")
        return report_path
    
    def create_comparison_visualizations(self, 
                                       output_dir: str = "results/statistical_analysis") -> List[str]:
        """
        Create visualizations comparing baseline and ensemble methods.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to created visualization files
        """
        if self.comparison_data is None:
            raise ValueError("No comparison data available. Load experiment results first.")
        
        print("📊 Creating comparison visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Box plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baseline vs Ensemble Performance Comparison', fontsize=16)
        
        metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted']
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Get best scores for each dataset
            best_scores = self.comparison_data.groupby(['dataset', 'method_type'])[metric].max().reset_index()
            
            sns.boxplot(data=best_scores, x='method_type', y=metric, ax=ax)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Method Type')
        
        plt.tight_layout()
        box_plot_path = os.path.join(output_dir, 'baseline_vs_ensemble_boxplots.png')
        plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(box_plot_path)
        
        # 2. Paired comparison plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get best F1 scores for each dataset
        best_f1 = self.comparison_data.groupby(['dataset', 'method_type'])['f1_weighted'].max().reset_index()
        pivot_f1 = best_f1.pivot(index='dataset', columns='method_type', values='f1_weighted').dropna()
        
        # Create scatter plot
        ax.scatter(pivot_f1['baseline'], pivot_f1['ensemble'], alpha=0.7, s=100)
        
        # Add diagonal line (y=x)
        min_val = min(pivot_f1.min())
        max_val = max(pivot_f1.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Equal performance')
        
        # Add dataset labels
        for dataset, row in pivot_f1.iterrows():
            ax.annotate(dataset, (row['baseline'], row['ensemble']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Baseline F1-weighted Score')
        ax.set_ylabel('Ensemble F1-weighted Score')
        ax.set_title('Baseline vs Ensemble F1-weighted Performance by Dataset')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_path = os.path.join(output_dir, 'baseline_vs_ensemble_scatter.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(scatter_path)
        
        # 3. Performance difference plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        differences = pivot_f1['ensemble'] - pivot_f1['baseline']
        datasets = differences.index
        
        colors = ['green' if d > 0 else 'red' for d in differences]
        bars = ax.bar(range(len(datasets)), differences, color=colors, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('F1-weighted Difference (Ensemble - Baseline)')
        ax.set_title('Performance Difference by Dataset')
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.005),
                   f'{diff:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        diff_path = os.path.join(output_dir, 'performance_differences.png')
        plt.savefig(diff_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(diff_path)
        
        print(f"✅ Created {len(saved_files)} visualization files in {output_dir}")
        return saved_files 