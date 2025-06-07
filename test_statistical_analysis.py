#!/usr/bin/env python3
"""
Test script for StatisticalAnalyzer functionality.

This creates sample data to test the statistical analysis capabilities.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.statistical_analysis import StatisticalAnalyzer


def create_sample_results():
    """Create sample baseline and ensemble results for testing."""
    
    # Sample baseline results
    baseline_results = {
        "iris": {
            "random_forest": {
                "accuracy": 0.95,
                "f1_weighted": 0.95,
                "f1_macro": 0.94,
                "precision_weighted": 0.96,
                "recall_weighted": 0.95,
                "n_samples_test": 30,
                "n_features": 4
            },
            "decision_tree": {
                "accuracy": 0.93,
                "f1_weighted": 0.93,
                "f1_macro": 0.92,
                "precision_weighted": 0.94,
                "recall_weighted": 0.93,
                "n_samples_test": 30,
                "n_features": 4
            }
        },
        "wine": {
            "random_forest": {
                "accuracy": 0.97,
                "f1_weighted": 0.97,
                "f1_macro": 0.97,
                "precision_weighted": 0.97,
                "recall_weighted": 0.97,
                "n_samples_test": 36,
                "n_features": 13
            },
            "svm": {
                "accuracy": 0.94,
                "f1_weighted": 0.94,
                "f1_macro": 0.94,
                "precision_weighted": 0.94,
                "recall_weighted": 0.94,
                "n_samples_test": 36,
                "n_features": 13
            }
        },
        "breast_cancer": {
            "logistic_regression": {
                "accuracy": 0.96,
                "f1_weighted": 0.96,
                "f1_macro": 0.95,
                "precision_weighted": 0.96,
                "recall_weighted": 0.96,
                "n_samples_test": 114,
                "n_features": 30
            },
            "svm": {
                "accuracy": 0.95,
                "f1_weighted": 0.95,
                "f1_macro": 0.94,
                "precision_weighted": 0.95,
                "recall_weighted": 0.95,
                "n_samples_test": 114,
                "n_features": 30
            }
        }
    }
    
    # Sample ensemble results (slightly better performance)
    ensemble_results = {
        "iris": {
            "ensemble_rf_dt": {
                "accuracy": 0.97,
                "f1_weighted": 0.97,
                "f1_macro": 0.96,
                "precision_weighted": 0.97,
                "recall_weighted": 0.97,
                "n_samples_test": 30,
                "n_features": 4,
                "ensemble_params": {"n_models": 5}
            }
        },
        "wine": {
            "ensemble_mixed": {
                "accuracy": 0.98,
                "f1_weighted": 0.98,
                "f1_macro": 0.98,
                "precision_weighted": 0.98,
                "recall_weighted": 0.98,
                "n_samples_test": 36,
                "n_features": 13,
                "ensemble_params": {"n_models": 10}
            }
        },
        "breast_cancer": {
            "ensemble_voting": {
                "accuracy": 0.97,
                "f1_weighted": 0.97,
                "f1_macro": 0.96,
                "precision_weighted": 0.97,
                "recall_weighted": 0.97,
                "n_samples_test": 114,
                "n_features": 30,
                "ensemble_params": {"n_models": 7}
            }
        }
    }
    
    return baseline_results, ensemble_results


def test_statistical_analyzer():
    """Test the StatisticalAnalyzer with sample data."""
    
    print("🧪 Testing StatisticalAnalyzer")
    print("=" * 40)
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    baseline_dir = os.path.join(temp_dir, "baseline")
    ensemble_dir = os.path.join(temp_dir, "custom")
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(ensemble_dir, exist_ok=True)
    
    try:
        # Create sample results
        baseline_results, ensemble_results = create_sample_results()
        
        # Save sample results
        baseline_file = os.path.join(baseline_dir, "baseline_results_20240101_120000.json")
        ensemble_file = os.path.join(ensemble_dir, "custom_results_20240101_120000.json")
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        with open(ensemble_file, 'w') as f:
            json.dump(ensemble_results, f, indent=2)
        
        print(f"✅ Created sample data files")
        print(f"   Baseline: {baseline_file}")
        print(f"   Ensemble: {ensemble_file}")
        
        # Initialize analyzer
        analyzer = StatisticalAnalyzer(alpha=0.05)
        
        # Load results
        print(f"\n📊 Loading results...")
        analyzer.load_experiment_results(baseline_dir, ensemble_dir)
        
        if analyzer.comparison_data is None:
            print("❌ Failed to load comparison data")
            return False
        
        print(f"✅ Loaded {len(analyzer.comparison_data)} records")
        print(f"   Datasets: {analyzer.comparison_data['dataset'].unique().tolist()}")
        
        # Test paired comparison
        print(f"\n🔬 Testing paired comparison...")
        f1_results = analyzer.perform_paired_comparison('f1_weighted')
        
        if 'error' not in f1_results:
            print(f"✅ Paired comparison successful:")
            print(f"   Test: {f1_results['test_name']}")
            print(f"   Mean difference: {f1_results['mean_difference']:.4f}")
            print(f"   P-value: {f1_results['p_value']:.6f}")
            print(f"   Significant: {f1_results['significant']}")
            print(f"   Effect size: {f1_results['effect_size']:.4f}")
        else:
            print(f"❌ Paired comparison failed: {f1_results['error']}")
        
        # Test multiple metric comparison
        print(f"\n🔢 Testing multiple metric comparison...")
        multi_results = analyzer.perform_multiple_metric_comparison()
        
        if 'individual_results' in multi_results:
            print(f"✅ Multiple metric comparison successful:")
            for metric, results in multi_results['individual_results'].items():
                if 'error' not in results:
                    print(f"   {metric}: Δ={results['mean_difference']:.4f}, p={results['p_value']:.4f}")
        else:
            print(f"❌ Multiple metric comparison failed")
        
        # Test win/loss/tie
        print(f"\n🏆 Testing win/loss/tie analysis...")
        wlt_results = analyzer.calculate_win_loss_ties('f1_weighted')
        
        if 'error' not in wlt_results:
            print(f"✅ Win/loss/tie analysis successful:")
            print(f"   Ensemble wins: {wlt_results['ensemble_wins']}/{wlt_results['total_comparisons']}")
            print(f"   Win rate: {wlt_results['ensemble_win_rate']:.1%}")
        else:
            print(f"❌ Win/loss/tie analysis failed: {wlt_results['error']}")
        
        # Test report generation
        print(f"\n📝 Testing report generation...")
        report_dir = os.path.join(temp_dir, "statistical_analysis")
        try:
            report_path = analyzer.generate_statistical_report(report_dir)
            print(f"✅ Report generated: {report_path}")
            
            # Verify report exists
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    content = f.read()
                print(f"   Report length: {len(content)} characters")
            else:
                print(f"❌ Report file not found")
                
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
        
        # Test visualization creation
        print(f"\n📊 Testing visualization creation...")
        try:
            viz_files = analyzer.create_comparison_visualizations(report_dir)
            print(f"✅ Created {len(viz_files)} visualization files:")
            for viz_file in viz_files:
                if os.path.exists(viz_file):
                    file_size = os.path.getsize(viz_file)
                    print(f"   📈 {os.path.basename(viz_file)} ({file_size} bytes)")
                else:
                    print(f"   ❌ {os.path.basename(viz_file)} (missing)")
                    
        except Exception as e:
            print(f"❌ Visualization creation failed: {e}")
        
        print(f"\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary files")
        except:
            print(f"⚠️  Could not clean up {temp_dir}")


def demonstrate_statistical_concepts():
    """Demonstrate key statistical concepts used in the analyzer."""
    
    print(f"\n\n📚 Statistical Concepts Demonstration")
    print("=" * 50)
    
    print(f"🔬 Key Statistical Tests Used:")
    print(f"")
    print(f"1. Paired t-test:")
    print(f"   • Use: Compare means of two related groups")
    print(f"   • Assumption: Differences are normally distributed")
    print(f"   • Example: Baseline vs Ensemble on same datasets")
    print(f"")
    print(f"2. Wilcoxon signed-rank test:")
    print(f"   • Use: Non-parametric alternative to paired t-test")  
    print(f"   • Assumption: Differences are symmetric")
    print(f"   • Example: When normality assumption fails")
    print(f"")
    print(f"3. Friedman test:")
    print(f"   • Use: Compare multiple algorithms across datasets")
    print(f"   • Assumption: Non-parametric, no distribution assumptions")
    print(f"   • Example: Comparing 3+ methods on multiple datasets")
    print(f"")
    print(f"📊 Effect Size Interpretation (Cohen's d):")
    print(f"   • |d| < 0.2: Negligible effect")
    print(f"   • 0.2 ≤ |d| < 0.5: Small effect")
    print(f"   • 0.5 ≤ |d| < 0.8: Medium effect")
    print(f"   • |d| ≥ 0.8: Large effect")
    print(f"")
    print(f"🔧 Multiple Comparison Correction:")
    print(f"   • Problem: Multiple tests increase Type I error")
    print(f"   • Bonferroni: α_corrected = α / n_comparisons")
    print(f"   • Example: 5 metrics, α=0.05 → α_corrected=0.01")
    print(f"")
    print(f"📈 Confidence Intervals:")
    print(f"   • 95% CI: Range containing true difference 95% of the time")
    print(f"   • If CI excludes 0: Significant difference")
    print(f"   • Width indicates precision of estimate")


if __name__ == "__main__":
    print("🧪 Statistical Analysis Testing Suite")
    print("=" * 60)
    
    # Run the main test
    success = test_statistical_analyzer()
    
    if success:
        # Demonstrate statistical concepts
        demonstrate_statistical_concepts()
        
        print(f"\n🎯 Usage Summary:")
        print(f"   1. Run baseline and ensemble experiments")
        print(f"   2. Use StatisticalAnalyzer to load results")
        print(f"   3. Perform statistical comparisons")
        print(f"   4. Generate reports and visualizations")
        print(f"   5. Interpret results with proper statistical rigor")
        
    else:
        print(f"\n❌ Testing failed - check the implementation")
        
    print(f"\n📚 For real analysis, run: python example_statistical_analysis.py") 