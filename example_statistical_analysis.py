#!/usr/bin/env python3
"""
Example script demonstrating statistical analysis of baseline vs ensemble results.

This script shows how to use the StatisticalAnalyzer class to perform comprehensive
statistical comparisons between baseline and ensemble methods.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.statistical_analysis import StatisticalAnalyzer


def run_comprehensive_statistical_analysis():
    """
    Run a comprehensive statistical analysis comparing baseline and ensemble methods.
    """
    print("📊 Statistical Analysis: Baseline vs Ensemble Methods")
    print("=" * 60)
    
    # Initialize the statistical analyzer
    analyzer = StatisticalAnalyzer(alpha=0.05)
    
    # Load experiment results
    print("\n1️⃣ Loading Experiment Results...")
    analyzer.load_experiment_results(
        baseline_dir="results/baseline",
        ensemble_dir="results/custom"
    )
    
    if analyzer.comparison_data is None:
        print("❌ No comparison data available. Make sure you have both baseline and ensemble results.")
        return False
    
    print(f"📊 Loaded {len(analyzer.comparison_data)} experiment records")
    print(f"🎯 Datasets: {analyzer.comparison_data['dataset'].unique().tolist()}")
    print(f"🔬 Method types: {analyzer.comparison_data['method_type'].unique().tolist()}")
    
    # Perform paired comparison on F1-weighted score
    print("\n2️⃣ Performing Paired Statistical Tests...")
    try:
        f1_results = analyzer.perform_paired_comparison('f1_weighted')
        
        if 'error' not in f1_results:
            print(f"📈 F1-weighted Score Analysis:")
            print(f"   Test: {f1_results['test_name']}")
            print(f"   Baseline mean: {f1_results['baseline_mean']:.4f}")
            print(f"   Ensemble mean: {f1_results['ensemble_mean']:.4f}")
            print(f"   Difference: {f1_results['mean_difference']:.4f}")
            print(f"   P-value: {f1_results['p_value']:.6f}")
            print(f"   Significant: {'✅ Yes' if f1_results['significant'] else '❌ No'}")
            print(f"   Effect size: {f1_results['effect_size']:.4f} ({f1_results['effect_interpretation']})")
            
            ci_lower, ci_upper = f1_results['confidence_interval']
            print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            print(f"⚠️  F1 comparison failed: {f1_results['error']}")
            
    except Exception as e:
        print(f"❌ Error in paired comparison: {e}")
    
    # Perform multiple metric comparison
    print("\n3️⃣ Multiple Metric Comparison...")
    try:
        multi_results = analyzer.perform_multiple_metric_comparison()
        
        if 'individual_results' in multi_results:
            print("📊 Results across multiple metrics:")
            for metric, results in multi_results['individual_results'].items():
                if 'error' not in results:
                    sig_marker = "✅" if results['significant'] else "❌"
                    bonf_marker = "✅" if results.get('bonferroni_significant', False) else "❌"
                    print(f"   {metric}: {results['mean_difference']:.4f} "
                          f"(p={results['p_value']:.4f}, {sig_marker}raw, {bonf_marker}corrected)")
            
            # Show Bonferroni correction info
            if 'bonferroni_correction' in multi_results:
                bc = multi_results['bonferroni_correction']
                print(f"\n🔧 Bonferroni Correction:")
                print(f"   Original α: {bc['original_alpha']}")
                print(f"   Corrected α: {bc['corrected_alpha']:.6f}")
                print(f"   Comparisons: {bc['n_comparisons']}")
                
    except Exception as e:
        print(f"❌ Error in multiple metric comparison: {e}")
    
    # Win/Loss/Tie analysis
    print("\n4️⃣ Win/Loss/Tie Analysis...")
    try:
        wlt_results = analyzer.calculate_win_loss_ties('f1_weighted')
        
        if 'error' not in wlt_results:
            print(f"🏆 Win/Loss/Tie Statistics (F1-weighted):")
            print(f"   Total comparisons: {wlt_results['total_comparisons']}")
            print(f"   Ensemble wins: {wlt_results['ensemble_wins']} ({wlt_results['ensemble_win_rate']:.1%})")
            print(f"   Baseline wins: {wlt_results['baseline_wins']} ({wlt_results['baseline_win_rate']:.1%})")
            print(f"   Ties: {wlt_results['ties']} ({wlt_results['tie_rate']:.1%})")
        else:
            print(f"⚠️  Win/Loss analysis failed: {wlt_results['error']}")
            
    except Exception as e:
        print(f"❌ Error in win/loss analysis: {e}")
    
    # Friedman test
    print("\n5️⃣ Friedman Test (Multiple Algorithm Comparison)...")
    try:
        friedman_results = analyzer.perform_friedman_test('f1_weighted')
        
        if 'error' not in friedman_results:
            print(f"🔬 Friedman Test Results:")
            print(f"   Test statistic: {friedman_results['statistic']:.4f}")
            print(f"   P-value: {friedman_results['p_value']:.6f}")
            print(f"   Significant: {'✅ Yes' if friedman_results['significant'] else '❌ No'}")
            print(f"   Datasets: {friedman_results['n_datasets']}")
            print(f"   Algorithms: {friedman_results['n_algorithms']}")
        else:
            print(f"⚠️  Friedman test failed: {friedman_results['error']}")
            
    except Exception as e:
        print(f"❌ Error in Friedman test: {e}")
    
    # Generate comprehensive report
    print("\n6️⃣ Generating Statistical Report...")
    try:
        report_path = analyzer.generate_statistical_report()
        print(f"✅ Comprehensive report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
    
    # Create visualizations
    print("\n7️⃣ Creating Visualizations...")
    try:
        viz_files = analyzer.create_comparison_visualizations()
        print(f"✅ Created {len(viz_files)} visualization files:")
        for viz_file in viz_files:
            print(f"   📊 {viz_file}")
            
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
    
    return True


def demonstrate_individual_analyses():
    """
    Demonstrate individual analysis functions.
    """
    print("\n\n🔬 Individual Analysis Demonstrations")
    print("=" * 50)
    
    analyzer = StatisticalAnalyzer(alpha=0.05)
    analyzer.load_experiment_results()
    
    if analyzer.comparison_data is None:
        print("❌ No data available for individual demonstrations")
        return
    
    # Test different metrics
    metrics_to_test = ['accuracy', 'f1_weighted', 'precision_weighted']
    
    for metric in metrics_to_test:
        print(f"\n🎯 Testing {metric}...")
        try:
            results = analyzer.perform_paired_comparison(metric, test_type='auto')
            if 'error' not in results:
                direction = "better" if results['mean_difference'] > 0 else "worse"
                print(f"   Ensemble performs {direction} (Δ={results['mean_difference']:.4f})")
                print(f"   Statistical significance: {'Yes' if results['significant'] else 'No'}")
                print(f"   Effect size: {results['effect_interpretation']}")
            else:
                print(f"   Error: {results['error']}")
        except Exception as e:
            print(f"   Error: {e}")


def show_data_summary():
    """
    Show summary of available data for analysis.
    """
    print("\n\n📊 Data Summary for Statistical Analysis")
    print("=" * 50)
    
    analyzer = StatisticalAnalyzer()
    analyzer.load_experiment_results()
    
    if analyzer.comparison_data is not None:
        df = analyzer.comparison_data
        
        print(f"📈 Total experiment records: {len(df)}")
        print(f"🎯 Unique datasets: {df['dataset'].nunique()}")
        print(f"🔬 Method types: {df['method_type'].unique().tolist()}")
        print(f"📝 Models per method type:")
        
        for method_type in df['method_type'].unique():
            method_data = df[df['method_type'] == method_type]
            models = method_data['model_name'].unique()
            print(f"   {method_type}: {len(models)} models ({models.tolist()})")
        
        print(f"\n📊 Available metrics: {[col for col in df.columns if col not in ['dataset', 'method_type', 'model_name']]}")
        
        # Show missing data
        print(f"\n⚠️  Missing data summary:")
        missing_summary = df.isnull().sum()
        for col, missing_count in missing_summary.items():
            if missing_count > 0:
                print(f"   {col}: {missing_count}/{len(df)} missing ({missing_count/len(df):.1%})")
    
    else:
        print("❌ No comparison data available")


if __name__ == "__main__":
    print("🔬 Statistical Analysis Example")
    print("=" * 70)
    
    # Show what data is available
    show_data_summary()
    
    # Run comprehensive analysis
    success = run_comprehensive_statistical_analysis()
    
    if success:
        # Demonstrate individual analyses
        demonstrate_individual_analyses()
        
        print("\n🎉 Statistical Analysis Complete!")
        print("\nKey outputs generated:")
        print("  📝 Comprehensive markdown report")
        print("  📊 Box plots comparing methods")
        print("  🎯 Scatter plot (baseline vs ensemble)")
        print("  📈 Performance difference bar chart")
        print("  📋 Win/loss/tie statistics")
        print("  🔬 Multiple statistical tests")
        print("\nFiles saved to: results/statistical_analysis/")
        
    else:
        print("\n❌ Statistical analysis could not be completed")
        print("Make sure you have both baseline and ensemble experiment results available.")
        
    print("\n💡 Tips for interpretation:")
    print("  • p < 0.05: Statistically significant difference")
    print("  • Effect size: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)")
    print("  • Confidence intervals: Range of plausible values for the true difference")
    print("  • Bonferroni correction: Controls for multiple comparisons") 