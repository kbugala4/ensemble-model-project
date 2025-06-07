#!/usr/bin/env python3
"""
Example usage of the EnsembleResultsVisualizer

This script demonstrates how to use the visualizer to create horizontal range plots
showing min/max performance ranges for ensemble methods.
"""

import sys
from pathlib import Path

# Add src to path so we can import our custom modules
sys.path.append(str(Path(__file__).parent / "src"))

from results_visualizer import EnsembleResultsVisualizer
import matplotlib.pyplot as plt


def main():
    """Example usage of the EnsembleResultsVisualizer."""
    
    print("🎯 Ensemble Results Visualizer Example")
    print("=" * 50)
    
    # Create visualizer instance
    visualizer = EnsembleResultsVisualizer(results_dir="results")
    
    try:
        # Load data and show summary
        print("📊 Loading data...")
        visualizer.load_data()
        visualizer.show_summary_stats()
        
        print("\n" + "=" * 50)
        print("📈 Creating custom experiment comparison plots...")
        
        # Create custom experiment comparison plots
        custom_figures = visualizer.create_custom_experiment_plots(save_plots=True)
        
        # Display the plots
        total_custom = sum(len(figs) for figs in custom_figures.values()) if custom_figures else 0
        
        if custom_figures:
            print(f"\n✅ Created {total_custom} custom comparison plots!")
            print("🖼️  Displaying plots...")
            
            # Show custom experiment plots
            for dataset, dataset_figs in custom_figures.items():
                for experiment, fig in dataset_figs.items():
                    fig.show()
                    
            plt.show()
        else:
            print("⚠️  No plots were created. Check your data and try again.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nPlease make sure you have:")
        print("1. CSV files in the results/baseline or results/custom directories")
        print("2. Required Python packages installed (pandas, matplotlib, seaborn)")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 