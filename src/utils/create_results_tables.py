#!/usr/bin/env python3
"""
Script to create results tables from baseline and custom experiment CSV files.

This script reads CSV summary files from results/baseline and results/custom directories
and creates formatted tables for each dataset with metrics as columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display

import warnings
import glob


def find_latest_csv_files(results_dir):
    """Find the most recent CSV files in the results directory."""
    csv_files = glob.glob(str(results_dir / "*.csv"))
    if not csv_files:
        return None
    
    # Return the most recent file based on modification time
    latest_file = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
    return Path(latest_file)


def load_experiment_data():
    """Load and combine data from baseline and custom CSV files."""
    
    baseline_dir = Path("results/baseline")
    custom_dir = Path("results/custom")
    
    combined_data = []
    
    # Load baseline data
    if baseline_dir.exists():
        baseline_csv = find_latest_csv_files(baseline_dir)
        if baseline_csv:
            print(f"📊 Loading baseline data from: {baseline_csv.name}")
            baseline_df = pd.read_csv(baseline_csv)
            baseline_df['experiment_type'] = 'BASELINE'
            combined_data.append(baseline_df)
        else:
            print("⚠️  Warning: No baseline CSV files found")
    else:
        print("⚠️  Warning: Baseline directory does not exist")
    
    # Load custom data
    if custom_dir.exists():
        custom_csv = find_latest_csv_files(custom_dir)
        if custom_csv:
            print(f"📊 Loading custom data from: {custom_csv.name}")
            custom_df = pd.read_csv(custom_csv)
            if 'experiment_type' not in custom_df.columns:
                custom_df['experiment_type'] = 'CUSTOM'
            combined_data.append(custom_df)
        else:
            print("⚠️  Warning: No custom CSV files found")
    else:
        print("⚠️  Warning: Custom directory does not exist")
    
    if not combined_data:
        raise ValueError("❌ No CSV data found in either baseline or custom directories")
    
    # Combine all data
    full_df = pd.concat(combined_data, ignore_index=True, sort=False)
    
    return full_df


def create_dataset_table(df, dataset_name, metrics):
    """Create a table for a specific dataset showing all metrics across experiments."""
    
    # Filter data for the specific dataset
    dataset_df = df[df['dataset'] == dataset_name].copy()
    
    if dataset_df.empty:
        print(f"⚠️  Warning: No data found for dataset '{dataset_name}'")
        return None
    
    # # Group by model to handle multiple runs
    # grouped = dataset_df.groupby('model').agg(['mean', 'std', 'count']).reset_index()
    
    # Create result table
    result_rows = []
    
    for model in dataset_df['model'].unique():
        model_data = dataset_df[dataset_df['model'] == model]
        row = {'Experiment Name': model}
        
        for metric_code, metric_title in metrics:
            test_metric = metric_code
            train_metric = f"{metric_code}_train"
            
            # Get test value
            test_val = None
            if test_metric in model_data.columns:
                test_vals = model_data[test_metric].dropna()
                if not test_vals.empty:
                    test_val = test_vals.mean()
            
            # Get train value
            train_val = None
            if train_metric in model_data.columns:
                train_vals = model_data[train_metric].dropna()
                if not train_vals.empty:
                    train_val = train_vals.mean()
            
            # Format the cell value as "train | test"
            if train_val is not None and test_val is not None:
                row[metric_title] = f"{train_val:.4f} | {test_val:.4f}"
            elif train_val is not None:
                row[metric_title] = f"{train_val:.4f} | N/A"
            elif test_val is not None:
                row[metric_title] = f"N/A | {test_val:.4f}"
            else:
                row[metric_title] = "N/A | N/A"
        
        result_rows.append(row)
    
    # Create DataFrame
    result_df = pd.DataFrame(result_rows)
    
    # Sort by experiment name for consistent ordering
    result_df = result_df.sort_values('Experiment Name').reset_index(drop=True)
    
    return result_df


def generate_results():
    """Main function to create and display results tables."""
    
    print("🎯 Results Table Generator")
    print("=" * 50)
    
    # Load data
    try:
        df = load_experiment_data()
        print(f"✅ Loaded data: {len(df)} records")
        print(f"📊 Datasets: {sorted(df['dataset'].unique())}")
        print(f"🔬 Models: {sorted(df['model'].unique())}")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    # Define metrics
    metrics = [
        ('accuracy', 'Accuracy'),
        ('f1_weighted', 'F1 Weighted'),
        ('f1_macro', 'F1 Macro'),
        ('precision_weighted', 'Precision Weighted'),
        ('recall_weighted', 'Recall Weighted'),
        ('roc_auc', 'AUC')
    ]
    
    # Get available datasets from the data
    available_datasets = sorted(df['dataset'].unique())
    
    # Create and display tables for each dataset
    print("📋 RESULTS TABLES BY DATASET")
    print("Format: Train Score | Test Score (N/A if not available)")
    print("=" * 80)
    print()
    
    for dataset in available_datasets:
        print(f"📊 Creating table for {dataset} dataset...")
        table = create_dataset_table(df, dataset, metrics)
        if table is not None:
            print(f"\n🔍 {dataset.upper()} Dataset Results:")
            print("-" * 50)
            display(table)
            print()
    
    # Additional statistics
    print("\n📈 SUMMARY STATISTICS")
    print("-" * 30)
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        print(f"\n{dataset.upper()}:")
        print(f"  Models tested: {len(dataset_df['model'].unique())}")
        print(f"  Records: {len(dataset_df)}")
        if 'accuracy' in dataset_df.columns:
            best_acc_idx = dataset_df['accuracy'].idxmax()
            if not pd.isna(best_acc_idx):
                print(f"  Best accuracy: {dataset_df['accuracy'].max():.4f} ({dataset_df.loc[best_acc_idx, 'model']})")