import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, Tuple, List
import random
import warnings

FLIGHTS_DATASET_TARGET_CLASS_NAME = "satisfaction"
CREDIT_CARDS_DATASET_TARGET_CLASS_NAME = "class"
HUMAN_ACTIVITY_DATASET_TARGET_CLASS_NAME = "Activity"

class DataSubsetGenerator:
    """
    Class for generating data subsets from a dataset.
    """

    def load_bootstrapped_dataset(self, dataset_conf: Dict, seed: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and bootstrap a dataset according to the given configuration.
        
        This function loads a dataset, optionally shrinks it, and randomly selects
        a subset of attributes for training. This is useful for ensemble methods
        and feature bootstrapping in advanced machine learning projects.
        
        Args:
            dataset_conf (Dict): Configuration dictionary containing:
                - 'dataset_name' (str): Path to the dataset file
                - 'shrunk_size' (int, optional): Number of rows to sample (None for full dataset)
                - 'n_attributes_max' (int): Maximum number of attributes to select
                - 'n_attributes_min' (int): Minimum number of attributes to select
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: 
                - Bootstrapped dataset with selected attributes
                - List of selected attribute names
                
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If configuration parameters are invalid
        """
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Extract configuration parameters
        dataset_name = dataset_conf.get('dataset_name')
        shrunk_size = dataset_conf.get('shrunk_size')
        n_attributes_max = dataset_conf.get('n_attributes_max')
        n_attributes_min = dataset_conf.get('n_attributes_min')
        
        #validate the dataset configuration object fields
        self._validate_dataset_conf(dataset_conf)
        
        # Load the dataset directly from the provided path
        df = self._load_dataset(dataset_name)
        print(f"Dataset columns before shrinking: {df.columns.tolist()}")
        
        # Apply dataset shrinking if specified
        if shrunk_size is not None:
            if shrunk_size > len(df):
                warnings.warn(f"Requested shrunk_size ({shrunk_size}) is larger than dataset size ({len(df)}). Using full dataset.")
                shrunk_size = len(df)
            
            df = df.sample(n=shrunk_size, random_state=seed).reset_index(drop=True)
            print(f"Dataset shrunk to: {df.shape}")
        
        # Get all available attributes (excluding potential target column)
        # Look for common target names
        potential_targets = [FLIGHTS_DATASET_TARGET_CLASS_NAME, 
                             CREDIT_CARDS_DATASET_TARGET_CLASS_NAME, 
                             HUMAN_ACTIVITY_DATASET_TARGET_CLASS_NAME]
        target_column = None
        
        # Find target column
        for col in potential_targets:
            if col in df.columns:
                target_column = col
                break
        
        # If no standard target found, assume last column is target
        if target_column is None and len(df.columns) > 1:
            target_column = df.columns[-1]
            print(f"Assuming '{target_column}' as target column")
        
        # Get feature columns (all except target)
        if target_column and target_column in df.columns:
            feature_columns = [col for col in df.columns if col != target_column]
        else:
            feature_columns = list(df.columns)
            target_column = None
        
        print(f"Available feature columns: {len(feature_columns)}")
        
        # Validate that we have enough attributes
        max_available = len(feature_columns)
        if n_attributes_min > max_available:
            raise ValueError(f"Requested minimum attributes ({n_attributes_min}) exceeds available features ({max_available})")
        
        # Adjust n_attributes_max if it exceeds available features
        n_attributes_max = min(n_attributes_max, max_available)
        
        # Randomly select number of attributes to use
        n_attributes_to_select = random.randint(n_attributes_min, n_attributes_max)
        print(f"Selected {n_attributes_to_select} attributes")
        
        # Randomly select the attributes
        selected_attributes = random.sample(feature_columns, n_attributes_to_select)
        
        print(f"Selected {n_attributes_to_select} attributes: {selected_attributes}")
        
        # Create the subset DataFrame
        subset_columns = selected_attributes.copy()
        if target_column:
            subset_columns.append(target_column)
        
        df_subset = df[subset_columns].copy()
        
        print(f"Final bootstrapped dataset shape: {df_subset.shape}")
        
        return df_subset, selected_attributes

    def create_multiple_bootstrap_samples(self, dataset_conf: Dict, n_samples: int, base_seed: int = 42) -> List[Tuple[pd.DataFrame, List[str]]]:
        """
        Create multiple bootstrap samples with different random seeds.
        
        Args:
            dataset_conf (Dict): Dataset configuration
            n_samples (int): Number of bootstrap samples to create
            base_seed (int): Base seed for generating different seeds
            
        Returns:
            List[Tuple[pd.DataFrame, List[str]]]: List of tuples (df_subset, selected_attributes) for each sample
        """
        bootstrap_samples = []
        
        for i in range(n_samples):
            sample_seed = base_seed + i
            df_subset, selected_attrs = self.load_bootstrapped_dataset(dataset_conf, seed=sample_seed)
            bootstrap_samples.append((df_subset, selected_attrs))
            print(f"Created bootstrap sample {i+1}/{n_samples} for seed {sample_seed}")
        
        return bootstrap_samples

    def _validate_dataset_conf(self, dataset_conf: Dict) -> None:
        """
        Validate the dataset configuration.
        """
        dataset_name = dataset_conf.get('dataset_name')
        n_attributes_max = dataset_conf.get('n_attributes_max')
        n_attributes_min = dataset_conf.get('n_attributes_min')
        
        # Validate required parameters
        if not dataset_name:
            raise ValueError("'dataset_name' must be specified in dataset_conf")
        if n_attributes_max is None or n_attributes_min is None:
            raise ValueError("Both 'n_attributes_max' and 'n_attributes_min' must be specified")
        if n_attributes_min > n_attributes_max:
            raise ValueError("'n_attributes_min' cannot be greater than 'n_attributes_max'")
        if n_attributes_min < 1:
            raise ValueError("'n_attributes_min' must be at least 1")
    
    def _load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a dataset from a CSV file.
        """
        try:
            print(f"Loading dataset from: {dataset_name}")
            df = pd.read_csv(dataset_name)
            print(f"Original dataset shape: {df.shape}")
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {dataset_name}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
