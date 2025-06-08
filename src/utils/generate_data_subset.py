import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, Tuple, List
import random
import warnings

from utils.common import COMMON_DATASET_CONFIGS

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
            select_type (str): Type of attribute selection strategy:
                - "standard-selection": Random selection within min/max range
                - "sqrt-selection": Select sqrt(n_features) attributes
                - "correlation-selection": Select attributes based on correlation analysis
                
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
        dataset_name = dataset_conf.get('dataset_path')
        shrunk_size = dataset_conf.get('shrunk_size')
        n_attributes_max = dataset_conf.get('n_attributes_max')
        n_attributes_min = dataset_conf.get('n_attributes_min')
        select_type = dataset_conf.get('select_type', "standard-selection")
        target_column = dataset_conf.get('target_column')
        
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
        
        
        # Get feature columns (all except target)
        if target_column and target_column in df.columns:
            feature_columns = [col for col in df.columns if col != target_column]
        else:
            feature_columns = list(df.columns)
            target_column = None
        
        print(f"Available feature columns: {len(feature_columns)}")
        
        # Select attributes using the specified selection strategy
        selected_attributes = self._select_attributes(
            df, feature_columns, n_attributes_min, n_attributes_max, select_type, target_column
        )
        
        print(f"Selected {len(selected_attributes)} attributes: {selected_attributes}")
        
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
            select_type (str): Type of attribute selection strategy
            
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

    def _select_attributes(self, df: pd.DataFrame, feature_columns: List[str], 
                          n_attributes_min: int, n_attributes_max: int, 
                          select_type: str, target_column: Optional[str] = None) -> List[str]:
        """
        Select attributes based on the specified selection strategy.
        
        Args:
            df (pd.DataFrame): The dataset
            feature_columns (List[str]): Available feature columns
            n_attributes_min (int): Minimum number of attributes to select
            n_attributes_max (int): Maximum number of attributes to select
            select_type (str): Selection strategy type
            target_column (str, optional): Target column name for correlation analysis
            
        Returns:
            List[str]: Selected attribute names
        """
        max_available = len(feature_columns)
        
        if select_type == "standard-selection":
            return self._standard_selection(feature_columns, n_attributes_min, n_attributes_max, max_available)
        
        elif select_type == "sqrt-selection":
            return self._sqrt_selection(feature_columns, max_available)
        else:
            raise ValueError(f"Unknown select_type: {select_type}")

    def _standard_selection(self, feature_columns: List[str], n_attributes_min: int, 
                           n_attributes_max: int, max_available: int) -> List[str]:
        """
        Standard random selection within min/max range.
        """
        # Validate that we have enough attributes
        if n_attributes_min > max_available:
            raise ValueError(f"Requested minimum attributes ({n_attributes_min}) exceeds available features ({max_available})")
        
        # Adjust n_attributes_max if it exceeds available features
        n_attributes_max = min(n_attributes_max, max_available)
        
        # Randomly select number of attributes to use
        n_attributes_to_select = random.randint(n_attributes_min, n_attributes_max)
        print(f"Standard selection: Selected {n_attributes_to_select} attributes")
        
        # Randomly select the attributes
        selected_attributes = random.sample(feature_columns, n_attributes_to_select)
        return selected_attributes

    def _sqrt_selection(self, feature_columns: List[str], max_available: int) -> List[str]:
        """
        Square root selection - select sqrt(n_features) attributes (Random Forest approach).
        """
        n_attributes_to_select = max(1, int(np.sqrt(max_available)))
        n_attributes_to_select = min(n_attributes_to_select, max_available)
        
        print(f"Sqrt selection: Selected {n_attributes_to_select} attributes (sqrt of {max_available})")
        
        # Randomly select the attributes
        selected_attributes = random.sample(feature_columns, n_attributes_to_select)
        return selected_attributes

    def _validate_dataset_conf(self, dataset_conf: Dict) -> None:
        """
        Validate the dataset configuration.
        """
        dataset_name = dataset_conf.get('dataset_name')
        n_attributes_max = dataset_conf.get('n_attributes_max')
        n_attributes_min = dataset_conf.get('n_attributes_min')
        target_column = dataset_conf.get('target_column')
        
        # Validate required parameters
        if not dataset_name:
            raise ValueError("'dataset_name' must be specified in dataset_conf")
        if n_attributes_max is None or n_attributes_min is None:
            raise ValueError("Both 'n_attributes_max' and 'n_attributes_min' must be specified")
        if n_attributes_min > n_attributes_max:
            raise ValueError("'n_attributes_min' cannot be greater than 'n_attributes_max'")
        if n_attributes_min < 1:
            raise ValueError("'n_attributes_min' must be at least 1")
        if target_column is None:
            raise ValueError("'target_column' must be specified in dataset_conf")

    def _load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a dataset from a CSV file.
        """
        try:
            # READ ONLY TRAIN DATASET FOR BOOTSTRAPPING
            # print(dataset_name)
            # dataset_path_name = COMMON_DATASET_CONFIGS[dataset_name]['path']
            # path = f"data/processed/{dataset_path_name}/train.csv"
            path = f"../{dataset_name}"

            print(f"Loading dataset from: {path}")
            df = pd.read_csv(path)
            print(f"Original dataset shape: {df.shape}")
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {dataset_name}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")