import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import os


class DatasetLoader:
    """
    Dataset loader for project datasets.
    Loads pre-processed datasets without additional preprocessing 
    (preprocessing is done in Jupyter notebooks).
    """
    
    def __init__(self, data_root: str = "data/processed"):
        self.data_root = data_root
        self.dataset_configs = {
            'flights': {
                'path': 'binary_balanced_airflight_satisfaction',
                'target_column': 'satisfaction',
                'type': 'binary'
            },
            'credit_cards': {
                'path': 'binary_unbalanced_credit_card_fraud', 
                'target_column': 'class',
                'type': 'binary'
            },
            'human_activity': {
                'path': 'multi_class_balanced_human_activity_recognition',
                'target_column': 'Activity', 
                'type': 'multiclass'
            }
        }
    
    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load a pre-processed dataset.
        
        Args:
            dataset_name (str): Name of dataset ('flights', 'credit_cards', 'human_activity')
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test, feature_names
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.dataset_configs.keys())}")
        
        config = self.dataset_configs[dataset_name]
        dataset_path = os.path.join(self.data_root, config['path'])
        
        # Load train and test files
        train_path = os.path.join(dataset_path, 'train.csv')
        test_path = os.path.join(dataset_path, 'test.csv')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        print(f"Loading dataset: {dataset_name}")
        print(f"  Train file: {train_path}")
        print(f"  Test file: {test_path}")
        
        # Load dataframes
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Extract features and target
        target_col = config['target_column']
        
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in train data. Available columns: {list(train_df.columns)}")
        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data. Available columns: {list(test_df.columns)}")
        
        # Get feature columns (all except target)
        feature_columns = [col for col in train_df.columns if col != target_col]
        
        # Extract arrays
        X_train = train_df[feature_columns].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_columns].values
        y_test = test_df[target_col].values
        
        print(f"Dataset {dataset_name} loaded successfully:")
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Classes: {len(np.unique(y_train))}, Features: {len(feature_columns)}")
        print(f"  Target column: {target_col}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def load_test_dataset(self, dataset_conf: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load only the test dataset according to the given configuration.
        
        Args:
            dataset_conf (Dict): Configuration dictionary containing:
                - 'dataset_name' (str): Name of the dataset
                - 'target_column' (str): Name of target column
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: X_test, y_test arrays
            
        Raises:
            FileNotFoundError: If test file doesn't exist
            ValueError: If dataset name is unknown or target column not found
        """
        dataset_name = dataset_conf.get('dataset_name')
        
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.dataset_configs.keys())}")
        
        config = self.dataset_configs[dataset_name]
        dataset_path = os.path.join(self.data_root, config['path'])
        test_path = os.path.join(dataset_path, 'test.csv')
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
            
        print(f"Loading test dataset: {dataset_name}")
        print(f"  Test file: {test_path}")
        
        # Load test dataframe
        test_df = pd.read_csv(test_path)
        
        # Extract features and target
        target_col = dataset_conf.get('target_column')
        
        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data. Available columns: {list(test_df.columns)}")
        
        # Get feature columns (all except target)
        feature_columns = [col for col in test_df.columns if col != target_col]
        
        # Extract arrays
        X_test = test_df[feature_columns].values
        y_test = test_df[target_col].values
        
        print(f"Test dataset loaded successfully:")
        print(f"  Shape: {X_test.shape}")
        print(f"  Classes: {len(np.unique(y_test))}")
        print(f"  Features: {len(feature_columns)}")
        
        return X_test, y_test
    
    def get_dataset_info(self) -> Dict:
        """Get information about available datasets."""
        return self.dataset_configs
    
    def list_available_datasets(self) -> List[str]:
        """List available dataset names."""
        return list(self.dataset_configs.keys())
    
    def get_dataset_summary(self, dataset_name: str) -> Dict:
        """
        Get summary information about a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict: Summary information
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        try:
            X_train, X_test, y_train, y_test, feature_names = self.load_dataset(dataset_name)
            
            return {
                'name': dataset_name,
                'type': self.dataset_configs[dataset_name]['type'],
                'target_column': self.dataset_configs[dataset_name]['target_column'],
                'n_features': len(feature_names),
                'n_classes': len(np.unique(y_train)),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_class_distribution': dict(zip(*np.unique(y_train, return_counts=True))),
                'test_class_distribution': dict(zip(*np.unique(y_test, return_counts=True))),
                'feature_names': feature_names
            }
        except Exception as e:
            return {
                'name': dataset_name,
                'error': str(e)
            }
