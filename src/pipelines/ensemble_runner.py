from typing import Callable, List, Dict, Any
import numpy as np
import pandas as pd
import os
import random
from models.base_model import BaseModel
from sklearn.model_selection import train_test_split
from utils.dataloader import DatasetLoader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


class EnsembleRunner:
    """
    Runner class responsible for managing the full experiment lifecycle:
    - building an ensemble of models,
    - training them,
    - making predictions,
    - evaluating performance.
    """

    def __init__(
        self,
        hyperparam_generator: Callable[[dict], dict],
        data_loader: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        data_sampler: Callable
    ):
        """
        Initializes the Runner.

        Args:
            hyperparam_generator (Callable): Function generating hyperparams from config.
            sampler (Callable): Function returning a bootstrap sample of (X, y).
        """
        self.hyperparam_generator = hyperparam_generator
        self.data_loader = data_loader
        self.data_sampler = data_sampler
        self.models: List[BaseModel] = []

    def build_ensemble(self, conf: dict, seed : int = 42) -> None:
        """
        Builds the ensemble by creating model instances with sampled data and hyperparameters.
        """

        self.models.clear()
        logs = {
            'train': {},
            'test': {}
        }


        ds_name = self.data_loader.list_available_datasets()[0]
        
        print(f'Using dataset: {ds_name}')

        X_train, X_test, y_train, y_test, feature_columns = self.data_loader.load_dataset(ds_name)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        for i in range(conf['n_models']):
            
            dataset_conf = conf['dataset_conf']
            sample, features = self.data_sampler.load_bootstrapped_dataset(dataset_conf=dataset_conf, seed=None)
            target_column = sample.columns[-1]

            X_sample = sample.drop(columns=[target_column])
            y_sample = sample[target_column]
            # X_train, X_test, y_train, y_test = train_test_split(
            # X, y, test_size=0.1, random_state=seed
            # )
            
            model_conf = conf['model_conf']

            hyperparams = self.hyperparam_generator.generate_hyperparams(model_conf, seed=None)
            model = model_conf['model_type'](hyperparams)
            # print(hyperparams)

            log_train = model.fit(X_sample, y_sample)
            print(f"Training finished!")
            # print(log_train)
            log_train = model.evaluate(X_sample, y_sample)
            print(log_train)

            logs['train'][f"{model_conf['model_name']}_{i}"] = log_train
            # logs['test'][f"{model_conf['model_name']}_{i}"] = log_test
            self.models.append(model)
        print(logs)
        return logs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts using the ensemble (majority voting or averaging).

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Aggregated predictions from all models.
        """
        predictions = np.array([model.predict(X) for model in self.models])
        # Majority voting (for classification)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric_fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
        """
        Evaluates the ensemble using a provided metric function.

        Args:
            X (np.ndarray): Evaluation features.
            y (np.ndarray): Ground truth labels.
            metric_fn (Callable): Metric function accepting y_true and y_pred.

        Returns:
            float: Evaluation score.
        """
        y_pred = self.predict(X)
        return metric_fn(y, y_pred)

    def get_models(self) -> List[BaseModel]:
        """
        Returns:
            List[BaseModel]: List of trained models.
        """
        return self.models
