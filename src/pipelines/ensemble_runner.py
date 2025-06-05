from typing import Callable, List, Dict, Any
import numpy as np
import os
import random
from models.base_model import BaseModel
from sklearn.model_selection import train_test_split


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
        data_sampler: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ):
        """
        Initializes the Runner.

        Args:
            hyperparam_generator (Callable): Function generating hyperparams from config.
            sampler (Callable): Function returning a bootstrap sample of (X, y).
        """
        self.hyperparam_generator = hyperparam_generator
        self.data_sampler = data_sampler
        self.models: List[BaseModel] = []

    def build_ensemble(self, model_conf: dict, seed : int = 42) -> None:
        """
        Builds the ensemble by creating model instances with sampled data and hyperparameters.
        """

        self.models.clear()
        logs = {}

        # TUTAJ CHCEMY ODCZYTAC TEST SET
        X_test, y_test = ...

        for i in range(model_conf['n_models']):

            # TU NIECH BIERZE CONFIG DATASETU, GENERUJE PODZBIOR I ZWRACA WYBRANE KOLUMNY PODZBIORU
            X_sample, y_sample, features = self.data_sampler.load_bootstrapped_dataset(model_conf['dataset_conf'], seed=seed)

            hyperparams = self.hyperparam_generator.generate_hyperparams(self.model_config, seed=seed)
            model = model_conf['model_class'](hyperparams)

            log_train = model.fit(X_sample, y_sample)
            log_test = model.evaluate(X_test[features], y_test)

            logs['train'][f"{model_conf['model_name']}_{i}"] = log_train
            logs['test'][f"{model_conf['model_name']}_{i}"] = log_test
            self.models.append(model)
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
