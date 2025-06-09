from typing import Callable, List, Dict, Any
import numpy as np
from scipy.stats import mode
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
import os
import random
# from models.base_model import BaseModel
from src.models.base_model import BaseModel
from sklearn.model_selection import train_test_split
# from utils.dataloader import DatasetLoader
from src.utils.dataloader import DatasetLoader
import sys
from collections import defaultdict
from src.utils.metrics_saver import MetricsSaver
from src.utils.common import COMMON_DATASET_CONFIGS

# from utils.metrics_saver import MetricsSaver
# from utils.common import COMMON_DATASET_CONFIGS
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
        data_sampler: Callable,
        metrics_saver: Callable,
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
        self.metrics_saver = metrics_saver

    def run_dataset_experiment(self, conf: dict, n_runs: int = 1, seed: int = 42) -> Dict:
        dataset_name = conf['dataset_conf']['dataset_shortname']
        for use_default in [False, True]:
            for i in range(n_runs):
                seed = seed + 1
                train_logs, test_logs = self.build_ensemble(conf, dataset_name, use_default=use_default, seed=seed)
                averages_train_logs = EnsembleRunner.average_model_metrics(train_logs)

                full_report = test_logs
                # Add training info
                for k, v in averages_train_logs['train'].items():
                    full_report[f'{k}_train'] = v

                name = f"{conf['model_name']}_run{i}"                
                if use_default:
                    name = name + "_default"
                else:
                    name = name + "_randomize"

                self.metrics_saver.add_experiment_results(
                    dataset_name,
                    name,
                    full_report
                    )
            
        # # Print summary and save results
        # self.metrics_saver.print_summary()
        # json_path, csv_path = self.metrics_saver.save_results()
        
        # print(f"\n✅ Full custom experiments completed!")
        # print(f"📄 Detailed results: {json_path}")
        # print(f"📊 Summary: {csv_path}")
        
        return self.metrics_saver.get_results()

    def build_ensemble(self, conf: dict, dataset_name: str, use_default: bool = False, seed : int = 42) -> Dict:
        """
        Builds the ensemble by creating model instances with sampled data and hyperparameters.
        """

        self.models.clear()
        logs = {}
        _, X_test, _, y_test, _   = self.data_loader.load_dataset(dataset_name)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        for i in range(conf['n_models']):
            model_logs = {}

            dataset_conf = conf['dataset_conf']
            sample, features = self.data_sampler.load_bootstrapped_dataset(dataset_conf=dataset_conf, seed=None)
            target_column = sample.columns[-1]
            model_logs['selected_features'] = features

            X_sample = sample.drop(columns=[target_column])
            y_sample = sample[target_column]
            
            model_conf = conf['model_conf']

            hyperparams = self.hyperparam_generator.generate_hyperparams(model_conf, use_default=use_default, seed=None)
            model = model_conf['model_type'](hyperparams)
            model.training_features = features
            model_logs['hyperparams'] = hyperparams


            log_train = model.fit(X_sample, y_sample)

            log_train = model.evaluate(X_sample, y_sample)
            log_test = model.evaluate(X_test[features], y_test)

            model_logs['train'] = log_train
            model_logs['test'] = log_test
            model_key = f"{conf['model_name']}_{i}"
            logs[model_key] = model_logs

            self.models.append(model)
        
        test_metrics = self.evaluate(X_test, y_test)
        return logs, test_metrics
    
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

    # def evaluate_model(self, X: np.ndarray, y: np.ndarray, ):
    #     """
    #     Evaluates the ensemble using a provided metric function.

    #     Args:
    #         X (np.ndarray): Evaluation features.
    #         y (np.ndarray): Ground truth labels.

    #     Returns:
    #         dict: Dictionary with evaluation metrics.
    #     """
    #     y_pred = self.predict(X)
    #     results = {
    #         'accuracy': float(accuracy_score(y, y_pred)),
    #         'precision_macro': float(precision_score(y, y_pred, average='macro', zero_division=0)),
    #         'precision_weighted': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
    #         'recall_macro': float(recall_score(y, y_pred, average='macro', zero_division=0)),
    #         'recall_weighted': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
    #         'f1_macro': float(f1_score(y, y_pred, average='macro', zero_division=0)),
    #         'f1_weighted': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
    #         'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
    #         'classification_report': classification_report(y, y_pred, output_dict=True),
    #     }
    #     return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluates an ensemble of models using majority voting and returns various metrics.

        Args:
            models (list): List of trained models implementing a .predict() method.
            X_test (np.ndarray or pd.DataFrame): Test features.
            y_test (np.ndarray or pd.Series): True labels.

        Returns:
            dict: Dictionary with evaluation metrics.
        """
        if len(self.models) == 0:
            raise "No model to evaluate"

        # Get predictions from all models
        predictions = np.array([model.predict(X[model.training_features]) for model in self.models])  # shape: (n_models, n_samples)

        # Majority voting
        majority_preds, _ = mode(predictions, axis=0, keepdims=False)  # shape: (n_samples,)

        # Calculate metrics
        y_pred = majority_preds

        results = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision_macro': float(precision_score(y, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            'recall_macro': float(recall_score(y, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(y, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True),
        }

        return results
        # """
        # Evaluates the ensemble using a provided metric function.

        # Args:
        #     X (np.ndarray): Evaluation features.
        #     y (np.ndarray): Ground truth labels.
        #     metric_fn (Callable): Metric function accepting y_true and y_pred.

        # Returns:
        #     float: Evaluation score.
        # """
        # y_pred = self.predict(X)
        # return metric_fn(y, y_pred)

    def get_models(self) -> List[BaseModel]:
        """
        Returns:
            List[BaseModel]: List of trained models.
        """
        return self.models

    @staticmethod
    def average_model_metrics(models_metrics: dict) -> dict:
        """
        Averages scalar performance metrics across all models in the ensemble.

        Args:
            models_metrics (dict): Dictionary of per-model metrics (like the one you provided).

        Returns:
            dict: Dictionary with averaged metrics under 'train' and 'test' keys.
        """
        train_sums = defaultdict(float)
        test_sums = defaultdict(float)
        n_models = len(models_metrics)

        scalar_metrics = [
            'accuracy',
            'precision_macro',
            'precision_weighted',
            'recall_macro',
            'recall_weighted',
            'f1_macro',
            'f1_weighted'
        ]

        for model_key, model_data in models_metrics.items():
            train = model_data.get('train', {})
            test = model_data.get('test', {})
            
            for metric in scalar_metrics:
                train_sums[metric] += train.get(metric, 0.0)
                test_sums[metric] += test.get(metric, 0.0)

        averaged = {
            'train': {metric: train_sums[metric] / n_models for metric in scalar_metrics},
            'test': {metric: test_sums[metric] / n_models for metric in scalar_metrics}
        }

        return averaged
