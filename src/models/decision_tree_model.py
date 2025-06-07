import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from models.base_model import BaseModel
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class DecisionTreeModel(BaseModel):
    """
    Wrapper class for sklearn's DecisionTreeClassifier, compliant with BaseModel interface.
    """

    def __init__(self, params: dict = None):
        """
        Initializes the DecisionTree model with given hyperparameters.

        Args:
            params (dict): Dictionary of hyperparameters for DecisionTreeClassifier.
        """
        # super().__init__(name="DecisionTree", params=params)
        self.params = params
        self.model = DecisionTreeClassifier(**(params or {}))
        self.training_features = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the DecisionTreeClassifier on the provided dataset.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for given input.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric: callable = None) -> float:
        """
        Evaluates the model.

        Args:
            X (np.ndarray): Evaluation features.
            y (np.ndarray): Ground truth labels.

        Returns:
            dict: Dictionary with evaluation metrics.
        """
        y_pred = self.predict(X)
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


    def save(self, path: str) -> None:
        """
        Saves the trained model to the specified file path.

        Args:
            path (str): Path to save the model file.
        """
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """
        Loads the model from the specified file path.

        Args:
            path (str): Path to the saved model file.
        """
        self.model = joblib.load(path)

    @property
    def name(self) -> str:
        """
        Returns the name of the model.

        Returns:
            str: Model name.
        """
        return self._name

    def get_params(self) -> dict:
        """
        Returns the model's hyperparameters.

        Returns:
            dict: Model parameters.
        """
        return self.model.get_params()
