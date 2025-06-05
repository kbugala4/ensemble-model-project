import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from models.base_model import BaseModel


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

    def fit(self, X: np.ndarray, y: np.ndarray, training_params: dict = None) -> None:
        """
        Trains the DecisionTreeClassifier on the provided dataset.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            training_params (dict, optional): Not used for sklearn tree.
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
        Evaluates the model using the specified metric function.

        Args:
            X (np.ndarray): Evaluation features.
            y (np.ndarray): True labels.
            metric (callable, optional): Evaluation function. Defaults to accuracy.

        Returns:
            float: Evaluation score.
        """
        y_pred = self.predict(X)
        metric_fn = metric if metric else accuracy_score
        return metric_fn(y, y_pred)

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
