from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for all classification models in the ensemble system.
    Unifies the API interface for runner class.

    All model implementations must subclass this and implement all abstract methods.
    """

    def __init__(self, name: str = "BaseModel", params: dict = None):
        """
        Initializes the base model with a name and optional hyperparameters.

        Args:
            name (str): The display name of the model.
            params (dict, optional): Dictionary of model hyperparameters.
        """
        self.name = name
        self.params = params or {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, training_params: dict = None) -> None:
        """
        Fits the model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
            training_params (dict, optional): Additional training-specific parameters,
                such as 'n_epochs', 'n_models', 'batch_size', etc. Defaults to None.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the given input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels or probabilities of shape (n_samples,).
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray, metric: callable) -> float:
        """
        Evaluates the model performance on the given test data using a specified metric function.

        Args:
            X (np.ndarray): Test feature set.
            y (np.ndarray): True labels.
            metric (callable): A scoring function.

        Returns:
            float: Evaluation score.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the model to the specified file path.

        Args:
            path (str): File path to save the model.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the model from the specified file path.

        Args:
            path (str): File path to load the model from.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Abstract property for the model's name.
        """
        pass


    def get_params(self) -> dict:
        """
        Returns the model's hyperparameters.

        Returns:
            dict: Dictionary of model hyperparameters.
        """
        return {}
