from typing import Dict, Any
import numpy as np
import pickle
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .base_model import BaseModel


class SklearnEnsembleWrapper(BaseModel):
    """
    Wrapper for sklearn ensemble models to make them compatible with BaseModel interface.
    """
    
    def __init__(self, name: str, model_class, params: dict = None):
        """
        Initialize sklearn model wrapper.
        
        Args:
            name (str): Model name
            model_class: Sklearn model class (e.g., BaggingClassifier)
            params (dict): Model hyperparameters
        """
        super().__init__(name, params)
        self.model_class = model_class
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, training_params: dict = None) -> None:
        """Fit the sklearn model."""
        # Initialize model with params
        self.model = self.model_class(**self.params)
        
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        print(f"{self.name} fitted on data shape {X.shape}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray, metric: callable) -> float:
        """Evaluate model performance."""
        predictions = self.predict(X)
        return metric(y, predictions)
        
    def save(self, path: str) -> None:
        """Save model to file."""
        model_state = {
            'name': self.name,
            'params': self.params,
            'model': self.model,
            'model_class': self.model_class,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
            
    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        self.name = model_state['name']
        self.params = model_state['params']
        self.model = model_state['model']
        self.model_class = model_state['model_class']
        self.is_fitted = model_state['is_fitted']
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter  
    def name(self, value: str):
        self._name = value

    def get_params(self) -> dict:
        """Returns the model's hyperparameters."""
        return self.params.copy()


def create_bagging_classifier(params: dict = None) -> SklearnEnsembleWrapper:
    """
    Create BaggingClassifier wrapper.
    
    Args:
        params (dict): Custom parameters for BaggingClassifier
        
    Returns:
        SklearnEnsembleWrapper: Wrapped BaggingClassifier
    """
    default_params = {
        'estimator': DecisionTreeClassifier(),
        'n_estimators': 10,
        'random_state': 42,
        'bootstrap': True,
        'bootstrap_features': False
    }
    if params:
        default_params.update(params)
    
    return SklearnEnsembleWrapper(
        name="BaggingClassifier",
        model_class=BaggingClassifier,
        params=default_params
    )


def create_random_forest_classifier(params: dict = None) -> SklearnEnsembleWrapper:
    """
    Create RandomForestClassifier wrapper.
    
    Args:
        params (dict): Custom parameters for RandomForestClassifier
        
    Returns:
        SklearnEnsembleWrapper: Wrapped RandomForestClassifier
    """
    default_params = {
        'n_estimators': 10,
        'random_state': 42,
        'bootstrap': True
    }
    if params:
        default_params.update(params)
        
    return SklearnEnsembleWrapper(
        name="RandomForestClassifier", 
        model_class=RandomForestClassifier,
        params=default_params
    ) 