import random
import numpy as np
import sys
sys.path.append("..") 

class HyperparamsGenerator:
    """
    Utility class for generating randomized hyperparameters based on a given configuration.

    This class supports generating model hyperparameters by applying either a user-defined
    randomization function or a default method that handles uniform sampling of integers,
    floats, or categorical options.
    """

    def __init__(self, randomize_f: callable = None, use_default: bool = False) -> None:
        """
        Initializes the HyperparamsGenerator.

        Args:
            randomize_f (callable, optional): A function that accepts a parameter description
                dictionary (containing metadata such as 'type', 'min', 'max', or 'options')
                and returns a randomized value. If None, a default uniform randomizer is used.
            use_default (bool): Flag if the values set for hyperparameters should be filled with the default value
        """
        if randomize_f is None:
            self.randomize_f = self.randomize_uniform
        else:
            self.randomize_f = randomize_f
        self.use_default = use_default

    def generate_hyperparams(self, model_conf: dict, use_default: bool = False, seed: int = None) -> dict:
        """
        Generates a dictionary of randomized hyperparameters according to the specified configuration.

        Args:
            model_conf (dict): A dictionary where keys are hyperparameter names and values are
                dictionaries defining the constraints for randomization (e.g., type, range, options).
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            dict: A dictionary mapping hyperparameter names to their randomized values.

        Raises:
            TypeError: If `model_conf` is not a dictionary.
        """
        if not isinstance(model_conf, dict):
            raise TypeError("Configuration for hyperparameters must be a dictionary.")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        model_hyperparams = {}
        for param_name, param_desc in model_conf.items():
            if param_name in ['model_name', 'model_id', 'model_type']:
                continue
            if self.use_default:
                model_hyperparams[param_name] = param_desc['default']
            else:
                model_hyperparams[param_name] = self.randomize_f(param_desc)

        return model_hyperparams

    # @staticmethod
    # def get_default(param_dict: dict)  -> int | float | str:
    #     return param_dict['default']
    #     param_type = param_dict['type']
    #     if param_type in ['float', 'int']:
    #         min_v = int(param_dict['min'])
    #         max_v = int(param_dict['max'])

    #         if param_type == 'int':
    #             return random.randint(min_v, max_v)
    #         elif param_type == 'float':
    #             return random.uniform(min_v, max_v)
    #     elif param_type == 'list':
    #         return random.choice(param_dict['options'])
    #     else:
    #         raise ValueError(f"Unsupported parameter type: {param_type}")

    @staticmethod
    def randomize_uniform(param_dict: dict) -> int | float | str:
        """
        Default randomization function that generates a randomized value based on parameter type.

        Supports:
        - 'int': uniform random integer between 'min' and 'max' (inclusive)
        - 'float': uniform random float between 'min' and 'max'
        - 'list': random choice from a list of options under 'options'

        Args:
            param_dict (dict): Parameter description dict containing:
                - 'type' (str): One of 'int', 'float', or 'list'
                - For 'int' and 'float': 'min' and 'max' keys defining range
                - For 'list': 'options' key with list of possible values

        Returns:
            int | float | str: Randomly selected value appropriate to the parameter type.

        Raises:
            KeyError: If required keys are missing in param_dict.
            ValueError: If 'type' is unsupported.
        """
        
        param_type = param_dict['type']
        if param_type in ['float', 'int']:
            min_v = int(param_dict['min'])
            max_v = int(param_dict['max'])

            if param_type == 'int':
                return random.randint(min_v, max_v)
            elif param_type == 'float':
                return random.uniform(min_v, max_v)
        elif param_type == 'list':
            return random.choice(param_dict['options'])
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
