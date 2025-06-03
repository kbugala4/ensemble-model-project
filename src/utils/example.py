from utils.generate_hyperparams import HyperparamsGenerator
from utils.generate_data_subset import DataSubsetGenerator


if __name__ == '__main__':
    # Example configuration matching your template
    example_dataset_conf = {
        'dataset_name': 'data/processed/binary_balanced_airflight_satisfaction/train.csv',
        'shrunk_size': 1000,
        'n_attributes_max': 10,
        'n_attributes_min': 5
    }
    
    try:
        # Create an instance of the class
        generator = DataSubsetGenerator()
        df_bootstrap, selected_features = generator.load_bootstrapped_dataset(example_dataset_conf, seed=42)
        print(f'\nBootstrap successful!')
        print(f'Selected features: {selected_features}')
        print(f'Final dataset shape: {df_bootstrap.shape}')
        print(f'Dataset columns: {list(df_bootstrap.columns)}')
        print(df_bootstrap.head())
        
    except Exception as e:
        print(f'Error: {e}')

    model_conf = {
        'model_name': 'DecisionTree',
        'max_depth': {
            'type': 'int',
            'min': 10,
            'max': 20
        },
        'min_samples_split': {
            'type': 'int',
            'min': 2,
            'max': 10
        },
        'min_samples_leaf': {
            'type': 'int',
            'min': 1,
            'max': 5
        },
        'max_features': {
            'type': 'list',
            'options': ['sqrt', 'log2', None]
        },
        'criterion': {
            'type': 'list',
            'options': ['gini', 'entropy']
        },
        'splitter': {
            'type': 'list',
            'options': ['best', 'random']
        }        
    }
    
    randomize_f = HyperparamsGenerator.randomize_uniform

    try:
        # Create an instance of the class
        hyperparam_generator = HyperparamsGenerator(randomize_f=randomize_f)
        
        hyperparams = hyperparam_generator.generate_hyperparams(model_conf, seed=42)
        
        print(f'\nHyperparameters generated!')
        for k,v in hyperparams.items():
            print(f'{k}: {v}')
       
    except Exception as e:
        print(f'Error: {e}')

