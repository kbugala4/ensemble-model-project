conf = {
    'run_id': 0,
    'seed': 0,
    
    'n_models': 10,
    
    'dataset_conf': {
        'dataset_name': "xxx",
        'shrunk_size': None,
        'n_attributes_max': 10,
        'n_attributes_min': 10,
        'select_type': 'standard-selection', # or 'sqrt-selection'
        'target_column': 'target'
    },
    
    'model_conf': {
        'model_name': 'XXXTree',

        'hyperparam_A': {
            'max_value': 100,
            'min_value': 100,
        }
        
        # 'hyperparam_B': {
        #     ...
        # }
        # ...
        
        }
}