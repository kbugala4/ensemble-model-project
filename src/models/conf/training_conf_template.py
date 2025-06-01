conf = {
    'run_id': 0,
    'seed': 0,
    
    'n_models': 10,
    
    'dataset_conf': {
        'dataset_name': "xxx.csv",
        'shrunk_size': None,
        'n_attributes_max': 10,
        'n_attributes_min': 10,
        # whatever conf value we need
        # '...': '...'
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