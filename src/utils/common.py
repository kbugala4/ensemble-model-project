FLIGHTS_DATASET_TARGET_CLASS_NAME = "satisfaction"
CREDIT_CARDS_DATASET_TARGET_CLASS_NAME = "class"
HUMAN_ACTIVITY_DATASET_TARGET_CLASS_NAME = "Activity"

COMMON_DATASET_CONFIGS = {
    'flights': {
        'path': 'binary_balanced_airflight_satisfaction',
        'target_column': 'satisfaction',
        'type': 'binary'
    },
    'credit_cards': {
        'path': 'binary_unbalanced_credit_card_fraud', 
        'target_column': 'class',
        'type': 'binary'
    },
    'human_activity': {
        'path': 'multi_class_balanced_human_activity_recognition',
        'target_column': 'Activity', 
        'type': 'multiclass'
    }
}