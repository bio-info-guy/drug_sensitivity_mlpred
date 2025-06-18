import json
import yaml
from scipy.stats import uniform, beta, norm, randint, expon, lognorm

# Define model names (these will be needed for validation in load_config)
MODEL_TYPES = {
    'XGBClassifier': None, # Placeholder, actual classes will be imported in skl_train_model.py
    'RandomForestClassifier': None,
    'SGDClassifier': None,
    'LGBMClassifier': None
}

# Define oversampler types (these will be needed for validation in load_config)
OVERSAMPLER_TYPES = {
    'RandomOverSampler': None, # Placeholder
    'SMOTE': None
}

# Define FLAML supported tasks and estimators
FLAML_TASKS = ['classification', 'regression']
FLAML_ESTIMATORS = ['lgbm', 'xgboost', 'xgb_limitdepth', 'rf', 'extra_tree', 'lrl1', 'lrl2', 
                   'catboost', 'kneighbor', 'prophet', 'arima', 'sarimax', 'holt-winters',
                   'sgd', 'svc', 'nb', 'dt']

def _parse_distribution(dist_str):
    """Parses a string representation of a scipy.stats distribution."""
    parts = dist_str.split('(')
    dist_name = parts[0]
    if dist_name not in globals():
        raise ValueError(f"Unsupported distribution: {dist_name}")
    
    dist_args_str = parts[1][:-1] # Remove closing parenthesis
    dist_args = [float(arg.strip()) if '.' in arg or 'e' in arg else int(arg.strip()) for arg in dist_args_str.split(',')]
    
    return globals()[dist_name](*dist_args)

def load_config(config_file):
    """Load configuration from JSON or YAML file."""
    with open(config_file, 'r') as f:
        if config_file.endswith('.json'):
            config = json.load(f)
        elif config_file.endswith(('.yml', '.yaml')):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON (.json) or YAML (.yml/.yaml)")
    
    # Validate required fields
    required_fields = ['model_type', 'grid_search_params', 'use_oversampling', 
                      'oversampler_type', 'oversampler_seed', 'train_test_split_seed']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Validate model type
    if config['model_type'] not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {config['model_type']}. Supported: {list(MODEL_TYPES.keys())}")
    
    # Validate oversampler type
    if config['use_oversampling'] and config['oversampler_type'] not in OVERSAMPLER_TYPES:
        raise ValueError(f"Unsupported oversampler type: {config['oversampler_type']}. Supported: {list(OVERSAMPLER_TYPES.keys())}")

    # Parse scipy distributions in grid_search_params
    if 'grid_search_params' in config:
        for param, value in config['grid_search_params'].items():
            if isinstance(value, str) and value.startswith(('uniform(', 'beta(', 'norm(', 'randint(', 'expon(', 'lognorm(')):
                config['grid_search_params'][param] = _parse_distribution(value)
    
    return config

def load_flaml_config(config_file):
    """Load FLAML configuration from JSON or YAML file."""
    with open(config_file, 'r') as f:
        if config_file.endswith('.json'):
            config = json.load(f)
        elif config_file.endswith(('.yml', '.yaml')):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON (.json) or YAML (.yml/.yaml)")
    
    # Validate required fields for FLAML
    required_fields = ['task', 'time_budget', 'metric']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in FLAML config: {field}")
    
    # Validate task type
    if config['task'] not in FLAML_TASKS:
        raise ValueError(f"Unsupported FLAML task: {config['task']}. Supported: {FLAML_TASKS}")
    
    # Validate estimator list if provided
    if 'estimator_list' in config:
        if not isinstance(config['estimator_list'], list):
            raise ValueError("estimator_list must be a list")
        for estimator in config['estimator_list']:
            if estimator not in FLAML_ESTIMATORS:
                raise ValueError(f"Unsupported FLAML estimator: {estimator}. Supported: {FLAML_ESTIMATORS}")
    
    # Validate time_budget
    if not isinstance(config['time_budget'], (int, float)) or config['time_budget'] <= 0:
        raise ValueError("time_budget must be a positive number")
    
    # Validate metric is a string
    if not isinstance(config['metric'], str):
        raise ValueError("metric must be a string")
    
    return config
