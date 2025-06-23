import json
import yaml
from scipy.stats import uniform, beta, norm, randint, expon, lognorm
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

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

def _parse_optuna_distribution(dist_str):
    """Parses a string representation of an optuna distribution."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not available. Please install optuna to use optuna distributions.")
    dist_str0 = dist_str
    # Remove 'optuna.distributions.' prefix if present
    if dist_str.startswith('optuna.distributions.'):
        dist_str = dist_str[len('optuna.distributions.'):]
        # Parse the distribution
    if dist_str.startswith('FloatDistribution('):
        return eval(dist_str0)
    elif dist_str.startswith('IntDistribution('):
        return eval(dist_str0)
    elif dist_str.startswith('CategoricalDistribution('):
        return eval(dist_str0)
    else:
        raise ValueError(f"Unsupported optuna distribution: {dist_str}")

def load_config(config_file, n_cores, device):
    """Load configuration from JSON or YAML file."""
    with open(config_file, 'r') as f:
        if config_file.endswith('.json'):
            config = json.load(f)
        elif config_file.endswith(('.yml', '.yaml')):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON (.json) or YAML (.yml/.yaml)")
    
    # Validate required fields
    required_fields = ['model_type', 'use_oversampling', 
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

    # Get search method and validate
    search_method = config.get('search_method', 'gridcv')
    valid_search_methods = ['gridcv', 'halvingrandomsearch', 'optuna']
    if search_method not in valid_search_methods:
        raise ValueError(f"Unsupported search method: {search_method}. Supported: {valid_search_methods}")
    
    # Determine which search parameters to use based on search method
    search_params_key = f"{search_method.replace('cv', '').replace('search', '')}_search_params"
    if search_method == 'gridcv':
        search_params_key = 'grid_search_params'
    elif search_method == 'halvingrandomsearch':
        search_params_key = 'halving_search_params'
    elif search_method == 'optuna':
        search_params_key = 'optuna_search_params'
    
    # Validate that the required search params exist
    if search_params_key not in config:
        raise ValueError(f"Missing search parameters for method '{search_method}': {search_params_key}")
    
    # Parse distributions based on search method
    if search_method in ['gridcv', 'halvingrandomsearch']:
        # Parse scipy distributions for grid and halving search
        for param, value in config[search_params_key].items():
            if isinstance(value, str) and value.startswith(('uniform(', 'beta(', 'norm(', 'randint(', 'expon(', 'lognorm(')):
                config[search_params_key][param] = _parse_distribution(value)
    elif search_method == 'optuna':
        # Parse optuna distributions
        for param, value in config[search_params_key].items():
            if isinstance(value, str) and 'optuna.distributions.' in value:
                config[search_params_key][param] = _parse_optuna_distribution(value)
    
    # Add the active search parameters to the config for easy access
    config['search_params'] = config[search_params_key]
    
    #handle gpu and cpu here
    if config.get('model_type') in ['XGBClassifier', 'RandomForestClassifier']:
        config['device'] = device
        if device == 'cuda':
            config['n_cores'] = 1
        else:
            config['n_cores'] = n_cores


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
