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
