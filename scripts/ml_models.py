import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
import sys
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier as skl_rf
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
sys.path.append('../drug_sensitivity_mlpred/')
from utils.misc import set_cuda_device
import logging

MODEL_TYPES = {
    'XGBClassifier': XGBClassifier,
    'RandomForestClassifier': skl_rf,
    'SGDClassifier': SGDClassifier,
    'LGBMClassifier': LGBMClassifier# Added for GPU RandomForest
}

# Define oversampler types
OVERSAMPLER_TYPES = {
    'RandomOverSampler': RandomOverSampler,
    'SMOTE': SMOTE
}

def get_model_for_device(config):

    model_name = config['model_type']
    device = config.get('device', 'cpu') # Default to CPU if not specified
    fixed_params = config.get('fixed_params', {})
    search_method = config.get('search_method', None) # Get search method from config
    n_cores = config.get('n_cores', 1)
    # Ensure device is 'cpu' for models that don't support GPU
    if model_name not in ['XGBClassifier', 'RandomForestClassifier']:
        device = 'cpu' # Force to CPU if GPU is requested for unsupported models
    if device == 'cuda':
        logging.info('running on gpu')
        set_cuda_device()
    # Determine n_jobs based on device and search_method
    if device == 'cpu':# CPU models
        if search_method in ['gridcv', 'optuna', 'halvingrandomsearch']: # If hyperparameter search is active
            fixed_params['n_jobs'] = max(n_cores // 2, 1)
        else: # No hyperparameter search
            fixed_params['n_jobs'] = n_cores
    else: # GPU models (device == 'cuda')
        fixed_params['n_jobs'] = 1 # n_jobs is effectively 1 for GPU models

    if model_name == 'XGBClassifier':
        if device == 'cuda':
            # For GPU XGBoost, set device and tree_method
            return XGBClassifier(device='cuda', tree_method='hist', **fixed_params)
        else:
            # For CPU XGBoost, use default or specified n_jobs
            return XGBClassifier(**fixed_params)
    elif model_name == 'RandomForestClassifier':
        if device == 'cuda':
            # For GPU RandomForest, use cuml's RandomForestClassifier
            from cuml import RandomForestClassifier as cu_rf
            return cu_rf(**fixed_params)
        else:
            # For CPU RandomForest, use scikit-learn's RandomForestClassifier
            return skl_rf(**fixed_params)
    elif model_name in MODEL_TYPES:
        # For other models, use the CPU version from MODEL_TYPES
        model_class = MODEL_TYPES[model_name]
        # Remove n_jobs if the model doesn't support it, as it might have been added for CPU path
        if not hasattr(model_class(), 'n_jobs') and 'n_jobs' in fixed_params:
            fixed_params.pop('n_jobs')
        return model_class(**fixed_params)
    else:
        raise ValueError(f'{model_name} type not supported')


def build_pipe(config, base_estimator):

    """
    Builds base pipeline based on config. Only implemented standardscaler

    Args:
        config: Configuration dictionary with oversampling parameters.
        base_estimator: The base estimator to be used in the pipeline.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        A tuple containing:
            - pipeline_estimator: A Pipeline with oversampler and estimator, or just the base_estimator.
            - X_res: Resampled X_train or original X_train.
            - y_res: Resampled y_train or original y_train.
    """
    oversample_flag = config['use_oversampling']
    scaler_flag = config['scale_data']
    scaler_class = StandardScaler()
    
    pipe_list = []
    if oversample_flag:
        oversampler_class = OVERSAMPLER_TYPES[config['oversampler_type']]
        oversampler = oversampler_class(random_state=config['oversampler_seed'])
        pipe_list.append(('sampling', oversampler))
    if scaler_flag:
        pipe_list.append(('scaling', scaler_class))
    
    pipe_list.append(('classifier', base_estimator))

    pipeline_estimator = Pipeline(
            pipe_list
        )
    return pipeline_estimator
