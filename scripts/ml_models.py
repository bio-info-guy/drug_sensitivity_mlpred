import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
import sys
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier as skl_rf
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
sys.path.append('../')
sys.path.append('./')
from parallel.device_pin import set_cuda_device, pick_free_gpu
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



class multi_EarlyStopCB_wrapper:
    def __init__(self, **params):
        self.escb_param = params
        self.CB_ = {}
    def __call__(self, env):

        if str(env.model) in self.CB_:
            self.CB_[str(env.model)](env)
        else:
            self.CB_[str(env.model)] = lgb.early_stopping(**self.escb_param)
            self.CB_[str(env.model)](env)


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
        #set_cuda_device()
        #set_cuda_device()
    # Determine n_jobs based on device and search_method
    if device == 'cpu':# CPU models
        if search_method in ['gridcv', 'optuna', 'halvingrandomsearch']: # If hyperparameter search is active
            fixed_params['n_jobs'] = max(n_cores // 2, 1)
        else: # No hyperparameter search
            fixed_params['n_jobs'] = n_cores
    else: # GPU models (device == 'cuda')
        fixed_params['n_jobs'] = 1 # n_jobs is effectively 1 for GPU models
    logging.info(f"using {fixed_params['n_jobs']} for model training")
    if model_name == 'XGBClassifier':
        if device == 'cuda':
            # For GPU XGBoost, set device and tree_method
            logging.info('XGBoost using cuda')
            logging.info('XGBoost using cuda')
            return XGBClassifier(device='cuda', tree_method='hist', **fixed_params), config
        else:
            # For CPU XGBoost, use default or specified n_jobs
            logging.info('XGBoost using cpu')
            logging.info('XGBoost using cpu')
            config['n_cores'] = 2
            return XGBClassifier(**fixed_params), config
    elif model_name == 'RandomForestClassifier':
        if device == 'cuda':
            # For GPU RandomForest, use cuml's RandomForestClassifier
            from cuml import RandomForestClassifier as cu_rf
            logging.info('RandomForest using cuda')
            return cu_rf(**fixed_params), config
        else:
            # For CPU RandomForest, use scikit-learn's RandomForestClassifier
            logging.info('RandomForest using cpu')
            return skl_rf(**fixed_params), config
    elif model_name == 'LGBMClassifier':
        if device == 'cuda':
            # For GPU RandomForest, use cuml's RandomForestClassifier
            from cuml import RandomForestClassifier as cu_rf
            logging.info('LGBM using cuda')
            return cu_rf(**fixed_params), config
        else:
            # For CPU RandomForest, use scikit-learn's RandomForestClassifier
            logging.info('LGBM using cpu')
            cb = []; fit_params = {}
            if config.get('early_stop', False) and config.get('test_set', None) != None:
                cb.append(multi_EarlyStopCB_wrapper(stopping_rounds=200, first_metric_only=True))
                fit_params['classifier__eval_set'] = config.get('test_set')
                fit_params['classifier__eval_metric'] = config.get('scoring_metric', 'f1')
            if config.get('lr_decay', False):
                def decay_rate(current_round):
                    return 0.1 * (0.995 ** current_round) 
                cb.append(lgb.reset_parameter(learning_rate=decay_rate))
            if len(cb) > 0:
                fit_params['classifier__callbacks'] = cb
            config['fit_params'] = fit_params
            config['n_cores'] = 2
            fixed_params['n_jobs'] = n_cores//2+1
            fixed_params['metric'] = None
            config.pop('test_set')
            return LGBMClassifier(**fixed_params), config
    elif model_name in MODEL_TYPES:
        # For other models, use the CPU version from MODEL_TYPES
        model_class = MODEL_TYPES[model_name]
        # Remove n_jobs if the model doesn't support it, as it might have been added for CPU path
        if not hasattr(model_class(), 'n_jobs') and 'n_jobs' in fixed_params:
            fixed_params.pop('n_jobs')
        return model_class(**fixed_params), config
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
    oversample_flag = config.get('use_oversampling', False)
    scaler_flag = config.get('scale_data', False)
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
