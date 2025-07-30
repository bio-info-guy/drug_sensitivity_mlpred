import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
import sys
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier as skl_rf
from sklearn.ensemble import RandomForestRegressor as skl_rfr
from sklearn.linear_model import SGDClassifier, SGDRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
sys.path.append('../')
sys.path.append('./')
from parallel.device_pin import set_cuda_device, pick_free_gpu
import logging


MODEL_TYPES = {
    'xgb_classifier': XGBClassifier,
    'xgb_regressor': XGBRegressor,
    'randomforest_classifier': skl_rf,
    'randomforest_regressor': skl_rfr,
    'sgd_classifier': SGDClassifier,
    'sgd_regressor': SGDRegressor,
    'lgbm_classifier': LGBMClassifier,
    'lgbm_regressor': LGBMRegressor
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


from sklearn.metrics import r2_score
def r2_scorer(pred, labels):
    return 'r2_score', r2_score(pred,labels), True

def get_model_for_device(config, y_type):
    """
    Get the model for the specified device and y_type.
    Args:
        config (dict): Configuration dictionary.
        y_type (str): Type of y, either 'binary' or 'continuous'.
    Returns:
        model: The model object.
        config: The updated config object.
    """
    model_base_name = config['model_type']
    device = config.get('device', 'cpu')  # Default to CPU if not specified
    # Determine if the task is classification or regression
    classification_metrics = ['balanced_accuracy', 'precision', 'average_precision', 'recall', 'f1', 'roc_auc']
    regression_metrics = ['r2',  'neg_mean_absolute_error', 'neg_root_mean_squared_error','neg_mean_absolute_percentage_error']


    if y_type == 'binary':
        model_name = f"{model_base_name}_classifier"
        default_scoring_metric = 'average_precision'
        valid_metrics = classification_metrics
    else:
        model_name = f"{model_base_name}_regressor"
        default_scoring_metric = r2_scorer
        valid_metrics = regression_metrics

    scoring_metric = config.get('scoring_metric', default_scoring_metric)

    # Check if the provided scoring_metric is valid for the given y_type
    if scoring_metric not in valid_metrics:
        logging.warning(
            f"Invalid scoring metric '{scoring_metric}' for {y_type} task. "
            f"Falling back to default: '{default_scoring_metric}'"
        )
        scoring_metric = default_scoring_metric
    
    config['scoring_metric'] = scoring_metric

    fixed_params = config.get('fixed_params', {})
    search_method = config.get('search_method', None) # Get search method from config
    n_cores = config.get('n_cores', 1)
    # Ensure device is 'cpu' for models that don't support GPU
    if model_base_name not in ['xgb', 'randomforest']:
        device = 'cpu'  # Force to CPU if GPU is requested for unsupported models
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
    
    model_class = MODEL_TYPES[model_name]
    
    if model_base_name == 'xgb':
        if device == 'cuda':
            logging.info('XGBoost using cuda')
            return model_class(device='cuda', tree_method='hist', **fixed_params), config
        else:
            logging.info('XGBoost using cpu')
            config['n_cores'] = 2
            return model_class(**fixed_params), config
    elif model_base_name == 'randomforest':
        if device == 'cuda':
            from cuml import RandomForestClassifier as cu_rf, RandomForestRegressor as cu_rfr
            logging.info('RandomForest using cuda')
            model_class = cu_rf if y_type == 'binary' else cu_rfr
            return model_class(**fixed_params), config
        else:
            logging.info('RandomForest using cpu')
            return model_class(**fixed_params), config
    elif model_base_name == 'lgbm':
        if device == 'cuda':
            # LightGBM can use GPU with device='gpu'
            logging.info('LGBM using cuda')
            fixed_params['device'] = 'gpu'
        else:
            logging.info('LGBM using cpu')
        
        cb = []
        fit_params = {}
        if config.get('early_stop', False) and config.get('test_set', None) is not None:
            cb.append(multi_EarlyStopCB_wrapper(stopping_rounds=200, first_metric_only=True))
            fit_params['model__eval_set'] = config.get('test_set')
            fit_params['model__eval_metric'] = config.get('scoring_metric', 'average_precision' if y_type == 'binary' else r2_scorer)

        if config.get('lr_decay', False):
            def decay_rate(current_round):
                return 0.1 * (0.995 ** current_round)
            cb.append(lgb.reset_parameter(learning_rate=decay_rate))
        
        if len(cb) > 0:
            fit_params['model__callbacks'] = cb
        
        config['fit_params'] = fit_params
        config['n_cores'] = 2
        fixed_params['n_jobs'] = max(n_cores // 2, 1)
        fixed_params['metric'] = 'custom'
        fixed_params['importance_type'] = 'gain'
        if 'test_set' in config:
            config.pop('test_set')
            
        return model_class(**fixed_params), config
    
    elif model_base_name in ['sgd']:
        # For other models, use the CPU version from MODEL_TYPES
        # Remove n_jobs if the model doesn't support it
        if not hasattr(model_class(), 'n_jobs') and 'n_jobs' in fixed_params:
            fixed_params.pop('n_jobs')
        return model_class(**fixed_params), config
    else:
        raise ValueError(f'{model_base_name} type not supported')


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
    
    pipe_list.append(('model', base_estimator))

    pipeline_estimator = Pipeline(
            pipe_list
        )
    return pipeline_estimator
