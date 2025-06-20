import os
import sys
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import yaml
try:
    from optuna.integration import OptunaSearchCV
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
from sklearn.metrics import make_scorer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
import argparse
import mlflow
import mlflow.sklearn
import logging
sys.path.append('../drug_sensitivity_mlpred/')
from utils.config_loader import load_config, MODEL_TYPES, OVERSAMPLER_TYPES # Import from new module
from cross_validation_utils import outer_cross_validate
from utils.misc import dummy_gc, random_name
from customio import read_data, write_drug_model_result
from plot_utils import plot_roc_aupr_curves

#mlflow.create_experiment(
 #   name="Drug_Sensitivity_Model_Training_test",
  #  artifact_location="mlruns/Drug_Sensitivity_Model_Training_test"
#)
#mlflow.autolog()




# Define model names
MODEL_TYPES = {
    'XGBClassifier': XGBClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'SGDClassifier': SGDClassifier,
    'LGBMClassifier': LGBMClassifier
}

# Define oversampler types
OVERSAMPLER_TYPES = {
    'RandomOverSampler': RandomOverSampler,
    'SMOTE': SMOTE
}

def handle_oversampling(config, base_estimator, X_train, y_train):
    """
    Handles oversampling based on configuration.

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
    oversampler_class = OVERSAMPLER_TYPES[config['oversampler_type']]
    oversampler = oversampler_class(random_state=config['oversampler_seed'])

    if oversample_flag:
        X_res, y_res = oversampler.fit_resample(X_train, y_train)
        pipeline_estimator = Pipeline([
            ('sampling', oversampler),
            ('classifier', base_estimator)
        ])
    else:
        X_res, y_res = X_train, y_train
        pipeline_estimator = base_estimator
    
    return pipeline_estimator, X_res, y_res


def perform_hyperparameter_search(  imba_pipeline, X, y, config, n_cores, kfold_inner, oversample_flag):
    """
    Performs hyperparameter search using GridSearchCV, HalvingRandomSearchCV, or OptunaSearchCV.

    Args:
        base_estimator: The base estimator or pipeline object.
        X: Feature matrix.
        y: Target vector.
        config: Configuration dictionary with model parameters.
        n_cores: Number of cores to use for parallel processing.
        kfold_inner: Inner cross-validation strategy.
        oversample_flag: Boolean indicating if oversampling is used.
        imba_pipeline: The pipeline used for hyperparameter search.

    Returns:
        A tuple containing:
            - best_estimator: The best estimator found by the search.
            - best_params: Dictionary of best parameters found, or None if no search.
    """
    search_params = config.get('search_params', {})
    search_method = config.get('search_method', 'gridcv')
    scoring_metric = config.get('scoring_metric', 'average_precision')
    cv_splits = config.get('cv_splits', 5)

    if not search_params:
        return imba_pipeline, None, None

    final_search_parameters = {}
    if oversample_flag:
        final_search_parameters = {'classifier__' + key: search_params[key] for key in search_params}
    else:
        final_search_parameters = search_params

    search_estimator = None
    if search_method == 'gridcv':
        search_estimator = GridSearchCV(
            imba_pipeline,
            param_grid=final_search_parameters,
            cv=kfold_inner,
            pre_dispatch=cv_splits,
            scoring=scoring_metric,
            n_jobs=n_cores // 2
        )
    elif search_method == 'halvingrandomsearch':
        search_estimator = HalvingRandomSearchCV(
            imba_pipeline,
            param_distributions=final_search_parameters,
            cv=kfold_inner,
            scoring=scoring_metric,
            n_jobs=cv_splits,
            verbose=2
        )
    elif search_method == 'optuna':
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not available. Please install optuna to use optuna search.")
        logging.info("This is the current search_parameters: " + str(final_search_parameters))
        hpo_n_trials = config.get('hpo_n_trials', 5)
        search_estimator = OptunaSearchCV(
            imba_pipeline,
            param_distributions=final_search_parameters,
            cv=kfold_inner,
            scoring=scoring_metric,
            n_jobs=n_cores // 2,
            n_trials=hpo_n_trials,
            verbose=2,
            callbacks=[dummy_gc]
        )
    else:
        raise ValueError(f"Unknown search method: {search_method}")

    # Fit the search estimator to get best parameters
    search_estimator.fit(X, y)

    # Extract best parameters
    if oversample_flag:
        best_params = {key.removeprefix('classifier__'): search_estimator.best_params_[key]
                       for key in search_estimator.best_params_}
    else:
        best_params = search_estimator.best_params_
    cv_estimator = search_estimator.best_estimator_

    return cv_estimator, search_estimator, best_params

# Basic function to handle sklearn and traditional model training and basic hyperparameter optimization
# TODO refactor this into a class maybe, class DrugModel
def skl_drug_model(X, Y, drug, config, n_cores=1):
    """
    Train a drug sensitivity model using configuration parameters.
    
    Args:
        X: Feature matrix
        Y: Target matrix (all drugs)
        drug: Specific drug name to model
        config: Configuration dictionary with model parameters
        n_cores: Number of cores to use for parallel processing
    """
    # get the exact model type
    model_name = config['model_type']
    assert model_name in MODEL_TYPES, f'{model_name} type not supported'
    model_class = MODEL_TYPES[model_name]

    # select drug column
    y = Y[drug]
    logging.info(drug+' has '+str(np.sum(y))+' sensitive samples')

    # Split X and y into training and testing sets for final classification report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['train_test_split_seed']
    )

    # get cross-validation seeds and parameters and set up cv objects
    cv_seed = config.get('cv_seed', 7)
    cv_splits = config.get('cv_splits', 5)
    kfold_outer = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
    kfold_inner = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed*2)

    # get fixed model parameters, hpo search method and whther to do nested_cv
    fixed_params = config.get('fixed_params', {})
    search_method = config.get('search_method', 'gridcv')
    nested_cv= config.get('nested_cv', True)

    # Add n_jobs to fixed_params if the model supports it
    if hasattr(model_class(), 'n_jobs'):
        fixed_params['n_jobs'] = n_cores//2 if search_method == 'optuna' else n_cores - cv_splits
    logging.info("This is the current config:\n"+yaml.dump(config))
    
    # Initialize base model
    model0 = model_class( **fixed_params)

    # Handle oversampling and get the initial pipeline and resampled data
    imba_pipeline, X_res, y_res = handle_oversampling(config, model0, X_train, y_train)
    oversample_flag = config['use_oversampling']
    
    # Perform hyperparameter search if needed
    search_estimator = None
    logging.info("This is the current search method: "+search_method)
    cv_estimator, search_estimator, best_params = perform_hyperparameter_search(
        imba_pipeline, X, y, config, n_cores, kfold_inner, oversample_flag
    )

    # Only perform nested cross validation if doing hyperparameter search to evaluate model
    if nested_cv and search_estimator is not None: 
        logging.info(f"Best param found using all data for {drug}: {best_params} ")
        logging.info('Performing nested cross-validation')
        # If hyperparameter search was performed, the best_estimator_from_search is the final pipeline
        # and we use it for outer cross-validation.
        cv_results = outer_cross_validate(
            search_estimator, X, y, 
            scoring=['balanced_accuracy', 'precision', 'recall', 'f1', 'average_precision', 'roc_auc'], 
            cv=kfold_outer
        )
    else:
        # No hyperparameter search, just cross-validate the base model or best hpo model
        cv_results = cross_validate(
            cv_estimator, X_train, y_train, 
            scoring=['balanced_accuracy', 'precision', 'average_precision', 'recall', 'f1', 'roc_auc'], 
            cv=kfold_outer
        )


    #cv_results = cross_validate(
     #   imba_pipeline, X_train, y_train, cv=kfold, 
      #  scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], 
       # n_jobs=n_cores
     #)
    cv_results = pd.DataFrame(cv_results)

    # Fit the classifier to the training data
    # model0 is the base model, which needs to be fitted on X_res, y_res
    # If best_estimator_from_search is a pipeline, it's already fitted during the search.
    # If no search, model0 is the base model.
    # If search was performed, best_estimator_from_search is the fitted pipeline
    model0 = cv_estimator # Use the best estimator from search as the final model
     # If no search, fit the initial model0 on resampled data
    model0.fit(X_train, y_train)
    y_pred = model0.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    model_report = classification_report(y_test, y_pred, output_dict=True, labels=np.unique(y_pred))
    model_report = pd.DataFrame(model_report).transpose()

    # final results in a dictionary
    final_results = {
        'best_model': model0, # This will be the best estimator from search or the original imba_pipeline
        'drug': drug,
        'model_class': model_name,
        'model_search': search_estimator, 
        'search_method': search_method,
        'X_test': X_test,
        'Y_test': y_test,
        'X_train': X_train,
        'Y_train': y_train,
        'cv_results': cv_results,
        'oversample': oversample_flag,
        'confusion_matrix':conf_mat,
        'report':model_report
    }

    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate drug sensitivity models.")
    parser.add_argument("--data_file", type=str, help="Path to input CSV data file")
    parser.add_argument("--drug_name", type=str, help="Drug name (column) to model")
    parser.add_argument("--config_file", type=str, help="Path to model configuration file (JSON or YAML)")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument("--n_cores", type=int, default=1, help="Number of cores to use for parallel processing (default: 1)")
    #parser.add_argument("--run", type=int, default=-1, help="run name")
    args = parser.parse_args()
    data_file = args.data_file
    drug_name = args.drug_name
    config_file = args.config_file
    out_dir = args.out_dir
    n_cores = args.n_cores
    #Load data
    X, y, drugs = read_data(data_file)
    #load config
    config = load_config(config_file)
    # generate a run name
    run_name = random_name(config, X, y)
    os.makedirs(f'./logs/{run_name}', exist_ok=True)

    logging.basicConfig(
        filename=f'./logs/{run_name}/{drug_name}.log',  # Specify the log file name
        level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log message format
    )
    logger = logging.getLogger(__name__)
    logging.info('finished read in of data')
    logging.info('finished read in of config')

    
    # Train model
    start_t = time.time()
    logging.info('started training')
    results = skl_drug_model(X, y, drug=drug_name, config=config, n_cores=n_cores)

    logging.info(f'Finished training in {time.time()-start_t}s')

    # Plot ROC and AUPR curves
    plot_roc_aupr_curves(
        best_model=results['best_model'],
        X_test=results['X_test'],
        y_test=results['Y_test'],
        drug=results['drug'],
        output_dir=f'{out_dir}/{run_name}' # Use the same output directory as other results
    )

    # Write results
    out_dir=f'{out_dir}/{run_name}'
    write_drug_model_result(results, out_dir=out_dir)
    shutil.copy2(config_file, out_dir)
