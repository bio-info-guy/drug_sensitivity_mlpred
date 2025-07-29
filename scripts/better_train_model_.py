import os
import sys
from parallel.device_pin import pick_resources
#pick_resources(
 #   cores_required=22,
  #  mem_required_gb=64,
   # lock_dir="/tmp",
    #wait_interval=5
#)
import time
import yaml
import shutil
import argparse
import pandas as pd
import numpy as np
from numpy import loadtxt
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error, mean_gamma_deviance
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
sys.path.append('../drug_sensitivity_mlpred/')
from utils.config_loader import load_config
from cross_validation_utils_ import outer_cross_validate
from utils.misc import random_name
from customio import read_data2, write_drug_model_result
from preprocess import filter_and_threshold_data, get_y_type, split_train_test_data, apply_pca_pipeline
from plot_utils import plot_roc_aupr_curves, plot_regression_evaluation
from eval_func import calculate_feature_importance
from ml_models_ import build_pipe, get_model_for_device
from hpo_ import perform_hyperparameter_search
import logging 

#import mlflow
#import mlflow.sklearn

#mlflow.create_experiment(
 #   name="Drug_Sensitivity_Model_Training_test",
  #  artifact_location="mlruns/Drug_Sensitivity_Model_Training_test"
#)
#mlflow.autolog()
 
# Basic function to handle sklearn and traditional model training and basic hyperparameter optimization
# TODO refactor this into a class maybe, class DrugModel
def train_drug_model(X, Y, drug, config, cell_lines=None):
    """
    Train a drug sensitivity model using configuration parameters.
    Args:
        X: Feature matrix
        Y: Target matrix (all drugs)
        drug: Specific drug name to model
        config: Configuration dictionary with model parameters
        cell_lines: Series of cell line names
    """
    # Preprocess Y based on the configuration
    y_transform_option = config.get('y_transform_option', 'raw')
    y_transform_quantile = config.get('y_transform_quantile', 0.25)
    y_transform_percentage_threshold = config.get('y_transform_percentage_threshold', 0.5)
    
    X, Y = filter_and_threshold_data(
        X, Y,
        option=y_transform_option,
        quantile=y_transform_quantile,
        threshold=y_transform_percentage_threshold
    )
    
    y = Y[[drug]]
    y_type = get_y_type(y)
    
    logging.info(f"Drug: {drug}, Y type: {y_type}")
    if y_type == 'binary':
        logging.info(f"{drug} has {y.sum().item()} sensitive samples")
    else:
        config['use_oversampling'] = False

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test_data(X, y, y_type, config, cell_lines)
    X_train_orig = X_train.copy()

    # Apply PCA if configured
    pca_model = None
    if config.get('pca', False):
        X_train, X_test, y_train, pca_model = apply_pca_pipeline(X_train, X_test, y_train, config)
        logging.info(f"PCA applied. New X_train shape: {X_train.shape}")


    # Get cross-validation seeds and parameters and set up cv objects
    cv_seed = config.get('cv_seed', 7)
    cv_splits = config.get('cv_splits', 5)
    config['test_set'] = [(X_test, y_test)]
    
    if y_type == 'binary':
        kfold_outer = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
        kfold_inner = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed * 2)
    else:
        kfold_outer = KFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
        kfold_inner = KFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed * 2)

    # get fixed model parameters, hpo search method and whther to do nested_cv
    search_method = config.get('search_method', 'optuna')
    nested_cv= config.get('nested_cv', True)

    
    
    # Initialize base model using the new function
    model0, config = get_model_for_device(config, y_type)
    logging.info("This is the current config:\n" + yaml.dump(config))

    # Handle oversampling and get the initial pipeline and resampled data
    imba_pipeline = build_pipe(config, model0)

    # Perform hyperparameter search if needed
    search_estimator = None
    logging.info("This is the current search method: "+search_method)

    # HPO
    cv_estimator, search_estimator, best_params = perform_hyperparameter_search(
        imba_pipeline, X_train, y_train.values.ravel(), config, kfold_inner, y_type
    )

    # Define scoring metrics based on model type
    if y_type == 'binary':
        scoring_metrics = ['balanced_accuracy', 'precision', 'average_precision', 'recall', 'f1', 'roc_auc']
    else:
        scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_gamma_deviance']

    # Only perform nested cross validation if doing hyperparameter search to evaluate model
    if nested_cv and search_estimator is not None:
        logging.info(f"Best param found using all data for {drug}: {best_params} ")
        logging.info('Performing nested cross-validation')
        cv_results = outer_cross_validate(
            search_estimator, X, y.values.ravel(),
            scoring=scoring_metrics,
            cv=kfold_outer,
            config=config,
            **config.get('fit_params', {})
        )
    else:
        cv_results = cross_validate(
            cv_estimator, X_train, y_train.values.ravel(),
            scoring=scoring_metrics,
            cv=kfold_outer
        )


    cv_results = pd.DataFrame(cv_results)
    model0 = cv_estimator  # Use the best estimator from search as the final model
    y_pred = model0.predict(X_test)

    final_results = {
        'best_model': model0,
        'drug': drug,
        'model_class': str(model0.named_steps['model']).split('(')[0],
        'model_search': search_estimator,
        'search_method': search_method,
        'X_test': X_test,
        'Y_test': y_test,
        'X_train': X_train,
        'Y_train': y_train,
        'cv_results': cv_results,
        'oversample': config.get('use_oversampling', False),
        'feature_importance': calculate_feature_importance(model0, X_train_orig, pca_object=pca_model.named_steps.get('pca') if pca_model else None),
        'pca_model': pca_model,
        'y_type': y_type
    }

    if y_type == 'binary':
        final_results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        final_results['report'] = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, labels=np.unique(y_pred))).transpose()
    else:
        final_results['report'] = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'gamma_deviance': mean_gamma_deviance(y_test, y_pred)
        }

    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate drug sensitivity models.")
    parser.add_argument("--x_file", type=str, required=True, help="Path to input CSV data file for features (X)")
    parser.add_argument("--y_file", type=str, required=True, help="Path to input CSV data file for labels (Y)")
    parser.add_argument("--drug_name", type=str, required=True, help="Drug name (column) to model")
    parser.add_argument("--config_file", type=str, required=True, help="Path to model configuration file (JSON or YAML)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n_cores", type=int, default=1, help="Number of cores to use for parallel processing (default: 1)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (e.g., 'cpu', 'cuda')")
    args = parser.parse_args()
    
    x_file = args.x_file
    y_file = args.y_file
    drug_name = args.drug_name
    config_file = args.config_file
    out_dir = args.out_dir
    n_cores = args.n_cores
    device = args.device
    
    # Load data
    X, y, drugs, cell_lines = read_data2(x_file, y_file)
    
    # Load config
    config = load_config(config_file, n_cores, device)
    # Only set device in config if model type is XGBoost or RandomForest
    # generate a run name
    run_name = config['model_type']+'_'+random_name(config_file, X, y)
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
    results = train_drug_model(X, y, drug=drug_name, config=config, cell_lines=cell_lines)

    logging.info(f'Finished training in {time.time()-start_t}s')

    

    # Write results
    out_dir=f'{out_dir}/{run_name}'
    write_drug_model_result(results, out_dir=out_dir)
    shutil.copy2(config_file, out_dir)

    # Plot evaluation curves based on y_type
    if results['y_type'] == 'binary':
        plot_roc_aupr_curves(
            best_model=results['best_model'],
            X_test=results['X_test'],
            y_test=results['Y_test'],
            drug=results['drug'],
            output_dir=out_dir,
            threshold_type='both'
        )
    else:
        plot_regression_evaluation(
            best_model=results['best_model'],
            X_test=results['X_test'],
            y_test=results['Y_test'],
            target_name=results['drug'],
            output_dir=out_dir
        )
