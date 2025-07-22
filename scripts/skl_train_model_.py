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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
sys.path.append('../drug_sensitivity_mlpred/')
from utils.config_loader import load_config# Import from new module
from cross_validation_utils_ import outer_cross_validate
from utils.misc import random_name
from customio import read_data, write_drug_model_result
from plot_utils import plot_roc_aupr_curves
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
def skl_drug_model(X, Y, drug, config):
    """
    Train a drug sensitivity model using configuration parameters.
    
    Args:
        X: Feature matrix
        Y: Target matrix (all drugs)
        drug: Specific drug name to model
        config: Configuration dictionary with model parameters
    """
    # select drug column
    y = Y[drug]
    logging.info(drug+' has '+str(np.sum(y))+' sensitive samples')

    # Split X and y into training and testing sets for final classification report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['train_test_split_seed']
    )

    # Apply PCA if configured
    pca_model = None
    if config.get('pca', False):
        n_components = min(X_train.shape[0], X_train.shape[1])
        pca_model = PCA(n_components=n_components)
        X_train = pca_model.fit_transform(X_train)
        X_test = pca_model.transform(X_test)
        logging.info(f"PCA applied with {n_components} components. New X_train shape: {X_train.shape}")
        # Convert X_train and X_test back to DataFrame to maintain column names for feature importance
        # This is a simplification, as PCA transforms to a new feature space.
        # The feature importance calculation will need to handle this.
        X_train = pd.DataFrame(X_train, columns=[f'PC_{i}' for i in range(X_train.shape[1])])
        X_test = pd.DataFrame(X_test, columns=[f'PC_{i}' for i in range(X_test.shape[1])])


    # get cross-validation seeds and parameters and set up cv objects
    cv_seed = config.get('cv_seed', 7)
    cv_splits = config.get('cv_splits', 5)
    config['test_set'] = [(X_test, y_test)]
    kfold_outer = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)
    kfold_inner = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed*2)

    # get fixed model parameters, hpo search method and whther to do nested_cv
    search_method = config.get('search_method', 'optuna')
    nested_cv= config.get('nested_cv', True)

    
    
    # Initialize base model using the new function
    model0, config = get_model_for_device(config)
    logging.info("This is the current config:\n"+yaml.dump(config))

    # Handle oversampling and get the initial pipeline and resampled data
    imba_pipeline = build_pipe(config, model0)

    # Perform hyperparameter search if needed
    search_estimator = None
    logging.info("This is the current search method: "+search_method)

    # HPO
    cv_estimator, search_estimator, best_params = perform_hyperparameter_search(
        imba_pipeline, X_train, y_train, config, kfold_inner
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
            cv=kfold_outer,
            config=config, # Pass the config dictionary
            **config.get('fit_params',{})
        )
    else:
        # No hyperparameter search, just cross-validate the base model or best hpo model
        cv_results = cross_validate(
            cv_estimator, X_train, y_train, 
            scoring=['balanced_accuracy', 'precision', 'average_precision', 'recall', 'f1', 'roc_auc'], 
            cv=kfold_outer
        )


    cv_results = pd.DataFrame(cv_results)

    model0 = cv_estimator # Use the best estimator from search as the final model
     # fit this hpo model on training data 
    #model0.fit(X_train, y_train, **config.get('fit_params', {}))
    y_pred = model0.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    model_report = classification_report(y_test, y_pred, output_dict=True, labels=np.unique(y_pred))
    model_report = pd.DataFrame(model_report).transpose()
    feature_importance = calculate_feature_importance(model0, X_train, pca_object=pca_model) # Pass pca_model
    # final results in a dictionary
    final_results = {
        'best_model': model0, # This will be the best estimator from search or the original imba_pipeline
        'drug': drug,
        'model_class': str(model0['classifier']).split('(')[0],
        'model_search': search_estimator, 
        'search_method': search_method,
        'X_test': X_test,
        'Y_test': y_test,
        'X_train': X_train,
        'Y_train': y_train,
        'cv_results': cv_results,
        'oversample': config['use_oversampling'],
        'confusion_matrix':conf_mat,
        'report':model_report,
        'feature_importance':feature_importance,
        'pca_model': pca_model
    }

    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate drug sensitivity models.")
    parser.add_argument("--data_file", type=str, help="Path to input CSV data file")
    parser.add_argument("--drug_name", type=str, help="Drug name (column) to model")
    parser.add_argument("--config_file", type=str, help="Path to model configuration file (JSON or YAML)")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument("--n_cores", type=int, default=1, help="Number of cores to use for parallel processing (default: 1)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (e.g., 'cpu', 'cuda')")
    #parser.add_argument("--run", type=int, default=-1, help="run name")
    args = parser.parse_args()
    data_file = args.data_file
    drug_name = args.drug_name
    config_file = args.config_file
    out_dir = args.out_dir
    n_cores = args.n_cores
    device = args.device
    #Load data
    X, y, drugs = read_data(data_file)
    #load config
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
    results = skl_drug_model(X, y, drug=drug_name, config=config)

    logging.info(f'Finished training in {time.time()-start_t}s')

    

    # Write results
    out_dir=f'{out_dir}/{run_name}'
    write_drug_model_result(results, out_dir=out_dir)
    shutil.copy2(config_file, out_dir)

    # Plot ROC and AUPR curves
    plot_roc_aupr_curves(
        best_model=results['best_model'],
        X_test=results['X_test'],
        y_test=results['Y_test'],
        drug=results['drug'],
        output_dir=out_dir,
        threshold_type='both'# Use the same output directory as other results
    )
