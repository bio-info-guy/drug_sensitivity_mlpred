import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
import argparse
import json
import yaml



def read_data(fpath: str):
    dataset = pd.read_csv(fpath)
    X = dataset.iloc[:, 1:-4686]
    drug_y_all = dataset.iloc[:, -4686:]
    drug_list = drug_y_all.columns.tolist()
    return X, drug_y_all, drug_list



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
    
    return config

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
    model_name = config['model_type']
    assert model_name in MODEL_TYPES, f'{model_name} type not supported'

    model_class = MODEL_TYPES[model_name]
    y = Y[drug]

    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config['train_test_split_seed']
    )

    # Set up oversampling if specified
    oversample_flag = config['use_oversampling']
    if oversample_flag:
        oversampler_class = OVERSAMPLER_TYPES[config['oversampler_type']]
        oversampler = oversampler_class(random_state=config['oversampler_seed'])
        X_res, y_res = oversampler.fit_resample(X_train, y_train)
        oversample_str = 'oversample'
    else:
        X_res, y_res = X_train, y_train
        oversample_str = ''

    # Set up cross-validation
    cv_seed = config.get('cv_seed', 7)
    cv_splits = config.get('cv_splits', 5)
    kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cv_seed)

    # Get fixed parameters and search parameters
    fixed_params = config.get('fixed_params', {})
    search_params = config.get('grid_search_params', {})
    search_method = config.get('search_method', 'gridcv')

    # Add n_jobs to fixed_params if the model supports it
    if hasattr(model_class(), 'n_jobs'):
        fixed_params['n_jobs'] = n_cores

    # Set up pipeline for grid search
    if oversample_flag:
        oversampler_for_cv = oversampler_class(random_state=config['oversampler_seed'])
        imba_pipeline = Pipeline([
            ('sampling', oversampler_for_cv), 
            ('classifier', model_class(**fixed_params))
        ])
        grid_search_parameters = {'classifier__' + key: search_params[key] for key in search_params}
    else:
        imba_pipeline = model_class(**fixed_params)
        grid_search_parameters = search_params

    # Initialize base model
    model0 = model_class(**fixed_params)

    # Perform grid search if needed
    grid_imba = None
    if search_method == 'gridcv' and search_params:
        scoring_metric = config.get('scoring_metric', 'precision')
        grid_imba = GridSearchCV(
            imba_pipeline, 
            param_grid=grid_search_parameters, 
            cv=kfold, 
            scoring=scoring_metric,
            return_train_score=True,
            n_jobs=n_cores
        )
        grid_imba.fit(X_train, y_train)

        # Extract best parameters
        if oversample_flag:
            best_params = {key.removeprefix('classifier__'): grid_imba.best_params_[key] 
                          for key in grid_imba.best_params_}
        else:
            best_params = grid_imba.best_params_

        print(f"Best parameters for {drug}: {best_params}")
        model0.set_params(**best_params)

    # Set up final pipeline for cross-validation
    if oversample_flag:
        final_oversampler = oversampler_class(random_state=config['oversampler_seed'])
        imba_pipeline = Pipeline([
            ('sampling', final_oversampler), 
            ('classifier', model0)
        ])
    else:
        imba_pipeline = model0

    # Generate cross validation results of best model with correct oversampling 
    # OVERSAMPLING must come after validation split for correct validation, thus the use of pipeline
    # cross_validate function will first split into train/validate, then feed training data into pipeline (oversampling + training)
    cv_results = cross_validate(
        imba_pipeline, X_train, y_train, cv=kfold, 
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], 
        n_jobs=n_cores
    )
    cv_results = pd.DataFrame(cv_results)

    # Fit the classifier to the training data
    model0.fit(X_res, y_res)

    final_results = {
        'best_model': model0,
        'drug': drug,
        'model_class': model_name,
        'model_search': grid_imba,
        'search_method': search_method,
        'X_test': X_test,
        'Y_test': y_test,
        'X_train': X_train,
        'Y_train': y_train,
        'cv_results': cv_results,
        'oversample': oversample_flag
    }

    return final_results


def write_drug_model_result(model_results, out_dir):
	 
	model0 = model_results['best_model']
	oversample = 'oversample' if model_results['oversample'] else ''
	drug = model_results['drug']
	model_name = model_results['model_class']

	

	full_out_dir = f'{out_dir}/{oversample}/{drug}/'
	X_test, y_test = model_results['X_test'], model_results['Y_test']
	X_train, y_train = model_results['X_train'], model_results['Y_train']
	y_pred = model0.predict(X_test)

	os.makedirs(full_out_dir, exist_ok=True)

	joblib.dump(model0, f'{full_out_dir}/{model_name}_{drug}.joblib')
	
	model_results['cv_results'].to_csv(f'{full_out_dir}/{model_name}_cv_results_{drug}.csv')

	print(drug,
		 f'{model_name}_model_parameters', model0, "\n",
		 "confusion_matrix:", "\n", confusion_matrix(y_test, y_pred), "\n",
		 file=open(f'{full_out_dir}/{model_name}_confusion_matrix.txt', "a"))

	model_report = classification_report(y_test, y_pred, output_dict=True, labels=np.unique(y_pred))
	model_report = pd.DataFrame(model_report).transpose()
	
	if (model_report.index == "1").any() == True:
		r1 = pd.DataFrame(model_report.loc["1"]).transpose()
		r1.to_csv(f'{full_out_dir}/{model_name}_classification_report_{drug}.csv')
	else:
		print(drug, "Nothing predicted as 1",
			   file=open(f'{out_dir}/{oversample}/{model_name}_classification_report_log.txt', "a"))

	 

	 # feature importance with XGBoost
	if model_name in ['XGBClassifier','RandomForestClassifier']:
		fi = pd.DataFrame({'feature': list(X_train.columns),
					'importances': model0.feature_importances_ * 100}).\
					 sort_values('importances', ascending = False)
		fi.to_csv(f'{full_out_dir}/{model_name}_feature_importance_{drug}.csv')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train and evaluate drug sensitivity models.")
    parser.add_argument("--data_file", type=str, help="Path to input CSV data file")
    parser.add_argument("--drug_name", type=str, help="Drug name (column) to model")
    parser.add_argument("--config_file", type=str, help="Path to model configuration file (JSON or YAML)")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument("--n_cores", type=int, default=1, help="Number of cores to use for parallel processing (default: 1)")

    args = parser.parse_args()

    data_file = args.data_file
    drug_name = args.drug_name
    config_file = args.config_file
    out_dir = args.out_dir
    n_cores = args.n_cores

    # Load configuration
    config = load_config(config_file)
    
    # Load data
    X, y, drugs = read_data(data_file)
    
    # Train model
    results = skl_drug_model(X, y, drug=drug_name, config=config, n_cores=n_cores)
    
    # Write results
    write_drug_model_result(results, out_dir=out_dir)
