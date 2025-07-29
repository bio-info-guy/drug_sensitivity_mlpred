"""
Cross-validation utilities for drug sensitivity modeling.

This module provides enhanced cross-validation functions that extend sklearn's
cross_validate functionality with support for parameter searchers and detailed timing.
"""

import numpy as np
import time
from sklearn.metrics import get_scorer
from sklearn.model_selection import check_cv, StratifiedKFold, GridSearchCV, KFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.base import clone
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from scripts.preprocess import get_y_type, apply_pca_pipeline
import pandas as pd

try:
    from optuna.integration import OptunaSearchCV
except ImportError:
    OptunaSearchCV = None
    OptunaSearchCV = None


def outer_cross_validate(estimator, X, y=None, cv=None, scoring=None, random_state=None, config=None, **fit_params):
    """
    Perform cross-validation similar to sklearn's cross_validate but also record best parameters if estimator is an HPO 
    
    Parameters:
    -----------
    estimator : estimator object
        The object to use to fit the data. Can be a parameter searcher like GridSearchCV.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,), default=None
        The target variable to try to predict.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on the test set.
    random_state : int, default=None
        Random state for cross-validation splitting.
    config : dict, default=None
        Configuration dictionary, used to check for PCA option.
        
    Returns:
    --------
    dict : Dictionary with keys as column headers and values as columns, that can be
           imported into a pandas DataFrame. Includes 'test_score' for each scoring metric,
           'fit_time', 'score_time', and 'best_params' if estimator is a parameter searcher.
    """
    
    # Set default cv if None
    if get_y_type(y.to_frame() if isinstance(y, pd.Series) else y) == 'binary':
        cv_fun = StratifiedKFold
        classify = True
    else:
        cv_fun = KFold
        classify = False

    if cv is None:
        cv = cv_fun(n_splits=5, shuffle=True, random_state=random_state)
    elif isinstance(cv, int):
        cv = cv_fun(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Convert cv to cross-validation generator
    cv = check_cv(cv, y, classifier=classify)
    
    # Handle scoring parameter
    if scoring is None:
        scoring = 'accuracy'
    
    # Convert scoring to list if it's a single string
    if isinstance(scoring, str):
        scoring = [scoring]
    elif isinstance(scoring, (list, tuple)):
        scoring = list(scoring)
    else:
        raise ValueError("scoring must be a string, list, or tuple")
    
    # Initialize results dictionary
    results = {}
    for score_name in scoring:
        results[f'test_{score_name}'] = []
    
    # Add timing results
    results['fit_time'] = []
    results['score_time'] = []
    
    # Check if estimator is a parameter searcher
    searcher_classes = [GridSearchCV, HalvingRandomSearchCV]
    if OptunaSearchCV is not None:
        searcher_classes.append(OptunaSearchCV)
    
    is_searcher = isinstance(estimator, tuple(searcher_classes))
    if is_searcher:
        results['best_params'] = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Clone the estimator for this fold
        estimator_fold = clone(estimator)
        
        # Split data
        if hasattr(X, 'iloc'):  # pandas DataFrame
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        else:  # numpy array
            X_train, X_test = X[train_idx], X[test_idx]
            
        if y is not None:
            if hasattr(y, 'iloc'):  # pandas Series
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:  # numpy array
                y_train, y_test = y[train_idx], y[test_idx]
        else:
            y_train, y_test = None, None

        # Apply PCA if configured for outer cross-validation
        if config and config.get('pca', False):
            X_train, X_test, y_train, pca_model = apply_pca_pipeline(X_train, X_test, y_train, config)

        # Fit the estimator and measure training time
        fit_params['model__eval_set'] = [(X_test, y_test)]
        start_fit_time = time.time()

        estimator_fold.fit(X_train, y_train, **fit_params)
        fit_time = time.time() - start_fit_time
        results['fit_time'].append(fit_time)
        
        # Get the best estimator if this is a parameter searcher
        if is_searcher:
            best_estimator = estimator_fold.best_estimator_
            results['best_params'].append(estimator_fold.best_params_)
        else:
            best_estimator = estimator_fold
        
        # Score on test set and measure scoring time
        start_score_time = time.time()
        for score_name in scoring:
            scorer = get_scorer(score_name)
            score = scorer(best_estimator, X_test, y_test)
            results[f'test_{score_name}'].append(score)
        score_time = time.time() - start_score_time
        results['score_time'].append(score_time)
    
    # Convert lists to numpy arrays
    for key in results:
        if key != 'best_params':  # best_params should remain as list of dicts
            results[key] = np.array(results[key])
    
    return results
