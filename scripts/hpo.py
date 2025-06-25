from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, GridSearchCV
try:
    from optuna.integration import OptunaSearchCV
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
import logging
from utils.misc import dummy_gc

def perform_hyperparameter_search( imba_pipeline, X, y, config, kfold_inner):
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
    n_cores = max(2, config.get('n_cores', 2))
    if not search_params:
        return imba_pipeline, None, None
    final_search_parameters = {'classifier__' + key: search_params[key] for key in search_params}

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
        search_estimator_best = search_estimator
    elif search_method == 'halvingrandomsearch':
        search_estimator = HalvingRandomSearchCV(
            imba_pipeline,
            param_distributions=final_search_parameters,
            cv=kfold_inner,
            scoring=scoring_metric,
            n_jobs=cv_splits,
            verbose=2
        )
        search_estimator_best = search_estimator
    elif search_method == 'optuna':
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not available. Please install optuna to use optuna search.")
        logging.info("This is the current search_parameters: " + str(final_search_parameters))
        logging.info(f"using {n_cores} for cross validation")
        hpo_n_trials = config.get('hpo_n_trials', 5)
        search_estimator = OptunaSearchCV(
            imba_pipeline,
            param_distributions=final_search_parameters,
            cv=kfold_inner,
            scoring=scoring_metric,
            n_jobs=n_cores // 2,
            n_trials=30, # at least 30 trials for optuna search, hardcoded for now
            verbose=2,
            callbacks=[dummy_gc]
        )
        search_estimator_best = OptunaSearchCV(
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
    search_estimator_best.fit(X, y)

    # Extract best parameters
    best_params = search_estimator_best.best_params_
    cv_estimator = search_estimator_best.best_estimator_

    return cv_estimator, search_estimator, best_params
