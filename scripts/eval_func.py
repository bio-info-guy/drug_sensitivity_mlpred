import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging

def calculate_feature_importance(pipeline_model, X_train):
    """
    Calculates feature importance based on model type and training data.

    Args:
        pipeline_model: A Pipeline object with a 'classifier' key pointing to the model.
        X_train: Training features (pandas DataFrame).

    Returns:
        A pandas DataFrame with feature importances.
    """
    classifier = pipeline_model.named_steps['classifier']
    feature_names = X_train.columns

    # Tree-based models
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return feature_importance_df
    
    # Linear models
    elif hasattr(classifier, 'coef_'):
        coef = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_

        # Check if StandardScaler was used in the pipeline
        scaler_used = False
        if 'scaling' in pipeline_model.named_steps:
            scaler = pipeline_model.named_steps['scaling']
            if isinstance(scaler, StandardScaler):
                scaler_used = True

        if not scaler_used:
            # Calculate standard deviation of each feature from X_train
            std_dev = X_train.std().values
            # Multiply coef_ by standard deviation
            feature_importances = np.abs(coef * std_dev)
        else:
            # If scaler was used, coef_ already reflects scaled importance
            feature_importances = np.abs(coef)
            std_dev = np.full(len(feature_names), np.nan) # No direct std_dev to multiply if scaled

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Original_Coef': coef,
            'Standard_Deviation': std_dev,
            'Final_Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Final_Importance', ascending=False).reset_index(drop=True)
        return feature_importance_df
    
    else:
        logging.warning(f"Model type {type(classifier).__name__} does not have feature_importances_ or coef_ attribute.")
        return pd.DataFrame()