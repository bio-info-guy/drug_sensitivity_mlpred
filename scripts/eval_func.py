import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging

def calculate_feature_importance(pipeline_model, X_train, pca_object=None):
    """
    Calculates feature importance based on model type and training data.

    Args:
        pipeline_model: A Pipeline object with a 'classifier' key pointing to the model.
        X_train: Training features (pandas DataFrame).
        pca_object: Optional PCA object if PCA was used for preprocessing.

    Returns:
        A pandas DataFrame with feature importances.
    """
    classifier = pipeline_model.named_steps['classifier']
    
    df_dict = {}
    
    # Tree-based models
    if hasattr(classifier, 'feature_importances_'):
        if pca_object is not None:
            # Calculate feature importance using PCA components
            df_dict['Feature'] = pca_object.feature_names_in_
            df_dict['Importance'] = ((classifier.feature_importances_* np.sqrt(pca_object.explained_variance_))[:, np.newaxis] * np.abs(pca_object.components_)).sum(axis=0)
            df_dict['pca_importance'] = (classifier.feature_importances_[:, np.newaxis] * np.abs(pca_object.components_)).sum(axis=0)
            df_dict['pca_importance_absum'] = np.abs((classifier.feature_importances_[:, np.newaxis] * pca_object.components_).sum(axis=0))
            df_dict['pca_importance_sd'] = ((classifier.feature_importances_* np.sqrt(pca_object.explained_variance_))[:, np.newaxis] * np.abs(pca_object.components_)).sum(axis=0)
            df_dict['pca_importance_var'] = ((classifier.feature_importances_* pca_object.explained_variance_)[:, np.newaxis] * np.abs(pca_object.components_)).sum(axis=0)
            df_dict['pca_importance_log'] = ((classifier.feature_importances_* np.log1p(pca_object.explained_variance_))[:, np.newaxis] * np.abs(pca_object.components_)).sum(axis=0)
            df_dict['pca_importance_ratio'] = ((classifier.feature_importances_* np.log1p(pca_object.explained_variance_ratio_))[:, np.newaxis] * np.abs(pca_object.components_)).sum(axis=0)

        else:
            df_dict['Feature'] = X_train.columns
            df_dict['Importance'] = classifier.feature_importances_


        feature_importance_df = pd.DataFrame(df_dict)
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return feature_importance_df
    
    # Linear models
    elif hasattr(classifier, 'coef_'):
        coef = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
        coef1 = np.full(len(feature_names), np.nan)
        if pca_object is not None:
            coef = np.abs(coef[:, np.newaxis] * pca_object.components_).sum(axis=0)
            coef1 = (coef[:, np.newaxis] * np.abs(pca_object.components_)).sum(axis=0)
            feature_names = pca_object.feature_names_in_
        else:
            feature_names = X_train.columns
        # Check if StandardScaler was used in the pipeline
        scaler_used = False
        if 'scaling' in pipeline_model.named_steps:
            scaler = pipeline_model.named_steps['scaling']
            if isinstance(scaler, StandardScaler):
                scaler_used = True
        # TODO add in implementation for pca to importance for models with coef
        if not scaler_used:
            
            # Calculate standard deviation of each feature from X_train
            std_dev = X_train.std().values
            # Multiply coef_ by standard deviation
            feature_importances = np.abs(coef * std_dev)
            feature_importances1 = np.abs(coef1 * std_dev)
        else:
            # If scaler was used, coef_ already reflects scaled importance
            feature_importances = np.abs(coef)
            feature_importances1 = np.full(len(feature_names), np.nan)
            std_dev = np.full(len(feature_names), np.nan) # No direct std_dev to multiply if scaled

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coef': coef,
            'Coef1': coef1,
            'Standard_Deviation': std_dev,
            'Importance': feature_importances,
            'Importance2': feature_importances1
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return feature_importance_df
    
    else:
        logging.warning(f"Model type {type(classifier).__name__} does not have feature_importances_ or coef_ attribute.")
        return pd.DataFrame()
