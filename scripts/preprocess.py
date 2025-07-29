import pandas as pd
import numpy as np

def filter_and_threshold_data(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    option: str,
    threshold: float = None,
    quantile: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters and thresholds feature (X) and label (Y) DataFrames.

    This function first removes rows from both X and Y where Y has any NA values.
    It then applies a specific thresholding strategy to Y based on the 'option' parameter.

    Args:
        X (pd.DataFrame): DataFrame of features with shape ($n \times p$).
        Y (pd.DataFrame): DataFrame of labels with shape ($n \times m$).
        option (str): The thresholding strategy. Must be one of:
                      'none', 'percentage', 'quantile'.
        threshold (float, optional): The value for the 'percentage' option. Defaults to None.
        quantile (float, optional): The quantile for the 'quantile' option. Defaults to 0.3.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple of the processed X and Y DataFrames.
    """
    if not X.index.equals(Y.index):
        raise ValueError("❌ X and Y must have the same index for sample correspondence.")

    # 1. Always filter out rows where Y has any NA values
    valid_rows_mask = Y.notna().all(axis=1)
    X_filtered = X.loc[valid_rows_mask].copy()
    Y_filtered = Y.loc[valid_rows_mask].copy()

    # 2. Apply the selected thresholding option
    if option == 'raw':
        # Option 'a': Return the data with only NA filtering
        return X_filtered, Y_filtered

    elif option == 'percentage':
        # Option 'b': Apply threshold if resulting positive rate is between 5% and 95%
        if threshold is None:
            raise ValueError("A 'threshold' value must be provided for the 'percentage' option.")

        Y_binary = Y_filtered < threshold
        positive_rate = Y_binary.sum().sum() / Y_binary.size

        if 0.05 <= positive_rate <= 0.95:
            return X_filtered, Y_binary  # Use the new binarized Y
        else:
            return X_filtered, Y_filtered  # Use the original Y values

    elif option == 'quantile':
        # Option 'c': Binarize Y based on negative values in the lowest quantile
        quantile_thresholds = Y_filtered.quantile(q=quantile, axis=0)
        # The condition is broadcasted column-wise
        Y_binary = (Y_filtered < 0) & (Y_filtered < quantile_thresholds)
        return X_filtered, Y_binary

    else:
        raise ValueError(f"Invalid option '{option}'. Choose from 'raw', 'percentage', or 'quantile'.")



def get_y_type(Y: pd.DataFrame) -> dict[str, str]:
    """
    Determines if each column in a DataFrame is binary, continuous, or constant.

    Args:
        Y (pd.DataFrame): The DataFrame of labels to analyze.

    Returns:
        dict[str, str]: A dictionary mapping column names to their determined data type.
    """
    if not isinstance(Y, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    column_types = {}
    for col_name in Y.columns:
        # nunique() counts distinct non-null values
        unique_values = Y[col_name].nunique()
        if unique_values == 2:
            return 'binary'
        elif unique_values > 2:
            return 'continuous'
        else:
            return 'constant'

def split_train_test_data(X, y, y_type, config, cell_lines=None):
    """
    Splits data into training and testing sets, handling predefined test cell lines.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.DataFrame): Label data.
        y_type (str): The type of the y variable ('binary' or 'continuous').
        config (dict): Configuration dictionary.
        cell_lines (pd.Series, optional): Series of cell line names. Defaults to None.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    from sklearn.model_selection import train_test_split
    
    # Handle predefined test set from cell lines
    test_cell_lines = config.get('test_cell_lines', [])
    X_test_predefined, y_test_predefined = None, None
    if test_cell_lines and cell_lines is not None:
        test_indices = cell_lines[cell_lines.isin(test_cell_lines)].index
        X_test_predefined = X.loc[test_indices]
        y_test_predefined = y.loc[test_indices]
        X = X.drop(test_indices)
        y = y.drop(test_indices)
        #logging.info(f"Removed {len(test_cell_lines)} cell lines for testing.")

    # Split X and y into training and testing sets
    split_ratio = config.get('train_test_split_ratio', 0.2)
    if split_ratio > 0 and not X.empty:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, random_state=config.get('train_test_split_seed', 42),
            stratify=y if y_type == 'binary' else None
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = pd.DataFrame(columns=X.columns), pd.DataFrame(columns=y.columns)

    if X_test_predefined is not None:
        X_test = pd.concat([X_test, X_test_predefined])
        y_test = pd.concat([y_test, y_test_predefined])
        
    return X_train, X_test, y_train, y_test

def apply_pca_pipeline(X_train, X_test, y_train, config):
    """
    Applies a PCA pipeline to the data, including oversampling and scaling if configured.

    Args:
        X_train (pd.DataFrame): Training feature data.
        X_test (pd.DataFrame): Testing feature data.
        y_train (pd.DataFrame): Training label data.
        config (dict): Configuration dictionary.

    Returns:
        tuple: A tuple containing the transformed X_train, X_test, the PCA model, and the updated config.
    """
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD
    from imblearn.over_sampling import RandomOverSampler, SMOTE

    local_config = config.copy()
    
    OVERSAMPLER_TYPES = {
        'RandomOverSampler': RandomOverSampler,
        'SMOTE': SMOTE
    }

    # Initialize variables
    pipeline_steps = []
    y_train_resampled = y_train  # Default to original y_train
    X_train_to_process = X_train # Default to original X_train

    # 1. Oversampling Step (handled outside the main pipeline)
    if local_config.get('use_oversampling', False):
        oversampler_type = local_config.get('oversampler_type')
        if not oversampler_type or oversampler_type not in OVERSAMPLER_TYPES:
            raise ValueError(f"Invalid 'oversampler_type': {oversampler_type}. "
                             f"Valid options are: {list(OVERSAMPLER_TYPES.keys())}")
        
        oversampler_class = OVERSAMPLER_TYPES[oversampler_type]
        oversampler_seed = local_config.get('oversampler_seed', 42)
        oversampler = oversampler_class(random_state=oversampler_seed)
        
        print("Applying oversampling...")
        # Resample both X and y train data
        X_train_to_process, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # 2. Scaling Step
    if local_config.get('scale_data', True):
        pipeline_steps.append(('scaling', StandardScaler()))

    # 3. PCA Step
    n_components = local_config.get('pca_n_components', 0.99999)
    
    if isinstance(n_components, int):
        max_components = min(X_train_to_process.shape)
        if n_components >= max_components:
            print(f"Warning: n_components ({n_components}) is >= min(n_samples, n_features) ({max_components}). "
                  f"Adjusting to {max_components - 1}.")
            n_components = max_components - 1

    pipeline_steps.append(('pca', PCA(n_components=n_components)))
    
    # Use a standard sklearn Pipeline now
    pca_pipeline = Pipeline(pipeline_steps)
    
    # Fit the pipeline on the (potentially resampled) training data.
    X_train_transformed_np = pca_pipeline.fit_transform(X_train_to_process)
    
    # Transform the original test data using the fitted pipeline.
    X_test_transformed_np = pca_pipeline.transform(X_test)
    
    n_final_components = pca_pipeline.named_steps['pca'].n_components_
    pca_columns = [f'PC_{i+1}' for i in range(n_final_components)]
    
    # Use the index from the processed (resampled) X_train for the new DataFrame
    X_train_transformed = pd.DataFrame(X_train_transformed_np, columns=pca_columns, index=X_train_to_process.index)
    X_test_transformed = pd.DataFrame(X_test_transformed_np, columns=pca_columns, index=X_test.index)

    return X_train_transformed, X_test_transformed, y_train_resampled, pca_pipeline
