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
