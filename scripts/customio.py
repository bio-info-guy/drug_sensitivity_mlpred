import sys
import os
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib as mtl
from matplotlib import pyplot as plt
import joblib

def read_data(fpath: str):
    dataset = pd.read_csv(fpath)
    
    # Identify drug columns based on the "BRD-" prefix
    all_columns = dataset.columns.tolist()
    drug_columns = [col for col in all_columns if col.startswith('BRD-')]
    
    # X consists of all columns except the first one (assumed to be an ID)
    # and the identified drug columns.
    X = dataset.drop(columns=drug_columns)
    if X.shape[1] > 0: # Ensure there are columns left after dropping drug columns
        X = X.iloc[:, 1:] # Drop the first column (ID column)
    else:
        # This case means all columns were either drug columns or the first ID column.
        # If there are no features left, X should be an empty DataFrame.
        X = pd.DataFrame() 

    drug_y_all = dataset[drug_columns]
    drug_list = drug_y_all.columns.tolist()
    
    return X, drug_y_all, drug_list


def read_data2(xpath: str, ypath:str):

    # unlike the previous function, X and y are different files now
    X = pd.read_csv(xpath)
    y = pd.read_csv(ypath)
    assert X.shape[0] == y.shape[0], 'X and y have different number of samples'
    
    cell_lines = None
    if 'cell_line_name' in X.columns:
        cell_lines = X['cell_line_name']
        X = X.drop(columns=['cell_line_name'])

    if 'cell_line_name' in y.columns:
        y = y.drop(columns=['cell_line_name'])
    
    drug_y_all = y
    drug_list = drug_y_all.columns.tolist()
    
    return X, drug_y_all, drug_list, cell_lines

def write_drug_model_result(model_results, out_dir):
    model0 = model_results['best_model']
    drug = model_results['drug']
    model_name = model_results['model_class']
    full_out_dir = f'{out_dir}/{drug}/'
    X_test, y_test = model_results['X_test'], model_results['Y_test']
    y_pred = model0.predict(X_test)

    os.makedirs(full_out_dir, exist_ok=True)
    joblib.dump(model0, f'{full_out_dir}/{model_name}_{drug}.joblib')
    model_results['cv_results'].to_csv(f'{full_out_dir}/{model_name}_cv_results_{drug}.csv')

    if 'confusion_matrix' in model_results:
        print(drug,
              f'{model_name}_model_parameters', model0, "\n",
              "confusion_matrix:", "\n", model_results['confusion_matrix'], "\n",
              file=open(f'{full_out_dir}/{model_name}_confusion_matrix.txt', "a"))
    
    if 'report' in model_results:
        model_report = model_results['report']
        if isinstance(model_report, pd.DataFrame):
            model_report.to_csv(f'{full_out_dir}/{model_name}_classification_report_{drug}.csv')
        else:
            with open(f'{full_out_dir}/{model_name}_regression_report_{drug}.txt', 'w') as f:
                f.write(str(model_report))

    fi = model_results['feature_importance']
    fi.to_csv(f'{full_out_dir}/{model_name}_feature_importance_{drug}.csv')

    if 'pca_model' in model_results and model_results['pca_model'] is not None:
        joblib.dump(model_results['pca_model'], f'{full_out_dir}/{model_name}_pca_model_{drug}.joblib')
