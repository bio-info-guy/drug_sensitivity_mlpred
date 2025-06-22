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

def write_drug_model_result(model_results, out_dir):
    model0 = model_results['best_model']
    oversample = model_results['oversample']
    drug = model_results['drug']
    model_name = model_results['model_class']
    full_out_dir = f'{out_dir}/{drug}/'
    X_test, y_test = model_results['X_test'], model_results['Y_test']
    X_train, y_train = model_results['X_train'], model_results['Y_train']
    y_pred = model0.predict(X_test)


    os.makedirs(full_out_dir, exist_ok=True)
    print(full_out_dir)
    joblib.dump(model0, f'{full_out_dir}/{model_name}_{drug}.joblib')
    model_results['cv_results'].to_csv(f'{full_out_dir}/{model_name}_cv_results_{drug}.csv')
    print(drug,
		 f'{model_name}_model_parameters', model0, "\n",
		 "confusion_matrix:", "\n", model_results['confusion_matrix'], "\n",
		 file=open(f'{full_out_dir}/{model_name}_confusion_matrix.txt', "a"))
    model_report = model_results['report']
    if (model_report.index == "1").any() == True:
        r1 = pd.DataFrame(model_report.loc["1"]).transpose()
        r1.to_csv(f'{full_out_dir}/{model_name}_classification_report_{drug}.csv')
    else:
        print(drug, "Nothing predicted as 1",
              file=open(f'{out_dir}/{model_name}_classification_report_log.txt', "a"))
    fi = model_results['feature_importance']
    fi.to_csv(f'{full_out_dir}/{model_name}_feature_importance_{drug}.csv')