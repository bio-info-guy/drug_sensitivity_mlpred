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



def read_data(fpath: str):
    dataset = pd.read_csv(fpath)
    X = dataset.iloc[:, 1:-4686]
    drug_y_all = dataset.iloc[:, -4686:]
    drug_list = drug_y_all.columns.tolist()
    return X, drug_y_all, drug_list



# Define model names
MODEL_TYPES= {
		  'XGBClassifier': XGBClassifier,
		  'RandomForestClassifier': RandomForestClassifier,
		  'SGDClassifier': SGDClassifier,
		  'LGBMClassifier': LGBMClassifier

}


# SOME PARAMETERS FOR GRIDSEARCH
GRID_SEARCH_PARAM = {
'xgboost': {
		"n_estimators": [100, 150],
 },
 'rf':{
	 "n_estimators": [100, 150, 250],
	 "max_depth" : [20, 50, 100, 200],
 },
 'sgd':{
	 'l1_ratio':[0.2, 0.15, 0.1, 0.05],
	 'alpha':[0.02, 0.05]
 }
}

# Basic function to handel sklearn and traditional model training and basic hyperparameter optimization
# TODO refactor this into a class maybe, class DrugModel
def skl_drug_model(X, Y, drug, model_name = 'xgboost', oversample = True, fixed_params={}, search_params = {}, search_method = 'gridcv'):

	assert model_name in MODEL_TYPES, f' {model_name} type not supported'

	model = MODEL_TYPES[model_name]
	
	y = Y[drug]

	 # split X and y into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


	if oversample:
		ros = SMOTE(random_state=72)
		X_res, y_res = ros.fit_resample(X_train, y_train)
		oversample = 'oversample'
	else:
		X_res, y_res = X_train, y_train
		oversample = ''


	 # instantiate the classifier
	 

	# k-fold cross validation using multiple metric evaluation
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
	 #cv_results = cross_validate(xgbc, X_train, y_train, cv=kfold, scoring= ['accuracy', 'precision', 'recall', 'f1'], n_jobs = 10)
	if oversample:
		imba_pipeline = Pipeline([('sampling', SMOTE(random_state=72)), 
							  ('classifier', model(n_jobs=20, **fixed_params))])
		grid_search_parameters = {'classifier__' + key: search_params[key] for key in search_params}
	else:
		imba_pipeline = model(**fixed_params, n_jobs=20)
	 #cross_val_score(imba_pipeline, X_train, y_train, scoring='recall', cv=kf)
	model0 = model(**fixed_params, n_jobs=20)

	 #perform gridsearch if needed
	if search_method == 'gridcv' and search_params:
		grid_imba = GridSearchCV(imba_pipeline, param_grid=grid_search_parameters, cv=kfold, scoring='precision',
						return_train_score=True)
		grid_imba.fit(X_train, y_train) 

		best_params = {key.removeprefix('classifier__'):grid_imba.best_params_[key] for key in grid_imba.best_params_}
		print(best_params)
		model0.set_params(**best_params)
	else:
		grid_imba = None
	 
	if oversample:
		imba_pipeline = Pipeline([('sampling', RandomOverSampler(random_state=72)), 
							  ('classifier', model0)])
	else:
		imba_pipeline = model0
		  
	# generate cross validation results of best model with correct oversampling 
	# OVERSAMPLING must come after validation split for correct validation , thus the use of pipeline
	# cross_validate function will first split into train/validate, then feed training data into pipeline (oversampling + training)
	cv_results = cross_validate(imba_pipeline, X_train, y_train, cv=kfold, scoring= ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], n_jobs = 20)
	cv_results = pd.DataFrame(cv_results)
	

	 # declare parameters
	 

	 # fit the classifier to the training data
	model0.fit(X_res, y_res)

	 # save the trained model
	
		  
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
		'cv_results':cv_results,
		'oversample': True if oversample else False
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
	parser.add_argument("data_file", type=str, help="Path to input CSV data file")
	parser.add_argument("drug_name", type=str, help="Drug name (column) to model")
	parser.add_argument("model_type", type=str, choices=MODEL_TYPES.keys(), help="Model type")
	parser.add_argument("out_dir", type=str, help="Output directory")
	parser.add_argument("oversample", type=lambda x: (str(x).lower() == 'true'), help="Whether to use oversampling (True/False)")

	args = parser.parse_args()

	data_file = args.data_file
	drug_name = args.drug_name
	model_type = args.model_type
	out_dir = args.out_dir
	oversample = args.oversample

	X, y, drugs = read_data(data_file)
	results = skl_drug_model(X, y, drug = drug_name, model = model_type)
	write_drug_model_result(results, out_dir = out_dir)