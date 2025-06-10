#!/usr/bin/env python3

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import loadtxt
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv("input/combined_DepMap_21Q3.csv")
num_gene = 17651

#dataset = pd.read_csv("input/combined_DepMap_21Q3_CCLE_expression.csv")
#num_gene = 19177

X = dataset.iloc[:, 1:num_gene+1]

drug_y_all = dataset.iloc[:, -4686:]
drug_list = drug_y_all.columns.tolist()

def rfc(drug):
    y = drug_y_all[drug]

    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    # instantiate the classifier
    rfc = RandomForestClassifier(n_estimators=40, random_state=7, n_jobs=10)

    # fit the classifier to the training data
    rfc.fit(X_train, y_train)

    # save the trained model
    joblib.dump(rfc, "output/RandomForest_%s.joblib" % drug)

    # make predictions on test data
    y_pred = rfc.predict(X_test)

    print(drug,
         "RandomForest_model_parameters", rfc, "\n",
         "confusion_matrix:", "\n", confusion_matrix(y_test, y_pred), "\n",
         file=open("output/RandomForest_confusion_matrix.txt", "a"))

    model_report = classification_report(y_test, y_pred, output_dict=True, labels=np.unique(y_pred))
    model_report = pd.DataFrame(model_report).transpose()

    if (model_report.index == "1").any() == True:
        r1 = pd.DataFrame(model_report.loc["1"]).transpose()
        r1.to_csv("output/RandomForest_classification_report_%s.csv" % drug)
    else:
        print(drug, "Nothing predicted as 1",
             file=open("output/RandomForest_classification_report_log.txt", "a"))

    # k-fold cross validation using multiple metric evaluation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    cv_results = cross_validate(rfc, X, y, cv=kfold, scoring= ['accuracy', 'precision', 'recall', 'f1'], n_jobs=10)
    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv("output/RandomForest_cv_results_%s.csv" % drug)

    # feature importance with RandomForest
    fi = pd.DataFrame({'feature': list(X_train.columns),
               'importances': rfc.feature_importances_ * 100}).\
                sort_values('importances', ascending = False)
    fi.to_csv("output/RandomForest_feature_importance_%s.csv" % drug)

starttime = time.time()
rfc(sys.argv[1])
endtime = time.time()
file_object = open('time_log.txt', 'a')
file_object.write(sys.argv[1] + '\t' + str(endtime-starttime)+'s\n')
file_object.close()
