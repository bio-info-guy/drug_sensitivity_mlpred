# Drug Sensitivity Prediction using Machine Learning

This project focuses on building machine learning models to predict drug sensitivity based on CRISPR-screening data. The goal is to leverage computational approaches to understand and forecast how different cell lines respond to various drugs.

## Project Overview

The core of this project involves:
- Utilizing CRISPR-screening data, specifically from datasets like DepMap, to serve as features for machine learning models.
- Developing and training predictive models that can accurately forecast drug sensitivity.
- Employing Automated Machine Learning (AutoML) tools, such as FLAML, to streamline the model selection and hyperparameter tuning process, ensuring efficient and robust model development.

## Data

The primary data source for this project is the DepMap (Cancer Dependency Map) dataset, which provides extensive CRISPR-screening data and drug sensitivity profiles.

## Models

Machine learning models are developed using the FLAML library, which supports various algorithms including LightGBM, XGBoost, SGD, LRL2, and SVC, as configured in the `notebooks/flaml_sensitivity.ipynb` notebook.

## Getting Started

Further details on data preprocessing, model training, and evaluation can be found in the `notebooks/` directory.
