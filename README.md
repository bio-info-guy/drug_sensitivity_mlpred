# Drug Sensitivity Prediction using Machine Learning

This project focuses on building robust machine learning models to predict drug sensitivity based on high-throughput CRISPR-screening data. The primary goal is to leverage advanced computational approaches to accurately understand and forecast how different cell lines respond to various therapeutic compounds, aiding in drug discovery and personalized medicine.

## Project Overview

The core of this project encompasses a comprehensive machine learning pipeline, from data preparation to model evaluation and reporting. Key aspects include:

-   **Data Integration and Feature Engineering**: Utilizing extensive CRISPR-screening data, primarily from datasets like DepMap, as features for predictive modeling.
-   **Flexible Model Development**: Implementing and training a variety of machine learning classifiers, with support for both CPU and GPU acceleration for enhanced performance.
-   **Advanced Hyperparameter Optimization (HPO)**: Employing sophisticated HPO techniques (e.g., Optuna, Grid Search, Halving Random Search) to fine-tune model parameters, ensuring optimal predictive performance.
-   **Rigorous Model Evaluation**: Conducting nested cross-validation and comprehensive test set evaluations using a suite of metrics (e.g., Average Precision, ROC AUC, F1-score, Recall, Precision).
-   **Automated Reporting and Visualization**: Generating detailed reports, feature importance analyses, and various plots (ROC, PR curves, Confusion Matrices) to provide deep insights into model performance and interpretability.
-   **Model Comparison**: Tools for comparing the performance of different model types across multiple drugs, facilitating informed model selection.

## Data

The primary data source for this project is the DepMap (Cancer Dependency Map) dataset, which provides extensive CRISPR-screening data and drug sensitivity profiles. The data processing pipeline is designed to handle tabular data where features are derived from cell line characteristics and drug sensitivity is represented by specific drug columns (identified by the "BRD-" prefix).

## Models

The project supports a range of scikit-learn compatible machine learning models, including:

-   **XGBoost (XGBClassifier)**: Gradient Boosting framework with optional GPU acceleration.
-   **Random Forest (RandomForestClassifier)**: Ensemble learning method with optional GPU acceleration (via `cuml` for CUDA).
-   **LightGBM (LGBMClassifier)**: High-performance gradient boosting framework.
-   **Stochastic Gradient Descent (SGDClassifier)**: Linear classifier with SGD training.

While Automated Machine Learning (AutoML) tools like FLAML can be integrated (as seen in `notebooks/flaml_sensitivity.ipynb`), the core scripts provide direct control over model selection, preprocessing, and hyperparameter tuning.

## Key Features

-   **GPU Acceleration**: Leverage CUDA-enabled GPUs for faster training of compatible models (XGBoost, RandomForest).
-   **Configurable Pipelines**: Easily configure data scaling (StandardScaler) and oversampling techniques (RandomOverSampler, SMOTE) within the model training pipeline.
-   **Multiple HPO Strategies**: Choose from Optuna, Grid Search, or Halving Random Search for efficient hyperparameter optimization.
-   **Comprehensive Metrics**: Evaluate models using a wide array of classification metrics, including Average Precision, ROC AUC, F1-score, Precision, and Recall.
-   **Detailed Feature Importance**: Analyze the contribution of individual features to model predictions for both tree-based and linear models.
-   **Automated Plotting**: Generate and save ROC curves, Precision-Recall curves, and Confusion Matrices for each drug and model.
-   **Cross-Model Comparison**: Summarize and visualize the performance of different model types to identify the best-performing approaches.

## Getting Started

To get started with this project:

1.  **Explore Configurations**: Review the `configs/` directory for example model configurations (e.g., `xgb_config.json`) that define model types, HPO settings, and preprocessing options.
2.  **Understand Core Logic**: Dive into the `scripts/` directory to understand the main functionalities:
    -   `ml_models.py`: Defines supported ML models and pipeline construction.
    -   `eval_func.py`: Handles feature importance calculation.
    -   `summarize_models.py`: Manages model summarization and comparison.
    -   `plot_utils.py`: Contains plotting utilities for evaluation metrics.
    -   `customio.py`: Manages data input/output operations.
3.  **Run Experiments**: Refer to the `notebooks/` directory for examples on how to run training and evaluation workflows.

## Project Structure

```
.
├── configs/                  # Configuration files for different ML models and HPO settings
├── notebooks/                # Jupyter notebooks for data exploration, model training examples
├── scripts/                  # Core Python scripts for ML pipeline components
│   ├── parallel/             # Utilities for parallel processing (e.g., GPU device pinning)
│   ├── customio.py           # Data reading/writing utilities
│   ├── eval_func.py          # Functions for model evaluation and feature importance
│   ├── ml_models.py          # Definitions of ML models and pipeline construction
│   ├── plot_utils.py         # Utilities for generating plots (ROC, PR curves, etc.)
│   ├── summarize_models.py   # Scripts for summarizing and comparing model results
│   └── ...                   # Other training/utility scripts
├── utils/                    # General utility functions
│   ├── config_loader.py      # Utility for loading configuration files
│   └── misc.py               # Miscellaneous helper functions
├── README.md                 # Project overview and documentation
└── ...                       # Other project files (e.g., Dockerfile, .gitignore)

```

## Miscellaneous Commands to Run Training and Plotting

1. submitting slurm training script
```
bash bash_submit_slurm/submit_all_drug_models_orion.sh --data input/combined_DepMap_21Q3.csv --config configs/lgbm_config_early_decay.json --n-cores 10 --slurm-logs-dir slurm_logs --out-dir output/
```

2. plotting summary plots and importance scores
```
python scripts/summarize_models.py --model_root_dirs output/LGBMClassifier_intelligent_kare/,output/RandomForestClassifier_fervent_davinci/,output/SGDClassifier_strange_visvesvaraya/ --data_file_in input/combined_DepMap_21Q3.csv --outdir compare_lgb_rf_sgd --num_top_drugs 10 --metric_for_top_drugs test_average_precision --plot_type violinplot
```
