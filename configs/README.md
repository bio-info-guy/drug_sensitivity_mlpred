# Model Configuration Files

This directory contains configuration files for training drug sensitivity models using the `skl_train_model.py` script. The configuration files allow you to specify model parameters, grid search settings, oversampling options, and random seeds in a structured format.

## Configuration File Format

Configuration files can be in JSON or YAML format. Both formats are supported.

### Required Fields

- `model_type`: The type of model to use (XGBClassifier, RandomForestClassifier, SGDClassifier, LGBMClassifier)
- `use_oversampling`: Boolean indicating whether to use oversampling
- `oversampler_type`: Type of oversampler (SMOTE, RandomOverSampler)
- `oversampler_seed`: Random seed for the oversampler
- `train_test_split_seed`: Random seed for train/test split
- `grid_search_params`: Dictionary of parameters for grid search

### Optional Fields

- `cv_seed`: Random seed for cross-validation (default: 7)
- `cv_splits`: Number of cross-validation splits (default: 5)
- `scoring_metric`: Metric to optimize during grid search (default: "precision")
- `search_method`: Search method to use (default: "gridcv")
- `fixed_params`: Dictionary of fixed parameters for the model

## Example Configurations

### XGBoost with SMOTE Oversampling (JSON)
```json
{
    "model_type": "XGBClassifier",
    "use_oversampling": true,
    "oversampler_type": "SMOTE",
    "oversampler_seed": 42,
    "train_test_split_seed": 20,
    "cv_seed": 7,
    "cv_splits": 5,
    "scoring_metric": "precision",
    "search_method": "gridcv",
    "fixed_params": {
        "random_state": 42,
        "eval_metric": "logloss"
    },
    "grid_search_params": {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.2]
    }
}
```

### Random Forest with Random Oversampling (JSON)
```json
{
    "model_type": "RandomForestClassifier",
    "use_oversampling": true,
    "oversampler_type": "RandomOverSampler",
    "oversampler_seed": 42,
    "train_test_split_seed": 20,
    "cv_seed": 7,
    "cv_splits": 5,
    "scoring_metric": "f1",
    "search_method": "gridcv",
    "fixed_params": {
        "random_state": 42
    },
    "grid_search_params": {
        "n_estimators": [100, 150, 250],
        "max_depth": [20, 50, 100, 200],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
}
```

### SGD Classifier without Oversampling (YAML)
```yaml
model_type: "SGDClassifier"
use_oversampling: false
oversampler_type: "SMOTE"
oversampler_seed: 42
train_test_split_seed: 20
cv_seed: 7
cv_splits: 5
scoring_metric: "accuracy"
search_method: "gridcv"
fixed_params:
  random_state: 42
  loss: "hinge"
  penalty: "elasticnet"
grid_search_params:
  l1_ratio: [0.05, 0.1, 0.15, 0.2]
  alpha: [0.02, 0.05, 0.1]
  max_iter: [1000, 2000]
```

## Usage

### Single Model Training
```bash
python scripts/skl_train_model.py \
    --data_file input/combined_DepMap_21Q3.csv \
    --drug_name "Drug_Name" \
    --config_file configs/xgb_config.json \
    --out_dir output/models \
    --n_cores 4
```

### Batch Training with Slurm
```bash
bash scripts/submit_all_drug_models.sh \
    --data input/combined_DepMap_21Q3.csv \
    --config configs/xgb_config.json \
    --out-dir output/models \
    --n-cores 8 \
    --slurm-logs-dir logs
```

## Available Models

- **XGBClassifier**: Extreme Gradient Boosting classifier
- **RandomForestClassifier**: Random Forest classifier
- **SGDClassifier**: Stochastic Gradient Descent classifier
- **LGBMClassifier**: Light Gradient Boosting Machine classifier

## Available Oversamplers

- **SMOTE**: Synthetic Minority Oversampling Technique
- **RandomOverSampler**: Random oversampling with replacement

## Scoring Metrics

Common scoring metrics for grid search optimization:
- `accuracy`: Classification accuracy
- `precision`: Precision score
- `recall`: Recall score
- `f1`: F1 score
- `roc_auc`: Area under the ROC curve

## Notes

- The `n_cores` parameter controls parallelization for model training and grid search
- When using Slurm, the script automatically allocates `n_cores + 1` CPUs and 24GB memory per task
- All random seeds should be set for reproducible results
- Grid search parameters should be lists of values to try
- Fixed parameters are passed directly to the model constructor
