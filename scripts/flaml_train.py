import argparse
import pandas as pd
import json
import os
import joblib
from flaml import AutoML

def main():
    parser = argparse.ArgumentParser(description="Train a FLAML model for drug sensitivity prediction.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the input data CSV file.")
    parser.add_argument("--drug_name", type=str, required=True, help="Name of the drug to train the model for.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the FLAML model configuration JSON file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the output model and results.")
    parser.add_argument("--num_gene", type=int, default=17651, help="Number of gene features in the dataset.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading data from {args.data_file}...")
    dataset = pd.read_csv(args.data_file)

    X = dataset.iloc[:, 1:args.num_gene+1]
    
    if args.drug_name not in dataset.columns:
        print(f"Error: Drug '{args.drug_name}' not found in the dataset.")
        return

    y = dataset[args.drug_name]

    print(f"Training FLAML model for drug: {args.drug_name}")
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    print(f"Loading model configuration from {args.config_file}...")
    with open(args.config_file, 'r') as f:
        model_config = json.load(f)

    automl = AutoML()

    # FLAML fit expects X and y, and then **kwargs for model configuration
    automl.fit(X, y, **model_config)

    print(f"FLAML training finished for {args.drug_name}.")

    # Prepare results to save
    results = {
        "drug_name": args.drug_name,
        "best_model": automl.best_model,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "best_metric": automl.best_metric,
        "training_time": automl.train_time,
        "model_config_used": model_config,
        "models_tried": automl.model_history # This contains details for each model tried
    }

    # Save the best model
    model_output_path = os.path.join(args.out_dir, f"flaml_model_{args.drug_name}.joblib")
    joblib.dump(automl, model_output_path)
    print(f"Best FLAML model saved to {model_output_path}")

    # Save the results to a JSON file
    results_output_path = os.path.join(args.out_dir, f"flaml_results_{args.drug_name}.json")
    with open(results_output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"FLAML results saved to {results_output_path}")

if __name__ == "__main__":
    main()
