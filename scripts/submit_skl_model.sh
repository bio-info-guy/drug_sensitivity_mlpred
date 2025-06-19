#!/bin/bash
#SBATCH --partition=ihc
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --account=ihc
#SBATCH --cpus-per-task=20



# Command to run your Python script
# Replace with the actual command to run skl_train_model.py
# Example: python scripts/skl_train_model.py --data_file /path/to/data.csv --drug_name DRUGX --config_file configs/xgb_config.json --out_dir results
echo "Starting model training..."
python scripts/skl_train_model.py  --data_file input/combined_DepMap_21Q3.csv --drug_name "BRD-A39255369-001-03-1::2.5::HTS" --config_file configs/xgb_config.json --out_dir output/models --n_cores 18

echo "Model training finished."
