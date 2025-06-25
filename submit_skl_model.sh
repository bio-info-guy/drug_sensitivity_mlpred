#!/bin/bash
#SBATCH --partition=Orion
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=24


echo $SLURM_SUBMIT_DIR
# Command to run your Python script
# Replace with the actual command to run skl_train_model.py
# Example: python scripts/skl_train_model.py --data_file /path/to/data.csv --drug_name DRUGX --config_file configs/xgb_config.json --out_dir results
 
python scripts/skl_train_model.py  --data_file input/combined_DepMap_21Q3.csv --drug_name "BRD-A25004090-001-08-4::2.5::HTS" --config_file configs/test_config/sgd_config_test.yaml --out_dir output/ --n_cores 20

echo "Model training finished."
