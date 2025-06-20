#!/bin/bash
#SBATCH --partition=Orion
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=24


echo $SLURM_SUBMIT_DIR
# Command to run your Python script
# Replace with the actual command to run skl_train_model.py
# Example: python scripts/skl_train_model.py --data_file /path/to/data.csv --drug_name DRUGX --config_file configs/xgb_config.json --out_dir results
 
python scripts/skl_train_model.py  --data_file input/combined_DepMap_21Q3.csv --drug_name "BRD-A39189014-003-22-7::2.5::HTS" --config_file configs/xgb_config_simple.json --out_dir output/opt_1_20/ --n_cores 20

echo "Model training finished."
