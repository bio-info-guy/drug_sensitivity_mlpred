#!/bin/bash
#SBATCH --partition=ihc
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --account=ihc
#SBATCH --cpus-per-task=24
#SBATCH --nodelist=ihc-grid-1-1-1

echo $TMPDIR
echo $SLURM_SUBMIT_DIR
# Command to run your Python script
# Replace with the actual command to run skl_train_model.py
# Example: python scripts/skl_train_model.py --data_file /path/to/data.csv --drug_name DRUGX --config_file configs/xgb_config.json --out_dir results
echo "Starting model training..."
python scripts/skl_train_model.py  --data_file input/combined_DepMap_21Q3.csv --drug_name "BRD-A39189014-003-22-7::2.5::HTS" --config_file configs/xgb_config.json --out_dir output/ --n_cores 22

echo "Model training finished."
