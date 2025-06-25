#!/bin/bash

# Default values
DATA_FILE="input/combined_DepMap_21Q3.csv"
CONFIG_FILE="configs/xgb_config.json"
OUT_DIR="output/skl_models"
SLURM_LOGS_DIR="slurm_logs"
N_CORES=16
ARRAY_INDEX=1
APPTAINER_IMAGE="pytorch-ml-container.sif"
DEVICE="cpu" # Default device

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --data)
            DATA_FILE="$2"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift
            ;;
        --slurm-logs-dir)
            SLURM_LOGS_DIR="$2"
            shift
            ;;
        --n-cores)
            N_CORES="$2"
            shift
            ;;
        --array)
            ARRAY_INDEX="$2"
            shift
            ;;
        --apptainer-image)
            APPTAINER_IMAGE="$2"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Submit a Slurm job array to train scikit-learn drug sensitivity models using Apptainer."
            echo ""
            echo "Options:"
            echo "  --data <file>         Path to the combined data file (default: input/combined_DepMap_21Q3.csv)"
            echo "  --config <file>       Path to the model configuration file (default: configs/xgb_config.json)"
            echo "  --out-dir <dir>       Directory to save trained models (default: output/skl_models)"
            echo "  --slurm-logs-dir <dir> Directory to save Slurm logs (default: slurm_logs)"
            echo "  --n-cores <num>       Number of CPU cores to use per task (default: 1)"
            echo "  --apptainer-image <file> Path to the Apptainer image (default: pytorch-ml-container.sif)"
            echo "  --device <cpu|cuda>   Device to use for training (default: cpu)"
            echo "  -h, --help            Display this help message and exit"
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Create necessary directories
mkdir -p "${SLURM_LOGS_DIR}"
mkdir -p "${OUT_DIR}"

# Calculate CPU and memory allocation
CPUS_PER_TASK=$((N_CORES + 4))
MEMORY="64G"

echo "Using parameters:"
echo "  Data File: ${DATA_FILE}"
echo "  Config File: ${CONFIG_FILE}"
echo "  Output Directory: ${OUT_DIR}"
echo "  Slurm Logs Directory: ${SLURM_LOGS_DIR}"
echo "  Number of Cores: ${N_CORES}"
echo "  CPUs per Task: ${CPUS_PER_TASK}"
echo "  Memory: ${MEMORY}"
echo "  Apptainer Image: ${APPTAINER_IMAGE}"
echo "  Device: ${DEVICE}"

echo "Generating drug list..."
# Get the list of drugs and count them
DRUG_LIST_FILE="${SLURM_LOGS_DIR}/drug_list.txt"
python scripts/get_drug_list.py "${DATA_FILE}" > "${DRUG_LIST_FILE}"
NUM_DRUGS=$(wc -l < "${DRUG_LIST_FILE}")
ARRAY_MAX_INDEX=$((NUM_DRUGS - 1))

if [ "$NUM_DRUGS" -eq 0 ]; then
    echo "Error: No drugs found in ${DATA_FILE}. Exiting."
    exit 1
fi

echo "Found ${NUM_DRUGS} drugs. Preparing Slurm job array..."

# Create the Slurm job script dynamically
SLURM_JOB_SCRIPT="${SLURM_LOGS_DIR}/skl_drug_model_array_apptainer_job.sh"

# Determine --nv flag for Apptainer
NV_FLAG=""
if [ "${DEVICE}" == "cuda" ]; then
    NV_FLAG="--nv"
fi

cat > "${SLURM_JOB_SCRIPT}" <<EOF
#!/bin/bash
#
#SBATCH --job-name=skl_drug_model_array_apptainer
#SBATCH --output=${SLURM_LOGS_DIR}/skl_drug_model_array_%A_%a.out
#SBATCH --error=${SLURM_LOGS_DIR}/skl_drug_model_array_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --mem=${MEMORY}
#SBATCH --nodes=1
#SBATCH --account=ihc
#SBATCH --nodelist=ihc-grid-1-1-1
#SBATCH --partition=ihc
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --array=40-${ARRAY_MAX_INDEX}%7

# Define paths and parameters (these are passed from the submission script)
DATA_FILE="${DATA_FILE}"
CONFIG_FILE="${CONFIG_FILE}"
OUT_DIR="${OUT_DIR}"
N_CORES="${N_CORES}"
DRUG_LIST_FILE="${DRUG_LIST_FILE}"
SLURM_LOGS_DIR="${SLURM_LOGS_DIR}" # Pass logs dir to inner script for output path
APPTAINER_IMAGE="${APPTAINER_IMAGE}"
DEVICE="${DEVICE}"

# Get the drug name for this specific job array task
# Read the drug list file and get the drug at the current array index
CURRENT_DRUG=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "\${DRUG_LIST_FILE}")

echo "Starting job for drug: \${CURRENT_DRUG} (Task ID: \${SLURM_ARRAY_TASK_ID})"
echo "Data file: \${DATA_FILE}"
echo "Config file: \${CONFIG_FILE}"
echo "Number of cores: \${N_CORES}"
echo "Output directory: \${OUT_DIR}"
echo "Apptainer Image: \${APPTAINER_IMAGE}"
echo "Device: \${DEVICE}"

module load apptainer 

# Execute the training script inside the Apptainer container
apptainer exec ${NV_FLAG} \\
    --env PYTHONUSERBASE=/tmp/python-packages \\
    --env PYTHONPATH=\$PYTHONUSERBASE:/mnt/host_repo \\
    --env CURRENT_DRUG="\${CURRENT_DRUG}" \\
    --bind $(pwd):/mnt/host_repo \\
    "\${APPTAINER_IMAGE}" bash -c ' \\
    mkdir -p /tmp/python-packages && \\
    pip install --user matplotlib seaborn sklearn-evaluation && \\
    cd /mnt/host_repo && \\
    python scripts/skl_train_model.py \\
    --data_file "${DATA_FILE}" \\
    --drug_name "\$CURRENT_DRUG" \\
    --config_file "${CONFIG_FILE}" \\
    --out_dir "${OUT_DIR}" \\
    --n_cores ${N_CORES} \\
    --device ${DEVICE} '

echo "Finished job for drug: \${CURRENT_DRUG}"
EOF

echo "Submitting Slurm job array..."
sbatch "${SLURM_JOB_SCRIPT}"

echo "Slurm job array submission complete. Check ${SLURM_LOGS_DIR} for logs."
