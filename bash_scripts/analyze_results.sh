#!/bin/sh
#SBATCH --partition=cpu
#SBATCH -c 2
#SBATCH --output=bootstrap%A.log
#SBATCH --mem=50gb

set -e
source activate hurtfulwords

BASE_DIR="/Users/aravind/Premnisha/MS/dlh/HurtfulWords"
OUTPUT_DIR="${BASE_DIR}/data"
cd "$BASE_DIR/scripts"

python analyze_results.py \
    --models_path "${OUTPUT_DIR}/models/" \
    --set_to_use "test" \
    --bootstrap
