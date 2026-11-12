#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -M mluczak2@nd.edu
#$ -m abe
#$ -pe smp 1
#$ -q gpu
#$ -l gpu_card=1

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: qsub -N job_name submit_job.sh FILENAME MODEL_OUTPUT_FOLDER"
    exit 1
fi

PYTHON_FILE="${1%.py}.py"
OUTPUT_FOLDER="$2"

if [[ ! "$OUTPUT_FOLDER" =~ ^model_outputs(_[a-zA-Z0-9_-]+)?$ ]]; then
    echo "Error: Invalid output folder name."
    exit 1
fi

OUTPUT_DIR="$HOME/$OUTPUT_FOLDER"

mkdir -p "$OUTPUT_DIR"

find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

exec > "$OUTPUT_DIR/job_${JOB_ID}.out" 2> "$OUTPUT_DIR/job_${JOB_ID}.err"

module load python
source ~/.venvs/ml-research/bin/activate

export MODEL_OUTPUT_DIR="$OUTPUT_DIR"

MPLBACKEND=Agg python -u "$PYTHON_FILE"