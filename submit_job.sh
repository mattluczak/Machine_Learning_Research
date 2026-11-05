#!/bin/bash

#$ -M mluczak2@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp 1         # Specify parallel environment and legal core size
#$ -q gpu            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -N job_name       # Specify job name

module load python  # Required modules

source ~/.venvs/ml-research/bin/activate

find ~/model_outputs -mindepth 1 -maxdepth 1 ! -name 'housing.data' -delete

MPLBACKEND=Agg python -u ${1}.py