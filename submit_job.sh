#!/bin/bash

#$ -M mluczak2@nd.edu   # Email address for job notification
#$ -m abe            # Send mail when job begins, ends and aborts
#$ -pe smp X         # Specify parallel environment and legal core size
#$ -q gpu            # Run on the GPU cluster
#$ -l gpu_card=1     # Run on 1 GPU card
#$ -q long           # Specify queue
#$ -N job_name       # Specify job name

module load python  # Required modules
conda activate ENVIRONMENT

cd ../
python ${1}.py