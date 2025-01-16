#!/bin/bash
# Job name:
#SBATCH --job-name=extr_nnunet
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio3_htc
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks needed for use case (example):
#SBATCH --ntasks=1
#
# Processors per task:
#SBATCH --cpus-per-task=2
#
# Wall clock limit:
#SBATCH --time=12:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=numi@berkeley.edu
#
## Command(s) to run (example):

source activate /global/scratch/users/numi/environments/seqseg2

python3  dataset_dirs/create_nnunet.py \
    -indir //global/scratch/users/numi/PARSE_dataset/local_extraction_fewer/ \
    -outdir /global/scratch/users/numi/nnUnet_data/nnUnet_raw/ \
    -dataset_number 40 \
    -name SEQPULMPARSE \
    -modality ct \
