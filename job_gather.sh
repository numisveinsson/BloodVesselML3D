#!/bin/bash
# Job name:
#SBATCH --job-name=gather_data
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio4_htc
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
#SBATCH --cpus-per-task=12
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=numi@berkeley.edu
#
## Command(s) to run (example):
module load gcc
module load cuda/10.0
module load cudnn/7.5
source activate /global/scratch/users/numi/environments/seqseg2
cd /global/scratch/users/numi/BloodVesselML3D/

python3  gather_sampling_data_parallel.py \
    -outdir /global/scratch/users/numi/aortaseg24/extraction_output/aortaseg24_labelll/ \
    -config_name global_fewer_samples \
    -num_cores 12 \
    
# -perc_dataset 0.15 \