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
#SBATCH --mail-user=stanleychmwong@berkeley.edu
#
## Command(s) to run (example):

module load python/3.11.6-gcc-11.4.0
module load ml/pytorch/2.3.1-py3.11.7
cd /global/scratch/users/stanleychmwong/BloodVesselML3D/

python3  gather_sampling_data_parallel.py \
    -outdir /global/scratch/users/stanleychmwong/datasets/aorta_datasets/mic23_aorta_dataset_updated_units/local_model_wrong_radii/training/ \
    -config_name global_more_samples_savio_vmr \
    -num_cores 12\
    
# -perc_dataset 0.15 \