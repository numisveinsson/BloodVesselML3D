#!/bin/bash
# Job name:
#SBATCH --job-name=create_nnunet_014
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
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=12:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stanleychmwong@berkeley.edu
#
## Command(s) to run (example):

module load python/3.11.6-gcc-11.4.0
module load ml/pytorch/2.3.1-py3.11.7
cd /global/scratch/users/stanleychmwong/BloodVesselML3D/dataset_dirs/

python3  create_nnunet.py \
    -outdir /global/scratch/users/stanleychmwong/nnUnet_data/nnUnet_raw/ \
    -indir /global/scratch/users/stanleychmwong/datasets/aorta_datasets/vmr_aorta_ct_dataset/global_model/training/ \
    -name AORTASGLOBAL \
    -dataset_number 14 \
    -modality ct \
    -start_from 84
    
# -perc_dataset 0.15 \

#/global/scratch/users/stanleychmwong/datasets/aorta_datasets/aortaseg24_dataset_updated_units
#/global/scratch/users/stanleychmwong/datasets/aorta_datasets/mic23_aorta_dataset_updated_units
#/global/scratch/users/stanleychmwong/datasets/aorta_datasets/vmr_aorta_ct_dataset