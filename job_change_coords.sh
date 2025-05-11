#!/bin/bash
# Job name:
#SBATCH --job-name=change_coords
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
#SBATCH --cpus-per-task=4
#
# Wall clock limit:
#SBATCH --time=2:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stanleychmwong@berkeley.edu
#
## Command(s) to run (example):

module load python/3.11.6-gcc-11.4.0
module load ml/pytorch/2.3.1-py3.11.7
cd /global/scratch/users/stanleychmwong/BloodVesselML3D/

python3  change_img_scale_coords.py
    
# -perc_dataset 0.15 \
