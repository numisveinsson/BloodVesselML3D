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
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=numi@berkeley.edu
#
## Command(s) to run (example):

source activate /global/scratch/users/numi/environments/seqseg2
cd /global/scratch/users/numi/BloodVesselML3D/

python3  gather_sampling_data_parallel.py \
    -outdir /global/scratch/users/numi/ASOCA/trajectories_train/ \
    -config_name trajectories_savio \
    -num_cores 1 \
    
# -perc_dataset 0.15 \
# -outdir /global/scratch/users/numi/vascular_data_3d/trajectories_aortas_train/ \

# -outdir /global/scratch/users/numi/ASOCA/trajectories/ \