#!/bin/bash
# Job name:
#SBATCH --job-name=3dunet_pred
#
# Account:
#SBATCH --account=fc_biome
#
# Partition:
#SBATCH --partition=savio2_htc
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=1
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#
# Wall clock limit:
#SBATCH --time=20:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=numi@berkeley.edu
#
## Command(s) to run (example):


conda activate ml_py_sitk1
conda list

unet3_dir=/Users/numisveinsson/Documents/Berkeley/Research/BloodVessel_UNet3D/
output_dir=/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/test5_trace_test1/
python3  trace_centerline.py \
    --image  /Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/images/OSMSC0176/OSMSC0176-cm.mha \
    --output ${output_dir}\
    --model ${unet3_dir}/output/test10 \
    --modality ct \
    --size 64 64 64 \
    --case 0176_0000\
    --threshold 0.5\
    --stepsize 1
deactivate
