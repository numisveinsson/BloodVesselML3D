# Blood Vessel Modeling Data Pre and Post Processing

This repository contains code to process data used to train machine learning methods for geometric modeling of blood vessels using medical image data. The data used can be:
    1. Medical image scans
    2. Surface meshes
    3. Centerlines

The fundamental idea is to 'piece up' vasculature into hundreds/thousands of vascular segments. These segments can be
    1. Image subvolumes/pathes (3D/2D)
    2. Local surface representations
    3. Local centerline segments
    4. Local outlet/bifurcation/size/orientation information
    etc.

## Running the code

This code relies on the VMR being stored in a particular folder structure.

All cases should be stored in separate folders under one main folder.
The main folder should have the path indicated by `DATA_DIR` in the config file `/config/global.yaml` in this repository. 'OUT_DIR' refers to where the samples will be saved.

The locations of the image volume files, ground-truth binary image volume files, path files and group folders should then be indicated in `.txt` files.
An example of this can be found in `/cases` in this repository.

Generating 3D data for Ml requires running two scripts

1. python3 generate_cases_3d.py

This collects the different vascular models we have and the different locations of all files

2. python3 gather_sampling_data.py

This collects data about all positions of sampling to take place and size of sub-volumes. Then runs through the locations and actually extracts the sub-volumes and saves them.

In the case of resampling use

3. python3 additional_pre_process.py

And in the case of generating labels run

4. python3 generate_labels.py
