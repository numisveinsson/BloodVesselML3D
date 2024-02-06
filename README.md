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

This code relies on the data being stored in a particular folder structure:
    - images
        - case0.x
    - centerlines
        - case0.vtp
    - truths
        - case0.x
    - surfaces (if applicable)

The main folder should have the path indicated by `DATA_DIR` in the config file `/config/global.yaml` in this repository. 'OUT_DIR' refers to where the samples will be saved.

1. python3 gather_sampling_data.py

This collects data about all positions of sampling to take place and size of sub-volumes. Then runs through the locations and actually extracts the sub-volumes and saves them.

In the case of resampling use

2. python3 additional_pre_process.py

And in the case of generating labels run

3. python3 generate_labels.py
