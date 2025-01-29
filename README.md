# Blood Vessel Modeling Data Pre and Post Processing

This repository contains code to process data used to train machine learning methods for geometric modeling of blood vessels using medical image data. The data used is:
    1. Medical image scans
    2. Ground truth segmentations
    3. Centerlines
    4. Surface meshes (if applicable)

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

This is the main script. This collects data about all positions of sampling to take place and size of sub-volumes. Then runs through the locations and actually extracts the sub-volumes and saves them.

In the case of resampling use

2. python3 additional_pre_process.py

And in the case of generating labels run

3. python3 generate_labels.py

### gather_sampling_data_parallel.py

This Python script is used for parallel processing of data sampling from multiple cases. It uses the multiprocessing library in Python to speed up the process by utilizing multiple cores of your CPU.

The script is designed to be run in a multiprocessing environment, where each process handles a different case. This allows for efficient use of computational resources when dealing with a large number of cases.

### global.yaml

This YAML file contains configuration settings for a data processing script. Here's a brief overview of what each variable does:

- `DATA_DIR`: This is the directory where the input data is stored. The script will look for data files in this directory.

- `OUT_DIR`: This is the directory where the script will write its output. The script will create this directory if it doesn't exist.

- `DATASET_NAME`: The name of the dataset being processed. This could be used to select specific processing routines based on the dataset.

- `TESTING`: A boolean flag indicating whether the script is in testing mode.

- `MODALITY`: The type of imaging modality of the data. This could be 'CT', 'MR', or both.

- `IMG_EXT`: The file extension of the image files.

- `SCALED`: A boolean flag indicating whether the images should be scaled.

- `CROPPED`: A boolean flag indicating whether the images should be cropped.

- `ANATOMY`: The type of anatomy that the images represent. This could be 'ALL' or a list of specific anatomies.

- `OUTLET_CLASSES`: A list of classes for outlet identification.

- `VALIDATION_PROP`: The proportion of the data that should be set aside for validation.

- `EXTRACT_VOLUMES`: A boolean flag indicating whether volumes should be extracted from the images.

- `ROTATE_VOLUMES`: A boolean flag indicating whether the volumes should be rotated.

- `RESAMPLE_VOLUMES`: A boolean flag indicating whether the volumes should be resampled.

- `RESAMPLE_SIZE`: The size to which volumes should be resampled.

- `AUGMENT_VOLUMES`: A boolean flag indicating whether data augmentation should be applied to the volumes.

- `WRITE_SAMPLES`: A boolean flag indicating whether samples should be written out.

- `WRITE_IMG`: A boolean flag indicating whether images should be written out.

- `REMOVE_OTHER`: A boolean flag indicating whether other unconnected vessels should be removed from the ground truth volume.

- `WRITE_SURFACE`: A boolean flag indicating whether the surface should be written out.

- `WRITE_CENTERLINE`: A boolean flag indicating whether the centerline should be written out.

- `WRITE_DISCRETE_CENTERLINE`: A boolean flag indicating whether a discrete centerline should be written out.

- `DISCRETE_CENTERLINE_N_POINTS`: The number of points in the discrete centerline.

- `WRITE_OUTLET_STATS`: A boolean flag indicating whether outlet statistics should be written out.

- `WRITE_OUTLET_IMG`: A boolean flag indicating whether outlet images should be written out.

These variables allow the user to customize the behavior of the script to suit their specific needs.