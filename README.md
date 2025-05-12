# Blood Vessel Modeling Data Pre and Post Processing

This repository contains code to process data used to train machine learning methods for geometric modeling of blood vessels using medical image data. The data used is:
    1. Medical image scans
    2. Ground truth segmentations
    3. Centerlines
    4. Surface meshes (if applicable)

## Primary Scripts Used

1. `python3 change_img_scale_coords.py`

This script changes the scale of the image in the path. This is done by scaling the image spacing but keeping everything else the same.

2. `python3 change_vtk_scale_coords.py `

This script changes the scale of the centerline in the path. This is done by scaling the overall size of the centerline but keeps everything else the same (including the radii, which should subsequently also be scaled in the gather script)

3. `python3 gather_sampling_data_parallel.py`

This is the main script. This collects data about all positions of sampling to take place and size of sub-volumes. Then runs through the locations and actually extracts the sub-volumes and saves them. This code relies on the data being stored in a particular folder structure:
    - images
        - case0.x
    - centerlines
        - case0.vtp
    - truths
        - case0.x
    - surfaces (if applicable)

The main folder should have the path indicated by `DATA_DIR` in the config file `/config/global_more_samples_savio.yaml` in this repository. This is also where the radii of the centerlines can be scaled prior to extracting of sub-volumes. 

4. `python3 create_nnunet.py`

This script organizes subvolume data into a format readable by nnUNet.

### global_more_samples_savio.yaml

This YAML file contains configuration settings for a data processing script. Here is a list of variables most pertinent to the dataset I was working with:

- `DATA_DIR`: This is the directory where the input data is stored. The script will look for data files in this directory.

- `OUT_DIR`: This is the directory where the script will write its output. The script will create this directory if it doesn't exist.

- `DATASET_NAME`: The name of the dataset being processed. This could be used to select specific processing routines based on the dataset.

- `TESTING`: A boolean flag indicating whether the script is in testing mode.

- `MODALITY`: The type of imaging modality of the data. This could be 'CT', 'MR', or both.

- `IMG_EXT`: The file extension of the image files.

- `SCALED`: A boolean flag indicating whether the images should be scaled.

- `ANATOMY`: The type of anatomy that the images represent. This could be 'ALL' or a list of specific anatomies.

- `RADIUS_SCALED`: How much to scale radii to correct bias

