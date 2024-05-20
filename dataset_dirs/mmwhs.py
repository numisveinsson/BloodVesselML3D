import SimpleITK as sitk
import os
import numpy as np

if __name__ == "__main__":
    """
    This script is used to move the raw image (ct.nii.gz) and heart seg (heart.nii.gz) files from a folder structure:

    directory_name
    ├── directory_name_1_image.nii.gz
    ├── directory_name_1_label.nii.gz
    ├── directory_name_2_image.nii.gz
    ├── directory_name_2_label.nii.gz

    Into a dataset that has the following structure:
    
    Output_Directory
    ├── imagesTr
        ├── directory_name_1.nii.gz
        ├── directory_name_2.nii.gz
    └── labelsTr
        ├── directory_name_1.nii.gz
        ├── directory_name_2.nii.gz

    Where the directory_name_1.nii.gz and directory_name_2.nii.gz are the raw images and the directory_name_1.nii.gz and directory_name_2.nii.gz are the heart segmentations.

    Additionally, the labels are binarized to 0 and 1. The original labels range from 1 to 850. Anything greater than 1 is set to 1.

    """
    
    directory = '/Users/numisveins/Documents/data_combo_paper/mr_train/'
    out_data_dir = '/Users/numisveins/Documents/data_combo_paper/DATASET/'

    # get list of folders in directory

    folders = os.listdir(directory)
    first_name = folders[0].split('_')[:2]
    first_name = '_'.join(first_name) + '_'
    folders = [f.split('_')[2] for f in folders if os.path.isfile(os.path.join(directory, f))]
    folders = np.unique(folders)
    # create new dataset directory

    try:
        os.mkdir(out_data_dir)
    except FileExistsError:
        print(f'Directory {out_data_dir} already exists')

    # create new dataset directory
    try:
        os.mkdir(os.path.join(out_data_dir, 'imagesTr'))
        os.mkdir(os.path.join(out_data_dir, 'labelsTr'))
    except FileExistsError:
        print(f'Directory {out_data_dir} already exists')

    for folder in folders:

        # get the raw image and heart seg files
        raw_img = os.path.join(directory, f'{first_name+folder}_image.nii.gz')
        heart_seg = os.path.join(directory, f'{first_name+folder}_label.nii.gz')
        print(f"The max label value is {sitk.GetArrayFromImage(sitk.ReadImage(heart_seg)).max()}")
        # copy the files to the new directory
        sitk.WriteImage(sitk.ReadImage(raw_img), os.path.join(out_data_dir, 'imagesTr', f'{folder}.nii.gz'))
        sitk.WriteImage(sitk.BinaryThreshold(sitk.ReadImage(heart_seg), lowerThreshold=1, upperThreshold=850, insideValue=1, outsideValue=0), os.path.join(out_data_dir, 'labelsTr', f'{folder}.nii.gz'))
        # also write original label
        # sitk.WriteImage(sitk.ReadImage(heart_seg), os.path.join(out_data_dir, 'labelsTr', f'{folder}_original.mha'))
        print(f'Copied {folder} to new directory')

    