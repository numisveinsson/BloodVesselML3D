import SimpleITK as sitk
import os
import numpy as np
import vtk 

import sys
sys.path.insert(0, '../..')
from modules.vtk_functions import vtk_marching_cube

if __name__ == "__main__":
    """
    This script is used to process the raw image (ct.nii.gz) and heart seg (heart.nii.gz) files from a folder structure:

    directory
    ├── images
        ├── subject001_CTA.mha
        ├── subject002_CTA.mha
    ├── labels
        ├── subject001_label.mha
        ├── subject002_label.mha

    and to add a new folder named binary_segs and surfaces with the following structure:
    
    directory
    ├── images
        ├── subject001_CTA.mha
        ├── subject002_CTA.mha
    ├── labels
        ├── subject001_label.mha
        ├── subject002_label.mha
    ├── binary_segs
        ├── subject001.mha
        ├── subject002.mha
    |── surfaces
        ├── subject001.vtp
        ├── subject002.vtp

    Where the subject001.nii.gz and subject002.nii.gz are the binarized heart segmentations.
    The original labels range from 1 to 850. Anything greater than 1 is set to 1.

    For each subject in the directories, the script:
        1. Reads in the 3D image and the 3D label.
        2. Binarizes the label.
        3. Saves the binarized label as a new 3D image in the binary_segs folder.
        4. Creates a surface from the binarized label and saves it as a .vtp file in the surfaces folder.

    """
    
    directory = '/Users/numisveins/Documents/aortaseg24/training/'
    out_data_dir = '/Users/numisveins/Documents/aortaseg24/process_binary/'

    # get list of folders in directory

    folders = os.listdir(os.path.join(directory, 'images'))
    folders = [f.split('_')[0] for f in folders if os.path.isfile(os.path.join(directory, 'images', f))]
    folders = np.unique(folders)
    # create new dataset directory

    try:
        os.mkdir(out_data_dir)
    except FileExistsError:
        print(f'Directory {out_data_dir} already exists')
    try:
        os.mkdir(os.path.join(out_data_dir, 'binary_segs'))
    except FileExistsError:
        print(f'Directory {out_data_dir} already exists')
    try:
        os.mkdir(os.path.join(out_data_dir, 'images'))
    except FileExistsError:
        print(f'Directory {out_data_dir} already exists')
    try:
        os.mkdir(os.path.join(out_data_dir, 'labels'))
    except FileExistsError:
        print(f'Directory {out_data_dir} already exists')


    for folder in folders:

        # get the raw image and heart seg files
        raw_img = os.path.join(directory, 'images', f'{folder}_CTA.mha')
        heart_seg = os.path.join(directory, 'labels', f'{folder}_label.mha')
        print(f"The max label value is {sitk.GetArrayFromImage(sitk.ReadImage(heart_seg)).max()}")

        # read in the binary segmentation and the image
        binary_seg = sitk.ReadImage(heart_seg)
        binary_np = sitk.GetArrayFromImage(binary_seg)
        binary_np[binary_np > 1] = 1
        binary_seg = sitk.GetImageFromArray(binary_np)
        binary_seg.CopyInformation(sitk.ReadImage(heart_seg))
        # cast to uint8
        binary_seg = sitk.Cast(binary_seg, sitk.sitkUInt8)

        # save the binary segmentation
        sitk.WriteImage(binary_seg, os.path.join(out_data_dir, 'binary_segs', f'{folder}.mha'))
        sitk.WriteImage(sitk.ReadImage(raw_img), os.path.join(out_data_dir, 'images', f'{folder}_CTA.mha'))
        sitk.WriteImage(sitk.ReadImage(heart_seg), os.path.join(out_data_dir, 'labels', f'{folder}_label.mha'))

        # create surface
        