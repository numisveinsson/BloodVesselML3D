import numpy as np
import os
import random
import SimpleITK as sitk
from modules import sitk_functions as sf

def random_files(directory, data_set_percent_size):

    #print(os.listdir(directory))

    # list all files in dir that are an image
    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]
    #print(files)

    # select a percent of the files randomly
    random_files = random.sample(files, int(len(files)*data_set_percent_size))
    #print(random_files)

    return random_files

def move_files(directory, target_directory, random_files):

    # move the randomly selected images by renaming directory
    for random_file_name in random_files:
        #print(directory+'/'+random_file_name)
        #print(target_directory+'/'+random_file_name)
        os.rename(directory+'/'+random_file_name, target_directory+'/'+random_file_name)

def change_values_images(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]

    for file in files:
        reader_im = sf.read_image(directory+'/'+file)
        img = reader_im.Execute()
        #sitk.Show(img, title="Image before "+file, debugOn=True)
        img = img//255
        #sitk.Show(img, title="Image after "+file, debugOn=True)
        sitk.WriteImage(img, directory+'/'+file)

if __name__=='__main__':
    #set directories
    directory = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train')
    target_directory = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val')

    directory_mask = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train_masks')
    target_directory_mask = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val_masks')

    data_set_percent_size = float(0.15)
    random_ct = random_files(directory, data_set_percent_size)
    import pdb; pdb.set_trace()
    move_files(directory, target_directory, random_ct)
    move_files(directory_mask, target_directory_mask, random_ct)

    # change_values_images(directory_mask)
    # change_values_images(target_directory_mask)
