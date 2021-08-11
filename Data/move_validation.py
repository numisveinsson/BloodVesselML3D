import numpy as np
import os
import random

#set directories
directory = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train')
target_directory = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_validation')

directory_mask = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train_masks')
target_directory_mask = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_validation_masks')

data_set_percent_size = float(0.15)

#print(os.listdir(directory))

# list all files in dir that are an image
files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]

#print(files)

# select a percent of the files randomly
random_files = random.sample(files, int(len(files)*data_set_percent_size))
#random_files = np.random.choice(files, int(len(files)*data_set_percent_size))

#print(random_files)

# move the randomly selected images by renaming directory

for random_file_name in random_files:
    #print(directory+'/'+random_file_name)
    #print(target_directory+'/'+random_file_name)
    os.rename(directory+'/'+random_file_name, target_directory+'/'+random_file_name)
    os.rename(directory_mask+'/'+random_file_name, target_directory_mask+'/'+random_file_name)
    
    continue
