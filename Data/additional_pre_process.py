import time
start_time = time.time()

import numpy as np
import os
import glob

from modules import io
from modules.pre_process import *

from vtk.util.numpy_support import vtk_to_numpy as v2n
import SimpleITK as sitk

def create_directories(output_folder, modality, fns):
    try:
        os.mkdir(output_folder)
    except Exception as e: print(e)
    for fn in fns:
        try:
            os.mkdir(output_folder+modality+fn)
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+modality+fn+'_masks')
        except Exception as e: print(e)

if __name__=='__main__':

    global_config_file = "./config/global.yaml"
    global_config = io.load_yaml(global_config_file)
    modalities = global_config['MODALITY']

    resample = False

    data_folder = '' #global_config['OUT_DIR']

    # data_folder = '/Users/numisveins/Library/Mobile Documents/com~apple~CloudDocs/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/test4/'
    # modality = 'ct'

    data_out = data_folder+'data_additional_processed/'
    fns = ['_train', '_val']

    # image_out_dir_train = out_dir+modality+'_train/'
    # seg_out_dir_train = out_dir+modality+'_train_masks/'
    # image_out_dir_val = out_dir+modality+'_val/'
    # seg_out_dir_val = out_dir+modality+'_val_masks/'
    #
    # folders = [image_out_dir_train, seg_out_dir_train, image_out_dir_val, seg_out_dir_val]

    size = [64, 64, 64]
    for modality in modalities:
        modality = modality.lower()
        create_directories(data_out, modality, fns)
        for fn in fns:
            imgVol_fn, seg_fn = [], []
            for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,modality+fn,'*.nii.gz')) \
                    +glob.glob(os.path.join(data_folder,modality+fn,'*.nii')) ):
                imgVol_fn.append(os.path.realpath(subject_dir))
            for subject_dir in natural_sort(glob.glob(os.path.join(data_folder,modality+fn+'_masks','*.nii.gz')) \
                    +glob.glob(os.path.join(data_folder,modality+fn+'_masks','*.nii')) ):
                seg_fn.append(os.path.realpath(subject_dir))
            print("number of training data %d" % len(imgVol_fn))
            print("number of training data segmentation %d" % len(seg_fn))
            assert len(seg_fn) == len(imgVol_fn)

            for i in range(len(imgVol_fn)):
                if i in range(0,len(imgVol_fn),len(imgVol_fn)//10): print('* ', end='', flush=True)
                img_path = imgVol_fn[i]
                seg_path = seg_fn[i]

                imgVol = sitk.ReadImage(img_path)
                #segVol = sitk.ReadImage(seg_path)
                
                if resample:
                    imgVol = resample_spacing(imgVol, template_size=size, order=1)[0]
                #segVol = resample_spacing(segVol, template_size=size, order=0)[0]

                sitk.WriteImage(imgVol, img_path.replace(data_folder, data_out))#.replace('nii.gz', 'vtk'))
                #sitk.WriteImage(segVol, seg_path.replace(data_folder, data_out))#.replace('nii.gz', 'vtk'))



    import pdb; pdb.set_trace()
