from datetime import datetime
now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")

import SimpleITK as sitk
import sys
sys.path.insert(0, './')

from modules import io
from modules.sampling_functions import *
from modules import sitk_functions as sf
from dataset_dirs.datasets import *

if __name__=='__main__':
    """
    Does same as gather_sampling_data.py but for global data
    So based on config/global.yaml, creates folders for training and testing
    Similarly, creates folders based on modalities
    This data then needs to be post-processed if to be used for eg nnUNet training
    """

    global_config_file = "./config/global.yaml"
    global_config = io.load_yaml(global_config_file)
    modalities = global_config['MODALITY']

    out_dir = global_config['OUT_DIR']

    # if not global_config['TESTING']:
    #     test_vars = [False]
    # else:
    #     test_vars = [False, True]

    # for testing in test_vars:

    testing = global_config['TESTING']

    for modality in modalities:

        cases = create_dataset(global_config, modality)
        modality = modality.lower()
        info_file_name = "info"+'_'+modality+dt_string+".txt"
        
        create_directories(out_dir, modality, global_config)

        image_out_dir_train = out_dir+modality+'_train/'
        seg_out_dir_train = out_dir+modality+'_train_masks/'
        image_out_dir_val = out_dir+modality+'_val/'
        seg_out_dir_val = out_dir+modality+'_val_masks/'

        image_out_dir_test = out_dir+modality+'_test/'
        seg_out_dir_test = out_dir+modality+'_test_masks/'

        #cases = Dataset.sort_cases(testing, global_config['TEST_CASES'])
        #cases = Dataset.check_which_cases_in_image_dir(cases)

        for i in cases:
            print(i)

        print_info_file(global_config, cases, global_config['TEST_CASES'], info_file_name)

        for i, case_fn in enumerate(cases):

            ## Load data
            case_dict = get_case_dict_dir(global_config['DATA_DIR'], case_fn, global_config['IMG_EXT'])
            print(f"\n {i+1}/{len(cases)}: {case_dict['NAME']}")

            name = case_dict['NAME']
            # Choose destination directory
            image_out_dir, seg_out_dir, val_port = choose_destination(testing, global_config['VALIDATION_PROP'], image_out_dir_test, seg_out_dir_test, 
                                                                        image_out_dir_val, seg_out_dir_val, image_out_dir_train, seg_out_dir_train, ip = None)
            
            reader_im, origin_im, size_im, spacing_im = sf.import_image(case_dict['IMAGE'])

            # Load image and segmentation
            img = sitk.ReadImage(case_dict['IMAGE'])
            try:
                seg = sitk.ReadImage(case_dict['SEGMENTATION'])
                # Make sure segmentation is binary
                seg = sitk.Cast(seg, sitk.sitkUInt8)
            except:
                print('No segmentation found')
                seg = None

            # check max and min values of img and seg
            max_val = sitk.GetArrayFromImage(img).max()
            min_val = sitk.GetArrayFromImage(img).min()
            print('Img Max value: ', max_val)
            print('Img Min value: ', min_val)
            if seg is not None:
                max_val = sitk.GetArrayFromImage(seg).max()
                min_val = sitk.GetArrayFromImage(seg).min()
                print('Seg Max value: ', max_val)
                print('Seg Min value: ', min_val)

            # If seg max value is over 1, then divide by max value
            if max_val > 1 and seg is not None:
                seg_np = sitk.GetArrayFromImage(seg)
                seg_np = seg_np/max_val
                file_reader = sf.read_image(case_dict['IMAGE'])
                seg = sf.create_new_from_numpy(file_reader, seg_np)
                seg = sitk.Cast(seg, sitk.sitkUInt8)
                print('Seg Max value: ', seg_np.max())
                print('Seg Min value: ', seg_np.min())

            if global_config['WRITE_SAMPLES']:
                sitk.WriteImage(img, image_out_dir + case_dict['NAME'] +'.nii.gz')
                if seg is not None:
                    sitk.WriteImage(seg, seg_out_dir + case_dict['NAME'] +'.nii.gz')
            if global_config['WRITE_VTK']:
                sitk.WriteImage(img, out_dir+'vtk_data/vtk_' + case_dict['NAME']+'.vtk')
                if seg is not None:
                    sitk.WriteImage(seg*255, out_dir+'vtk_data/vtk_mask_'+ case_dict['NAME']+'.vtk')

            print(f"\n Finished: ' {case_dict['NAME']}, {size_im}")