# import time
# start_time = time.time()

import os
import glob
from modules.pre_process import resample
import SimpleITK as sitk


if __name__=='__main__':

    testing_samples = [
        # '0139_1001',
        # '0141_1001',
        # '0144_1001',
        # '0146_1001',
        # '0150_0001',
        # '0151_0001',
    ]

    # Resampling configuration - choose either 'size' or 'spacing'
    resample_mode = 'spacing'  # 'size' or 'spacing'
    
    # If resample_mode is 'size', specify target size
    target_size = [512, 512, 512]  # [512, 512, 512] or [256, 256, 256]
    
    # If resample_mode is 'spacing', specify target spacing in mm
    target_spacing = [
                    0.03,
                    0.03,
                    0.03]  # [1.0, 1.0, 1.0] for 1mm isotropic spacing
    
    input_format = '.mha'  # '.mha' or '.vti'

    data_folder = '/Users/nsveinsson/Documents/datasets/vmr/images/'
    out_folder = data_folder + 'resampled/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]

    # sort the files
    imgs = sorted(imgs)

    # filter the images if testing_samples is not empty
    if testing_samples:
        imgs = [img for img in imgs if any(testing_sample in img for testing_sample in testing_samples)]
            
    print(f'Found {len(imgs)} images to resample')
    print(f'Images to resample: {imgs}')

    for img in imgs:
        img_path = os.path.join(data_folder, img)
        img_out_path = os.path.join(out_folder, img)

        # read the image
        img_sitk = sitk.ReadImage(img_path)

        print(f'Image {img} read')
        print(f"Image {img} shape: {img_sitk.GetSize()}")
        print(f"Image {img} spacing: {img_sitk.GetSpacing()}")

        # Calculate new resolution based on mode
        if resample_mode == 'size':
            # Calculate new spacing to achieve target size
            new_res = [img_sitk.GetSize()[0] / target_size[0],
                       img_sitk.GetSize()[1] / target_size[1],
                       img_sitk.GetSize()[2] / target_size[2]]
            new_res = [img_sitk.GetSpacing()[0] * new_res[0],
                       img_sitk.GetSpacing()[1] * new_res[1],
                       img_sitk.GetSpacing()[2] * new_res[2]]
            print(f"Image {img} target size: {target_size}")
            print(f"Image {img} calculated spacing: {new_res}")
        elif resample_mode == 'spacing':
            # Use target spacing directly
            new_res = target_spacing
            print(f"Image {img} target spacing: {target_spacing}")
        else:
            raise ValueError(f"Invalid resample_mode: {resample_mode}. Must be 'size' or 'spacing'")

        # resample the image
        img_sitk = resample(img_sitk, resolution = new_res, order=2, dim=3)

        print(f"Image {img} resampled shape: {img_sitk.GetSize()}")
        print(f"Image {img} resampled spacing: {img_sitk.GetSpacing()}")

        # write the image
        sitk.WriteImage(img_sitk, img_out_path)
        print(f'Image {img} resampled and saved to {img_out_path}')
