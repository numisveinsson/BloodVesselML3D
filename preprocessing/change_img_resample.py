# import time
# start_time = time.time()

import os
import glob
from modules.pre_process import resample
import SimpleITK as sitk


def resample_image(img_sitk, target_size=None, target_spacing=None, order=1):
    """
    Resample a SimpleITK image to a target size or spacing.
    
    Args:
        img_sitk: SimpleITK image to resample
        target_size: Target image size [x, y, z]. If provided, target_spacing is ignored.
        target_spacing: Target voxel spacing [x, y, z] in mm
        order: Interpolation order (0=nearest, 1=linear, 2=bspline)
    
    Returns:
        Resampled SimpleITK image
    
    Raises:
        ValueError: If neither target_size nor target_spacing is provided
    """
    if target_size is None and target_spacing is None:
        raise ValueError("Either target_size or target_spacing must be provided")
    
    if target_size is not None:
        # Calculate new spacing to achieve target size
        new_res = [img_sitk.GetSize()[0] / target_size[0],
                   img_sitk.GetSize()[1] / target_size[1],
                   img_sitk.GetSize()[2] / target_size[2]]
        new_res = [img_sitk.GetSpacing()[0] * new_res[0],
                   img_sitk.GetSpacing()[1] * new_res[1],
                   img_sitk.GetSpacing()[2] * new_res[2]]
    else:
        # Use target spacing directly
        new_res = target_spacing
    
    # Resample the image
    resampled_img = resample(img_sitk, resolution=new_res, order=order, dim=3)
    
    return resampled_img


def resample_images_batch(data_folder, out_folder, input_format='.mha', 
                          target_size=None, target_spacing=None, order=1,
                          testing_samples=None, skip_existing=True):
    """
    Batch resample all images in a folder.
    
    Args:
        data_folder: Input folder containing images
        out_folder: Output folder for resampled images
        input_format: Image file extension (e.g., '.mha', '.vti')
        target_size: Target image size [x, y, z]
        target_spacing: Target voxel spacing [x, y, z] in mm
        order: Interpolation order (0=nearest, 1=linear, 2=bspline)
        testing_samples: Optional list of sample names to filter
        skip_existing: Skip processing if output file already exists
    
    Returns:
        List of processed image filenames
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]
    imgs = sorted(imgs)
    
    # Filter images if testing_samples is provided
    if testing_samples:
        imgs = [img for img in imgs if any(ts in img for ts in testing_samples)]
    
    print(f'Found {len(imgs)} images to resample')
    print(f'Images to resample: {imgs}')
    
    processed = []
    
    for img in imgs:
        img_path = os.path.join(data_folder, img)
        img_out_path = os.path.join(out_folder, img)
        
        # Skip if output already exists
        if skip_existing and os.path.exists(img_out_path):
            print(f'Image {img} already processed, skipping...')
            continue
        
        # Read the image
        img_sitk = sitk.ReadImage(img_path)
        
        print(f'Image {img} read')
        print(f"Image {img} shape: {img_sitk.GetSize()}")
        print(f"Image {img} spacing: {img_sitk.GetSpacing()}")
        
        # Resample the image
        if target_size is not None:
            print(f"Image {img} target size: {target_size}")
            img_sitk = resample_image(img_sitk, target_size=target_size, order=order)
        else:
            print(f"Image {img} target spacing: {target_spacing}")
            img_sitk = resample_image(img_sitk, target_spacing=target_spacing, order=order)
        
        print(f"Image {img} resampled shape: {img_sitk.GetSize()}")
        print(f"Image {img} resampled spacing: {img_sitk.GetSpacing()}")
        
        # Write the image
        sitk.WriteImage(img_sitk, img_out_path)
        print(f'Image {img} resampled and saved to {img_out_path}')
        
        processed.append(img)
    
    return processed


if __name__=='__main__':

    testing_samples = [
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
    order = 1  # interpolation order: 0=nearest, 1=linear, 2=bspline

    data_folder = '/Users/nsveinsson/Documents/datasets/ASOCA_dataset/cm/images/'
    out_folder = data_folder.replace('images', 'images_03')
    
    # Determine target based on mode
    target_size_arg = target_size if resample_mode == 'size' else None
    target_spacing_arg = target_spacing if resample_mode == 'spacing' else None
    
    # Process batch
    resample_images_batch(
        data_folder=data_folder,
        out_folder=out_folder,
        input_format=input_format,
        target_size=target_size_arg,
        target_spacing=target_spacing_arg,
        order=order,
        testing_samples=testing_samples if testing_samples else None,
        skip_existing=True
    )
