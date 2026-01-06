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
    
    from modules.logger import get_logger
    logger = get_logger(__name__)
    
    logger.info(f'Found {len(imgs)} images to resample')
    logger.debug(f'Images to resample: {imgs}')
    
    processed = []
    
    for img in imgs:
        img_path = os.path.join(data_folder, img)
        img_out_path = os.path.join(out_folder, img)
        
        # Skip if output already exists
        if skip_existing and os.path.exists(img_out_path):
            logger.info(f'Image {img} already processed, skipping...')
            continue
        
        # Read the image
        img_sitk = sitk.ReadImage(img_path)
        
        logger.debug(f'Image {img} read')
        logger.debug(f"Image {img} shape: {img_sitk.GetSize()}")
        logger.debug(f"Image {img} spacing: {img_sitk.GetSpacing()}")
        
        # Resample the image
        if target_size is not None:
            logger.debug(f"Image {img} target size: {target_size}")
            img_sitk = resample_image(img_sitk, target_size=target_size, order=order)
        else:
            logger.debug(f"Image {img} target spacing: {target_spacing}")
            img_sitk = resample_image(img_sitk, target_spacing=target_spacing, order=order)
        
        logger.debug(f"Image {img} resampled shape: {img_sitk.GetSize()}")
        logger.debug(f"Image {img} resampled spacing: {img_sitk.GetSpacing()}")
        
        # Write the image
        sitk.WriteImage(img_sitk, img_out_path)
        logger.info(f'Image {img} resampled and saved to {img_out_path}')
        
        processed.append(img)
    
    return processed


if __name__=='__main__':
    import argparse
    import ast
    
    parser = argparse.ArgumentParser(
        description='Resample images to target size or spacing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resample to target spacing:
  python change_img_resample.py --input_dir /path/to/images --output_dir /path/to/output --target_spacing 0.03 0.03 0.03
  
  # Resample to target size:
  python change_img_resample.py --input_dir /path/to/images --target_size 512 512 512
  
  # Using default directory:
  python change_img_resample.py --target_spacing 1.0 1.0 1.0
        """
    )
    parser.add_argument('--input_dir', '--input-dir',
                       type=str,
                       default=None,
                       help='Directory containing input images. '
                            'Defaults to ./data/images/')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Directory to write resampled images. '
                            'Defaults to inferred from input_dir')
    parser.add_argument('--input_format', '--input-format',
                       type=str,
                       default='.mha',
                       help='Input file extension (default: .mha)')
    parser.add_argument('--target_size', '--target-size',
                       type=int,
                       nargs=3,
                       metavar=('X', 'Y', 'Z'),
                       default=None,
                       help='Target image size [x, y, z]. Mutually exclusive with --target_spacing')
    parser.add_argument('--target_spacing', '--target-spacing',
                       type=float,
                       nargs=3,
                       metavar=('X', 'Y', 'Z'),
                       default=None,
                       help='Target voxel spacing [x, y, z] in mm. Mutually exclusive with --target_size')
    parser.add_argument('--order',
                       type=int,
                       default=1,
                       choices=[0, 1, 2],
                       help='Interpolation order: 0=nearest, 1=linear, 2=bspline (default: 1)')
    parser.add_argument('--testing_samples', '--testing-samples',
                       type=str,
                       nargs='+',
                       default=None,
                       help='Optional list of sample names to filter (process only these)')
    parser.add_argument('--no_skip_existing', '--no-skip-existing',
                       dest='skip_existing',
                       action='store_false',
                       help='Re-process files even if output already exists')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.target_size and args.target_spacing:
        parser.error("--target_size and --target_spacing are mutually exclusive")
    if not args.target_size and not args.target_spacing:
        parser.error("Either --target_size or --target_spacing must be provided")
    
    # Use command-line arguments (required or default)
    data_folder = args.input_dir or './data/images/'
    out_folder = args.output_dir or data_folder.replace('images', 'images_resampled')
    
    # Validate directories
    if not os.path.exists(data_folder):
        raise ValueError(f"Input directory not found: {data_folder}. "
                        f"Provide --input_dir argument.")
    
    # Process batch
    resample_images_batch(
        data_folder=data_folder,
        out_folder=out_folder,
        input_format=args.input_format,
        target_size=args.target_size,
        target_spacing=args.target_spacing,
        order=args.order,
        testing_samples=args.testing_samples,
        skip_existing=args.skip_existing
    )
