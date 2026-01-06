import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sys
import os

# Add modules directory to path
modules_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

import vtk_functions as vf
import sitk_functions as sf


def rotate_mesh(mesh, vtkLabel, center=None):
    """
    Rotate the mesh by 90 degrees
    around the origin of the image y-axis

    The steps are as follows:
    1. Specify the axis of rotation
    2. Create a transform
    3. Apply the transform to the mesh

    Args:
        mesh: vtk PolyData
        vtkLabel: vtk ImageData
    Returns:
        mesh: rotated vtk PolyData
    """

    # Get the center of the image
    if center is None:
        center = vtkLabel.GetCenter()

    # Specify the axis of rotation
    axis = [0, 1, 0]
    # Create a transform
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(-center[0], -center[1], -center[2])
    transform.RotateWXYZ(90, axis[0], axis[1], axis[2])
    transform.Translate(center[0], center[1], center[2])

    # Apply the transform to the mesh
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    return transformFilter.GetOutput()


# All other functions moved to modules - import from there
# vtk_marching_cube_multi -> vf.vtk_marching_cube_multi
# eraseBoundary -> sf.eraseBoundary
# surface_to_image -> vf.surface_to_image
# convert_seg_to_surfs -> sf.convert_seg_to_surfs
# build_transform_matrix -> vf.build_transform_matrix
# exportSitk2VTK -> vf.exportSitk2VTK
# vtkImageResample -> vf.vtkImageResample
# vtk_marching_cube -> vf.vtk_discrete_marching_cube
# exportPython2VTK -> vf.exportPython2VTK
# smooth_polydata -> vf.smooth_polydata
# decimation -> vf.decimation
# appendPolyData -> vf.appendPolyData
# bound_polydata_by_image -> vf.bound_polydata_by_image
# convertPolyDataToImageData -> vf.convertPolyDataToImageData


if __name__ == '__main__':

    if_smooth = False
    if_keep_largest = False

    if_spacing_file = False
    spacing_file = '/Users/nsveinsson/Documents/datasets/CAS_cerebral_dataset/CAS2023_trainingdataset/meta.csv'
    
    # Filter option: only process images containing this string
    # Set to None or empty string to process all images
    filter_string = ''  # None e.g., 'aorta', '001', 'case_'

    # Let's create surfaces from segmentations
    dir_segmentations = '/Users/nsveinsson/Documents/datasets/vmr/truths/'

    img_ext = '.mha'
    img_ext_out = '.mha'
    # Which folder to write surfaces to
    out_dir = dir_segmentations.replace('truths', 'surfaces_mc/')
    try:
        os.mkdir(out_dir)
    except Exception as e:
        print(e)

    # all segmentations we have, create surfaces for each
    imgs = os.listdir(dir_segmentations)
    imgs = [img for img in imgs if img.endswith(img_ext)]
    
    # Filter images by string if specified
    if filter_string:
        original_count = len(imgs)
        imgs = [img for img in imgs if filter_string in img]
        print(f"Filtered from {original_count} to {len(imgs)} images containing '{filter_string}'")
    else:
        print(f"Processing all {len(imgs)} images")
    
    imgs.sort()
    
    if len(imgs) == 0:
        print("No images to process!")
        exit()

    if if_spacing_file:
        import pandas as pd
        spacing_df = pd.read_csv(spacing_file)
        # only keep 'spacing', they are sorted
        spacing_values = spacing_df['spacing'].values
        # read as tuples
        spacing_values = [tuple(map(float, x[1:-1].split(','))) for x in spacing_values]

    for img in imgs:
        print("Starting case: ", img)
        # Load segmentation
        seg = sitk.ReadImage(dir_segmentations+img)
        origin = seg.GetOrigin()

        if if_spacing_file:
            # set the spacing
            seg.SetSpacing(spacing_values[imgs.index(img)])
            sitk.WriteImage(seg, out_dir+img.replace(img_ext, img_ext_out))

        print(f"Image size: {seg.GetSize()}")
        print(f"Image spacing: {seg.GetSpacing()}")
        # Create surfaces
        # poly = sf.convert_seg_to_surfs(seg, new_spacing=[.5,.5,.5], target_node_num=1e5, bound=False)
        poly = vf.vtk_marching_cube_multi(vf.exportSitk2VTK(seg)[0], 0, rotate=False, center=origin)

        if if_keep_largest:
            # keep only the largest connected component
            poly = vf.get_largest_connected_polydata(poly)

        if if_smooth:
            # smooth the surface
            poly = vf.smooth_polydata(poly, iteration=50)
        # Write surfaces
        vf.write_geo(out_dir+img.replace(img_ext, '.vtp'), poly)
        print("Finished case: ", img)
    print("All done.")
