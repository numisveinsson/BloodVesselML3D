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


def create_seg_from_surface(surface, image):
    """
    Check all voxels:
    if voxel inside surface: voxel = 1
    if outside: voxel = 0
    Args:
        surface: VTK PolyData
        image: Sitk Image
    """

    # Assemble all points in image
    img_size = image.GetSize()
    points = vtk.vtkPoints()
    count = 0
    for i in range(img_size[0]):
        print('i is ', i)
        for j in range(img_size[1]):
            for k in range(img_size[2]):
                point = image.TransformIndexToPhysicalPoint((i,j,k))
                points.InsertNextPoint(point)
                count += 1

    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)

    # Create filter to check inside/outside
    enclosed_filter = vtk.vtkSelectEnclosedPoints()
    enclosed_filter.SetTolerance(0.001)
    # enclosed_filter.SetSurfaceClosed(True)
    # enclosed_filter.SetCheckSurface(True)

    enclosed_filter.SetInputData(pointsPolydata)
    enclosed_filter.SetSurfaceData(surface)
    enclosed_filter.Update()

    # Create new image to assemble

    for i in range(img_size[0]):
        print('i is ', i)
        for j in range(img_size[1]):
            for k in range(img_size[2]):
                point = image.TransformIndexToPhysicalPoint((i,j,k))
                is_inside = enclosed_filter.IsInsideSurface(point[0], point[1], point[2])
                if is_inside:
                    print('Inside')
                    image[i, j, k] = 1
                else:
                    image[i, j, k] = 0

    import pdb; pdb.set_trace()
    return image


# All functions moved to modules - import from there
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

    # Let's create GT segmentations from surfaces
    img_ext = '.mha'
    output_ext = '.mha'  # Output format: '.vti', '.mha', '.nii.gz', etc.
    dir_surfaces = '/Users/nsveinsson/Documents/datasets/ASOCA_dataset/cm/surfaces/'
    dir_imgs = '/Users/nsveinsson/Documents/datasets/ASOCA_dataset/cm/images/'
    # Which folder to write segs to
    out_dir = '/Users/nsveinsson/Documents/datasets/ASOCA_dataset/cm/truths/'

    # create output directory if it does not exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # all imgs we have, create segs for them
    imgs = os.listdir(dir_imgs)
    imgs = [img for img in imgs if img.endswith(img_ext)]
    # import pdb; pdb.set_trace()
    for img in imgs:
        # Check for both .vtp and .stl surface files
        surf_path_vtp = dir_surfaces + img.replace(img_ext, '.vtp')
        surf_path_stl = dir_surfaces + img.replace(img_ext, '.stl')
        
        # Determine which surface file exists
        if os.path.exists(surf_path_vtp):
            surf_path = surf_path_vtp
        elif os.path.exists(surf_path_stl):
            surf_path = surf_path_stl
        else:
            print(f"Skipping case {img}: No surface file (.vtp or .stl) found")
            continue
        
        output_path = out_dir + img.replace(img_ext, output_ext)

        # Check if output file already exists
        if os.path.exists(output_path):
            print(f"Skipping case {img}: Output file {output_path} already exists")
            continue
        
        # Read surface based on file type
        if surf_path.endswith('.stl'):
            reader = vtk.vtkSTLReader()
            reader.SetFileName(surf_path)
            reader.Update()
            surf_vtp = reader.GetOutput()
        else:  # .vtp file
            surf_vtp = vf.read_geo(surf_path).GetOutput()
        
        img_sitk = sitk.ReadImage(dir_imgs+img)
        img_vtk = vf.exportSitk2VTK(img_sitk)[0]
        # img_vtk = vf.read_img(dir_imgs+img).GetOutput()
        # seg = vf.convertPolyDataToImageData(surf_vtp, img_vtk)
        seg = vf.convertPolyDataToImageData(surf_vtp, img_vtk)
        
        # Write output in the specified format
        if output_ext == '.vti':
            vf.write_img(output_path, seg)
        else:
            # Convert VTK image to SITK and save
            # Get numpy array from VTK image
            vtk_array = vtk_to_numpy(seg.GetPointData().GetScalars())
            dims = seg.GetDimensions()
            vtk_array = vtk_array.reshape(dims, order='F')
            
            # Create SITK image from numpy array
            seg_sitk = sitk.GetImageFromArray(vtk_array.transpose(2, 1, 0))
            
            # Copy metadata from original image
            seg_sitk.SetOrigin(img_sitk.GetOrigin())
            seg_sitk.SetSpacing(img_sitk.GetSpacing())
            seg_sitk.SetDirection(img_sitk.GetDirection())
            
            seg_sitk = sitk.Cast(seg_sitk, sitk.sitkUInt8)
            sitk.WriteImage(seg_sitk, output_path)
        
        # vf.change_vti_vtk(output_path)
        print("Done case: ", img)
