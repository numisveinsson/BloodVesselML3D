import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, get_vtk_array_type
import sys
import os

# Add modules directory to path
modules_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

import vtk_functions as vf


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


def eraseBoundary(labels, pixels, bg_id):
    """
    Erase anything on the boundary by a specified number of pixels
    Args:
        labels: python nd array
        pixels: number of pixel width to erase
        bg_id: id number of background class
    Returns:
        labels: editted label maps
    """
    x, y, z = labels.shape
    labels[:pixels,:,:] = bg_id
    labels[-pixels:,:,:] = bg_id
    labels[:,:pixels,:] = bg_id
    labels[:,-pixels:,:] = bg_id
    labels[:,:,:pixels] = bg_id
    labels[:,:,-pixels:] = bg_id

    return labels


def surface_to_image(mesh, image):
    """
    Find the corresponding pixel of the mesh vertices,
    create a new image delineate the surface for testing

    Args:
        mesh: VTK PolyData
        image: VTK ImageData or Sitk Image
    """
    mesh_coords = vtk_to_numpy(mesh.GetPoints().GetData())
    if type(image) == vtk.vtkImageData:
        indices = ((mesh_coords - image.GetOrigin())/image.GetSpacing()).astype(int)

        py_im = np.zeros(image.GetDimensions(), dtype=np.int32)
        for i in indices:
            py_im[i[0], i[1], i[2]] = 1

        new_image = vtk.vtkImageData()
        new_image.DeepCopy(image)
        new_image.GetPointData().SetScalars(numpy_to_vtk(py_im.flatten('F')))
    elif type(image) == sitk.Image:
        matrix = build_transform_matrix(image)
        mesh_coords = np.append(mesh_coords, np.ones((len(mesh_coords),1)),axis=1)
        matrix = np.linalg.inv(matrix)
        indices = np.matmul(matrix, mesh_coords.transpose()).transpose().astype(int)
        py_im = sitk.GetArrayFromImage(image).transpose(2,1,0).astype(np.int32)
        py_im.fill(0)  # Initialize with zeros
        for i in indices:
            py_im[i[0], i[1], i[2]] = 1
        new_image = sitk.GetImageFromArray(py_im.transpose(2,1,0))
        new_image.SetOrigin(image.GetOrigin())
        new_image.SetSpacing(image.GetSpacing())
        new_image.SetDirection(image.GetDirection())
    return new_image


def convert_seg_to_surfs(seg, new_spacing=[1., 1., 1.], target_node_num=2048, bound=False):
    py_seg = sitk.GetArrayFromImage(seg).astype(np.int32)
    py_seg = eraseBoundary(py_seg, 1, 0)
    labels = np.unique(py_seg)
    for i, l in enumerate(labels):
        py_seg[py_seg==l] = i
    seg2 = sitk.GetImageFromArray(py_seg.astype(np.int32))
    seg2.CopyInformation(seg)

    seg_vtk,_ = exportSitk2VTK(seg2)
    seg_vtk = vtkImageResample(seg_vtk,new_spacing,'NN')
    poly_l = []
    for i, _ in enumerate(labels):
        if i==0:
            continue
        p = vtk_marching_cube(seg_vtk, 0, i)
        p = smooth_polydata(p, iteration=50)
        rate = max(0., 1. - float(target_node_num)/float(p.GetNumberOfPoints()))
        p = decimation(p, rate)
        arr = np.ones(p.GetNumberOfPoints())*i
        arr_vtk = numpy_to_vtk(arr)
        arr_vtk.SetName('RegionId')
        p.GetPointData().AddArray(arr_vtk)
        poly_l.append(p)
    poly = appendPolyData(poly_l)
    if bound:
        poly = bound_polydata_by_image(seg_vtk, poly, 1.5)
    return poly


def build_transform_matrix(image):
    matrix = np.eye(4)
    matrix[:-1,:-1] = np.matmul(np.reshape(image.GetDirection(), (3,3)), np.diag(image.GetSpacing()))
    matrix[:-1,-1] = np.array(image.GetOrigin())
    return matrix


def exportSitk2VTK(sitkIm, spacing=None):
    """
    This function creates a vtk image from a simple itk image
    Args:
        sitkIm: simple itk image
    Returns:
        imageData: vtk image
        import SimpleITK as sitk
    """
    if not spacing:
        spacing = sitkIm.GetSpacing()
    img = sitk.GetArrayFromImage(sitkIm).transpose(2,1,0)
    # Ensure segmentation data is integer type
    if img.dtype == np.float64 or img.dtype == np.float32:
        # Only convert if it looks like segmentation data (contains only integer values)
        if np.allclose(img, img.astype(np.int32)):
            img = img.astype(np.int32)
    vtkArray = exportPython2VTK(img)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(sitkIm.GetSize())
    imageData.GetPointData().SetScalars(vtkArray)
    imageData.SetOrigin([0.,0.,0.])
    imageData.SetSpacing(spacing)
    matrix = build_transform_matrix(sitkIm)
    space_matrix = np.diag(list(spacing)+[1.])
    matrix = np.matmul(matrix, np.linalg.inv(space_matrix))
    matrix = np.linalg.inv(matrix)
    vtkmatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkmatrix.SetElement(i, j, matrix[i,j])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(imageData)
    reslice.SetResliceAxes(vtkmatrix)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    imageData = reslice.GetOutput()
    # imageData.SetDirectionMatrix(sitkIm.GetDirection())

    return imageData, vtkmatrix


def vtkImageResample(image, spacing, opt):
    """
    Resamples the vtk image to the given dimenstion
    Args:
        image: vtk Image data
        spacing: image new spacing
        opt: interpolation option: linear, NN, cubic
    Returns:
        image: resampled vtk image data
    """
    reslicer = vtk.vtkImageReslice()
    reslicer.SetInputData(image)
    if opt=='linear':
        reslicer.SetInterpolationModeToLinear()
    elif opt=='NN':
        reslicer.SetInterpolationModeToNearestNeighbor()
    elif opt=='cubic':
        reslicer.SetInterpolationModeToCubic()
    else:
        raise ValueError("interpolation option not recognized")

    # size = np.array(image.GetSpacing())*np.array(image.GetDimensions())
    # new_spacing = size/np.array(dims)

    reslicer.SetOutputSpacing(*spacing)
    reslicer.Update()

    return reslicer.GetOutput()


def vtk_marching_cube(vtkLabel, bg_id, seg_id, smooth=None):
    """
    Use the VTK marching cube to create isosrufaces for all classes excluding the background
    Args:
        labels: vtk image contraining the label map
        bg_id: id number of background class
        smooth: smoothing iteration
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    contour.SetValue(0, seg_id)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh


def exportPython2VTK(img):
    """
    This function creates a vtk image from a python array
    Args:
        img: python ndarray of the image
    Returns:
        imageData: vtk image
    """
    vtkArray = numpy_to_vtk(num_array=img.flatten('F'), deep=True, array_type=get_vtk_array_type(img.dtype))
    #vtkArray = numpy_to_vtk(img.flatten())
    return vtkArray


def smooth_polydata(poly, iteration=25, boundary=False, feature=False, smoothingFactor=0.):
    """
    This function smooths a vtk polydata
    Args:
        poly: vtk polydata to smooth
        boundary: boundary smooth bool
    Returns:
        smoothed: smoothed vtk polydata
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetPassBand(pow(10., -4. * smoothingFactor))
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    smoothed = smoother.GetOutput()

    return smoothed


def decimation(poly, rate):
    """
    Simplifies a VTK PolyData
    Args:
        poly: vtk PolyData
        rate: target rate reduction
    """
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(poly)
    decimate.AttributeErrorMetricOn()
    decimate.ScalarsAttributeOn()
    decimate.SetTargetReduction(rate)
    decimate.VolumePreservationOff()
    decimate.Update()
    output = decimate.GetOutput()
    #output = cleanPolyData(output, 0.)
    return output


def appendPolyData(poly_list):
    """
    Combine two VTK PolyData objects together
    Args:
        poly_list: list of polydata
    Return:
        poly: combined PolyData
    """
    appendFilter = vtk.vtkAppendPolyData()
    for poly in poly_list:
        appendFilter.AddInputData(poly)
    appendFilter.Update()
    out = appendFilter.GetOutput()
    return out


def bound_polydata_by_image(image, poly, threshold):
    bound = vtk.vtkBox()
    image.ComputeBounds()
    b_bound = image.GetBounds()
    b_bound = [b+threshold if (i % 2) ==0 else b-threshold for i, b in enumerate(b_bound)]
    #print("Bounding box: ", b_bound)
    bound.SetBounds(b_bound)
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(bound)
    clipper.SetInputData(poly)
    clipper.InsideOutOn()
    clipper.Update()
    return clipper.GetOutput()


def convertPolyDataToImageData(poly, ref_im):
    """
    Convert the vtk polydata to imagedata
    Args:
        poly: vtkPolyData
        ref_im: reference vtkImage to match the polydata with
    Returns:
        output: resulted vtkImageData
    """

    ref_im.GetPointData().SetScalars(numpy_to_vtk(np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape, dtype=np.int32)))
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.05)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing())
    ply2im.SetInformationInput(ref_im)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im)
    stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    output = stencil.GetOutput()

    # Convert output to integer type
    output_array = vtk_to_numpy(output.GetPointData().GetScalars()).astype(np.int32)
    output.GetPointData().SetScalars(numpy_to_vtk(output_array))

    return output


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
        img_vtk = exportSitk2VTK(img_sitk)[0]
        # img_vtk = vf.read_img(dir_imgs+img).GetOutput()
        # seg = convertPolyDataToImageData(surf_vtp, img_vtk)
        seg = convertPolyDataToImageData(surf_vtp, img_vtk)
        
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
