import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, get_vtk_array_type
import sys
import os
# add the path to the modules ../modules/
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'modules'))
import vtk_functions as vf


def vtk_marching_cube_multi(vtkLabel, bg_id, smooth=None, rotate=False, center=None):
    """
    Note: another function from Arjun's code

    Use the VTK marching cube to create isosrufaces
    for all classes excluding the background
    Args:
        labels: vtk image containing the label map
        bg_id: id number of background class
        smooth: smoothing iteration
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    ids = np.unique(vtk_to_numpy(vtkLabel.GetPointData().GetScalars()))
    ids = np.delete(ids, np.where(ids == bg_id))

    # smooth the label map
    # vtkLabel = utils.gaussianSmoothImage(vtkLabel, 2.)

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    for index, i in enumerate(ids):
        print("Setting iso-contour value: ", i)
        contour.SetValue(index, i)
    contour.Update()
    mesh = contour.GetOutput()

    if rotate:
        mesh = rotate_mesh(mesh, vtkLabel, center=None)

    return mesh


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
    # Get the origin of the image
    origin = vtkLabel.GetOrigin()
    # Get the center of the image
    if center is None:
        center = vtkLabel.GetCenter()
    # Get the spacing of the image
    spacing = vtkLabel.GetSpacing()

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
    x,y,z = labels.shape
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

        py_im = np.zeros(image.GetDimensions())
        for i in indices:
            py_im[i[0], i[1], i[2]] = 1.

        new_image = vtk.vtkImageData()
        new_image.DeepCopy(image)
        new_image.GetPointData().SetScalars(numpy_to_vtk(py_im.flatten('F')))
    elif type(image) == sitk.Image:
        matrix = build_transform_matrix(image)
        mesh_coords = np.append(mesh_coords, np.ones((len(mesh_coords),1)),axis=1)
        matrix = np.linalg.inv(matrix)
        indices = np.matmul(matrix, mesh_coords.transpose()).transpose().astype(int)
        py_im = sitk.GetArrayFromImage(image).transpose(2,1,0)
        for i in indices:
            py_im[i[0], i[1], i[2]] = 1000.
        new_image = sitk.GetImageFromArray(py_im.transpose(2,1,0))
        new_image.SetOrigin(image.GetOrigin())
        new_image.SetSpacing(image.GetSpacing())
        new_image.SetDirection(image.GetDirection())
    return new_image

def convert_seg_to_surfs(seg, new_spacing=[1.,1.,1.], target_node_num=2048, bound=False):
    py_seg = sitk.GetArrayFromImage(seg)
    py_seg = eraseBoundary(py_seg, 1, 0)
    labels = np.unique(py_seg)
    for i, l in enumerate(labels):
        py_seg[py_seg==l] = i
    seg2 = sitk.GetImageFromArray(py_seg)
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
        # p = smooth_polydata(p, iteration=50)
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

def exportSitk2VTK(sitkIm,spacing=None):
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
    #imageData.SetDirectionMatrix(sitkIm.GetDirection())


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

    #size = np.array(image.GetSpacing())*np.array(image.GetDimensions())
    #new_spacing = size/np.array(dims)

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

def smooth_polydata(poly, iteration=25, boundary=False, feature=False, smoothingFactor=0.4):
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

    ref_im.GetPointData().SetScalars(numpy_to_vtk(np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape)))
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

    return output

if __name__=='__main__':

    if_smooth = False
    if_keep_largest = True

    if_spacing_file = False
    spacing_file = '/Users/numisveins/Documents/datasets/CAS_dataset/CAS2023_trainingdataset/meta.csv'

    # Let's create surfaces from segmentations
    dir_segmentations = '/Users/numisveins/Documents/aortaseg24/process_binary/binary_segs/'
    dir_segmentations = '/Users/numisveins/Downloads/segmentation/new_format/'
    dir_segmentations = '/Users/numisveins/Documents/data_combo_paper/ct_data/Ground truth cardiac segmentations/'
    dir_segmentations = '/Users/numisveins/Documents/datasets/CAS_dataset/CAS2023_trainingdataset/truths/'
    dir_segmentations = '/Users/numisveins/Documents/datasets/ASOCA_dataset/truths/'
    img_ext = '.nrrd'
    # Which folder to write surfaces to
    out_dir = dir_segmentations + 'surfaces_largest/'
    try:
        os.mkdir(out_dir)
    except Exception as e:
        print(e)

    # all segmentations we have, create surfaces for each
    imgs = os.listdir(dir_segmentations)
    imgs = [img for img in imgs if img.endswith(img_ext)]
    imgs.sort()

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
        print(f"Image size: {seg.GetSize()}")
        print(f"Image spacing: {seg.GetSpacing()}")
        # Create surfaces
        # poly = convert_seg_to_surfs(seg, new_spacing=[.5,.5,.5], target_node_num=1e5, bound=False)
        poly = vtk_marching_cube_multi(exportSitk2VTK(seg)[0], 0, rotate=False, center=origin)

        if if_keep_largest:
            # keep only the largest connected component
            poly = vf.get_largest_connected_polydata(poly)

        if if_smooth:
            # smooth the surface
            poly = smooth_polydata(poly, iteration=50)
        # Write surfaces
        vf.write_geo(out_dir+img.replace(img_ext, '.vtp'), poly)
        print("Finished case: ", img)
    print("All done.")
