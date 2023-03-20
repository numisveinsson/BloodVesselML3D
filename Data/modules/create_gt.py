import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, get_vtk_array_type
import vtk_functions as vf
import sitk_functions as sf
import os

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
    #enclosed_filter.SetSurfaceClosed(True)
    #enclosed_filter.SetCheckSurface(True)

    enclosed_filter.SetInputData(pointsPolydata)
    enclosed_filter.SetSurfaceData(surface)
    enclosed_filter.Update()

    import pdb; pdb.set_trace()
    # Create new image to assemble

    for i in range(img_size[0]):
        print('i is ', i)
        for j in range(img_size[1]):
            for k in range(img_size[2]):
                point = image.TransformIndexToPhysicalPoint((i,j,k))
                is_inside = enclosed_filter.IsInsideSurface(point[0], point[1], point[2])
                if is_inside:
                    print('Inside')
                    image[i,j,k] = 1
                else: image[i,j,k] = 0

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

    # Let's create GT segmentations from surfaces
    dir_surfaces = '/Users/numisveinsson/Documents_numi/vmr_data_new/surfaces/'
    dir_imgs = '/Users/numisveinsson/Documents_numi/vmr_data_new/scaled_images/'
    # Which folder to write segs to
    out_dir = '/Users/numisveinsson/Documents_numi/vmr_data_new/truths/'

    # all imgs we have, create segs for them
    imgs = os.listdir(dir_imgs)
    imgs = [img for img in imgs if 'aorta.vtk' in img]

    for img in imgs:
        surf_vtp = vf.read_geo(dir_surfaces+img.replace('.vtk', '.vtp')).GetOutput()
        img_sitk = sitk.ReadImage(dir_imgs+img)
        img_vtk = exportSitk2VTK(img_sitk)[0]
        seg = convertPolyDataToImageData(surf_vtp, img_vtk)
        vf.write_img(out_dir+img.replace('.vtk', '.vti'), seg)
        vf.change_vti_vtk(out_dir+img.replace('.vtk', '.vti'))
        print("Done case: ", img)
