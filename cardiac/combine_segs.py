import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, get_vtk_array_type
import numpy as np
import os
import SimpleITK as sitk
# now import the functions from modules/vtk_functions.py
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import modules.vtk_functions as vf

def vtk_marching_cube_multi(vtkLabel, bg_id, smooth=None):
    """
    Use the VTK marching cube to create isosrufaces for all classes excluding the background
    Args:
        labels: vtk image containing the label map
        bg_id: id number of background class
        smooth: smoothing iteration
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    ids = np.unique(vtk_to_numpy(vtkLabel.GetPointData().GetScalars()))
    ids = np.delete(ids, np.where(ids==bg_id))

    #smooth the label map
    #vtkLabel = utils.gaussianSmoothImage(vtkLabel, 2.)

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkLabel)
    for index, i in enumerate(ids):
        print("Setting iso-contour value: ", i)
        contour.SetValue(index, i)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh

def multiclass_convert_polydata_to_imagedata(poly, ref_im):
    poly = get_all_connected_polydata(poly)
    out_im_py = np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape)
    c = 0
    poly_i = thresholdPolyData(poly, 'RegionId', (c, c), 'point')
    while poly_i.GetNumberOfPoints() > 0:
        poly_im = convertPolyDataToImageData(poly_i, ref_im)
        poly_im_py = vtk_to_numpy(poly_im.GetPointData().GetScalars())
        mask = (poly_im_py==1) & (out_im_py==0) if c==6 else poly_im_py==1
        out_im_py[mask] = c+1
        c+=1
        poly_i = thresholdPolyData(poly, 'RegionId', (c, c), 'point')
    im = vtk.vtkImageData()
    im.DeepCopy(ref_im)
    im.GetPointData().SetScalars(numpy_to_vtk(out_im_py))
    return im

def get_all_connected_polydata(poly):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    #connectivity.MarkVisitedPointIdsOn()
    connectivity.Update()
    poly = connectivity.GetOutput()
    return poly

def thresholdPolyData(poly, attr, threshold, mode):
    """
    Get the polydata after thresholding based on the input attribute
    Args:
        poly: vtk PolyData to apply threshold
        atrr: attribute of the cell array
        threshold: (min, max)
    Returns:
        output: resulted vtk PolyData
    """
    surface_thresh = vtk.vtkThreshold()
    surface_thresh.SetInputData(poly)
    surface_thresh.ThresholdBetween(*threshold)
    if mode=='cell':
        surface_thresh.SetInputArrayToProcess(0, 0, 0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, attr)
    else:
        surface_thresh.SetInputArrayToProcess(0, 0, 0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, attr)
    surface_thresh.Update()
    surf_filter = vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetInputData(surface_thresh.GetOutput())
    surf_filter.Update()
    return surf_filter.GetOutput()

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

def combine_segs_aorta_keep(segmentation, vascular, label = 6, vascular_label = 1, keep_labels = [1,3,8]):
    """
    Combine the segmentation with the vascular segmentation
    Args:
        segmentation: vtkImageData, segmentation of the cardiac mesh
        vascular: vtkImageData, segmentation of the vascular mesh
        label: int, label of the cardiac segmentation
        vascular_label: int, label of the vascular segmentation
        keep_labels: list of int, labels to keep from the original segmentation
    Returns:
        output: vtkImageData, combined segmentation
    """
    # seg_old = vtk_to_numpy(segmentation.GetPointData().GetScalars())
    seg_new = vtk_to_numpy(segmentation.GetPointData().GetScalars())
    vas = vtk_to_numpy(vascular.GetPointData().GetScalars())
    
    # remove the vascular label pixels that are in keep_labels
    for l in keep_labels:
        vas[seg_new==l] = 0

    seg_new[vas==vascular_label] = label

    segmentation.GetPointData().SetScalars(numpy_to_vtk(seg_new))

    return segmentation

def get_bounding_box(seg_new, valve_label):
    """
    Get the bounding box of the aorta valve in the segmentation
    Args:
        seg_new: numpy array, segmentation of the cardiac mesh
        valve_label: int, label of the aorta valve
    Returns:
        bounds: list, bounding box of the aorta valve
    """
    # Get the indices of the aorta valve
    indices = np.where(seg_new==valve_label)
    # Get the bounds of the aorta valve
    bounds = [np.min(indices[0]), np.max(indices[0]), np.min(indices[1]), np.max(indices[1]), np.min(indices[2]), np.max(indices[2])]
    return bounds

def combing_segs_aorta_area(segmentation, vascular, label = 6, vascular_label = 1, valve_label = 8, keep_labels = [1,3,8]):
    """
    Combine the segmentation with the vascular segmentation
    It goes as follows:
    1. Get the bounding box of the aorta valve in segmentation, label is valve_label
    2. Remove the vascular segmentation inside the bounding box, set it to 0
    3. Combine the segmentations

    Args:
        segmentation: vtkImageData, segmentation of the cardiac mesh
        vascular: vtkImageData, segmentation of the vascular mesh
        label: int, label of the cardiac segmentation
        vascular_label: int, label of the vascular segmentation
        valve_label: int, label of the aorta valve
    Returns:
        output: vtkImageData, combined segmentation
    """
    # Create numpy arrays from the segmentations
    seg_new = vtk_to_numpy(segmentation.GetPointData().GetScalars())
    vas = vtk_to_numpy(vascular.GetPointData().GetScalars())

    # Reshape the segmentations to 3D
    dims = segmentation.GetDimensions()
    # Flip the dimensions
    dims = (dims[2], dims[1], dims[0])
    print(f"Dimensions of the segmentations: {dims}")

    seg_new = seg_new.reshape(dims)
    vas = vas.reshape(dims)
    print(f"Number of pixels in vas as vascular label before: {np.sum(vas==vascular_label)}")

    # Get the bounding box of the aorta valve
    bounds = get_bounding_box(seg_new, valve_label)
    print(f"Bounding box of the aorta valve: {bounds}")

    # Add the N pixels to the bounds, N/2 to z
    N = 30
    bounds[0] = max(0, bounds[0]-N//2)
    bounds[1] = min(dims[0]-1, bounds[1]+N//2)
    bounds[2] = max(0, bounds[2]-N)
    bounds[3] = min(dims[1]-1, bounds[3]+N)
    bounds[4] = max(0, bounds[4]-N)
    bounds[5] = min(dims[2]-1, bounds[5]+N)
    print(f"Bounding box of the aorta valve with N pixels added: {bounds}")

    # remove the vascular label pixels that are inside the bounding box
    vas[bounds[0]:bounds[1], bounds[2]:bounds[3], bounds[4]:bounds[5]] = 2
    print(f"Number of pixels in vas as vascular label after: {np.sum(vas==vascular_label)}")

    # remove the vascular label pixels that are in keep_labels
    for l in keep_labels:
        vas[seg_new==l] = 0

    # Combine the segmentations
    seg_new[vas==vascular_label] = label

    # Make valve label also vascular label
    seg_new[seg_new==valve_label] = label

    # Reshape the segmentation to 1D
    seg_new = seg_new.reshape(-1)
    vas = vas.reshape(-1)

    # Set the new segmentation to the vtkImageData
    segmentation.GetPointData().SetScalars(numpy_to_vtk(seg_new))
    vascular.GetPointData().SetScalars(numpy_to_vtk(vas))

    return segmentation, vascular

def combine_blood_aorta(combined_seg, labels_keep = [3,6]):
    """
    This function takes in a combined segmentation and creates a blood pool and aorta polydata
    Blood pool and aorta are given by labels 3 and 6
    We only keep the labels 3 and 6 in the combined segmentation

    Args:
        combined_seg: vtkImageData, combined segmentation
        labels_keep: list, labels to keep
    Returns:
        blood_aorta: vtkPolyData, blood pool and aorta polydata
    """
    # Create a polydata from the combined segmentation
    poly = vtk_marching_cube_multi(combined_seg, 0)

    # Get the labels
    labels = np.unique(vtk_to_numpy(combined_seg.GetPointData().GetScalars()))
    labels = [l for l in labels if l in labels_keep]

    # Create a new segmentation with only the labels 3 and 6
    combined_seg_new = vtk.vtkImageData()
    combined_seg_new.DeepCopy(combined_seg)
    seg_new = vtk_to_numpy(combined_seg_new.GetPointData().GetScalars())
    seg_new[~np.isin(seg_new, labels)] = 0
    combined_seg_new.GetPointData().SetScalars(numpy_to_vtk(seg_new))

    # Create a polydata from the new combined segmentation
    poly = vtk_marching_cube_multi(combined_seg_new, 0)

    return poly

if __name__ == "__main__":

    # Directory of cardiac meshes polydata
    directory = '/Users/numisveins/Documents/Heartflow/output_cardiac-meshes/'
    meshes = os.listdir(directory)
    meshes = [f for f in meshes if f.endswith('.vtp')]
    # sort meshes
    meshes = sorted(meshes)

    # Directory of cardiac images
    img_dir = directory
    img_ext = '.vti'
    imgs = os.listdir(img_dir)
    imgs = [f for f in imgs if f.endswith(img_ext)]
    # only keep images that have corresponding meshes
    imgs = [f for f in imgs if f.replace(img_ext, '.vtp') in meshes]
    # sort images
    imgs = sorted(imgs)

    # Directory of vascular segmentations
    vascular_dir = '/Users/numisveins/Documents/Heartflow/output_cardiac_2000_steps/new_format/'
    vascular_ext = '.vti'
    vascular_imgs = os.listdir(vascular_dir)
    vascular_imgs = [f for f in vascular_imgs if f.endswith(vascular_ext)]
    vascular_imgs = [f.replace('_seg_rem_3d_fullres_0', '') for f in vascular_imgs]
    vascular_imgs = [f for f in vascular_imgs if f in imgs]
    # sort vascular segmentations
    vascular_imgs = sorted(vascular_imgs)

    # Only keep the images that have corresponding vascular segmentations and meshes
    imgs = [f for f in imgs if f in vascular_imgs]
    meshes = [f for f in meshes if f.replace('.vtp', '.vti') in vascular_imgs]
    print(f"Number of meshes: {len(meshes)}")
    print(f"Number of images: {len(imgs)}")
    print(f"Number of vascular segmentations: {len(vascular_imgs)}")

    vascular_imgs = [vascular_dir+f for f in vascular_imgs]
    imgs = [img_dir+f for f in imgs]
    meshes = [directory+f for f in meshes]

    # create folder for segmentations
    segs_dir = directory + 'segmentations/'
    if not os.path.exists(segs_dir):
        os.makedirs(segs_dir)

    for i in range(len(imgs)):
        # Read image
        img = vf.read_img(imgs[i]).GetOutput()
        print(f"Processing image {imgs[i]}")
        
        # Read polydata
        poly = vf.read_geo(meshes[i]).GetOutput()
        print(f"Processing mesh {meshes[i]}")

        # Convert polydata to imagedata
        segmentation = multiclass_convert_polydata_to_imagedata(poly, img)
        # Save segmentation
        vf.write_img(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_seg.vti'), segmentation)

        # Create a polydata from the segmentation
        poly = vtk_marching_cube_multi(segmentation, 0)

        # Save polydata
        vf.write_geo(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_seg.vtp'), poly)

        # Read vascular segmentation
        vascular = vf.read_img(vascular_imgs[i]).GetOutput()
        print(f"Processing vascular segmentation {vascular_imgs[i]}")

        # Now we need to combine the segmentations
        # We will combine the vascular label 1 with the segmentation label 6
        # combined_seg = combine_segs_aorta_keep(segmentation, vascular)

        # # Save combined segmentation
        # vf.write_img(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_seg_combined.vti'), combined_seg)

        # # Create a polydata from the combined segmentation
        # poly = vtk_marching_cube_multi(combined_seg, 0)

        # # Save polydata
        # vf.write_geo(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_seg_combined.vtp'), poly)

        # Now we will combine the segmentations by keeping the original segmentation around the aorta valve
        combined_seg, new_vasc = combing_segs_aorta_area(segmentation, vascular)

        # Save combined segmentation
        vf.write_img(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_seg_combined_area.vti'), combined_seg)
        vf.write_img(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_vasc_combined_area.vti'), new_vasc)

        # Create a polydata from the combined segmentation
        poly = vtk_marching_cube_multi(combined_seg, 0)

        # Save polydata
        vf.write_geo(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_seg_combined_area.vtp'), poly)

        # Smooth the polydata
        poly = smooth_polydata(poly, iteration=25, boundary=False, feature=False, smoothingFactor=0.5)

        # Save smoothed polydata
        vf.write_geo(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_seg_combined_area_smoothed.vtp'), poly)

        # Create blood pool and aorta mesh
        combined_seg_blood_aorta = combine_blood_aorta(combined_seg)

        # Save blood pool and aorta mesh
        vf.write_geo(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_seg_combined_area_blood_aorta.vtp'), combined_seg_blood_aorta)

        # Save smoothed blood pool and aorta mesh
        combined_seg_blood_aorta = smooth_polydata(combined_seg_blood_aorta, iteration=25, boundary=False, feature=False, smoothingFactor=0.5)
        vf.write_geo(segs_dir + imgs[i].split('/')[-1].replace('.vti', '_seg_combined_area_blood_aorta_smoothed.vtp'), combined_seg_blood_aorta)