import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import os
import sys
from os.path import dirname, join, abspath

# Add modules directory to path
modules_path = join(dirname(dirname(abspath(__file__))), 'modules')
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

import vtk_functions as vf


# vtk_marching_cube_multi moved to vf.vtk_marching_cube_multi


def multiclass_convert_polydata_to_imagedata(poly, ref_im):
    poly = get_all_connected_polydata(poly)
    out_im_py = np.zeros(vtk_to_numpy(
                ref_im.GetPointData().GetScalars()).shape)
    c = 0
    poly_i = vf.thresholdPolyData(poly, 'RegionId', (c, c), 'point')
    while poly_i.GetNumberOfPoints() > 0:
        poly_im = vf.convertPolyDataToImageData(poly_i, ref_im)
        poly_im_py = vtk_to_numpy(poly_im.GetPointData().GetScalars())
        mask = ((poly_im_py == 1) & (out_im_py == 0)
                if c == 6 else poly_im_py == 1)
        out_im_py[mask] = c + 1
        c += 1
        poly_i = vf.thresholdPolyData(poly, 'RegionId', (c, c), 'point')
    im = vtk.vtkImageData()
    im.DeepCopy(ref_im)
    im.GetPointData().SetScalars(numpy_to_vtk(out_im_py))
    return im


def get_all_connected_polydata(poly):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    # connectivity.MarkVisitedPointIdsOn()
    connectivity.Update()
    poly = connectivity.GetOutput()
    return poly


# thresholdPolyData moved to vf.thresholdPolyData


# smooth_polydata moved to vf.smooth_polydata


# convertPolyDataToImageData moved to vf.convertPolyDataToImageData


def combine_segs_aorta_keep(segmentation, vascular, label=6,
                            vascular_label=1, keep_labels=[1, 3, 8]):
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
    for label0 in keep_labels:
        vas[seg_new == label0] = 0

    seg_new[vas == vascular_label] = label

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
    indices = np.where(seg_new == valve_label)
    # Get the bounds of the aorta valve
    bounds = [np.min(indices[0]), np.max(indices[0]), np.min(indices[1]),
              np.max(indices[1]), np.min(indices[2]), np.max(indices[2])]

    return bounds


def combing_segs_aorta_area(segmentation, vascular, label=6, vascular_label=1,
                            valve_label=8, keep_labels=[1, 3, 8]):
    """
    Combine the segmentation with the vascular segmentation
    It goes as follows:
    1. Get the bounding box of the aorta valve in segmentation,
       label is valve_label
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
    print(f"""Number of pixels in vas as vascular label before:
          {np.sum(vas==vascular_label)}""")

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
    print(f"""Number of pixels in vas as vascular label after:
          {np.sum(vas==vascular_label)}""")

    # remove the vascular label pixels that are in keep_labels
    for label0 in keep_labels:
        vas[seg_new == label0] = 0

    # Combine the segmentations
    seg_new[vas == vascular_label] = label

    # Make valve label also vascular label
    seg_new[seg_new == valve_label] = label

    # Reshape the segmentation to 1D
    seg_new = seg_new.reshape(-1)
    vas = vas.reshape(-1)

    # Set the new segmentation to the vtkImageData
    segmentation.GetPointData().SetScalars(numpy_to_vtk(seg_new))
    vascular.GetPointData().SetScalars(numpy_to_vtk(vas))

    return segmentation, vascular


def combine_blood_aorta(combined_seg, labels_keep=[3, 6]):
    """
    This function takes in a combined segmentation and creates a blood pool
    and aorta polydata
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
    labels = [label for label in labels if label in labels_keep]

    # Create a new segmentation with only the labels in labels_keep
    combined_seg_new = vtk.vtkImageData()
    combined_seg_new.DeepCopy(combined_seg)
    seg_new = vtk_to_numpy(combined_seg_new.GetPointData().GetScalars())
    seg_new[~np.isin(seg_new, labels)] = 0
    combined_seg_new.GetPointData().SetScalars(numpy_to_vtk(seg_new))

    # Create a polydata from the new combined segmentation
    poly = vtk_marching_cube_multi(combined_seg_new, 0)

    return poly, combined_seg_new


def fully_combine_blood_aorta(combined_seg_blood_aorta_vti,
                              combine_labels=[3, 6]):
    """
    Combine the blood pool and aorta into one label
    Args:
        combined_seg_blood_aorta_vti: vtkImageData,
                    blood pool and aorta segmentation
        combine_labels: list, labels to combine
    Returns:
        combined_seg_blood_aorta_vti: vtkImageData, combined blood pool
                    and aorta segmentation
    """
    seg = vtk_to_numpy(combined_seg_blood_aorta_vti.GetPointData().GetScalars())
    seg = np.where(np.isin(seg, combine_labels), 1, seg)
    combined_seg_blood_aorta_vti.GetPointData().SetScalars(numpy_to_vtk(seg))

    # create a polydata from the combined segmentation
    poly = vf.vtk_marching_cube_multi(combined_seg_blood_aorta_vti, 0)

    # define normals for the polydata
    poly = vtk_normals(poly)

    return poly, combined_seg_blood_aorta_vti


def vtk_normals(poly):
    """
    This function calculates the normals of a vtk polydata
    Args:
        poly: vtkPolyData, polydata to calculate normals
    Returns:
        poly: vtkPolyData, polydata with normals
    """
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.FlipNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()

    return normals.GetOutput()


def remove_cells_with_region_3(polydata):
    # Create an empty vtkPolyData object to store the filtered cells
    filtered_polydata = vtk.vtkPolyData()
    filtered_cells = vtk.vtkCellArray()
    filtered_points = vtk.vtkPoints()
    filtered_scalars = vtk.vtkFloatArray()

    # Get the scalars (labels) from the original polydata
    scalars = polydata.GetCellData().GetScalars()

    # Map to track the original point indices to new indices
    point_map = {}

    # Iterate through all cells in the polydata
    for cell_id in range(polydata.GetNumberOfCells()):
        # Get the label of the current cell
        label = scalars.GetTuple1(cell_id)

        # If the cell's label is not region 3, keep it
        if label != 3:
            cell = polydata.GetCell(cell_id)
            cell_points = cell.GetPoints()

            # Add points to the filtered points list
            new_cell_point_ids = []
            for i in range(cell_points.GetNumberOfPoints()):
                point = cell_points.GetPoint(i)
                point_id = polydata.FindPoint(point)

                # Check if the point has already been added to the new points list
                if point_id in point_map:
                    new_point_id = point_map[point_id]
                else:
                    new_point_id = filtered_points.InsertNextPoint(point)
                    point_map[point_id] = new_point_id

                new_cell_point_ids.append(new_point_id)

            # Create the new cell with the filtered points
            filtered_cells.InsertNextCell(len(new_cell_point_ids), new_cell_point_ids)
            filtered_scalars.InsertNextValue(label)

    # Set the points, cells, and scalars in the filtered_polydata
    filtered_polydata.SetPoints(filtered_points)
    filtered_polydata.SetPolys(filtered_cells)  # or SetLines, SetVerts depending on cell type
    filtered_polydata.GetCellData().SetScalars(filtered_scalars)

    return filtered_polydata


def update_labels_based_on_polydata2(polydata1, polydata2):
    # Remove cells with region 3
    polydata2 = remove_cells_with_region_3(polydata2)

    # Get the scalars (labels) from both polydata1 and polydata2
    labels_polydata1 = polydata1.GetCellData().GetScalars()
    labels_polydata2 = polydata2.GetCellData().GetScalars()

    # Create a point locator for polydata2 to quickly find corresponding points
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(polydata2)
    point_locator.BuildLocator()

    # Iterate through all cells in polydata1
    for cell_id in range(polydata1.GetNumberOfCells()):
        # Get the label of the current cell in polydata1
        label_polydata1 = labels_polydata1.GetTuple1(cell_id)

        # Process only cells labeled as region 3 in polydata1
        if label_polydata1 == 3:
            cell = polydata1.GetCell(cell_id)
            points = cell.GetPoints()

            # Assume that the first point can be used to find the corresponding cell
            match_found = False
            for point_id in range(points.GetNumberOfPoints()):
                point = points.GetPoint(point_id)
                closest_point_id = point_locator.FindClosestPoint(point)

                if closest_point_id >= 0:
                    # Get the cell in polydata2 that contains this point
                    cell_ids = vtk.vtkIdList()
                    polydata2.GetPointCells(closest_point_id, cell_ids)

                    for i in range(cell_ids.GetNumberOfIds()):
                        corresponding_cell_id = cell_ids.GetId(i)
                        label_polydata2 = labels_polydata2.GetTuple1(corresponding_cell_id)

                        if label_polydata2 in [2, 6]:
                            # If a matching cell is found in polydata2, update the label in polydata1
                            if label_polydata2 == 2:
                                labels_polydata1.SetTuple1(cell_id, 9)
                            elif label_polydata2 == 6:
                                labels_polydata1.SetTuple1(cell_id, 10)
                            match_found = True
                            break

                    if match_found:
                        break  # Stop checking further points if a match is found

    # Update the polydata
    polydata1.Modified()

    # Smooth the polydata
    polydata1 = vf.smooth_polydata(polydata1, iteration=25, boundary=False,
                                feature=False, smoothingFactor=0.5)

    # Rename scalar array to 'ModelFaceID'
    polydata1.GetCellData().GetScalars().SetName('ModelFaceID')

    # Make 'ModelFaceID' array an integer array
    polydata1 = convert_modelfaceid_to_int(polydata1)

    # Add new 'CapID' array that is the same as 'ModelFaceID' except scaled to start from 1
    polydata1 = add_cap_id(polydata1)

    # Define normals for the polydata
    # for model_face_id in [1, 2, 3]:
    #     polydata1 = ensure_outward_normals(polydata1, model_face_id)

    return polydata1


def convert_modelfaceid_to_int(polydata):
    # Get the scalars (labels) from the polydata
    labels = polydata.GetCellData().GetScalars()

    # Create a new array to store the rescaled labels
    model_face_id = vtk.vtkIntArray()
    model_face_id.SetName('CapID')
    model_face_id.SetNumberOfComponents(1)
    model_face_id.SetNumberOfTuples(labels.GetNumberOfTuples())

    # Rescale the labels to start from 1 and increment by 1
    for i in range(labels.GetNumberOfTuples()):
        label = labels.GetTuple1(i)
        model_face_id.SetTuple1(i, int(label))

    # Set the new labels in the polydata
    polydata.GetCellData().AddArray(model_face_id)

    return polydata


def add_cap_id(polydata):
    # Get the scalars (labels) from the polydata
    labels = polydata.GetCellData().GetScalars()

    # Create a new array to store the rescaled labels
    cap_id = vtk.vtkIntArray()
    cap_id.SetName('ModelFaceID')
    cap_id.SetNumberOfComponents(1)
    cap_id.SetNumberOfTuples(labels.GetNumberOfTuples())

    # Define a function to rescale the labels
    def label_id(label):
        # 3, 6, 9, 10
        if label == 3:
            return 1
        elif label == 6:
            return 1
        elif label == 9:
            return 3
        elif label == 10:
            return 2
        else:
            return 0

    # Rescale the labels to start from 1 and increment by 1
    for i in range(labels.GetNumberOfTuples()):
        label = labels.GetTuple1(i)
        cap_id.SetTuple1(i, label_id(label))

    # Set the new labels in the polydata
    polydata.GetCellData().AddArray(cap_id)

    return polydata


# bound_polydata_by_image_extended moved to vf.bound_polydata_by_image_extended
# define_bounding_box moved to vf.define_bounding_box
# get_threshold moved to vf.get_threshold



if __name__ == "__main__":

    write_all = True
    no_valve = True

    # Directory of cardiac meshes polydata
    directory = '/Users/numisveins/Documents/Heartflow/output_cardiac-meshes/'
    directory = '/Users/numisveins/Documents/data_combo_paper/ct_data/meshes/'
    meshes = os.listdir(directory)
    meshes = [f for f in meshes if f.endswith('.vtp')]
    # sort meshes
    meshes = sorted(meshes)

    # Directory of images
    img_dir = '/Users/numisveins/Documents/data_combo_paper/ct_data/images_vti/'
    img_ext = '.vti'
    imgs = os.listdir(img_dir)
    imgs = [f for f in imgs if f.endswith(img_ext)]
    # only keep images that have corresponding meshes
    imgs = [f for f in imgs if f.replace(img_ext, '.vtp') in meshes]
    # sort images
    imgs = sorted(imgs)

    # Directory of vascular segmentations
    vascular_dir = """/Users/numisveins/Documents/Heartflow/
                   output_cardiac_2000_steps/new_format/"""
    vascular_dir = '/Users/numisveins/Documents/data_combo_paper/ct_data/vascular_segs/vascular_segs_vti/'
    vascular_ext = '.vti'
    vascular_imgs = os.listdir(vascular_dir)
    vascular_imgs = [f for f in vascular_imgs if f.endswith(vascular_ext)]
    # vascular_imgs = [f.replace('_seg_rem_3d_fullres_0', '')
    #                  for f in vascular_imgs]
    # import pdb; pdb.set_trace()

    vascular_imgs = [f for f in vascular_imgs if f.replace(vascular_ext, img_ext).replace('_seg_rem_3d_fullres_0', '') in imgs]
    # sort vascular segmentations
    vascular_imgs = sorted(vascular_imgs)

    # Only keep the images that have corresponding vascular
    # segmentations and meshes
    # imgs = [f for f in imgs if f in vascular_imgs]
    meshes = [f for f in meshes if f.replace('.vtp', img_ext) in imgs]
    print(f"Number of meshes: {len(meshes)}")
    print(f"Number of images: {len(imgs)}")
    print(f"Number of vascular segmentations: {len(vascular_imgs)}")

    vascular_imgs = [vascular_dir+f for f in vascular_imgs]
    imgs = [img_dir+f for f in imgs]
    meshes = [directory+f for f in meshes]

    # create folder for segmentations
    segs_dir = directory + 'output/'
    if not os.path.exists(segs_dir):
        os.makedirs(segs_dir)

    for i in range(len(imgs)):
        # Read image
        img = vf.read_img(imgs[i]).GetOutput()
        print(f"Processing image {imgs[i]}")

        # Read polydata
        poly_mesh = vf.read_geo(meshes[i]).GetOutput()
        print(f"Processing mesh {meshes[i]}")

        # Save only region 6
        poly_mesh_6 = vf.thresholdPolyData(poly_mesh, 'RegionId', (6, 6), 'point')

        # Save polydata
        if write_all:
            vf.write_geo(segs_dir + imgs[i].split('/')[-1]
                         .replace(img_ext, '_region6.vtp'), poly_mesh_6)

        # Convert polydata to imagedata
        segmentation = multiclass_convert_polydata_to_imagedata(poly_mesh, img)
        # Save segmentation
        if write_all:
            vf.write_img(segs_dir + imgs[i].split('/')[-1]
                         .replace(img_ext, '_seg.vti'), segmentation)

        # Create a polydata from the segmentation
        poly = vf.vtk_marching_cube_multi(segmentation, 0)

        # Save polydata
        if write_all:
            vf.write_geo(segs_dir + imgs[i].split('/')[-1]
                         .replace(img_ext, '_seg.vtp'), poly)

        # Read vascular segmentation
        vascular = vf.read_img(vascular_imgs[i]).GetOutput()
        print(f"Processing vascular segmentation {vascular_imgs[i]}")

        # Now we need to combine the segmentations
        # We will combine the vascular label 1 with the segmentation label 6
        # combined_seg = combine_segs_aorta_keep(segmentation, vascular)

        # # Save combined segmentation
        # vf.write_img(segs_dir + imgs[i].split('/')[-1]
        #              .replace('.vti', '_seg_combined.vti'), combined_seg)

        # # Create a polydata from the combined segmentation
        # poly = vtk_marching_cube_multi(combined_seg, 0)

        # # Save polydata
        # vf.write_geo(segs_dir + imgs[i].split('/')[-1]
        #              .replace('.vti', '_seg_combined.vtp'), poly)

        # Now we will combine the segmentations by keeping the original
        # segmentation around the aorta valve
        combined_seg, new_vasc = combing_segs_aorta_area(segmentation,
                                                         vascular)

        # Save combined segmentation
        if write_all:
            vf.write_img(segs_dir + imgs[i].split('/')[-1]
                         .replace(img_ext, '_seg_combined_area.vti'),
                         combined_seg)
        vf.write_img(segs_dir + imgs[i].split('/')[-1]
                     .replace(img_ext, '_vasc_combined_area.vti'),
                     new_vasc)

        # Create a polydata from the combined segmentation
        poly = vf.vtk_marching_cube_multi(combined_seg, 0)

        # Save polydata - combined model
        if write_all:
            vf.write_geo(segs_dir + imgs[i].split('/')[-1]
                         .replace(img_ext, '_combined_model_unsmoothed.vtp'), poly)

        # Smooth the polydata
        poly = vf.smooth_polydata(poly, iteration=25, boundary=False,
                               feature=False, smoothingFactor=0.5)

        # Save smoothed polydata - combined model
        if write_all:
            vf.write_geo(segs_dir + imgs[i].split('/')[-1]
                         .replace(img_ext, '_combined_model_smoothed.vtp'),
                         poly)

        # Create blood pool and aorta mesh
        (combined_blood_aorta_vtp,
         combined_blood_aorta_vti) = combine_blood_aorta(combined_seg)

        # Save blood pool and aorta mesh
        if write_all:
            vf.write_geo(segs_dir + imgs[i].split('/')[-1]
                         .replace(img_ext,
                                  '_blood_aorta.vtp'),
                         combined_blood_aorta_vtp)

        if no_valve:
            # Combine the segmentations into one
            (fully_combined_blood_aorta_vtp,
             fully_combined_blood_aorta_vti
             ) = fully_combine_blood_aorta(combined_blood_aorta_vti)
            # Save fully combined blood pool and aorta mesh
            vf.write_geo(segs_dir + imgs[i].split('/')[-1]
                         .replace(img_ext,
                                  '_blood_aorta_no_valve.vtp'),
                         fully_combined_blood_aorta_vtp)

        # Label the valve cells
        blood_aorta_valve = update_labels_based_on_polydata2(
            combined_blood_aorta_vtp, poly)

        # Save blood pool and aorta mesh with valve labels
        vf.write_geo(segs_dir + imgs[i].split('/')[-1]
                     .replace(img_ext,
                              '_simulation_w_valve.vtp'),
                     blood_aorta_valve)

        # Bound the polydata by the image volume
        blood_aorta_valve = vf.bound_polydata_by_image_extended(img, blood_aorta_valve,
                                                    threshold=10,
                                                    name=imgs[i].split('/')[-1]
                                                    )

        # Save bounded polydata
        vf.write_geo(segs_dir + imgs[i].split('/')[-1]
                     .replace(img_ext, '_simulation_w_valve_bounded.vtp'),
                     blood_aorta_valve)

        if no_valve:
            # Bound the polydata by the image volume
            (fully_combined_blood_aorta_vtp
             ) = vf.bound_polydata_by_image_extended(img, fully_combined_blood_aorta_vtp,
                                         threshold=10,
                                         name=imgs[i].split('/')[-1])

            # Save bounded polydata
            vf.write_geo(segs_dir + imgs[i].split('/')[-1].replace(img_ext,
                         '_fully_combined_blood_aorta_bounded.vtp'),
                         fully_combined_blood_aorta_vtp)

            # Smooth the polydata
            fully_combined_blood_aorta_vtp = vf.smooth_polydata(
                fully_combined_blood_aorta_vtp, iteration=25, boundary=False,
                feature=False, smoothingFactor=0.5)

            # Save smoothed polydata
            vf.write_geo(segs_dir + imgs[i].split('/')[-1].replace(img_ext,
                         '_fully_combined_blood_aorta_bounded_smoothed.vtp'),
                         fully_combined_blood_aorta_vtp)
