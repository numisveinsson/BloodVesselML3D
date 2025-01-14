import SimpleITK as sitk
import numpy as np
import os
import random
from .vtk_functions import (
    collect_arrays, calc_normal_vectors, get_location_cells, clean_boundaries,
    exportSitk2VTK, bound_polydata_by_image, bound_polydata_by_sphere,
    get_largest_connected_polydata, points2polydata, write_geo, calc_caps
    )
from .sitk_functions import (
    extract_volume, rotate_volume_tangent, remove_other_vessels,
    connected_comp_info)
from vtk.util.numpy_support import vtk_to_numpy as v2n

import time
np.random.seed(0)
random.seed(0)


def print_info_file(global_config, cases, testing_samples, info_file_name):

    if not global_config['TESTING']:
        f = open(global_config['OUT_DIR'] + info_file_name, "w+")
        for key in global_config.keys():
            f.write("\n " + key + " :")
            f.write("\n     " + str(global_config[key]))
        f.write("\n Testing models not included: ")
        for i in range(len(testing_samples)):
            f.write(" \n     Model " + str(i) + " : " + testing_samples[i])
        f.write(f"\n Number of training models: {len(cases)}")
        f.write("\n Training models included: ")
        for i in range(len(cases)):
            f.write(" \n     Model " + str(i) + " : " + cases[i])
        f.close()
    else:
        f = open(global_config['OUT_DIR'] + info_file_name, "w+")
        for key in global_config.keys():
            f.write("\n " + key + " :")
            f.write("\n     " + str(global_config[key]))
        f.write("\n Testing models included: ")
        for i in range(len(testing_samples)):
            f.write(" \n     Model " + str(i) + " : " + testing_samples[i])
        f.close()


def create_directories(output_folder, modality, global_config):

    # create out folder if it doesnt exist
    try:
        os.mkdir(output_folder)
    except Exception as e:
        print(e)

    if global_config['TESTING']:
        fns = ['_test']
    elif global_config['VALIDATION_PROP'] == 0:
        fns = ['_train']
    else:
        fns = ['_train', '_val']

    if global_config['WRITE_IMG']:
        for fn in fns:
            try:
                os.mkdir(output_folder+modality+fn)
            except Exception as e: print(e)
            try:
                os.mkdir(output_folder+modality+fn+'_masks')
            except Exception as e: print(e)

    if global_config['WRITE_VTK']:
        try:
            os.mkdir(output_folder+'vtk_data')
        except Exception as e: print(e)

    if global_config['WRITE_SURFACE']:
        for fn in fns:
            try:
                os.mkdir(output_folder+modality+fn+'_masks_surfaces')
            except Exception as e: print(e)
            # try:
            #     os.mkdir(output_folder+modality+fn+'_masks_surfaces_box')
            # except Exception as e: print(e)
    if global_config['WRITE_CENTERLINE']:
        for fn in fns:
            try:
                os.mkdir(output_folder+modality+fn+'_masks_centerlines')
            except Exception as e: print(e)
    if global_config['WRITE_OUTLET_STATS']:
        for fn in fns:
            try:
                os.mkdir(output_folder+modality+fn+'_img_outlet_detection')
                os.mkdir(output_folder+modality+fn+'_masks_img_outlet_detection')
            except Exception as e: print(e)


def create_vtk_dir(output_folder, case_name, throwout=0.0):

    os.mkdir(output_folder+'vtk_data/vtk_' + case_name)
    os.mkdir(output_folder+'vtk_data/vtk_mask_' + case_name)
    if throwout:
        os.mkdir(output_folder+'vtk_data/vtk_throwout_' + case_name)


def get_cent_ids(num_points, cent_id, ip):

    try:
        # ids of points belonging to centerline ip
        ids = [i for i in range(num_points) if cent_id[i, ip] == 1]
    except Exception as e:
        print(e)
        ids = [i for i in range(num_points)]
    return ids


def get_surf_caps(surface):
    """
    Function to get the locations of the cells
    on the caps of the surface
    """
    surf_data = collect_arrays(surface.GetCellData())       # collect arrays of cell data
    surf_locs = get_location_cells(surface)                 # get locations of cell centroids
    try:
        ids_caps = np.where(surf_data['CapID']!= 0)[0]      # get ids of cells on caps
    except Exception as e:
        print(e)
        ids_caps = np.where(surf_data['BC_FaceID']!= 0)[0]
    cap_locs = surf_locs[ids_caps]                          # get locations of cap cells
    return cap_locs


def sort_centerline(centerline):
    """
    Function to sort the centerline data
    """

    num_points = centerline.GetNumberOfPoints()               # number of points in centerline
    cent_data = collect_arrays(centerline.GetPointData())
    c_loc = v2n(centerline.GetPoints().GetData())             # point locations as numpy array
    try:
        radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
    except Exception as e:
        print(e)
        radii = cent_data['f']
    # get cent_ids, a list of lists
    # each list is the ids of the points belonging to a centerline
    try:
        cent_ids = get_point_ids_post_proc(centerline)
        bifurc_id = cent_data['BifurcationIdTmp']
    except Exception as e:
        print(e)
        # centerline hasnt been processed
        cent_ids = get_point_ids_no_post_proc(centerline)
        bifurc_id = np.zeros(num_points)
        print("\nCenterline has not been processed, no known bifurcations\n")

    # check if there are duplicate points
    if np.unique(c_loc, axis=0).shape[0] != c_loc.shape[0]:
        # remove duplicate points
        print("\nCenterline has duplicate points, removing them\n")
        _, unique_ids = np.unique(c_loc, axis=0, return_index=True)
        # same for cent_ids, but keep same order
        cent_ids_new = []
        for i in range(len(cent_ids)):
            cent_ids_new.append([])
            for j in range(len(cent_ids[i])):
                if cent_ids[i][j] in unique_ids:
                    cent_ids_new[i].append(cent_ids[i][j])
        cent_ids = cent_ids_new

    num_cent = len(cent_ids)
    # print(f"Num branches {num_cent}, Num points: {num_points}")

    return num_points, c_loc, radii, cent_ids, bifurc_id, num_cent


def get_point_ids_post_proc(centerline_poly):

    cent = centerline_poly
    # number of points in centerline
    num_points = cent.GetNumberOfPoints()
    # number of points in centerline
    cent_data = collect_arrays(cent.GetPointData())

    # cell_data = collect_arrays(cent.GetCellData())
    # points_in_cells = get_points_cells_pd(cent)

    cent_id = cent_data['CenterlineId']
    # number of centerlines (one is assembled of multiple)
    try:
        num_cent = len(cent_id[0])
    except:
        num_cent = 1  # in the case of only one centerline

    point_ids_list = []
    for ip in range(num_cent):
        try:
            # ids of points belonging to centerline ip
            ids = [i for i in range(num_points) if cent_id[i, ip] == 1]
        except:
            ids = [i for i in range(num_points)]
        point_ids_list.append(ids)

    return point_ids_list


def get_point_ids_no_post_proc(centerline_poly):
    """
    For this case, the polydata does not have CenterlineIds,
    so we need to find the centerline ids manually based on the
    connectivity of the points
    Args:
        centerline_poly: vtk polydata of centerline
    Returns:
        point_ids: point ids of centerline (list of lists)
    """
    # the centerline is composed of vtk lines
    # Get the lines from the polydata
    point_ids_list = []
    # Iterate through cells and extract lines
    for i in range(centerline_poly.GetNumberOfCells()):
        cell = centerline_poly.GetCell(i)
        if cell.GetCellType() == 4:
            point_ids = []
            for j in range(cell.GetNumberOfPoints()):
                point_id = cell.GetPointId(j)
                # point = centerline_poly.GetPoint(point_id)
                point_ids.append(point_id)
            point_ids_list.append(point_ids)

    return point_ids_list


def choose_destination(trace_testing, val_prop, img_test, seg_test, img_val,
                       seg_val, img_train, seg_train, ip=None):
    # If tracing test, save to test
    if trace_testing:
        image_out_dir = img_test
        seg_out_dir = seg_test
        val_port = False
    # Else, have a probability to save to validation
    else:
        rand = random.uniform(0, 1)
        # print(" random is " + str(rand))
        if rand < val_prop and ip != 0:
            image_out_dir = img_val
            seg_out_dir = seg_val
            val_port = True  # label to say this sample is validation
        else:
            image_out_dir = img_train
            seg_out_dir = seg_train
            val_port = False
    return image_out_dir, seg_out_dir, val_port


def get_tangent(locs, count):
    """
    Function to calculate the tangent
    """
    if count == 0:
        tangent = locs[count+1] - locs[count]
    elif count == len(locs)-1:
        tangent = locs[count] - locs[count-1]
    else:
        tangent = locs[count+1] - locs[count-1]

    return tangent


def calc_samples(count, bifurc, locs, rads, global_config):
    """
    Function to calculate the number of samples
    and their locations and sizes
    """
    if bifurc[count] == 2:
        save_bif = 1
        n_samples = global_config['NUMBER_SAMPLES_BIFURC']
    else:
        save_bif = 0
        n_samples = global_config['NUMBER_SAMPLES']
    # if in the beginning of centerline, have more 
    if count < len(locs)/20:
        n_samples = global_config['NUMBER_SAMPLES_START']

    # Sample size(s) and shift(s)
    sizes = np.random.normal(global_config['MU_SIZE'],
                             global_config['SIGMA_SIZE'], n_samples)
    shifts = np.random.normal(global_config['MU_SHIFT'],
                              global_config['SIGMA_SHIFT'], n_samples)
    # Make first be correct size, Make first be on centerline
    sizes[0], shifts[0] = global_config['MU_SIZE'], global_config['MU_SHIFT']
    # print("sizes are: " + str(sizes))
    # print("shifts are: " + str(shifts))

    # Calculate vectors
    if not global_config['ROTATE_VOLUMES']:
        if count < len(locs)/2:
            vec0 = locs[count+1] - locs[count]
        else:
            vec0 = locs[count] - locs[count-1]
    else:
        # vec0 is x-axis
        vec0 = np.array([1, 0, 0])

    vec1, vec2 = calc_normal_vectors(vec0)

    # Shift centers
    centers = []
    for sample in range(n_samples):
        value = random.uniform(0, 1)
        vector = vec1*value + vec2*(1-value)
        centers.append(locs[count]+shifts[sample]*vector*rads[count])
    # print("Number of centers are " + str(len(centers)))

    return centers, sizes, save_bif, n_samples, vec0


def rotate_volumes(reader_im, reader_seg, tangent, point):
    """
    Function to rotate the volumes
    Inputs are:
        reader_im: sitk image reader
        reader_seg: sitk image reader
        tangent: tangent vector
        point: point to rotate around
    """
    # read in the volumes
    reader_im = reader_im.Execute()
    reader_seg = reader_seg.Execute()

    # rotate the volumes
    new_img = rotate_volume_tangent(reader_im, tangent, point)
    new_seg = rotate_volume_tangent(reader_seg, tangent, point)
    origin_im = np.array(list(new_img.GetOrigin()))

    return new_img, new_seg, origin_im


def extract_subvolumes(reader_im, reader_seg, index_extract, size_extract,
                       origin_im, spacing_im, location, radius, size_r, number, name,
                       O=None, global_img=False,
                       remove_others=True,
                       binarize=True):
    """"
    Function to extract subvolumes
    Both image data and GT segmentation
    Also calculates some statistics on
        the subvolumes of interest
    """

    index_extract = index_extract.astype(int).tolist()
    size_extract = size_extract.astype(int).tolist()

    new_img = extract_volume(reader_im, index_extract, size_extract)
    new_seg = extract_volume(reader_seg, index_extract, size_extract)
    im_np = sitk.GetArrayFromImage(new_img)
    seg_np = sitk.GetArrayFromImage(new_seg)

    if seg_np.max() > 1:
        new_seg_bin = seg_np / float(seg_np.max()*1.0)
        # make unsigned int
        new_seg_bin = sitk.Cast(new_seg_bin, sitk.sitkUInt8)
    else:
        new_seg_bin = new_seg

    # print("Original Seg")
    # labels, means, _ = connected_comp_info(new_seg, False)
    # print("Seg w removed bodies")
    # labels1, means1 = connected_comp_info(removed_seg)

    seed = np.rint(np.array(size_extract)/2).astype(int).tolist()
    removed_seg_bin = remove_other_vessels(new_seg_bin, seed)
    # labels, means, _ = connected_comp_info(removed_seg, True)
    # mask seg with removed seg
    if binarize:
        removed_seg = removed_seg_bin
    else:
        removed_seg = sitk.Mask(new_seg, removed_seg_bin)

    rem_np = sitk.GetArrayFromImage(removed_seg)
    blood_np = im_np[seg_np > 0.1]
    ground_truth = rem_np[seg_np > 0.1]

    center_volume = (seed)*spacing_im + origin_im
    stats = create_base_stats(number, name, size_r, radius, size_extract,
                              origin_im.tolist(), spacing_im.tolist(),
                              index_extract, center_volume)
    stats = add_image_stats(stats, im_np)

    if not global_img:
        diff_cent = np.linalg.norm(location - center_volume)
        labels, means, _ = connected_comp_info(new_seg, False)
        stats, O = add_local_stats(stats, location, diff_cent, blood_np,
                                   ground_truth, means, removed_seg, im_np, O)
        # mul_l.append(case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')
        # print("The sample has more than one label: " + case_dict['NAME'] +'_'+ str(N-n_old))

    if remove_others:
        new_seg = removed_seg

    return stats, new_img, new_seg, O


def resample_vol(removed_seg, resample_size):
    """
    Function to resample the volume

    Still in development
    """
    from modules.pre_process import resample_spacing
    removed_seg1 = resample_spacing(removed_seg, template_size=resample_size, order=1)[0]
    # if min(size_extract)<resample_size[0]:
    import pdb; pdb.set_trace()
    removed_seg = clean_boundaries(sitk.GetArrayFromImage(removed_seg))
    return removed_seg


def define_cases(global_config, modality):

    cases_dir = global_config['CASES_DIR']+'_'+modality
    cases_raw = os.listdir(cases_dir)
    cases = [cases_dir+'/'+f for f in cases_raw if 'case.' in f]

    testing_cases_raw = global_config['TEST_CASES']
    testing_samples = ['./cases'+'_'+modality+'/case.' + i + '.yml' for i in testing_cases_raw]

    bad_cases_raw = global_config['BAD_CASES']
    bad_cases = ['./cases'+'_'+modality+'/case.' + i + '.yml' for i in bad_cases_raw]

    for case in bad_cases:
        if case in cases:
            cases.remove(case)

    if global_config['TESTING']:
        cases = [i for i in cases if i in testing_samples]
    else:
        cases = [i for i in cases if i not in testing_samples]

    return cases, testing_samples, bad_cases


def extract_surface(img, surface, center, size):
    """
    Function to cut global surface
    into a local part
    size: radius of sphere to cut
    """
    stats = {}
    vtkimage = exportSitk2VTK(img)
    surface_local_box = bound_polydata_by_image(vtkimage[0], surface, 0)    
    surface_local_sphere = bound_polydata_by_sphere(surface, center, size)
    # surface_local_box = get_largest_connected_polydata(surface_local_box)
    surface_local_sphere = get_largest_connected_polydata(surface_local_sphere)

    outlets, outlet_areas = calc_caps(surface_local_sphere)
    stats['OUTLETS'] = outlets
    stats['NUM_OUTLETS'] = len(outlets)
    stats['OUTLET_AREAS'] = outlet_areas

    return stats, surface_local_box, surface_local_sphere


def extract_centerline(img, centerline):
    """
    Function to cut global centerline
    into a local part
    """
    stats = {}
    vtkimage = exportSitk2VTK(img)
    cent_local = bound_polydata_by_image(vtkimage[0], centerline, 0)
    # cent_local = get_largest_connected_polydata(cent_local)
    print(f"Skipping keeping largest connected centerline")

    return stats, cent_local


def clean_cent_ids(cent_id, num_cent):
    """
    Function to clean centerline ids
    """
    columns_to_delete = []
    for i in range(num_cent):
        if np.sum(cent_id[:, i]) == 0:
            columns_to_delete.append(i)
            num_cent -= 1
    cent_id = np.delete(cent_id, columns_to_delete, axis=1) # delete columns with no centerline

    return cent_id, num_cent


def get_bounds(img):
    """
    Function to get the bounds of the image
    """
    bounds = np.zeros((2,3))
    for i in range(3):
        bounds[0,i] = img.TransformIndexToPhysicalPoint((0,0,0))[i]
        bounds[1,i] = img.TransformIndexToPhysicalPoint((img.GetSize()[0]-1,img.GetSize()[1]-1,img.GetSize()[2]-1))[i]
    return bounds


def transform_to_ref(locs, bounds):
    """
    Function to transform the locations
    to the reference frame
    """
    delta = bounds[1,:] - bounds[0,:] # size of image
    locs = (locs - bounds[0,:])/delta # transform to [0,1]

    return locs


def transform_from_ref(locs, bounds):
    """
    Function to transform the locations
    from the reference frame
    """
    delta = bounds[1,:] - bounds[0,:] # size of image
    locs = locs*delta + bounds[0,:]

    return locs


def discretize_centerline(centerline, img, N = None, sub = None, name = None, outdir = None, num_discr_points=10):
    """
    Function to discretize centerline mesh into points
    with labels
    Input: centerline .vtp
    Output: stats dictionary

    Args:
        centerline: vtk polydata of centerline
        img: sitk image
        N: number of centerline
        sub: number of subcenterline
        name: name of the centerline
        outdir: output directory
        num_discr_points: number of discretized points
    Returns:
        stats: dictionary with statistics
    """
    bounds = get_bounds(img)

    num_points, c_loc, radii, cent_id, bifurc_id, num_cent = sort_centerline(centerline)
    
    c_loc = transform_to_ref(c_loc, bounds)
    # cent_id, num_cent = clean_cent_ids(cent_id, num_cent)
    total_ids = []
    stats = {}
    if name:
        stats['NAME'] = name
    if N:
        stats['No'] = N
    steps = np.empty((0, 6))
    for ip in range(num_cent):

        ids = cent_id[ip]
        # Remove ids that are already in the total_ids
        ids = [i for i in ids if i not in total_ids]
        total_ids.extend(ids)
        # print(f'Number of points in centerline {ip}: {len(ids)}')

        locs, rads, bifurc = c_loc[ids], radii[ids], bifurc_id[ids]
        num_ids = len(ids)
        if num_ids > num_discr_points:
            ids = np.linspace(0, num_ids-1, num_discr_points).astype(int)
            locs, rads, bifurc = locs[ids], rads[ids], bifurc[ids]
        else:
            num_cent -= 1
            continue
        # create polydata from locs
        if outdir:
            locs_pd = points2polydata(locs)
            write_geo(outdir+'/vtk_data/vtk_' + name[:9] +'/' +str(N)+'_'+str(sub)+'_'+str(ip)+ '.vtp', locs_pd)

        steps = create_steps(steps, locs, rads, bifurc, ip)
    # print(steps[:,-2:])
    # print(f'Number of centerlines: {num_cent}')
    stats['NUM_CENT'] = num_cent
    stats['STEPS'] = steps
    return stats


def get_outlet_stats(stats, img, seg, upsample=False):
    """
    Function to get outlet statistics
    """

    planes_img = get_outside_volume(img)
    planes_seg = get_outside_volume(seg)

    if upsample:
        planes_img = upsample_planes(planes_img, size=200, seg=False)
        planes_seg = upsample_planes(planes_seg, size=200, seg=True)

    centers, widths, pos_example, planes_seg = get_boxes(planes_seg)

    names_add = ['x0', 'x1', 'y0', 'y1', 'z0', 'z1']
    stats_all = []
    for i in range(6):

        name = stats['NAME'] + '_' + names_add[i]
        stats_new = {}
        stats_new['NAME'] = name
        stats_new['CENTER'] = centers[i]
        stats_new['WIDTH'] = widths[i]
        stats_new['SIZE'] = planes_seg[i].shape

        stats_all.append(stats_new)

    return stats_all, planes_img, planes_seg, pos_example


def upsample_planes(planes, size=480, seg=False):
    """
    Function to resample images to size

    Args:
        planes: list of 2d images
        size: size to resample to

    Returns:
        planes_up: list of upsampled images
    """
    import cv2

    if seg:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_CUBIC

    planes_up = []
    for i in range(len(planes)):
        plane = planes[i]
        input_size = plane.shape
        # get the minimum size
        min_size = min(input_size)
        # get the size to upsample to
        size_hor = size; size_vert = size
        # size_hor = int(input_size[0]/min_size*size)
        # size_vert = int(input_size[1]/min_size*size)
        # upsample
        # import pdb; pdb.set_trace()
        # change to int16
        plane = plane.astype(np.int16)
        plane = cv2.resize(plane, (size_hor, size_vert), interpolation=interp)
        planes_up.append(plane)

    return planes_up


def write_2d_planes(planes, stats_out, image_out_dir):
    """
    Function to write 2d planes
    Values are normalized to 0-255
    Written out as png
    """
    import cv2

    add = '_img_outlet_detection'
    # add to dir name end
    image_out_dir = image_out_dir[:-1]+ add + '/'

    for i in range(len(planes)):
        plane = planes[i]
        # # make sure channel is first
        # if len(plane.shape) == 3:
        #     if min(plane.shape) != plane.shape[0] and min(plane.shape) != plane.shape[1]:
        #         plane = np.moveaxis(plane, -1, 0)
        #     elif min(plane.shape) != plane.shape[0] and min(plane.shape) != plane.shape[2]:
        #         plane = np.moveaxis(plane, -2, 0)
        # shift to positive
        # plane = plane - np.amin(plane)
        # # make rane 0-255
        # plane = plane/max([np.amax(plane),1])*255
        # # make unsigned int
        # plane = plane.astype(np.uint8)
        
        plane = cv2.convertScaleAbs(plane, alpha=(255.0/np.amax(plane)))
        # print(f"Shape of plane: {plane.shape}")
        fn_name = image_out_dir + stats_out[i]['NAME']+'.png'
        cv2.imwrite(fn_name, plane)


def get_boxes(planes):

    centers, widths = [], []
    pos_example = 0
    for i in range(len(planes)):
        centers.append([])
        widths.append([])
        # check if plane is empty
        if np.sum(planes[i]) == 0:
            continue
        else:
            pos_example += 1
            # check how many connected components
            img_plane = sitk.GetImageFromArray(planes[i])
            labels, means, connected_seg = connected_comp_info(img_plane,
                                                               False)
            connected_seg = sitk.GetArrayFromImage(connected_seg)
            # print(f"Max of Connected Seg: {np.amax(connected_seg)}, Min of Connected Seg: {np.amin(connected_seg)}")
            for j, label in enumerate(labels):
                # if the connected component is has N or less pixels, skip
                if np.sum(connected_seg == label) <= 50:
                    connected_seg[connected_seg == label] = 0
                    continue

                # get the pixel in the center of the connected component
                center = np.array(np.where(connected_seg == label)).mean(axis=1).astype(int).tolist()
                # get the width and length of the connected component
                width = np.array(np.where(connected_seg == label)).max(axis=1) - np.array(np.where(connected_seg==label)).min(axis=1)
                # have at least 3 pixels in width and length
                width = np.where(width < 3, 3, width).tolist()
                # append to lists
                centers[i].append(center)
                widths[i].append(width)
            planes[i] = connected_seg

            # print(f"Shape: {connected_seg.shape}, Center: {center}, Width/Length: {width}")

    return centers, widths, pos_example, planes


def get_outside_volume(seg):
    """
    Function to get outside sides of volume
    """
    seg_np = sitk.GetArrayFromImage(seg)
    planes = []
    planes.append(seg_np[0, :, :])
    planes.append(seg_np[-1, :, :])
    planes.append(seg_np[:, 0, :])
    planes.append(seg_np[:, -1, :])
    planes.append(seg_np[:, :, 0])
    planes.append(seg_np[:, :, -1])

    return planes


def get_outside_volume_3(seg):
    """
    Function to get outside sides of volume
    The last three channels
    Make channels last
    """
    seg_np = sitk.GetArrayFromImage(seg)
    planes = []
    planes.append(seg_np[:3,:,:].transpose(1,2,0))
    planes.append(seg_np[-3:,:,:].transpose(1,2,0))
    planes.append(seg_np[:,:3,:].transpose(0,2,1))
    planes.append(seg_np[:,-3:,:].transpose(0,2,1))
    planes.append(seg_np[:,:,:3].transpose(0,1,2))
    planes.append(seg_np[:,:,-3:].transpose(0,1,2))

    return planes


def create_steps(steps, locs, rads, bifurc, ip):
    """
    Function to create a step
    Input:
        steps: np array to add to
        locs: locations of points
        rads: rads of points
        bifurc: bifurcations label of points
        ip: index of centerline
    """
    bifurc = binarize_bifurc(bifurc)
    step_add = np.zeros((len(locs), 6))
    step_add[:, 0:3] = locs
    step_add[:, 3] = rads
    step_add[:, 4] = bifurc
    step_add[-1, 5] = 1
    steps = np.append(steps, step_add, axis=0)

    steps_list = steps.tolist()
    for i in range(len(steps_list)):
        steps_list[i][0] = round(steps_list[i][0], 3)
        steps_list[i][1] = round(steps_list[i][1], 3)
        steps_list[i][2] = round(steps_list[i][2], 3)
        steps_list[i][3] = round(steps_list[i][3], 3)
        steps_list[i][4] = int(steps_list[i][4])
        steps_list[i][5] = int(steps_list[i][5])

    return steps_list


def binarize_bifurc(bifurc):
    "Function to binarize bifurcation labels"
    bifurc[bifurc >= 0] = 1
    bifurc[bifurc < 0] = 0

    return bifurc


def add_local_stats(stats, location, diff_cent, blood_np, ground_truth, means, removed_seg, im_np, O):
    """
    Function to add local stats to the stats dictionary
    """    
    stats.update({"DIFF_CENT": diff_cent, "POINT_CENT": location.tolist(),
                  "BLOOD_MEAN": np.mean(blood_np),     "BLOOD_MIN": np.amin(blood_np),
                  "BLOOD_STD": np.std(blood_np),       "BLOOD_MAX": np.amax(blood_np),  
                  "GT_MEAN": np.mean(ground_truth),   "GT_STD": np.std(ground_truth),     
                  "GT_MAX": np.amax(ground_truth),    "GT_MIN": np.amin(ground_truth) 
                  })
    if len(means) != 1:
        larg_np = sitk.GetArrayFromImage(removed_seg)
        rem_np = im_np[larg_np > 0.1]
        stats_rem = {
            "LARGEST_MEAN": np.mean(rem_np),"LARGEST_STD": np.std(rem_np),
            "LARGEST_MAX": np.amax(rem_np), "LARGEST_MIN": np.amin(rem_np)}
        stats.update(stats_rem)
        O += 1
    return stats, O


def add_tangent_stats(stats, vec0, save_bif):
    stats.update({"TANGENTX": (vec0/np.linalg.norm(vec0))[0], "TANGENTY": (vec0/np.linalg.norm(vec0))[1], 
                  "TANGENTZ": (vec0/np.linalg.norm(vec0))[2], "BIFURCATION": save_bif})
    return stats


def add_image_stats(stats, im_np):
    stats.update({
        "IM_MEAN": np.mean(im_np), "IM_MIN": np.amin(im_np), 
        "IM_STD": np.std(im_np),   "IM_MAX": np.amax(im_np),
    })
    return stats


def create_base_stats(N, name, size_r, radius, size_extract, origin_im, spacing_im, index_extract, center_volume):
    stats = {"No": N,
             "NAME": name,
             "SIZE": size_r*radius,
             "RADIUS": radius,
             "RESOLUTION": size_extract,
             "ORIGIN": origin_im,
             "SPACING": spacing_im,
             "INDEX": index_extract,   
             "VOL_CENT": center_volume.tolist(),
             "NUM_VOX": size_extract[0]*size_extract[1]*size_extract[2]}
    return stats


def append_stats(stats, csv_list, csv_list_val, val_port):
    if val_port:
        csv_list_val.append(stats)
    else:
        csv_list.append(stats)
    return csv_list, csv_list_val


def find_next_point(count, locs, rads, bifurc, global_config, on_cent):
    """
    Function to find the next point to move to
    """
    lengths = np.cumsum(np.insert(np.linalg.norm(np.diff(locs[count:], axis=0),
                                                 axis=1), 0, 0))
    move = 1
    count = count+1
    if count == len(locs):
        on_cent = False
        return count, on_cent
    move_distance = global_config['MOVE_DIST']*rads[count]
    if rads[count] >= 0.4:  # Move slower for larger vessels
        move_distance = move_distance * global_config['MOVE_SLOWER_LARGE']
        # print("slowerrrrr")
    if bifurc[count] == 2:  # Move slower for bifurcating vessels
        move_distance = move_distance * global_config['MOVE_SLOWER_BIFURC']
    while lengths[move] < move_distance:
        count = count+1
        move = move+1
        if count == len(locs):
            on_cent = False
            break
    return count, on_cent


def print_model_info(case_name, N, n_old, M, m_old):
    print(case_name)
    print("\n****************** All done for this model! ******************")
    print("****************** " + str(N-n_old) + " extractions! ******************")
    print("****************** " + str(M-m_old) + " throwouts! ****************** \n")


def print_into_info(info_file_name, case_name, N, n_old, M, m_old, K, k_old,
                    out_dir):
    f = open(out_dir + info_file_name, 'a')
    f.write("\n " + case_name)
    f.write("\n " + str([N-n_old, M-m_old, K-k_old]))
    f.write("\n ")
    f.close()


def print_into_info_all_done(info_file_name, N, M, K, O, out_dir,
                             start_time=None):
    f = open(out_dir + info_file_name, 'a')
    f.write("\n *** " + str(N) + " extractions! ***")
    f.write("\n *** " + str(M) + " throwouts! ***")
    f.write("\n *** " + str(K) + " errors in saving! ***")
    f.write("\n *** " + str(O) + " have more than one label! ***")
    if start_time:
        f.write(f"\n *** Time: {(time.time()-start_time)/60} minutes ***")
    f.close()


def print_all_done(info, N, M, K, O, mul_l=None):
    for i in info:
        print(i)
        print(info[i])
        print(" ")

    print("\n**** All done for all models! ****")
    print("**** " + str(N) + " extractions! ****")
    print("**** " + str(M) + " throwouts! ****")
    print("**** " + str(K) + " errors in saving! **** \n")
    print("**** O: " + str(O) + " have more than one label! They are: **** \n")

    if mul_l:
        for i in mul_l:
            print(i)


def write_vtk(new_img, removed_seg, out_dir, case_name, N, n_old, sub):
    # write vtk, if N is a multiple of 10
    # if N-n_old%10 == 0:
    sitk.WriteImage(new_img, out_dir+'vtk_data/vtk_' + case_name + '/' + str(N-n_old)+'_'+str(sub)+ '.mha')
    if sitk.GetArrayFromImage(removed_seg).max() == 1:
        removed_seg *= 255
    sitk.WriteImage(removed_seg, out_dir+'vtk_data/vtk_mask_' + case_name + '/' + str(N-n_old)+'_'+str(sub)+ '.mha')


def write_vtk_throwout(reader_seg, index_extract, size_extract, out_dir,
                       case_name, N, n_old, sub):
    new_seg = extract_volume(reader_seg, index_extract.astype(int).tolist(), size_extract.astype(int).tolist())
    sitk.WriteImage(new_seg, out_dir+'vtk_data/vtk_throwout_' + case_name +'/'+str(N-n_old)+ '_'+str(sub)+'.mha')


def write_img(new_img, removed_seg, image_out_dir, seg_out_dir, case_name, N,
              n_old, sub, binarize=True):
    print(f"Max seg value: {sitk.GetArrayFromImage(removed_seg).max()}")
    sitk.WriteImage(new_img, image_out_dir + case_name + '_' + str(N-n_old) + '_' + str(sub)+'.nii.gz')
    max_seg_value = sitk.GetArrayFromImage(removed_seg).max()
    if max_seg_value != 1 and binarize:
        removed_seg /= float(max_seg_value*1.0)
        print(f"Max seg value after scaling: {sitk.GetArrayFromImage(removed_seg).max()}")

    # make image unsigned int, removed_seg is sitk image
    removed_seg = sitk.Cast(removed_seg, sitk.sitkUInt8)
    # assert max_seg_value is 1
    if binarize:
        assert sitk.GetArrayFromImage(removed_seg).max() == 1
    sitk.WriteImage(removed_seg, seg_out_dir + case_name + '_' + str(N-n_old)
                    + '_' + str(sub)+'.nii.gz')


def write_surface(new_surf_box, new_surf_sphere, seg_out_dir, case_name, N,
                  n_old, sub):
    # write_geo(seg_out_dir.replace('masks','masks_surfaces_box') + case_dict['NAME']+'_' +str(N-n_old)+'_'+str(sub)+ '.vtp', new_surf_box)
    write_geo(seg_out_dir.replace('masks','masks_surfaces') + case_name + '_' + str(N-n_old)+'_'+str(sub)+ '.vtp', new_surf_sphere)


def write_centerline(new_cent, seg_out_dir, case_name, N, n_old, sub):
    write_geo(seg_out_dir.replace('masks','masks_centerlines') + case_name + '_' + str(N-n_old)+'_'+str(sub)+ '.vtp', new_cent)
    # pts_pd = points2polydata(stats_surf['OUTLETS'])
    # write_geo(out_dir+'vtk_data/vtk_' + case_dict['NAME']+'/' +str(N-n_old)+'_'+str(sub)+ '_caps.vtp', pts_pd)


def write_csv(csv_list, csv_list_val, modality, global_config):
    import csv
    csv_file = "_Sample_stats.csv"
    if global_config['TESTING']:
        csv_file = '_test'+csv_file
    else:
        csv_file = '_train'+csv_file

    csv_columns = ["No",            "NAME",         "SIZE",     "RESOLUTION",   "ORIGIN", 
                   "SPACING",       "POINT_CENT",   "INDEX",    "SIZE_EXTRACT", "VOL_CENT", 
                   "DIFF_CENT",     "IM_MEAN",      "IM_STD",   "IM_MAX",       "IM_MIN",
                   "BLOOD_MEAN",    "BLOOD_STD",    "BLOOD_MAX", "BLOOD_MIN",    "GT_MEAN", 
                   "GT_STD",        "GT_MAX",       "GT_MIN",   "LARGEST_MEAN", "LARGEST_STD",
                   "LARGEST_MAX",   "LARGEST_MIN",  "RADIUS",   "TANGENTX",     "TANGENTY", 
                   "TANGENTZ",      "BIFURCATION",  "NUM_VOX",  "OUTLETS",      "NUM_OUTLETS"]
    with open(global_config['OUT_DIR']+modality+csv_file, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_list:
            writer.writerow(data)
    if not global_config['TESTING'] and global_config['VALIDATION_PROP'] > 0:
        with open(global_config['OUT_DIR']+modality+csv_file.replace('train', 'val'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_list_val:
                writer.writerow(data)


def write_csv_discrete_cent(csv_discrete_centerline,
                            csv_discrete_centerline_val, modality,
                            global_config):
    import csv
    csv_file = "_Discrete_Centerline.csv"
    if global_config['TESTING']:
        csv_file = '_test'+csv_file
    else:
        csv_file = '_train'+csv_file

    csv_columns = ["No", "NAME", "NUM_CENT", "STEPS"]
    with open(global_config['OUT_DIR']+modality+csv_file, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_discrete_centerline:
            writer.writerow(data)
    if not global_config['TESTING'] and global_config['VALIDATION_PROP'] > 0:
        with open(global_config['OUT_DIR']+modality+csv_file.replace('train', 'val'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_discrete_centerline_val:
                writer.writerow(data)


def write_csv_outlet_stats(csv_outlet_stats, csv_outlet_stats_val, modality,
                           global_config):

    import csv
    csv_file = "_Outlet_Stats.csv"
    if global_config['TESTING']:
        csv_file = '_test'+csv_file
    else:
        csv_file = '_train'+csv_file

    csv_columns = ["NAME", "CENTER", "WIDTH", "SIZE"]
    with open(global_config['OUT_DIR']+modality+csv_file, 'a+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_outlet_stats:
            writer.writerow(data)
    if not global_config['TESTING'] and global_config['VALIDATION_PROP'] > 0:
        with open(global_config['OUT_DIR']+modality+csv_file.replace('train','val'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_outlet_stats_val:
                writer.writerow(data)


def write_pkl_outlet_stats(pkl_outlet_stats, pkl_outlet_stats_val, modality,
                           global_config):
    import pickle
    pkl_file = "_Outlet_Stats.pkl"
    if global_config['TESTING']:
        pkl_file = '_test'+pkl_file
    else:
        pkl_file = '_train'+pkl_file

    with open(global_config['OUT_DIR']+modality+pkl_file, 'wb') as f:
        pickle.dump(pkl_outlet_stats, f)
    if not global_config['TESTING'] and global_config['VALIDATION_PROP'] > 0:
        with open(global_config['OUT_DIR']+modality+pkl_file.replace('train', 'val'), 'wb') as f:
            pickle.dump(pkl_outlet_stats_val, f)