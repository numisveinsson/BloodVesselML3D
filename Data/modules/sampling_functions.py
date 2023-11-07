import SimpleITK as sitk
import numpy as np
import os
import random
from .vtk_functions import *
from .sitk_functions import *
from vtk.util.numpy_support import vtk_to_numpy as v2n

np.random.seed(0)
random.seed(0)

def create_dataset(global_config, modality):

    dataset_name = global_config['DATASET_NAME']
    if dataset_name == 'vmr':
        from dataset_dirs.datasets import VMR_dataset
        Dataset = VMR_dataset(global_config['DATA_DIR'], [modality], global_config['ANATOMY'])
        cases = Dataset.sort_cases(global_config['TESTING'], global_config['TEST_CASES'])
        cases = Dataset.check_which_cases_in_image_dir(cases)
    elif dataset_name == 'miccai_aortas':
        from dataset_dirs.miccai_aortas import get_miccai_aorta_dataset_names
        cases = get_miccai_aorta_dataset_names(global_config['DATA_DIR'])

    else:
        print("Dataset not found")
        exit()

    return cases

def print_info_file(global_config, cases, testing_samples, info_file_name):

    if not global_config['TESTING']:
        f = open(global_config['OUT_DIR'] + info_file_name,"w+")
        for key in global_config.keys():
            f.write("\n " + key + " :")
            f.write("\n     " + str(global_config[key]))
        f.write("\n Testing models not included: ")
        for i in range(len(testing_samples)):
            f.write(" \n     Model " + str(i) + " : " + testing_samples[i])
        f.write("\n Training models included: ")
        for i in range(len(cases)):
            f.write(" \n     Model " + str(i) + " : " + cases[i])
        f.close()
    else:
        f = open(global_config['OUT_DIR'] + info_file_name,"w+")
        for key in global_config.keys():
            f.write("\n " + key + " :")
            f.write("\n     " + str(global_config[key]))
        f.write("\n Testing models included: ")
        for i in range(len(testing_samples)):
            f.write(" \n     Model " + str(i) + " : " + testing_samples[i])
        f.close()

def create_directories(output_folder, modality, global_config):

    if global_config['TESTING']:
        fns = ['_test']
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

def create_vtk_dir(output_folder, case_name):
    
    os.mkdir(output_folder+'vtk_data/vtk_' + case_name)
    os.mkdir(output_folder+'vtk_data/vtk_mask_' + case_name)
    os.mkdir(output_folder+'vtk_data/vtk_throwout_' + case_name)

def get_cent_ids(num_points, cent_id, ip):
    
    try:
        ids = [i for i in range(num_points) if cent_id[i,ip]==1]    # ids of points belonging to centerline ip
    except:
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
    except:
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
    radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
    cent_id = cent_data['CenterlineId']
    bifurc_id = cent_data['BifurcationIdTmp']
    try:
        num_cent = len(cent_id[0]) # number of centerlines (one is assembled of multiple)
    except:
        num_cent = 1 # in the case of only one centerline
    
    return num_points, c_loc, radii, cent_id, bifurc_id, num_cent

def choose_destination(trace_testing, val_prop, img_test, seg_test, img_val, seg_val, img_train, seg_train, ip = None):
    ## If tracing test, save to test
    if trace_testing:
        image_out_dir = img_test
        seg_out_dir = seg_test
        val_port = False
    ## Else, have a probability to save to validation
    else:
        rand = random.uniform(0,1)
        print(" random is " + str(rand))
        if rand < val_prop and ip != 0:
            image_out_dir = img_val
            seg_out_dir = seg_val
            val_port = True # label to say this sample is validation
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

    # Sample size(s) and shift(s)
    sizes = np.random.normal(global_config['MU_SIZE'], global_config['SIGMA_SIZE'], n_samples)
    shifts = np.random.normal(global_config['MU_SHIFT'], global_config['SIGMA_SHIFT'], n_samples)
    sizes[0], shifts[0] = global_config['MU_SIZE'], global_config['MU_SHIFT'] # Make first be correct size, Make first be on centerline
    #print("sizes are: " + str(sizes))
    #print("shifts are: " + str(shifts))

    # Calculate vectors
    if not global_config['ROTATE_VOLUMES']:
        if count < len(locs)/2:
            vec0 = locs[count+1] - locs[count]
        else:
            vec0 = locs[count] - locs[count-1]
    else:
        # vec0 is x-axis
        vec0 = np.array([1,0,0])

    vec1, vec2 = calc_normal_vectors(vec0)

    # Shift centers
    centers = []
    for sample in range(n_samples):
        value = random.uniform(0,1)
        vector = vec1*value + vec2*(1-value)
        centers.append(locs[count]+shifts[sample]*vector*rads[count])
    #print("Number of centers are " + str(len(centers)))

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

def extract_subvolumes(reader_im, reader_seg, index_extract, size_extract, origin_im, spacing_im, location, radius, size_r, number, name, O=None, global_img=False):
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
        
    #print("Original Seg")
    labels, means, _ = connected_comp_info(new_seg, False)
    #print("Seg w removed bodies")
    #labels1, means1 = connected_comp_info(removed_seg)
    
    seed = np.rint(np.array(size_extract)/2).astype(int).tolist()
    removed_seg = remove_other_vessels(new_seg, seed)

    im_np = sitk.GetArrayFromImage(new_img)
    seg_np = sitk.GetArrayFromImage(new_seg)
    rem_np = sitk.GetArrayFromImage(removed_seg)
    blood_np = im_np[seg_np>0.1]
    ground_truth = rem_np[seg_np>0.1]

    center_volume = (seed)*spacing_im + origin_im
    stats = create_base_stats(number, name, size_r, radius, size_extract, origin_im.tolist(), spacing_im.tolist(), index_extract, center_volume)
    stats = add_image_stats(stats, im_np)

    if not global_img:
        diff_cent = np.linalg.norm(location - center_volume)
        stats, O = add_local_stats(stats, location, diff_cent, blood_np, ground_truth, means, removed_seg, im_np, O)
        #mul_l.append(case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')
        #print("The sample has more than one label: " + case_dict['NAME'] +'_'+ str(N-n_old))

    return stats, new_img, new_seg, removed_seg, O

def resample_vol(removed_seg, resample_size):
    """
    Function to resample the volume
    """
    from modules.pre_process import resample_spacing
    removed_seg1 = resample_spacing(removed_seg, template_size=resample_size, order=1)[0]
    #if min(size_extract)<resample_size[0]:
    import pdb; pdb.set_trace()
    removed_seg = clean_boundaries(sitk.GetArrayFromImage(removed_seg))
    return removed_seg

def define_cases(global_config, modality):

    cases_dir = global_config['CASES_DIR']+'_'+modality
    cases_raw = os.listdir(cases_dir)
    cases = [cases_dir+'/'+f for f in cases_raw if 'case.' in f]

    testing_cases_raw = global_config['TEST_CASES']
    testing_samples = ['./cases'+'_'+modality+'/case.'+ i + '.yml' for i in testing_cases_raw]

    bad_cases_raw = global_config['BAD_CASES']
    bad_cases = ['./cases'+'_'+modality+'/case.'+ i + '.yml' for i in bad_cases_raw]

    for case in bad_cases:
        if case in cases: cases.remove(case)

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
    #import pdb; pdb.set_trace()
    stats = {}
    vtkimage = exportSitk2VTK(img)
    surface_local_box = bound_polydata_by_image(vtkimage[0], surface, 0)    
    surface_local_sphere = bound_polydata_by_sphere(surface, center, size)
    #surface_local_box = get_largest_connected_polydata(surface_local_box)
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
    cent_local = get_largest_connected_polydata(cent_local)

    return stats, cent_local

def clean_cent_ids(cent_id, num_cent):
    """
    Function to clean centerline ids
    """
    columns_to_delete = []
    for i in range(num_cent):
        if np.sum(cent_id[:,i]) == 0:
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

def discretize_centerline(centerline, img, N, sub, name, outdir, num_discr_points=10):
    """
    Function to discretize centerline mesh into points
    with labels
    Input: centerline .vtp
    Output: stats dictionary
    """
    bounds = get_bounds(img)

    num_points, c_loc, radii, cent_id, bifurc_id, num_cent = sort_centerline(centerline)
    
    c_loc = transform_to_ref(c_loc, bounds)

    cent_id, num_cent = clean_cent_ids(cent_id, num_cent)
    total_ids = []
    stats = {}
    stats['NAME'] = name
    stats['No'] = N
    steps = np.empty((0,6))
    for ip in range(num_cent):

        ids = get_cent_ids(num_points, cent_id, ip)
        # Remove ids that are already in the total_ids
        ids = [i for i in ids if i not in total_ids]
        total_ids.extend(ids)
        #print(f'Number of points in centerline {ip}: {len(ids)}')

        locs, rads, bifurc = c_loc[ids], radii[ids], bifurc_id[ids]
        num_ids = len(ids)
        if num_ids > num_discr_points:
            ids = np.linspace(0, num_ids-1, num_discr_points).astype(int)
            locs, rads, bifurc = locs[ids], rads[ids], bifurc[ids]
        else:
            num_cent -= 1
            continue
        # create polydata from locs
        locs_pd = points2polydata(locs)
        write_geo(outdir+'/vtk_data/vtk_' + name[:9] +'/' +str(N)+'_'+str(sub)+'_'+str(ip)+ '.vtp', locs_pd)

        steps = create_steps(steps, locs, rads, bifurc, ip)
    #print(steps[:,-2:])
    #print(f'Number of centerlines: {num_cent}')
    stats['NUM_CENT'] = num_cent
    stats['STEPS'] = steps
    #print(f'Shape of steps: {steps.shape}')
    return stats

def get_outlet_stats(stats, img, seg, outlet_classes, global_surface, global_centerline, center, size):
    """
    Function to get outlet statistics
    """
    # _ , new_cent = extract_centerline(img, global_centerline)
    # # get centerline stats
    # cent_data = collect_arrays(new_cent.GetPointData())
    # cent_locs = v2n(new_cent.GetPoints().GetData())

    # stats_surf, new_surf_box, new_surf_sphere = extract_surface(img, global_surface, center, size)
    # num_out = len(stats_surf['OUTLETS'])
    # outlet_locs = stats_surf['OUTLETS']
    # outlet_areas = stats_surf['OUTLET_AREAS']

    # # get the pixel values of the outlets
    # outlet_indx = []
    # for i in range(num_out):
    #     index = img.TransformPhysicalPointToIndex(stats_surf['OUTLETS'][i].tolist())
    #     outlet_indx.append(index)
    # outlet_indx = np.array(outlet_indx)

    # # create sizes to check
    # for prop in [0.5, 0.6, 0.7, 0.8, 0.9]:
    #     stats_surf, new_surf_box, new_surf_sphere = extract_surface(img, global_surface, center, size*prop)
    #     new_num_out = len(stats_surf['OUTLETS'])
    #     if num_out != new_num_out:
    #         print(f"Num outlets don't match when changing size")

    planes_img = get_outside_volume_3(img)
    planes_seg = get_outside_volume(seg)
    centers, widths, pos_example = get_boxes(planes_seg)

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

def write_2d_planes(planes, stats_out, image_out_dir):
    """
    Function to write 2d planes
    Values are normalized to 0-255
    Written out as png
    """
    add = '_img_outlet_detection'
    # add to dir name end
    image_out_dir = image_out_dir[:-1]+ add + '/'

    for i in range(len(planes)):
        plane = planes[i]
        # make sure channel is first
        if len(plane.shape) == 3:
            if min(plane.shape) != plane.shape[0] and min(plane.shape) != plane.shape[1]:
                plane = np.moveaxis(plane, -1, 0)
            elif min(plane.shape) != plane.shape[0] and min(plane.shape) != plane.shape[2]:
                plane = np.moveaxis(plane, -2, 0)
        # shift to positive
        plane = plane - np.amin(plane)
        # make rane 0-255
        plane = plane/max([np.amax(plane),1])*255
        # make unsigned int
        plane = plane.astype(np.uint8)
        # print(f"Shape of plane: {plane.shape}")

        sitk.WriteImage(sitk.GetImageFromArray(plane), image_out_dir + stats_out[i]['NAME']+'.png')

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
            labels, means, connected_seg = connected_comp_info(img_plane, False)
            connected_seg = sitk.GetArrayFromImage(connected_seg)
            for label in labels:
                # get the pixel in the center of the connected component
                center = np.array(np.where(connected_seg==label)).mean(axis=1).astype(int).tolist()
                # get the width and length of the connected component
                width = np.array(np.where(connected_seg==label)).max(axis=1) - np.array(np.where(connected_seg==label)).min(axis=1)
                # have at least 3 pixels in width and length
                width = np.where(width<3, 3, width).tolist()
                # append to lists
                centers[i].append(center)
                widths[i].append(width)

                # print(f"Shape: {connected_seg.shape}, Center: {center}, Width/Length: {width}")
                
    return centers, widths, pos_example

def get_outside_volume(seg):
    """
    Function to get outside sides of volume
    """
    seg_np = sitk.GetArrayFromImage(seg)
    planes = []
    planes.append(seg_np[0,:,:])
    planes.append(seg_np[-1,:,:])
    planes.append(seg_np[:,0,:])
    planes.append(seg_np[:,-1,:])
    planes.append(seg_np[:,:,0])
    planes.append(seg_np[:,:,-1])

    # for i in range(len(planes)):
    #     print(np.sum(planes[i]))

    return planes

def get_outside_volume_3(seg):
    """
    Function to get outside sides of volume
    The last three channels
    """
    seg_np = sitk.GetArrayFromImage(seg)
    planes = []
    planes.append(seg_np[:3,:,:])
    planes.append(seg_np[-3:,:,:])
    planes.append(seg_np[:,:3,:])
    planes.append(seg_np[:,-3:,:])
    planes.append(seg_np[:,:,:3])
    planes.append(seg_np[:,:,-3:])

    # for i in range(len(planes)):
    #     print(np.sum(planes[i]))

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
    step_add[:,0:3] = locs
    step_add[:,3] = rads
    step_add[:,4] = bifurc
    step_add[-1,5] = 1
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
    bifurc[bifurc>=0] = 1
    bifurc[bifurc<0] = 0

    return bifurc


def add_local_stats(stats, location, diff_cent, blood_np, ground_truth, means, removed_seg, im_np, O):
    """
    Function to add local stats to the stats dictionary
    """    
    stats.update({"DIFF_CENT": diff_cent, "POINT_CENT": location.tolist(),
                "BLOOD_MEAN":np.mean(blood_np),     "BLOOD_MIN":np.amin(blood_np),
                "BLOOD_STD":np.std(blood_np),       "BLOOD_MAX":np.amax(blood_np),  
                "GT_MEAN": np.mean(ground_truth),   "GT_STD": np.std(ground_truth),     
                "GT_MAX": np.amax(ground_truth),    "GT_MIN": np.amin(ground_truth) 
                })
    if len(means) != 1:
            larg_np = sitk.GetArrayFromImage(removed_seg)
            rem_np = im_np[larg_np>0.1]
            stats_rem = {
                "LARGEST_MEAN":np.mean(rem_np),"LARGEST_STD":np.std(rem_np),
                "LARGEST_MAX":np.amax(rem_np), "LARGEST_MIN":np.amin(rem_np)}
            stats.update(stats_rem)
            O += 1
    return stats, O

def add_tangent_stats(stats, vec0, save_bif):
    stats.update({"TANGENTX": (vec0/np.linalg.norm(vec0))[0], "TANGENTY": (vec0/np.linalg.norm(vec0))[1], 
                  "TANGENTZ": (vec0/np.linalg.norm(vec0))[2], "BIFURCATION":save_bif})
    return stats

def add_image_stats(stats, im_np):
    stats.update({
    "IM_MEAN":np.mean(im_np), "IM_MIN":np.amin(im_np), 
    "IM_STD":np.std(im_np),   "IM_MAX":np.amax(im_np),
    })
    return stats

def create_base_stats(N, name, size_r, radius, size_extract, origin_im, spacing_im, index_extract, center_volume):
    stats = {"No":N, 
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
    lengths = np.cumsum(np.insert(np.linalg.norm(np.diff(locs[count:], axis=0), axis=1), 0, 0))
    move = 1
    count = count+1
    if count == len(locs):
        on_cent = False
        return count, on_cent
    move_distance = global_config['MOVE_DIST']*rads[count]
    if rads[count] >= 0.4: # Move slower for larger vessels
        move_distance = move_distance * global_config['MOVE_SLOWER_LARGE']
        #print("slowerrrrr")
    if bifurc[count] == 2: # Move slower for bifurcating vessels
        move_distance = move_distance* global_config['MOVE_SLOWER_BIFURC']
    while lengths[move] < move_distance :
        count = count+1
        move = move+1
        if count == len(locs):
            on_cent = False
            break
    return count, on_cent

def print_model_info(case_name, N, n_old, M, m_old):
    print(case_name)
    print("\n****************** All done for this model! ******************")
    print("****************** " + str(N-n_old) +" extractions! ******************")
    print("****************** " + str(M-m_old) +" throwouts! ****************** \n")
    
def print_into_info(info_file_name, case_name, N, n_old, M, m_old, K, k_old, out_dir):
    f = open(out_dir +info_file_name,'a')
    f.write("\n "+ case_name)
    f.write("\n "+ str([ N-n_old, M-m_old, K-k_old ]))
    f.write("\n ")
    f.close()

def print_into_info_all_done(info_file_name, N, M, K, O, out_dir):
    f = open(out_dir +info_file_name,'a')
    f.write("\n *** " + str(N) +" extractions! ***")
    f.write("\n *** " + str(M) +" throwouts! ***")
    f.write("\n *** " + str(K) +" errors in saving! ***")
    f.write("\n *** " + str(O) +" have more than one label! ***")
    f.close()

def print_all_done(info, N, M, K, O, mul_l):
    for i in info:
        print(i)
        print(info[i])
        print(" ")

    print("\n**** All done for all models! ****")
    print("**** " + str(N) +" extractions! ****")
    print("**** " + str(M) +" throwouts! ****")
    print("**** " + str(K) +" errors in saving! **** \n")
    print("**** O: " + str(O) +" have more than one label! They are: **** \n")
    for i in mul_l:
        print(i)

def write_vtk(new_img, removed_seg, out_dir, case_name, N, n_old, sub):
    sitk.WriteImage(new_img, out_dir+'vtk_data/vtk_' + case_name +'/' +str(N-n_old)+'_'+str(sub)+ '.vtk')
    sitk.WriteImage(removed_seg*255, out_dir+'vtk_data/vtk_mask_'+ case_name +'/' +str(N-n_old)+'_'+str(sub)+ '.vtk')

def write_vtk_throwout(reader_seg, index_extract, size_extract, out_dir, case_name, N, n_old, sub):
    new_seg = extract_volume(reader_seg, index_extract.astype(int).tolist(), size_extract.astype(int).tolist())
    sitk.WriteImage(new_seg, out_dir+'vtk_data/vtk_throwout_' +case_name+'/'+str(N-n_old)+ '_'+str(sub)+'.vtk')

def write_img(new_img, removed_seg, image_out_dir, seg_out_dir, case_name, N, n_old, sub):
    sitk.WriteImage(new_img, image_out_dir + case_name +'_'+ str(N-n_old) +'_'+str(sub)+'.nii.gz')
    sitk.WriteImage(removed_seg, seg_out_dir + case_name +'_'+ str(N-n_old) +'_'+str(sub)+'.nii.gz')
    
def write_surface(new_surf_box, new_surf_sphere, seg_out_dir, case_name, N, n_old, sub):
    #write_geo(seg_out_dir.replace('masks','masks_surfaces_box') + case_dict['NAME']+'_' +str(N-n_old)+'_'+str(sub)+ '.vtp', new_surf_box)
    write_geo(seg_out_dir.replace('masks','masks_surfaces') + case_name +'_' +str(N-n_old)+'_'+str(sub)+ '.vtp', new_surf_sphere)

def write_centerline(new_cent, seg_out_dir, case_name, N, n_old, sub):
    write_geo(seg_out_dir.replace('masks','masks_centerlines') + case_name +'_' +str(N-n_old)+'_'+str(sub)+ '.vtp', new_cent)
    #pts_pd = points2polydata(stats_surf['OUTLETS'])
    #write_geo(out_dir+'vtk_data/vtk_' + case_dict['NAME']+'/' +str(N-n_old)+'_'+str(sub)+ '_caps.vtp', pts_pd)

def write_csv(csv_list, csv_list_val, out_dir, modality, trace_testing):
    import csv
    csv_file = "_Sample_stats.csv"
    if trace_testing: csv_file = '_test'+csv_file
    else: csv_file = '_train'+csv_file

    csv_columns = ["No",            "NAME",         "SIZE",     "RESOLUTION",   "ORIGIN", 
                    "SPACING",       "POINT_CENT",   "INDEX",    "SIZE_EXTRACT", "VOL_CENT", 
                    "DIFF_CENT",     "IM_MEAN",      "IM_STD",   "IM_MAX",       "IM_MIN",
                    "BLOOD_MEAN",    "BLOOD_STD",    "BLOOD_MAX","BLOOD_MIN",    "GT_MEAN", 
                    "GT_STD",        "GT_MAX",       "GT_MIN",   "LARGEST_MEAN", "LARGEST_STD",
                    "LARGEST_MAX",   "LARGEST_MIN",  "RADIUS",   "TANGENTX",     "TANGENTY", 
                    "TANGENTZ",      "BIFURCATION",  "NUM_VOX",  "OUTLETS",      "NUM_OUTLETS"]
    with open(out_dir+modality+csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_list:
            writer.writerow(data)
    if not trace_testing:
        with open(out_dir+modality+csv_file.replace('train','val'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_list_val:
                writer.writerow(data)

def write_csv_discrete_cent(csv_discrete_centerline, csv_discrete_centerline_val, outdir, modality, trace_testing):
    import csv
    csv_file = "_Discrete_Centerline.csv"
    if trace_testing: csv_file = '_test'+csv_file
    else: csv_file = '_train'+csv_file

    csv_columns = ["No", "NAME", "NUM_CENT", "STEPS"]
    with open(outdir+modality+csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_discrete_centerline:
            writer.writerow(data)
    if not trace_testing:
        with open(outdir+modality+csv_file.replace('train','val'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_discrete_centerline_val:
                writer.writerow(data)

def write_csv_outlet_stats(csv_outlet_stats, csv_outlet_stats_val, out_dir, modality, trace_testing):
    
    import csv
    csv_file = "_Outlet_Stats.csv"
    if trace_testing: csv_file = '_test'+csv_file
    else: csv_file = '_train'+csv_file
    import pdb; pdb.set_trace()
    csv_columns = ["NAME", "CENTER", "WIDTH", "SIZE"]
    with open(out_dir+modality+csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_outlet_stats:
            writer.writerow(data)
    if not trace_testing:
        with open(out_dir+modality+csv_file.replace('train','val'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_outlet_stats_val:
                writer.writerow(data)

def write_pkl_outlet_stats(pkl_outlet_stats, pkl_outlet_stats_val, out_dir, modality, trace_testing):
    import pickle
    pkl_file = "_Outlet_Stats.pkl"
    if trace_testing: pkl_file = '_test'+pkl_file
    else: pkl_file = '_train'+pkl_file

    with open(out_dir+modality+pkl_file, 'wb') as f:
        pickle.dump(pkl_outlet_stats, f)
    if not trace_testing:
        with open(out_dir+modality+pkl_file.replace('train','val'), 'wb') as f:
            pickle.dump(pkl_outlet_stats_val, f)