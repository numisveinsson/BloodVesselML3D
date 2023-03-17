import time
start_time = time.time()
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")

import numpy as np

import os
import random
import csv

from modules import vtk_functions as vf
from modules import sitk_functions as sf
from modules import io

from vtk.util.numpy_support import vtk_to_numpy as v2n
import SimpleITK as sitk

def print_info_file(global_config, cases, testing_samples, info_file_name):

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

def create_directories(output_folder, modality, trace_testing, write_vtk_samples, extract_vtps=False):

    if trace_testing:
        fns = ['_test']
    else:
        fns = ['_train', '_val']

    for fn in fns:

        try:
                os.mkdir(output_folder+modality+fn)
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+modality+fn+'_masks')
        except Exception as e: print(e)

    if write_vtk_samples:
        try:
            os.mkdir(output_folder+'vtk_data')
        except Exception as e: print(e)

    if extract_vtps:
        for fn in fns:
            try:
                os.mkdir(output_folder+modality+fn+'_masks_surfaces_sphere')
            except Exception as e: print(e)
            try:
                os.mkdir(output_folder+modality+fn+'_masks_surfaces_box')
            except Exception as e: print(e)
            try:
                os.mkdir(output_folder+modality+fn+'_masks_centerlines')
            except Exception as e: print(e)

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

def extract_surface(img, surface, center, radius):
    """
    Function to cut global surface
    into a local part
    """
    #import pdb; pdb.set_trace()
    stats = {}
    vtkimage = vf.exportSitk2VTK(img)
    surface_local_box = vf.bound_polydata_by_image(vtkimage[0], surface, 0)    
    surface_local_sphere = vf.bound_polydata_by_sphere(surface, center, radius)
    #surface_local_box = vf.get_largest_connected_polydata(surface_local_box)
    surface_local_sphere = vf.get_largest_connected_polydata(surface_local_sphere)

    outlets = vf.calc_caps(surface_local_sphere)
    stats['OUTLETS'] = outlets
    stats['NUM_OUTLETS'] = len(outlets)

    return stats, surface_local_box, surface_local_sphere

def extract_centerline(img, centerline):
    """
    Function to cut global centerline
    into a local part
    """
    stats = {}
    vtkimage = vf.exportSitk2VTK(img)
    cent_local = vf.bound_polydata_by_image(vtkimage[0], centerline, 0)
    cent_local = vf.get_largest_connected_polydata(cent_local)

    return stats, cent_local


def extract_subvolumes(reader_im, reader_seg, index_extract, size_extract, origin_im, spacing_im, number, name, O=None, sub=None, global_img=False):
    """"
    Function to extract subvolumes
    Both image data and GT segmentation
    Also calculates some statistics on
        the subvolumes of interest
    """

    index_extract = index_extract.astype(int).tolist()
    size_extract = size_extract.astype(int).tolist()

    new_img = sf.extract_volume(reader_im, index_extract, size_extract)
    new_seg = sf.extract_volume(reader_seg, index_extract, size_extract)

    #print("Original Seg")
    labels, means = sf.connected_comp_info(new_seg, False)
    #print("Seg w removed bodies")
    #labels1, means1 = sf.connected_comp_info(removed_seg)
    
    seed = np.rint(np.array(size_extract)/2).astype(int).tolist()
    removed_seg = sf.remove_other_vessels(new_seg, seed)

    im_np = sitk.GetArrayFromImage(new_img)
    seg_np = sitk.GetArrayFromImage(new_seg)
    rem_np = sitk.GetArrayFromImage(removed_seg)
    blood_np = im_np[seg_np>0.1]
    ground_truth = rem_np[seg_np>0.1]

    center_volume = (seed)*spacing_im + origin_im
    stats = {
    "No":number,                        "NAME": name, 
    "RESOLUTION": size_extract,
    "ORIGIN": origin_im.tolist(),       "SPACING": spacing_im.tolist(),              
    "INDEX": index_extract,             "SIZE_EXTRACT": size_extract,       
    "VOL_CENT": center_volume.tolist(), 
    "IM_MEAN":np.mean(im_np),           "IM_MIN":np.amin(im_np), 
    "IM_STD":np.std(im_np),             "IM_MAX":np.amax(im_np),
    }

    if not global_img:
        diff_cent = np.linalg.norm(locs[count] - center_volume)
        stats.update({"DIFF_CENT": diff_cent, "POINT_CENT": locs[count].tolist(), "SIZE": size_r*rads[count],
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
            O = O+1
        #mul_l.append(case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')
        #print("The sample has more than one label: " + case_dict['NAME'] +'_'+ str(N-n_old))

    return stats, new_img, new_seg, removed_seg, O


if __name__=='__main__':

    np.random.seed(0)
    random.seed(0)

    extract_volumes = True

    extract_vtps = True
    outlet_types_allowed = [3]

    write_samples = True
    write_vtk_samples = True
    write_vtk_throwout = True
    show_samples = False
    rotate_samples = False
    gt_resample = False
    resample_size = [64, 64, 64]

    global_config_file = "./config/global.yaml"
    global_config = io.load_yaml(global_config_file)
    modalities = global_config['MODALITY']

    for modality in modalities:
        
        modality = modality.lower()
        info_file_name = "info"+'_'+modality+dt_string+".txt"

        #type = global_config['TYPE'].lower()
        #modality = modality+'_'+type
        trace_testing = global_config['TESTING']
        out_dir = global_config['OUT_DIR']
        create_directories(out_dir, modality, trace_testing, write_vtk_samples, extract_vtps)

        mu_size = global_config['MU_SIZE']
        sigma_size = global_config['SIGMA_SIZE']
        mu_shift = global_config['MU_SHIFT']
        sigma_shift = global_config['SIGMA_SHIFT']

        image_out_dir_train = out_dir+modality+'_train/'
        seg_out_dir_train = out_dir+modality+'_train_masks/'
        image_out_dir_val = out_dir+modality+'_val/'
        seg_out_dir_val = out_dir+modality+'_val_masks/'

        image_out_dir_test = out_dir+modality+'_test/'
        seg_out_dir_test = out_dir+modality+'_test_masks/'

        cases, test_samples, bad_samples = define_cases(global_config, modality)

        info = {}
        N = 0 # keep total of extractions
        M = 0 # keep total of throwouts
        K = 0 # keep total of errors
        O = 0 # keep total of samples with multiple labels
        mul_l = []
        csv_list = []

        for i in cases:
            print(i)

        print_info_file(global_config, cases, test_samples, info_file_name)

        for case_fn in cases:

            ## Load data
            case_dict = io.load_yaml(case_fn)
            print(case_dict['NAME'])

            if write_vtk_samples:
                try:
                    os.mkdir(out_dir+'vtk_data/vtk_' + case_dict['NAME'])
                    os.mkdir(out_dir+'vtk_data/vtk_mask_' + case_dict['NAME'])
                    os.mkdir(out_dir+'vtk_data/vtk_throwout_' + case_dict['NAME'])
                except Exception as e: print(e)

            reader_seg = sf.read_image(case_dict['SEGMENTATION'])
            reader_im, origin_im, size_im, spacing_im = sf.import_image(case_dict['IMAGE'])

            ## Surface Caps
            global_surface = vf.read_geo(case_dict['SURFACE']).GetOutput()    # read in geometry
            surf_data = vf.collect_arrays(global_surface.GetCellData())       # collect arrays of cell data
            surf_locs = vf.get_location_cells(global_surface)                 # get locations of cell centroids
            try:
                ids_caps = np.where(surf_data['CapID']!= 0)[0]      # get ids of cells on caps
            except:
                ids_caps = np.where(surf_data['BC_FaceID']!= 0)[0]
            cap_locs = surf_locs[ids_caps]                          # get locations of cap cells

            ## Centerline
            global_centerline = vf.read_geo(case_dict['CENTERLINE']).GetOutput()         # read in geometry
            num_points = global_centerline.GetNumberOfPoints()               # number of points in centerline
            cent_data = vf.collect_arrays(global_centerline.GetPointData())
            c_loc = v2n(global_centerline.GetPoints().GetData())             # point locations as numpy array
            radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
            cent_id = cent_data['CenterlineId']
            bifurc_id = cent_data['BifurcationIdTmp']

            try:
                num_cent = len(cent_id[0]) # number of centerlines (one is assembled of multiple)
            except:
                num_cent = 1 # in the case of only one centerline

            ids_total = []
            m_old = M
            n_old = N
            k_old = K

            for ip in range(num_cent):

                ## If tracing test, save to test
                if trace_testing:
                    image_out_dir = image_out_dir_test
                    seg_out_dir = seg_out_dir_test
                ## Else, every 20 extractions have a probability to save to validation
                else:
                    rand = random.uniform(0,1)
                    print(" random is " + str(rand))
                    if rand < global_config['VALIDATION_PROP'] and ip != 0:
                        image_out_dir = image_out_dir_val
                        seg_out_dir = seg_out_dir_val
                    else:
                        image_out_dir = image_out_dir_train
                        seg_out_dir = seg_out_dir_train

                try:
                    ids = [i for i in range(num_points) if cent_id[i,ip]==1]    # ids of points belonging to centerline ip
                except:
                    ids = [i for i in range(num_points)]
                locs = c_loc[ids] # locations of those points
                rads = radii[ids] # radii at those locations
                bifurc = bifurc_id[ids]

                on_cent = True
                count = 0 # the point along centerline
                lengths = [0]
                lengths_prev = [0]
                print("\n ** Ip is " + str(ip)+"\n")
                while on_cent:
                    #print("\n--- %s seconds ---" % (time.time() - start_time))

                    if not (ids[count] in ids_total):
                        print('The point # along centerline is ' + str(count))
                        #print('The location is ' + str(locs[count]))

                        if bifurc[count] == 2:
                            save_bif = 1
                            n_samples = global_config['NUMBER_SAMPLES_BIFURC']
                        else:
                            save_bif = 0
                            n_samples = global_config['NUMBER_SAMPLES']

                        # Sample size(s) and shift(s)
                        sizes = np.random.normal(mu_size, sigma_size, n_samples)
                        sizes[0] = mu_size # Make first be correct size
                        shifts = np.random.normal(mu_shift, sigma_shift, n_samples)
                        shifts[0] = mu_shift # Make first be on centerline
                        #print("sizes are: " + str(sizes))
                        #print("shifts are: " + str(shifts))

                        # Calculate vectors
                        if count < len(locs)/2:
                            vec0 = locs[count+1] - locs[count]
                        else:
                            vec0 = locs[count] - locs[count-1]
                        vec1, vec2 = vf.calc_normal_vectors(vec0)

                        # Shift centers
                        centers = []
                        for sample in range(n_samples):
                            value = random.uniform(0,1)
                            vector = vec1*value + vec2*(1-value)
                            centers.append(locs[count]+shifts[sample]*vector*rads[count])

                        #print("Number of centers are " + str(len(centers)))
                        sub = 0
                        for sample in range(n_samples):

                            center = centers[sample]
                            size_r = sizes[sample]

                            size_extract, index_extract, voi_min, voi_max = sf.map_to_image(center, rads[count], size_r, origin_im, spacing_im, global_config['CAPFREE_PROP'])
                            is_inside = vf.voi_contain_caps(voi_min, voi_max, cap_locs)

                            if not is_inside: #1 and far_from:
                                print("*", end =" ")
                                try:
                                    name = case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub)
                                    if extract_volumes:
                                        stats, new_img, new_seg, removed_seg, O = extract_subvolumes(reader_im, reader_seg, index_extract, size_extract, origin_im, spacing_im, N, name, O, sub)
                                        if gt_resample:
                                            from modules.pre_process import resample_spacing
                                            removed_seg1 = resample_spacing(removed_seg, template_size=resample_size, order=1)[0]
                                            #if min(size_extract)<resample_size[0]:
                                            import pdb; pdb.set_trace()
                                            removed_seg = vf.clean_boundaries(sitk.GetArrayFromImage(removed_seg))
                                        
                                        if extract_vtps:
                                            stats_surf, new_surf_box, new_surf_sphere = extract_surface(new_img, global_surface, center, size_r*rads[count]/1.5)
                                            num_out = len(stats_surf['OUTLETS'])
                                            print(f"Outlets are: {num_out}")
                                            
                                            stats_cent, new_cent = extract_centerline(new_img, global_centerline)
                                            
                                            stats.update(stats_surf)
                                    else:
                                        stats = {"No":N, "NAME": name, "SIZE": size_r*rads[count], "RESOLUTION": size_extract,"ORIGIN": origin_im, "SPACING": spacing_im,}
                                    stats.update({"NUM_VOX": size_extract[0]*size_extract[1]*size_extract[2], "TANGENTX": (vec0/np.linalg.norm(vec0))[0], "TANGENTY": (vec0/np.linalg.norm(vec0))[1], "TANGENTZ": (vec0/np.linalg.norm(vec0))[2], "RADIUS":rads[count], "BIFURCATION":save_bif})

                                    if not extract_vtps or num_out in outlet_types_allowed:
                                        
                                        print('Printing sample')
                                        if write_vtk_samples:
                                            sitk.WriteImage(new_img, out_dir+'vtk_data/vtk_' + case_dict['NAME']+'/' +str(N-n_old)+'_'+str(sub)+ '.vtk')
                                            sitk.WriteImage(removed_seg*255, out_dir+'vtk_data/vtk_mask_'+ case_dict['NAME']+'/' +str(N-n_old)+'_'+str(sub)+ '.vtk')

                                        if write_samples:
                                            sitk.WriteImage(new_img, image_out_dir + case_dict['NAME'] +'_'+ str(N-n_old) +'_'+str(sub)+'.nii.gz')
                                            sitk.WriteImage(removed_seg, seg_out_dir + case_dict['NAME'] +'_'+ str(N-n_old) +'_'+str(sub)+'.nii.gz')
                                            if extract_vtps:
                                                vf.write_geo(seg_out_dir.replace('masks','masks_surfaces_box') + case_dict['NAME']+'_' +str(N-n_old)+'_'+str(sub)+ '.vtp', new_surf_box)
                                                vf.write_geo(seg_out_dir.replace('masks','masks_surfaces_sphere') + case_dict['NAME']+'_' +str(N-n_old)+'_'+str(sub)+ '.vtp', new_surf_sphere)
                                                vf.write_geo(seg_out_dir.replace('masks','masks_centerlines') + case_dict['NAME']+'_' +str(N-n_old)+'_'+str(sub)+ '.vtp', new_cent)
                                                pts_pd = vf.points2polydata(stats_surf['OUTLETS'])
                                                vf.write_geo(out_dir+'vtk_data/vtk_' + case_dict['NAME']+'/' +str(N-n_old)+'_'+str(sub)+ '_caps.vtp', pts_pd)

                                        if show_samples:
                                            sitk.Show(new_img, title="Image"+str(N), debugOn=True)
                                            sitk.Show(new_seg, title="Seg"+str(N), debugOn=True)
                                            sitk.Show(removed_seg, title="Removed Seg"+str(N), debugOn=True)

                                        #print('Finished: ' + case_dict['NAME'] +'_'+ str(N-n_old)+'_'+str(sub))
                                        csv_list.append(stats)

                                except Exception as e:
                                    print(e)
                                    #print("\n*****************************ERROR: did not save files for " +case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub))
                                    K = K+1

                            else:
                                print(".", end =" ")
                                #print(" No save - cap inside")
                                try:
                                    if write_vtk_samples and write_vtk_throwout:
                                        new_seg = sf.extract_volume(reader_seg, index_extract.astype(int).tolist(), size_extract.astype(int).tolist())
                                        sitk.WriteImage(new_seg, out_dir+'vtk_data/vtk_' + 'throwout_' +case_dict['NAME']+'/'+str(N-n_old)+ '_'+str(sub)+'.vtk')

                                    M=M+1
                                except Exception as e:
                                    #print("\n*****************************ERROR: did not save throwout for " +case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub))
                                    K = K+1
                            sub = sub +1

                        #if sub != 0:
                        print('\n Finished: ' + case_dict['NAME'] +'_'+ str(N-n_old))
                        #print(" " + str(sub) + " variations")
                        N = N+1
                    lengths_prev = np.cumsum(np.insert(np.linalg.norm(np.diff(locs[:count], axis=0), axis=1), 0, 0))
                    lengths = np.cumsum(np.insert(np.linalg.norm(np.diff(locs[count:], axis=0), axis=1), 0, 0))
                    move = 1
                    count = count+1
                    if count == len(locs):
                        on_cent = False
                        break
                    move_distance = global_config['MOVE_DIST']*rads[count]
                    if rads[count] >= 0.4: # Move slower for larger vessels
                        move_distance = move_distance * global_config['MOVE_SLOWER_LARGE']
                        #print("slowerrrrr")

                    while lengths[move] < move_distance :
                        count = count+1
                        move = move+1
                        if count == len(locs):
                            on_cent = False
                            break

                    #if on_cent:
                        #print('Next we move ' + str(lengths[move]))
                        #print(" ")

                ids_total.extend(ids)           # keep track of ids that have already been operated on

            print(case_dict['NAME'])
            print("\n****************** All done for this model! ******************")
            print("****************** " + str(N-n_old) +" extractions! ******************")
            print("****************** " + str(M-m_old) +" throwouts! ****************** \n")
            # f = open(out_dir +info_file_name, 'a')
            # f.write("\n ")
            # f,write("\n "+case_dict['NAME'])
            # f.write("\n "+str(N-n_old) +" extractions!")
            # f.write("\n "+str(M-m_old) +" throwouts!")
            # f.close()
            info[case_dict['NAME']] = [ N-n_old, M-m_old, K-k_old]
            f = open(out_dir +info_file_name,'a')
            f.write("\n "+ case_dict['NAME'])
            f.write("\n "+ str([ N-n_old, M-m_old, K-k_old ]))
            f.write("\n ")
            f.close()

        for i in info:
            print(i)
            print(info[i])
            print(" ")

        print("\n****************** All done for all models! ******************")
        print("****************** " + str(N) +" extractions! ******************")
        print("****************** " + str(M) +" throwouts! ******************")
        print("****************** " + str(K) +" errors in saving! ****************** \n")
        print("****************** O: " + str(O) +" have more than one label! They are: ****************** \n")
        for i in mul_l:
            print(i)

        f = open(out_dir +info_file_name,'a')
        f.write("\n *** " + str(N) +" extractions! ***")
        f.write("\n *** " + str(M) +" throwouts! ***")
        f.write("\n *** " + str(K) +" errors in saving! ***")
        f.write("\n *** " + str(O) +" have more than one label! ***")
        f.close()

        print("\n--- %s seconds ---" % (time.time() - start_time))
        print("Continue to write out csv file w info")

        csv_file = "_Sample_stats.csv"
        if trace_testing: csv_file = '_test'+csv_file

        csv_columns = ["No", "NAME", "SIZE","RESOLUTION", "ORIGIN", "SPACING", "POINT_CENT", "INDEX", "SIZE_EXTRACT", "VOL_CENT", "DIFF_CENT", "IM_MEAN",
        "IM_STD","IM_MAX","IM_MIN","BLOOD_MEAN","BLOOD_STD","BLOOD_MAX","BLOOD_MIN","GT_MEAN", "GT_STD", "GT_MAX", "GT_MIN",
        "LARGEST_MEAN","LARGEST_STD","LARGEST_MAX","LARGEST_MIN", 'RADIUS', 'TANGENTX', 'TANGENTY', 'TANGENTZ', 'BIFURCATION', 'NUM_VOX', 'OUTLETS', 'NUM_OUTLETS']
        with open(out_dir+modality+csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_list:
                writer.writerow(data)
