import time
start_time = time.time()

import numpy as np
import os
import random
import csv

from modules import vtk_functions as vf
from modules import sitk_functions as sf
from modules import io

from vtk.util.numpy_support import vtk_to_numpy as v2n
import SimpleITK as sitk

def create_directories(output_folder, modality, trace_testing, write_vtk_samples):

    if trace_testing:
        try:
            os.mkdir(output_folder+modality+'_test')
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+modality+'_test_masks')
        except Exception as e: print(e)
    else:
        try:
            os.mkdir(output_folder+modality+'_train')
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+modality+'_train_masks')
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+modality+'_val')
        except Exception as e: print(e)
        try:
            os.mkdir(output_folder+modality+'_val_masks')
        except Exception as e: print(e)

    if write_vtk_samples:
        try:
            os.mkdir(output_folder+'vtk_data')
        except Exception as e: print(e)


def extract_subvolumes(reader_im, reader_seg, index_extract, size_extract, origin_im, spacing_im, O):
    """"
    Function to extract subvolumes
    Both image data and GT segmentation
    Also calculates some statistics on
        the subvolumes of interest
    """
    new_img = sf.extract_volume(reader_im, index_extract.astype(int).tolist(), size_extract.astype(int).tolist())
    new_seg = sf.extract_volume(reader_seg, index_extract.astype(int).tolist(), size_extract.astype(int).tolist())

    seed = np.rint(np.array(size_extract)/2).astype(int).tolist()
    removed_seg = sf.remove_other_vessels(new_seg, seed)

    #print("Original Seg")
    labels, means = sf.connected_comp_info(new_seg, False)
    #print("Seg w removed bodies")
    #labels1, means1 = sf.connected_comp_info(removed_seg)

    im_np = sitk.GetArrayFromImage(new_img)
    seg_np = sitk.GetArrayFromImage(new_seg)
    rem_np = sitk.GetArrayFromImage(removed_seg)
    blood_np = im_np[seg_np>0.1]
    ground_truth = rem_np[seg_np>0.1]

    center_volume = (seed)*spacing_im + origin_im

    stats = {"No":N, "NAME": case_dict['NAME']+'_'+str(N-n_old), "SIZE": size_r*rads[count], "RESOLUTION": size_extract,
    "ORIGIN": origin_im,            "SPACING": spacing_im,              "POINT_CENT": locs[count],
    "VOL_CENT": center_volume,      "DIFF_CENT": np.linalg.norm(locs[count] - center_volume),
    "IM_MEAN":np.mean(im_np),       "IM_STD":np.std(im_np),             "IM_MAX":np.amax(im_np),
    "IM_MIN":np.amin(im_np),        "BLOOD_MEAN":np.mean(blood_np),     "BLOOD_STD":np.std(blood_np),
    "BLOOD_MAX":np.amax(blood_np),  "BLOOD_MIN":np.amin(blood_np),      "GT_MEAN": np.mean(ground_truth),
    "GT_STD": np.std(ground_truth), "GT_MAX": np.amax(ground_truth),    "GT_MIN": np.amin(ground_truth) }

    if len(means) != 1:
        larg_np = sitk.GetArrayFromImage(removed_seg)
        rem_np = im_np[larg_np>0.1]
        stats_rem = {"LARGEST_MEAN":np.mean(rem_np),"LARGEST_STD":np.std(rem_np),
        "LARGEST_MAX":np.amax(rem_np),"LARGEST_MIN":np.amin(rem_np)}
        stats.update(stats_rem)
        O = O+1
        #mul_l.append(case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')
        #print("The sample has more than one label: " + case_dict['NAME'] +'_'+ str(N-n_old))

    return stats, new_img, new_seg, removed_seg, O


if __name__=='__main__':

    write_samples = True
    write_vtk_samples = True
    write_vtk_throwout = True
    show_samples = False
    rotate_samples = False
    trace_testing = False

    global_config_file = "./config/global.yaml"
    global_config = io.load_yaml(global_config_file)
    modality = global_config['MODALITY'].lower()

    out_dir = global_config['OUT_DIR']
    create_directories(out_dir, modality, trace_testing, write_vtk_samples)

    size_validation = 0.15
    testing_samples = ['./cases/case.0176_0000.yml', './cases/case.0146_1001.yml', './cases/case.0002_0001.yml', './cases/case.0005_1001.yml']

    prop = 0.25 # how much of volume must be cap free, per sidelength
    move_slower = 1/4 # how much slower for bigger vessels
    mu_size = 5
    sigma_size = 1
    mu_shift = 0
    sigma_shift = 0.8
    n_samples = 2

    image_out_dir_train = out_dir+modality+'_train/'
    seg_out_dir_train = out_dir+modality+'_train_masks/'
    image_out_dir_val = out_dir+modality+'_val/'
    seg_out_dir_val = out_dir+modality+'_val_masks/'

    image_out_dir_test = out_dir+modality+'_test/'
    seg_out_dir_test = out_dir+modality+'_test_masks/'

    cases = os.listdir(global_config['CASES_DIR'])
    cases = [global_config['CASES_DIR']+'/'+f for f in cases if 'case.' in f]

    # cases.remove('./cases/case.0005_1001.yml')
    # cases.remove('./cases/case.0119_0001.yml')
    # cases.remove('./cases/case.0158_0001.yml')
    cases.remove('./cases/case.0140_2001.yml') # faulty example
    cases = [i for i in cases if i not in testing_samples]

    if trace_testing:
        cases = testing_samples

    info = {}
    N = 0 # keep total of extractions
    M = 0 # keep total of throwouts
    K = 0 # keep total of errors
    O = 0 # keep total of samples with multiple labels
    mul_l = []
    csv_list = []

    for i in cases:
        print(i)

    f = open(out_dir +"info.txt","w+")
    f.write("Proportion cap free: " + str(prop))
    f.write("\n Relative slower for large vessels: " + str(move_slower))
    f.write("\n Testing models not included: ")
    for i in range(len(testing_samples)):
        f.write(" \n   Model " + str(i) + " : " + testing_samples[i])
    f.close()
    import pdb; pdb.set_trace()
    for case_fn in cases[-2:]:

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
        surf = vf.read_geo(case_dict['SURFACE']).GetOutput()    # read in geometry
        surf_data = vf.collect_arrays(surf.GetCellData())       # collect arrays of cell data
        surf_locs = vf.get_location_cells(surf)                 # get locations of cell centroids
        try:
            ids_caps = np.where(surf_data['CapID']!= 0)[0]      # get ids of cells on caps
        except:
            ids_caps = np.where(surf_data['BC_FaceID']!= 0)[0]
        cap_locs = surf_locs[ids_caps]                          # get locations of cap cells

        ## Centerline
        cent = vf.read_geo(case_dict['CENTERLINE']).GetOutput()         # read in geometry
        num_points = cent.GetNumberOfPoints()               # number of points in centerline
        cent_data = vf.collect_arrays(cent.GetPointData())
        c_loc = v2n(cent.GetPoints().GetData())             # point locations as numpy array
        radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
        cent_id = cent_data['CenterlineId']

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
                if rand < size_validation and ip != 0:
                    image_out_dir = image_out_dir_val
                    seg_out_dir = seg_out_dir_val
                else:
                    image_out_dir = image_out_dir_train
                    seg_out_dir = seg_out_dir_train

            try:
                ids = [i for i in range(num_points) if cent_id[i,ip]==1]    # ids of points belonging to centerline ip
            except:
                ids = [i for i in range(num_points)]
            locs = c_loc[ids]                                           # locations of those points
            rads = radii[ids]                                           # radii at those locations

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

                    # Sample size(s) and shift(s)
                    sizes = np.random.normal(mu_size, sigma_size, n_samples)
                    shifts = np.random.normal(mu_shift, sigma_shift, n_samples)
                    #print("sizes are: " + str(sizes))
                    #print("shifts are: " + str(shifts))

                    # Calculate vectors
                    if count < len(locs)/2:
                        vec0 = locs[count+1] - locs[count]
                    else:
                        vec0 = locs[count] - locs[count-1]
                    vec1, vec2 = vf.calc_normal_vectors(vec0)

                    # Shift centers
                    centers = [] #[locs[count]]
                    for sample in range(n_samples):
                        value = random.uniform(0,1)
                        vector = vec1*value + vec2*(1-value)
                        centers.append(locs[count]+shifts[sample]*vector*rads[count])

                    from test import points2polydata
                    vf.write_geo('/Users/numisveinsson/Downloads/' + str(N)+'point_centerline.vtp', points2polydata(centers))
                    #print("Number of centers are " + str(len(centers)))
                    sub = 0
                    for sample in range(n_samples):

                        center = centers[sample]
                        size_r = sizes[sample]

                        size_extract, index_extract, voi_min, voi_max = sf.map_to_image(center, rads[count], size_r, origin_im, spacing_im, prop)
                        is_inside = vf.voi_contain_caps(voi_min, voi_max, cap_locs)
                        # is_inside1 = True
                        # if is_inside:
                        #     size_extract, index_extract, voi_min, voi_max = sf.map_to_image(center, rads[count], size_r, origin_im, spacing_im, prop2)
                        #     is_inside1 = vf.voi_contain_caps(voi_min, voi_max, cap_locs)
                        #     far_from = lengths_prev[-1] > rads[count] and lengths[-1] > rads[count]
                        ## Check if cap is in volume
                        if not is_inside: #1 and far_from:
                            print("*", end =" ")
                            try:
                                stats, new_img, new_seg, removed_seg, O = extract_subvolumes(reader_im, reader_seg, index_extract, size_extract, origin_im, spacing_im, O)
                                stats.update({"TANGENT": vec0/np.linalg.norm(vec0), "RADIUS":rads[count]})
                                if write_vtk_samples:
                                    sitk.WriteImage(new_img, out_dir+'vtk_data/vtk_' + case_dict['NAME']+'/' +str(N-n_old)+'_'+str(sub)+ '.vtk')
                                    sitk.WriteImage(removed_seg*255, out_dir+'vtk_data/vtk_mask_'+ case_dict['NAME']+'/' +str(N-n_old)+'_'+str(sub)+ '.vtk')

                                if write_samples:
                                    sitk.WriteImage(new_img, image_out_dir + case_dict['NAME'] +'_'+ str(N-n_old) +'_'+str(sub)+'.nii.gz')
                                    sitk.WriteImage(removed_seg, seg_out_dir + case_dict['NAME'] +'_'+ str(N-n_old) +'_'+str(sub)+'.nii.gz')

                                if show_samples:
                                    sitk.Show(new_img, title="Image"+str(N), debugOn=True)
                                    sitk.Show(new_seg, title="Seg"+str(N), debugOn=True)
                                    sitk.Show(removed_seg, title="Removed Seg"+str(N), debugOn=True)

                                #print('Finished: ' + case_dict['NAME'] +'_'+ str(N-n_old)+'_'+str(sub))
                                csv_list.append(stats)
                            except Exception as e:
                                #print(e)
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
                                #print(e)
                                #print("\n*****************************ERROR: did not save throwout for " +case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub))
                                K = K+1
                        sub = sub +1

                    if sub != 0:
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
                    move_distance = move_distance * move_slower
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
        # f = open(out_dir +"info.txt", 'a')
        # f.write("\n ")
        # f,write("\n "+case_dict['NAME'])
        # f.write("\n "+str(N-n_old) +" extractions!")
        # f.write("\n "+str(M-m_old) +" throwouts!")
        # f.close()
        info[case_dict['NAME']] = [ N-n_old, M-m_old, K-k_old]
        f = open(out_dir +"info.txt",'a')
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

    print("\n--- %s seconds ---" % (time.time() - start_time))

    csv_file = "Sample_stats.csv"
    csv_columns = ["No", "NAME", "SIZE","RESOLUTION", "ORIGIN", "SPACING", "POINT_CENT", "VOL_CENT", "DIFF_CENT", "IM_MEAN",
    "IM_STD","IM_MAX","IM_MIN","BLOOD_MEAN","BLOOD_STD","BLOOD_MAX","BLOOD_MIN","GT_MEAN", "GT_STD", "GT_MAX", "GT_MIN",
    "LARGEST_MEAN","LARGEST_STD","LARGEST_MAX","LARGEST_MIN", 'RADIUS', 'TANGENT']
    with open(image_out_dir_train.replace(modality+'_train','vtk_data')+csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in csv_list:
            writer.writerow(data)

    import pdb; pdb.set_trace()
    # img = sitk.ReadeImage('my_input.png')
    # sitk.WriteImage(img, 'my_output.jpg')


    # def trace_choose_subvolumes(global_config, testing_samples, trace_testing =False, prop_cap_free =1, move_slower_larger =1 ):
    #     """
    #     Function to trace the centerlines and get
    #     data on centers and size of subvolumes to
    #     extract
    #     args:
    #
    #     return:
    #
    #     """
    #     if trace_testing:
    #         # Trace testing models
    #         cases = testing_samples
    #     else:
    #         # Trace models for training/validation
    #         cases = os.listdir(global_config['CASES_DIR'])
    #         cases = [global_config['CASES_DIR']+'/'+f for f in cases if 'case.' in f]
    #         cases = [i for i in cases if i not in testing_samples]
    #
    #     size_r = global_config['SIZE_RADIUS'] # Size of volume in radii at each point
    #
    #     for case_fn in cases:
    #
    #         ## Load file locations
    #         case_dict = io.load_yaml(case_fn)
    #         print(case_dict['NAME'])
    #         ## Load image info
    #         reader_im, origin_im, size_im, spacing_im = sf.import_image(case_dict['IMAGE'])
    #
    #         ## Load surface caps data
    #         surf = vf.read_geo(case_dict['SURFACE']).GetOutput()    # read in geometry
    #         surf_data = vf.collect_arrays(surf.GetCellData())       # collect arrays of cell data
    #         surf_locs = vf.get_location_cells(surf)                 # get locations of cell centroids
    #         try:
    #             ids_caps = np.where(surf_data['CapID']!= 0)[0]      # get ids of cells on caps
    #         except:
    #             ids_caps = np.where(surf_data['BC_FaceID']!= 0)[0]
    #         cap_locs = surf_locs[ids_caps]                          # get locations of cap cells
    #
    #         ## Load centerline data
    #         cent = vf.read_geo(case_dict['CENTERLINE']).GetOutput() # read in geometry
    #         num_points = cent.GetNumberOfPoints()                   # number of points in centerline
    #         cent_data = vf.collect_arrays(cent.GetPointData())
    #         c_loc = v2n(cent.GetPoints().GetData())                 # point locations as numpy array
    #         radii = cent_data['MaximumInscribedSphereRadius']       # Max Inscribed Sphere Radius as numpy array
    #         cent_id = cent_data['CenterlineId']
    #         try:
    #             num_cent = len(cent_id[0])                          # number of centerlines (one is assembled of multiple)
    #         except:
    #             num_cent = 1                                        # in the case of only one centerline
    #
    #         ids_total = [] # Keep track of ids already used
    #         ## For each subcenterline
    #         for ip in range(num_cent):
    #             try:
    #                 ids = [i for i in range(num_points) if cent_id[i,ip]==1]    # ids of points belonging to centerline ip
    #             except:
    #                 ids = [i for i in range(num_points)]
    #             locs = c_loc[ids]                                           # locations of those points
    #             rads = radii[ids]                                           # radii at those locations
    #
    #             on_cent = True
    #             count = 0
    #
    #             ## While on the subcenterline
    #             while on_cent:
    #                 ## If id has not been used before
    #                 if not (ids[count] in ids_total):
    #                     ## Map point and radius to subvolume location and size
    #                     size_extract, index_extract, voi_min, voi_max = sf.map_to_image(locs[count], rads[count], size_r, origin_im, spacing_im, prop_cap_free)
    #                     ## Check if caps
    #                     if not vf.voi_contain_caps(voi_min, voi_max, cap_locs):
    #
    #     return 0
    #
    # def extract_subvolumes():
    #
    #     return
