import time
start_time = time.time()

import numpy as np
import os

from modules import vtk_functions as vf
from modules import sitk_functions as sf
from modules import io
from modules import vascular_data as vd

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
import vtk
import SimpleITK as sitk

if __name__=='__main__':

    write_samples = True
    write_vtk_samples = True
    show_samples = False
    rotate_samples = False

    testing_samples = ['./cases/case.0176_0000.yml', './cases/case.0146_1001.yml', './cases/case.0002_0001.yml', './cases/case.0118_1000.yml']

    prop = 0.5 # how much of volume must be cap free, per sidelength
    move_slower = 1/4 # how much slower for bigger vessels

    global_config_file = "./config/global.yaml"
    global_config = io.load_yaml(global_config_file)

    image_out_dir = global_config['IMAGE_OUT_DIR'] +'/'
    seg_out_dir = global_config['SEG_OUT_DIR'] +'/'

    cases = os.listdir(global_config['CASES_DIR'])
    cases = [global_config['CASES_DIR']+'/'+f for f in cases if 'case.' in f]

    # cases.remove('./cases/case.0005_1001.yml')
    # cases.remove('./cases/case.0119_0001.yml')
    # cases.remove('./cases/case.0158_0001.yml')
    cases.remove('./cases/case.0140_2001.yml')
    cases = [i for i in cases if i not in testing_samples]

    cases = testing_samples

    info = {}
    N = 0 # keep total of extractions
    M = 0 # keep total of throwouts
    K = 0 # keep total of errors
    O = 0 # keep total of samples with multiple labels
    mul_l = []
    csv_list = []
    # print(cases)

    for case_fn in cases:

        ## Load data
        case_dict = io.load_yaml(case_fn)
        print(case_dict['NAME'])

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

        size_r = global_config['SIZE_RADIUS'] # Size of volume in radii at each point

        for ip in range(num_cent):

            try:
                ids = [i for i in range(num_points) if cent_id[i,ip]==1]    # ids of points belonging to centerline ip
            except:
                ids = [i for i in range(num_points)]
            locs = c_loc[ids]                                           # locations of those points
            rads = radii[ids]                                           # radii at those locations

            on_cent = True
            count = 0

            while on_cent:

                if not (ids[count] in ids_total):
                    #print('The point # along centerline is ' + str(count))
                    #print('The location is ' + str(locs[count]))

                    size_extract, index_extract, voi_min, voi_max = sf.map_to_image(locs[count], rads[count], size_r, origin_im, spacing_im, prop)

                    cap_in_volume = vf.voi_contain_caps(voi_min, voi_max, cap_locs)

                    if not cap_in_volume:
                        #print("\n It's cap free!\n")
                        #print('The size of VOI is ' + str(size_r*rads[count]))

                        try:

                            # import pdb; pdb.set_trace()
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
                            "ORIGIN": origin_im, "SPACING": spacing_im, "POINT_CENT": locs[count],
                            "VOL_CENT": center_volume, "DIFF_CENT": np.linalg.norm(locs[count] - center_volume)}
                            stats.update({"IM_MEAN":np.mean(im_np), "IM_STD":np.std(im_np), "IM_MAX":np.amax(im_np),
                            "IM_MIN":np.amin(im_np),"BLOOD_MEAN":np.mean(blood_np),"BLOOD_STD":np.std(blood_np),
                            "BLOOD_MAX":np.amax(blood_np),"BLOOD_MIN":np.amin(blood_np), "GT_MEAN": np.mean(ground_truth),
                            "GT_STD": np.std(ground_truth), "GT_MAX": np.amax(ground_truth), "GT_MIN": np.amin(ground_truth) })

                            if len(means) != 1:
                                larg_np = sitk.GetArrayFromImage(removed_seg)
                                rem_np = im_np[larg_np>0.1]
                                stats_rem = {"LARGEST_MEAN":np.mean(rem_np),"LARGEST_STD":np.std(rem_np),
                                "LARGEST_MAX":np.amax(rem_np),"LARGEST_MIN":np.amin(rem_np)}
                                stats.update(stats_rem)
                                O = O+1
                                mul_l.append(case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')
                                #print("The sample has more than one label: " + case_dict['NAME'] +'_'+ str(N-n_old))

                            if write_vtk_samples:
                                sitk.WriteImage(new_img, image_out_dir.replace('ct_train','vtk_data').replace('ct_val', 'vtk_data') + case_dict['NAME']+'_' +str(N-n_old)+ '.vtk')
                                #sitk.WriteImage(new_seg, image_out_dir.replace('ct_train','vtk_data') +'mask_'+ case_dict['NAME']+'_' +str(N-n_old)+ '.vtk')
                                sitk.WriteImage(removed_seg*255, image_out_dir.replace('ct_train','vtk_data').replace('ct_val', 'vtk_data') +'removed_mask_'+ case_dict['NAME']+'_' +str(N-n_old)+ '.vtk')

                            if write_samples:
                                sitk.WriteImage(new_img, image_out_dir + case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')
                                sitk.WriteImage(removed_seg, seg_out_dir + case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')

                            if show_samples:
                                sitk.Show(new_img, title="Image"+str(N), debugOn=True)
                                sitk.Show(new_seg, title="Seg"+str(N), debugOn=True)
                                sitk.Show(removed_seg, title="Removed Seg"+str(N), debugOn=True)
                                import pdb; pdb.set_trace()

                            N = N+1
                            print('Finished: ' + case_dict['NAME'] +'_'+ str(N-n_old))
                            csv_list.append(stats)
                        except:
                            print("\n*****************************ERROR: did not save files for " +case_dict['NAME']+'_'+str(N-n_old))
                            K = K+1

                    else:
                        #print("\n There's a cap inside! \n")
                        try:
                            new_seg = sf.extract_volume(case_dict['SEGMENTATION'], index_extract.astype(int).tolist(), size_extract.astype(int).tolist())

                            if write_vtk_samples:
                                sitk.WriteImage(new_seg, image_out_dir.replace('ct_train','vtk_throwouts') + case_dict['NAME']+'_'+str(N-n_old)+ '.vtk')

                            M=M+1
                        except:
                            print("\n*****************************ERROR: did not save throwout for " +case_dict['NAME']+'_'+str(N-n_old))
                            K = K+1

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

        info[case_dict['NAME']] = [ N-n_old, M-m_old, K-k_old]


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

    import csv
    csv_columns = ["No", "NAME", "SIZE","RESOLUTION", "ORIGIN", "SPACING", "POINT_CENT", "VOL_CENT", "DIFF_CENT", "IM_MEAN",
    "IM_STD","IM_MAX","IM_MIN","BLOOD_MEAN","BLOOD_STD","BLOOD_MAX","BLOOD_MIN","GT_MEAN", "GT_STD", "GT_MAX", "GT_MIN",
    "LARGEST_MEAN","LARGEST_STD","LARGEST_MAX","LARGEST_MIN"]
    csv_file = "Sample_stats.csv"
    try:
        with open(image_out_dir.replace('ct_train','vtk_data').replace('ct_val','vtk_data')+csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_list:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    import pdb; pdb.set_trace()
    # img = sitk.ReadeImage('my_input.png')
    # sitk.WriteImage(img, 'my_output.jpg')
