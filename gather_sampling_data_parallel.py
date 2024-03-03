import time
start_time = time.time()
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")

import sys
sys.stdout.flush()

from modules import vtk_functions as vf
from modules import sitk_functions as sf
from modules import io
from modules.sampling_functions import *
from modules.pre_process import *
from dataset_dirs.datasets import *

import multiprocessing

def sample_case(case_fn, global_config, out_dir, image_out_dir_train, seg_out_dir_train, image_out_dir_val, seg_out_dir_val, image_out_dir_test, seg_out_dir_test, info_file_name, modality):
    
    """ Sample a case and write out the results """
    N, M, K, O = 0, 0, 0, 0
    csv_list, csv_list_val = [], []

    # if global_config['WRITE_DISCRETE_CENTERLINE']:
    csv_discrete_centerline, csv_discrete_centerline_val = [], []
    
    # if global_config['WRITE_OUTLET_STATS']:
    csv_outlet_stats, csv_outlet_stats_val = [], []
    total_num_examples, total_num_examples_pos = 0, 0
    
    ## Load data
    case_dict = get_case_dict_dir(global_config['DATA_DIR'], case_fn, global_config['IMG_EXT'])

    ## Print case name and core label
    print(f"\n I am process {multiprocessing.current_process().name}")
    print(f"Starting case: {case_dict['NAME']}")
    time_now_case = time.time()

    if global_config['WRITE_VTK']:
        try:
            create_vtk_dir(out_dir, case_dict['NAME'])
        except Exception as e: print(e)

    ## Read Image Metadata
    reader_seg0 = sf.read_image(case_dict['SEGMENTATION'])
    reader_im0, origin_im0, size_im, spacing_im = sf.import_image(case_dict['IMAGE'])

    ## Surface Caps
    global_surface = vf.read_geo(case_dict['SURFACE']).GetOutput()
    if global_config['CAPFREE']:
        cap_locs = get_surf_caps(global_surface)
    
    ## Centerline
    global_centerline = vf.read_geo(case_dict['CENTERLINE']).GetOutput()
    num_points, c_loc, radii, cent_ids, bifurc_id, num_cent = sort_centerline(global_centerline)

    ids_total = []
    m_old = M
    n_old = N
    k_old = K

    for ip in range(num_cent):
        # Choose destination directory
        image_out_dir, seg_out_dir, val_port = choose_destination(global_config['TESTING'], global_config['VALIDATION_PROP'], image_out_dir_test, seg_out_dir_test, 
                                                                    image_out_dir_val, seg_out_dir_val, image_out_dir_train, seg_out_dir_train, ip)
        # Get ids on this centerline
        ids = cent_ids[ip]
        # skip if empty
        if len(ids) == 0:
            continue
        # Get info of those ids
        locs, rads, bifurc = c_loc[ids], radii[ids], bifurc_id[ids] # locations of those points, radii and bifurcation ids at those locations
        # Continue taking steps while still on centerline
        on_cent, count = True, 0 # count is the point along centerline
        print(f"\n--- {case_dict['NAME']} ---")
        print(f"--- Ip is {ip} / {num_cent} ---\n")
        while on_cent:
            # Only continue if we've not walked this centerline before
            if not (ids[count] in ids_total):
                print('The point # along centerline is ' + str(count))
                time_now = time.time()
                # check if we need to rotate the volume
                if global_config['ROTATE_VOLUMES']:
                    tangent = get_tangent(locs, count)
                    reader_im, reader_seg, origin_im = rotate_volumes(reader_im0, reader_seg0, tangent, locs[count])
                else:
                    reader_im, reader_seg = reader_im0, reader_seg0
                    origin_im = origin_im0

                # Calculate centers and sizes of samples for this point
                centers, sizes, save_bif, n_samples, vec0 = calc_samples(count, bifurc, locs, rads, global_config)
                sub = 0 # In the case of multiple samples at this point
                for sample in range(n_samples):
                    # Map each center and size to image data
                    center, size_r = centers[sample], sizes[sample]
                    size_extract, index_extract, voi_min, voi_max = sf.map_to_image(center, rads[count], size_r, origin_im, spacing_im, size_im, global_config['CAPFREE_PROP'])
                    # Check if a surface cap is in volume
                    if global_config['CAPFREE']:
                        is_inside = vf.voi_contain_caps(voi_min, voi_max, cap_locs)
                    else:
                        is_inside = False
                    # Continue if surface cap is not present
                    if not is_inside:
                        print("*", end =" ")
                        try:
                            name = case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub)
                            
                            # Extract volume
                            if global_config['EXTRACT_VOLUMES']:
                                stats, new_img, removed_seg, O = extract_subvolumes(reader_im, reader_seg, index_extract, size_extract, 
                                                                    origin_im, spacing_im, locs[count], rads[count], size_r, N, name, O, 
                                                                    remove_others=global_config['REMOVE_OTHER'])
                                if global_config['WRITE_SURFACE']:
                                    stats_surf, new_surf_box, new_surf_sphere = extract_surface(new_img, global_surface, center, size_r*rads[count])
                                    num_out = len(stats_surf['OUTLETS'])
                                    #print(f"Outlets are: {num_out}")
                                    stats.update(stats_surf)
                            else:
                                stats = create_base_stats(N, name, size_r, rads[count], size_extract, origin_im, spacing_im, index_extract, center)

                            stats = add_tangent_stats(stats, vec0, save_bif)
                            
                            if global_config['WRITE_SAMPLES']:
                                    # Write surface and centerline vtps
                                if (global_config['WRITE_SURFACE'] or global_config['WRITE_CENTERLINE']) and num_out in global_config['OUTLET_CLASSES']:
                                    if global_config['WRITE_SURFACE']:
                                        write_surface(new_surf_box, new_surf_sphere, seg_out_dir, case_dict['NAME'], N, n_old, sub)
                                    if global_config['WRITE_CENTERLINE']:
                                        _ , new_cent = extract_centerline(new_img, global_centerline)
                                        write_centerline(new_cent, seg_out_dir, case_dict['NAME'], N, n_old, sub)
                                if global_config['WRITE_IMG']:
                                    if global_config['RESAMPLE_VOLUMES']:
                                        removed_seg_re  = resample_spacing(removed_seg, template_size=global_config['RESAMPLE_SIZE'], order=1)[0]
                                        new_img_re  = resample_spacing(new_img, template_size=global_config['RESAMPLE_SIZE'], order=1)[0]
                                        write_img(new_img_re, removed_seg_re, image_out_dir, seg_out_dir, case_dict['NAME'], N, n_old, sub)
                                    else:
                                        write_img(new_img, removed_seg, image_out_dir, seg_out_dir, case_dict['NAME'], N, n_old, sub)
                                    
                                if global_config['WRITE_VTK']:
                                    write_vtk(new_img, removed_seg, out_dir, case_dict['NAME'], N, n_old, sub)

                            # Discretize centerline
                            if global_config['WRITE_DISCRETE_CENTERLINE']:
                                _ , new_cent = extract_centerline(new_img, global_centerline)
                                cent_stats = discretize_centerline(new_cent, new_img, N-n_old, sub, name, out_dir, global_config['DISCRETE_CENTERLINE_N_POINTS'])
                                if val_port:
                                    csv_discrete_centerline_val.append(cent_stats)
                                else:
                                    csv_discrete_centerline.append(cent_stats)
                            
                            # Outlet stats
                            if global_config['WRITE_OUTLET_STATS']:
                                stats_out, planes, planes_seg, pos_example = get_outlet_stats(stats, new_img, removed_seg, upsample=global_config['UPSAMPLE_OUTLET_IMG'])
                                
                                total_num_examples_pos += pos_example
                                total_num_examples += 6
                                print(f"Ratio of positive examples: {total_num_examples_pos/total_num_examples * 100:.2f}")
                                if global_config['WRITE_OUTLET_IMG']:
                                    write_2d_planes(planes, stats_out, image_out_dir)
                                    write_2d_planes(planes_seg, stats_out, seg_out_dir)
                                if val_port:
                                    for out_stats in stats_out:
                                        csv_outlet_stats_val.append(out_stats)
                                else:
                                    for out_stats in stats_out:
                                        csv_outlet_stats.append(out_stats)

                            # Append stats to csv list        
                            csv_list, csv_list_val = append_stats(stats, csv_list, csv_list_val, val_port)
                        
                        except Exception as e:
                            print(e)
                            #print("\n*****************************ERROR: did not save files for " +case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub))
                            K+=1

                    else:
                        print(".", end =" ")
                        #print(" No save - cap inside")
                        try:
                            if global_config['WRITE_VTK'] and global_config['WRITE_VTK_THROWOUT']:
                                write_vtk_throwout(reader_seg, index_extract, size_extract, out_dir, case_dict['NAME'], N, n_old, sub)
                            M+=1
                        except Exception as e:
                            print(e)
                            #print("\n*****************************ERROR: did not save throwout for " +case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub))
                            K+=1
                    sub +=1

                print('\n Finished: ' + case_dict['NAME'] +'_'+ str(N-n_old))
                print(f"Time for this point: {(time.time() - time_now):.2f} sec")
                #print(" " + str(sub) + " variations")
                N+=1

            count, on_cent = find_next_point(count, locs, rads, bifurc, global_config, on_cent)
            
        ids_total.extend(ids)           # keep track of ids that have already been operated on

    print_model_info(case_dict['NAME'],  N, n_old, M, m_old)
    # info[case_dict['NAME']] = [ N-n_old, M-m_old, K-k_old]
    print(f"Total time for this case: {(time.time() - time_now_case):.2f} sec")
    print_into_info(info_file_name, case_dict['NAME'], N, n_old, M, m_old, K, k_old, out_dir)
    write_csv(csv_list, csv_list_val, modality, global_config)
    if global_config['WRITE_DISCRETE_CENTERLINE']:
        write_csv_discrete_cent(csv_discrete_centerline, csv_discrete_centerline_val, modality, global_config)

    if global_config['WRITE_OUTLET_STATS']:
        write_csv_outlet_stats(csv_outlet_stats, csv_outlet_stats_val, modality, global_config)
        write_pkl_outlet_stats(csv_outlet_stats, csv_outlet_stats_val, modality, global_config)

    return csv_list, csv_list_val, csv_discrete_centerline, csv_discrete_centerline_val, csv_outlet_stats, csv_outlet_stats_val

if __name__=='__main__':

    global_config = io.load_yaml("./config/global.yaml")
    modalities = global_config['MODALITY']

    out_dir = global_config['OUT_DIR']
    # sys.stdout = open(out_dir+"/log.txt", "w")

    for modality in modalities:

        cases = create_dataset(global_config, modality)

        modality = modality.lower()
        info_file_name = "info"+'_'+modality+dt_string+".txt"
        
        create_directories(out_dir, modality, global_config)

        image_out_dir_train = out_dir+modality+'_train/'
        seg_out_dir_train = out_dir+modality+'_train_masks/'
        image_out_dir_val = out_dir+modality+'_val/'
        seg_out_dir_val = out_dir+modality+'_val_masks/'

        image_out_dir_test = out_dir+modality+'_test/'
        seg_out_dir_test = out_dir+modality+'_test_masks/'

        info = {}
        N, M, K, O = 0,0,0,0 # keep total of extractions, throwouts, errors, total of samples with multiple labels
        csv_list, csv_list_val = [], []

        print(f"\n--- {modality} ---")
        print(f"--- {len(cases)} cases ---")
        for i in cases:
            print(f"Case: {i}")
        
        print_info_file(global_config, cases, global_config['TEST_CASES'], info_file_name)

        # Multiprocessing
        num_cores = 5 #multiprocessing.cpu_count()
        print(f"Number of cores: {num_cores}")
        pool = multiprocessing.Pool(num_cores)
        results = [pool.apply_async(sample_case, args=(case, global_config, out_dir, image_out_dir_train, seg_out_dir_train, image_out_dir_val, seg_out_dir_val, image_out_dir_test, seg_out_dir_test, info_file_name, modality)) for case in cases]
        pool.close()
        pool.join()
        # Collect results
        # for result in results:
        #     csv_list, csv_list_val, csv_discrete_centerline, csv_discrete_centerline_val, csv_outlet_stats, csv_outlet_stats_val = result.get()

        # print_all_done(info, N, M, K, O)
            # write_csv(csv_list, csv_list_val, modality, global_config)
            


        print_into_info_all_done(info_file_name, N, M, K, O, out_dir, start_time=start_time)
        print(f"\n--- {(time.time() - start_time)/60:.2f} min ---")
        print("Continue to write out csv file w info")