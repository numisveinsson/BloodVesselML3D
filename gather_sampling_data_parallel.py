import time
from datetime import datetime

import argparse
import sys
import random
import os
import pandas as pd

from modules import vtk_functions as vf
from modules import sitk_functions as sf
from modules import io
from modules.sampling_functions import (
    create_vtk_dir, get_surf_caps, sort_centerline, choose_destination,
    get_tangent, rotate_volumes, calc_samples, extract_subvolumes,
    extract_surface, get_outlet_stats, write_2d_planes,
    write_img, write_vtk, write_vtk_throwout, find_next_point,
    create_base_stats, add_tangent_stats, extract_centerline,
    discretize_centerline, write_surface, write_centerline, write_csv,
    write_csv_discrete_cent, write_csv_outlet_stats, write_pkl_outlet_stats,
    print_model_info, print_info_file, get_cross_sectional_planes,
    print_into_info, print_into_info_all_done,
    append_stats, create_directories, print_csv_stats,
    get_longest_centerline, sort_centerline_by_length, flip_radius,
    get_proj_traj
    )
from modules.pre_process import resample_spacing
from dataset_dirs.datasets import get_case_dict_dir, create_dataset

import multiprocessing

start_time = time.time()
now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
sys.stdout.flush()


def sample_case(case_fn, global_config, out_dir, image_out_dir_train,
                seg_out_dir_train, image_out_dir_val, seg_out_dir_val,
                image_out_dir_test, seg_out_dir_test, info_file_name,
                modality):

    """ Sample a case and write out the results """

    # Check if case is in done.txt
    if os.path.exists(out_dir+"done.txt"):
        with open(out_dir+"done.txt", "r") as f:
            done = f.read().splitlines()
            f.close()
        if case_fn in done:
            print(f"Skipping {case_fn}")
            return (case_fn, [], [], [], [], [], [])

    if global_config['WRITE_TRAJECTORIES']:
        traj_list = []
        num_trajs = 0

    N, M, K, O, skipped = 0, 0, 0, 0, 0
    csv_list, csv_list_val = [], []

    # If global_config['WRITE_DISCRETE_CENTERLINE']:
    csv_discrete_centerline, csv_discrete_centerline_val = [], []

    # if global_config['WRITE_OUTLET_STATS']:
    csv_outlet_stats, csv_outlet_stats_val = [], []
    total_num_examples, total_num_examples_pos = 0, 0

    # Load data
    case_dict = get_case_dict_dir(global_config['DATA_DIR'], case_fn,
                                  global_config['IMG_EXT'])

    # Print case name and core label
    print(f"\n I am process {multiprocessing.current_process().name}")
    print(f"Starting case: {case_dict['NAME']}")
    time_now_case = time.time()

    if global_config['WRITE_VTK']:
        try:
            create_vtk_dir(out_dir, case_dict['NAME'], global_config['CAPFREE'])
        except Exception as e:
            print(e)

    # Read Image Metadata
    reader_seg0 = sf.read_image(case_dict['SEGMENTATION'])
    (reader_im0, origin_im0,
     size_im, spacing_im) = sf.import_image(case_dict['IMAGE'])

    # Surface Caps
    if global_config['CAPFREE'] or global_config['WRITE_SURFACE']:
        global_surface = vf.read_geo(case_dict['SURFACE']).GetOutput()
        if global_config['CAPFREE']:
            cap_locs = get_surf_caps(global_surface)

    # Centerline
    global_centerline = vf.read_geo(case_dict['CENTERLINE']).GetOutput()
    (_, c_loc, radii, cent_ids,
     bifurc_id, num_cent) = sort_centerline(global_centerline)

    # Check radii and add if necessary
    radii += global_config['RADIUS_ADD']
    radii *= global_config['RADIUS_SCALE']

    ids_total = []
    m_old = M
    n_old = N
    k_old = K

    ip_longest = get_longest_centerline(cent_ids, c_loc)
    print(f"Case: {case_fn}: Longest centerline is {ip_longest}"
          + f" with {len(cent_ids[ip_longest])} points")

    # Sort centerlines by length, starting with longest
    ips_sorted_length = sort_centerline_by_length(cent_ids, c_loc)

    # Make all cent_ids start where the radius is larger
    cent_ids = flip_radius(cent_ids, radii)

    # Loop over centerlines
    for ip in ips_sorted_length:
        # Choose destination directory
        (image_out_dir, seg_out_dir,
         val_port) = choose_destination(global_config['TESTING'],
                                        global_config['VALIDATION_PROP'],
                                        image_out_dir_test, seg_out_dir_test,
                                        image_out_dir_val, seg_out_dir_val,
                                        image_out_dir_train, seg_out_dir_train,
                                        ip)
        # Get ids on this centerline
        ids = cent_ids[ip]
        # skip if empty
        if len(ids) == 0:
            continue
        # Get info of those ids
        # locations of those points, radii
        # and bifurcation ids at those locations
        locs, rads, bifurc = c_loc[ids], radii[ids], bifurc_id[ids]
        # Continue taking steps while still on centerline
        on_cent, count = True, 0  # count is the point along centerline
        print(f"\n--- {case_dict['NAME']} ---")
        print(f"--- Ip is {ip} / {num_cent} ---\n")
        while on_cent:
            # Only continue if we've not walked this centerline before
            if not (ids[count] in ids_total):

                print('The point # along centerline is ' + str(count))
                print('The radius is ' + str(rads[count]))

                time_now = time.time()
                # check if we need to rotate the volume
                if global_config['ROTATE_VOLUMES']:
                    print("Rotating volume")
                    tangent = get_tangent(locs, count)
                    (reader_im, reader_seg,
                     origin_im, y_vec, z_vec,
                     rot_matrix) = rotate_volumes(
                         reader_im0, reader_seg0,
                         tangent, locs[count], outdir=out_dir)
                else:
                    reader_im, reader_seg = reader_im0, reader_seg0
                    origin_im = origin_im0
                    tangent, y_vec, z_vec, rot_matrix = None, None, None, None

                # Calculate centers and sizes of samples for this point
                (centers, sizes, save_bif,
                 n_samples, vec0) = calc_samples(count, bifurc, locs, rads,
                                                 global_config)
                print(f"Vec0 is {vec0}")
                sub = 0  # In the case of multiple samples at this point
                for sample in range(n_samples):
                    # Map each center and size to image data
                    center, size_r = centers[sample], sizes[sample]
                    # Get subvolume info
                    (size_extract, index_extract, voi_min,
                     voi_max) = sf.map_to_image(center, rads[count],
                                                size_r, origin_im,
                                                spacing_im, size_im,
                                                global_config['CAPFREE_PROP'])
                    # if any dim is less than 5, skip
                    # if np.any(size_extract < 5):
                    #     print("Size extract too small, skipping")
                    #     skipped += 1
                    #     continue

                    # Check if a surface cap is in volume
                    if global_config['CAPFREE']:
                        is_inside = vf.voi_contain_caps(voi_min, voi_max,
                                                        cap_locs)
                    else:
                        is_inside = False
                    # Continue if surface cap is not present
                    if not is_inside:
                        print("*", end=" ")
                        try:
                            name = (case_dict['NAME']+'_'+str(N-n_old)
                                    + '_'+str(sub))

                            # Extract volume
                            if global_config['EXTRACT_VOLUMES']:
                                (stats, new_img, removed_seg, O
                                 ) = extract_subvolumes(
                                     reader_im, reader_seg,
                                     index_extract,
                                     size_extract,
                                     origin_im, spacing_im,
                                     locs[count],
                                     rads[count], size_r, N,
                                     name, O,
                                     remove_others=global_config['REMOVE_OTHER'],
                                     binarize=global_config['BINARIZE'],
                                     rotate=global_config['ROTATE_VOLUMES'],
                                     orig_im=reader_im0, orig_seg=reader_seg0,
                                     outdir=out_dir
                                     )

                                if global_config['WRITE_SURFACE']:
                                    (stats_surf, new_surf_box, new_surf_sphere
                                     ) = extract_surface(
                                         new_img,
                                         global_surface,
                                         center,
                                         size_r*rads[count])
                                    num_out = len(stats_surf['OUTLETS'])
                                    # print(f"Outlets are: {num_out}")
                                    stats.update(stats_surf)
                                else:
                                    num_out = 0
                            else:
                                stats = create_base_stats(N, name, size_r,
                                                          rads[count],
                                                          size_extract,
                                                          origin_im,
                                                          spacing_im,
                                                          index_extract,
                                                          center)

                            stats = add_tangent_stats(stats, vec0, save_bif)

                            if global_config['WRITE_SAMPLES']:
                                # Write surface and centerline vtps
                                if ((global_config['WRITE_SURFACE']
                                    or global_config['WRITE_CENTERLINE'])
                                   and num_out in global_config['OUTLET_CLASSES']):
                                    if global_config['WRITE_SURFACE']:
                                        write_surface(new_surf_box,
                                                      new_surf_sphere,
                                                      seg_out_dir,
                                                      case_dict['NAME'],
                                                      N, n_old, sub)
                                    if global_config['WRITE_CENTERLINE']:
                                        _, new_cent = extract_centerline(
                                            new_img, global_centerline)
                                        write_centerline(new_cent,
                                                         seg_out_dir,
                                                         case_dict['NAME'],
                                                         N, n_old, sub)
                                if global_config['WRITE_IMG']:
                                    if global_config['RESAMPLE_VOLUMES']:
                                        removed_seg_re = resample_spacing(
                                            removed_seg,
                                            template_size=global_config['RESAMPLE_SIZE'],
                                            order=1)[0]
                                        new_img_re = resample_spacing(
                                            new_img,
                                            template_size=global_config['RESAMPLE_SIZE'],
                                            order=1)[0]
                                        write_img(new_img_re, removed_seg_re,
                                                  image_out_dir, seg_out_dir,
                                                  case_dict['NAME'],
                                                  N, n_old, sub, global_config['BINARIZE'])
                                    else:
                                        write_img(new_img, removed_seg,
                                                  image_out_dir, seg_out_dir,
                                                  case_dict['NAME'],
                                                  N, n_old, sub, global_config['BINARIZE'])

                                if global_config['WRITE_VTK']:
                                    write_vtk(new_img, removed_seg,
                                              out_dir, case_dict['NAME'],
                                              N, n_old, sub)

                            # Discretize centerline
                            if global_config['WRITE_DISCRETE_CENTERLINE']:
                                _, new_cent = extract_centerline(
                                    new_img,
                                    global_centerline,
                                    tangent=tangent)
                                cent_stats = discretize_centerline(
                                    new_cent,
                                    new_img,
                                    N-n_old,
                                    sub, name,
                                    out_dir,
                                    global_config['DISCRETE_CENTERLINE_N_POINTS'])
                                if val_port:
                                    csv_discrete_centerline_val.append(cent_stats)
                                else:
                                    csv_discrete_centerline.append(cent_stats)

                            # Outlet stats
                            if global_config['WRITE_OUTLET_STATS']:
                                (stats_out, planes, planes_seg, pos_example
                                 ) = get_outlet_stats(
                                     stats, new_img,
                                     removed_seg,
                                     upsample=global_config['UPSAMPLE_OUTLET_IMG'])

                                total_num_examples_pos += pos_example
                                total_num_examples += 6
                                print(f"Ratio of positive examples: {total_num_examples_pos/total_num_examples * 100:.2f}")
                                if global_config['WRITE_OUTLET_IMG']:
                                    write_2d_planes(planes, stats_out,
                                                    image_out_dir)
                                    write_2d_planes(planes_seg, stats_out,
                                                    seg_out_dir)
                                if val_port:
                                    for out_stats in stats_out:
                                        csv_outlet_stats_val.append(out_stats)
                                else:
                                    for out_stats in stats_out:
                                        csv_outlet_stats.append(out_stats)

                            # Append stats to csv list
                            csv_list, csv_list_val = append_stats(stats, csv_list, csv_list_val, val_port)

                            # Cross sectional planes
                            if global_config['WRITE_CROSS_SECTIONAL']:
                                (stats_out, planes_img, planes_seg
                                 ) = get_cross_sectional_planes(
                                     stats, new_img, removed_seg,
                                     upsample=global_config['RESAMPLE_CROSS_IMG'])
                                # write cross sectional planes
                                write_2d_planes(planes_img, stats_out,
                                                image_out_dir, add='_cross_rot')
                                write_2d_planes(planes_seg, stats_out,
                                                seg_out_dir, add='_cross_rot')
                            if global_config['WRITE_TRAJECTORIES']:
                                (traj_list,
                                 num_trajs) = get_proj_traj(
                                    stats, new_img,
                                    global_centerline,
                                    traj_list,
                                    num_trajs,
                                    tangent=tangent,
                                    y_vec=y_vec,
                                    z_vec=z_vec,
                                    rot_point=locs[count],
                                    rot_matrix=rot_matrix,
                                    outdir=out_dir,
                                    planes_img=planes_img,
                                    planes_seg=planes_seg,
                                    visualize=True,
                                    img_size=global_config['RESAMPLE_CROSS_IMG'])

                        except Exception as e:
                            print(e)
                            # print("\n*****************************ERROR: did not save files for " +case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub))
                            K += 1

                    else:
                        print(".", end=" ")
                        # print(" No save - cap inside")
                        try:
                            if (global_config['WRITE_VTK']
                               and global_config['WRITE_VTK_THROWOUT']):
                                write_vtk_throwout(reader_seg, index_extract,
                                                   size_extract, out_dir,
                                                   case_dict['NAME'], N,
                                                   n_old, sub)
                            M += 1
                        except Exception as e:
                            print(e)
                            # print("\n*****************************ERROR: did not save throwout for " +case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub))
                            K += 1
                    sub += 1

                print('\n Finished: ' + case_dict['NAME'] + '_' + str(N-n_old))
                print(f"Time for this point: {(time.time() - time_now):.2f} sec")
                # print(" " + str(sub) + " variations")
                N += 1

                if N*n_samples - skipped > global_config['MAX_SAMPLES']:
                    print("Max samples reached")
                    on_cent = False
                    break

            count, on_cent = find_next_point(count, locs, rads, bifurc,
                                             global_config, on_cent)
        # keep track of ids that have already been operated on   
        ids_total.extend(ids)

        if N*n_samples - skipped > global_config['MAX_SAMPLES']:
            print("Max samples reached")
            break

    print_model_info(case_dict['NAME'],  N, n_old, M, m_old)
    # info[case_dict['NAME']] = [ N-n_old, M-m_old, K-k_old]
    print(f"Total time for this case: {(time.time() - time_now_case):.2f} sec")
    print_into_info(info_file_name, case_dict['NAME'], N, n_old, M, m_old, K,
                    k_old, out_dir)
    write_csv(csv_list, csv_list_val, modality, global_config)
    if global_config['WRITE_DISCRETE_CENTERLINE']:
        write_csv_discrete_cent(csv_discrete_centerline,
                                csv_discrete_centerline_val,
                                modality, global_config)

    if global_config['WRITE_OUTLET_STATS']:
        write_csv_outlet_stats(csv_outlet_stats, csv_outlet_stats_val,
                               modality, global_config)
        write_pkl_outlet_stats(csv_outlet_stats, csv_outlet_stats_val,
                               modality, global_config)

    # TODO: if trajectories, add to df file
    if global_config['WRITE_TRAJECTORIES']:
        print(f"Number of trajectories for {case_dict['NAME']}: {num_trajs}")
        column_names = ['frame', 'trackId', 'x', 'y', 'sceneId', 'metaId']
        # check if files exist
        if os.path.exists(out_dir+"trajectories.pkl"):
            df = pd.read_pickle(out_dir+"trajectories.pkl")
            df = df.append(pd.DataFrame(traj_list, columns=column_names))
            df.to_pickle(out_dir+"trajectories.pkl")
        else:
            df = pd.DataFrame(traj_list, columns=column_names)
            df.to_pickle(out_dir+"trajectories.pkl")

        # test read in
        df = pd.read_pickle(out_dir+"trajectories.pkl")
        print(df[df['metaId'] == 0])

    # write to done.txt the name of the case
    with open(out_dir+"done.txt", "a") as f:
        f.write(case_dict['NAME']+'\n')
        f.close()

    return (case_fn, csv_list, csv_list_val, csv_discrete_centerline,
            csv_discrete_centerline_val, csv_outlet_stats,
            csv_outlet_stats_val, traj_list, num_trajs)


if __name__ == '__main__':
    """ Set up

    Example:

    python3 gather_sampling_data_parallel.py \
        -outdir ./extracted_data/ \
        -config_name config \
        -perc_dataset 1.0 \
        -num_cores 1 \
        -start_from 0 \
        -end_at -1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-outdir', '--outdir',
                        default='./extracted_data/',
                        type=str,
                        help='Output directory')
    parser.add_argument('-config_name', '--config_name',
                        type=str,
                        help='Name of configuration file')
    parser.add_argument('-perc_dataset', '--perc_dataset',
                        default=1.0,
                        type=float,
                        help='Percentage of dataset to use')
    parser.add_argument('-num_cores', '--num_cores',
                        default=1,
                        type=int,
                        help='Number of cores to use')
    parser.add_argument('-start_from', '--start_from',
                        default=0,
                        type=int,
                        help='Start from case number')
    parser.add_argument('-end_at', '--end_at',
                        default=-1,
                        type=int,
                        help='End at case number')
    args = parser.parse_args()

    print(args)

    global_config = io.load_yaml("./config/"+args.config_name+".yaml")
    modalities = global_config['MODALITY']

    out_dir = args.outdir  # global_config['OUT_DIR']
    global_config['OUT_DIR'] = out_dir
    # sys.stdout = open(out_dir+"/log.txt", "w")

    # make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for modality in modalities:

        cases = create_dataset(global_config, modality)

        # shuffle cases
        # set random seed
        random.seed(42)
        random.shuffle(cases)
        # percentage of dataset to use
        cases = cases[:int(args.perc_dataset*len(cases))]

        # start from case number
        if args.end_at != -1:
            cases = cases[args.start_from:args.end_at]
        else:
            cases = cases[args.start_from:]

        # skip ones in done.txt if it exists
        if os.path.exists(out_dir+"done.txt"):
            with open(out_dir+"done.txt", "r") as f:
                done = f.read().splitlines()
                f.close()
            for case in done:
                print(f"Skipping {case}")
            cases = [case for case in cases if case not in done]

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
        num_cores_pos = multiprocessing.cpu_count()
        print(f"Number of possible cores: {num_cores_pos}")

        if args.num_cores > 1:
            pool = multiprocessing.Pool(args.num_cores)
            results = [pool.apply_async(sample_case, args=(case, global_config, out_dir, image_out_dir_train, seg_out_dir_train, image_out_dir_val, seg_out_dir_val, image_out_dir_test, seg_out_dir_test, info_file_name, modality)) for case in cases]
            pool.close()
            pool.join()

        else:
            for case in cases:
                results = sample_case(case, global_config, out_dir, image_out_dir_train, seg_out_dir_train, image_out_dir_val, seg_out_dir_val, image_out_dir_test, seg_out_dir_test, info_file_name, modality)

        if global_config['WRITE_TRAJECTORIES']:
            traj_list_all = []
            num_trajs = 0
            if args.num_cores > 1:
                for result in results:
                    case_fn, csv_list, csv_list_val, csv_discrete_centerline, csv_discrete_centerline_val, csv_outlet_stats, csv_outlet_stats_val, traj_list, num_trajs = result
                    traj_list_all.extend(traj_list)
                    num_trajs += num_trajs
            else:
                case_fn, csv_list, csv_list_val, csv_discrete_centerline, csv_discrete_centerline_val, csv_outlet_stats, csv_outlet_stats_val, traj_list, num_trajs = results
                traj_list_all.extend(traj_list)
                num_trajs += num_trajs
            # # write as pandas dataframe
            # column_names = ['frame', 'trackId', 'x', 'y', 'sceneId', 'metaId']
            # df = pd.DataFrame(traj_list_all, columns=column_names)
            # # write as pickle
            # df.to_pickle(out_dir+"trajectories.pkl")

            # # test read in
            # df = pd.read_pickle(out_dir+"trajectories.pkl")
            # print(df[df['metaId'] == 0])

        # Collect results
        # for result in results:
        #     csv_list, csv_list_val, csv_discrete_centerline, csv_discrete_centerline_val, csv_outlet_stats, csv_outlet_stats_val = result.get()

        # print_all_done(info, N, M, K, O)
            # write_csv(csv_list, csv_list_val, modality, global_config)

        print_into_info_all_done(info_file_name, N, M, K, O, out_dir, start_time=start_time)
        print(f"\n--- {(time.time() - start_time)/60:.2f} min ---")
        print(f"--- {(time.time() - start_time)/3600:.2f} hours ---")

        print_csv_stats(out_dir, global_config, modality)

    print("All done")
