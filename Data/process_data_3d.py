import os
import argparse
from modules import io
from modules import vascular_data as sv
import scipy
import numpy as np
from imageio import imsave

from modules import volume_functions as vf

global_config_file = "./config/global.yaml"
case_config_file   = "./config/case.yaml"


global_config = io.load_yaml(global_config_file)
case_config   = io.load_yaml(case_config_file)

####################################
# Get necessary params
####################################
cases = os.listdir(global_config['CASES_DIR'])
cases = [global_config['CASES_DIR']+'/'+f for f in cases if 'case.' in f]

spacing_vec = [case_config['SPACING']]*2
dims_vec    = [case_config['DIMS']]*2
ext_vec     = [case_config['DIMS']-1]*2
path_start  = case_config['PATH_START']

files = open(case_config['DATA_DIR']+'/files.txt','w')

for i, case_fn in enumerate(cases):
    case_dict = io.load_yaml(case_fn)
    print(case_dict['NAME'])

    image_dir = case_config['DATA_DIR']+'/'+case_dict['NAME']
    sv.mkdir(image_dir)

    image        = sv.read_mha(case_dict['IMAGE'])
    #image        = sv.resample_image(image,case_config['SPACING'])

    segmentation = sv.read_mha(case_dict['SEGMENTATION'])
    #segmentation = sv.resample_image(segmentation,case_config['SPACING'])

    path_dict    = sv.parsePathFile(case_dict['PATHS'])
    group_dir    = case_dict['GROUPS']

    im_np  = sv.vtk_image_to_numpy(image)
    seg_np = sv.vtk_image_to_numpy(segmentation)
    blood_np = im_np[seg_np>0.1]

    stats = {"MEAN":np.mean(im_np), "STD":np.std(im_np), "MAX":np.amax(im_np),
    "MIN":np.amin(im_np),
    "BLOOD_MEAN":np.mean(blood_np),
    "BLOOD_STD":np.std(blood_np),
    "BLOOD_MAX":np.amax(blood_np),
    "BLOOD_MIN":np.amin(blood_np)}

    for grp_id in path_dict.keys():
        path_info      = path_dict[grp_id]
        path_points    = path_info['points']
        group_name     = path_info['name']
        group_filename = group_dir +'/'+group_name

        if not os.path.exists(group_filename): continue

        group_dict = sv.parseGroupFile(group_filename)

        group_points = sorted(group_dict.keys())

        if len(group_points) < 4: continue

        tup = sv.get_segs(path_points,group_dict,
            [case_config['DIMS']]*2, [case_config['SPACING']]*2,
            case_config['NUM_CONTOUR_POINTS'])

        if tup == None: continue

        group_data_dir = image_dir+'/'+group_name
        sv.mkdir(group_data_dir)

        segs,norm_grps,interp_grps,means = tup

        im_slices  = []
        seg_slices = []

        for i,I in enumerate(group_points[path_start:-path_start]):
            j = i+path_start

            v = path_points[I]
            im_slice = sv.getImageReslice(image, ext_vec,
                v[:3],v[3:6],v[6:9], case_config['SPACING'],True)
            seg_slice = sv.getImageReslice(segmentation, ext_vec,
                v[:3],v[3:6],v[6:9], case_config['SPACING'],True)

            try:
                x_path  = '{}/{}.X.npy'.format(group_data_dir,I)
                y_path  = '{}/{}.Y.npy'.format(group_data_dir,I)
                yc_path = '{}/{}.Yc.npy'.format(group_data_dir,I)
                c_path  = '{}/{}.C.npy'.format(group_data_dir,I)
                ci_path =  '{}/{}.C_interp.npy'.format(group_data_dir,I)

                yaml_path = '{}/{}.yaml'.format(group_data_dir,I)

                radius = np.sqrt((1.0*np.sum(segs[j]))/np.pi)

                yaml_dict = {}
                yaml_dict['X'] = x_path
                yaml_dict['Y'] = y_path
                yaml_dict['Yc'] = yc_path
                yaml_dict['C'] = c_path
                yaml_dict['C_interp'] = ci_path
                yaml_dict['point'] = I
                yaml_dict['path_name'] = group_name
                yaml_dict['path_id'] = grp_id
                yaml_dict['image'] = case_dict['NAME']
                yaml_dict['extent'] = case_config['DIMS']
                yaml_dict['dimensions'] = case_config['CROP_DIMS']
                yaml_dict['spacing'] = case_config['SPACING']
                yaml_dict['radius'] = float(radius)
                for k,v in stats.items():
                    yaml_dict[k] = float(v)

                io.save_yaml(yaml_path, yaml_dict)

                np.save(x_path, im_slice)
                np.save(y_path, seg_slice)
                np.save(yc_path, segs[j])
                np.save(c_path, norm_grps[j])
                np.save(ci_path, interp_grps[j])

                imsave('{}/{}.X.png'.format(group_data_dir,I),im_slice)
                imsave('{}/{}.Y.png'.format(group_data_dir,I),seg_slice)
                imsave('{}/{}.Yc.png'.format(group_data_dir,I),segs[j])

                files.write(yaml_path+'\n')
            except:
                print( "failed to save {}/{}".format(group_data_dir,I))

        io.write_csv(image_dir+'/'+'image_stats.csv',stats)

files.close()
