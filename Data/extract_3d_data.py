import numpy as np
import os

from modules import io
from modules import vascular_data as sv

#location = "/Users/numi/Documents/Berkeley/Research/BloodVesselML3D/Data/"

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
    segmentation = sv.read_mha(case_dict['SEGMENTATION'])

    im_np  = sv.vtk_image_to_numpy(image)
    seg_np = sv.vtk_image_to_numpy(segmentation)

    import pdb; pdb.set_trace()
