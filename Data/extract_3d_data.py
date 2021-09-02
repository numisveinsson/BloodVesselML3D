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

global_config_file = "./config/global.yaml"
global_config = io.load_yaml(global_config_file)

image_out_dir = global_config['IMAGE_OUT_DIR'] +'/'
seg_out_dir = global_config['SEG_OUT_DIR'] +'/'

cases = os.listdir(global_config['CASES_DIR'])
cases = [global_config['CASES_DIR']+'/'+f for f in cases if 'case.' in f]

# cases.remove('./cases/case.0005_1001.yml')
# cases.remove('./cases/case.0119_0001.yml')
# cases.remove('./cases/case.0158_0001.yml')
#
# cases = ['./cases/case.0150_0001.yml', './cases/case.0151_0001.yml', './cases/case.0145_1001.yml', './cases/case.0138_1001.yml', './cases/case.0141_1001.yml', './cases/case.0142_1001.yml', './cases/case.0176_0000.yml', './cases/case.0149_1001.yml', './cases/case.0175_0000.yml', './cases/case.0139_1001.yml', './cases/case.0146_1001.yml', './cases/case.0174_0000.yml', './cases/case.0148_1001.yml', './cases/case.0156_0001.yml', './cases/case.0157_0000.yml', './cases/case.0129_0000.yml']
#cases = ['./cases/case.0158_0001.yml']
info = {}
N = 0
M = 0
K = 0
P = 0
O = 0
mul_l = []
for case_fn in cases:

    case_dict = io.load_yaml(case_fn)
    print(case_dict['NAME'])

    reader_im = sf.read_image(case_dict['IMAGE'])
    reader_seg = sf.read_image(case_dict['SEGMENTATION'])

    origin_im = np.array(list(reader_im.GetOrigin()))
    size_im = np.array(list(reader_im.GetSize()))
    spacing_im = np.array(list(reader_im.GetSpacing()))

    ## Surface Caps
    surf = vf.read_geo(case_dict['SURFACE']).GetOutput()    # read in geometry
    surf_data = vf.collect_arrays(surf.GetCellData())       # collect arrays of cell data
    surf_locs = vf.get_location_cells(surf)                 # get locations of cell centroids
    try:
        ids_caps = np.where(surf_data['CapID']!= 0)[0]      # get ids of cells on caps
        print("CapID not BC")
    except:
        ids_caps = np.where(surf_data['BC_FaceID']!= 0)[0]
        print("BC instead of CapID")
    cap_locs = surf_locs[ids_caps]                          # get locations of cap cells

    ## Centerline
    cent = vf.read_geo(case_dict['CENTERLINE']).GetOutput()         # read in geometry
    num_points = cent.GetNumberOfPoints()               # number of points in centerline
    cent_data = vf.collect_arrays(cent.GetPointData())
    c_loc = v2n(cent.GetPoints().GetData())             # point locations as numpy array
    radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
    cent_id = cent_data['CenterlineId']

    try:
        num_cent = len(cent_id[0])
    except:
        num_cent = 1
        print("\n Only one centerline to follow \n")

    ids_total = []
    m_old = M
    n_old = N
    k_old = K
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
                print('The point # along centerline is ' + str(count))
                print('The location is ' + str(locs[count]))

                size_extract = (4*rads[count]//spacing_im +1).astype(int).tolist()
                index_extract = ((locs[count]-origin_im)//spacing_im - 2*rads[count]//spacing_im +1).astype(int).tolist()

                voi_min = locs[count] - 2*rads[count]
                voi_max = locs[count] + 2*rads[count]

                cap_in_volume = vf.voi_contain_caps(voi_min, voi_max, cap_locs)

                if not cap_in_volume:
                    print("\n It's cap free!\n")
                    print('The size of VOI is ' + str(4*rads[count]))

                    try:
                        reader_im = sf.read_image(case_dict['IMAGE'])
                        reader_seg = sf.read_image(case_dict['SEGMENTATION'])

                        reader_im.SetExtractIndex(index_extract)
                        reader_im.SetExtractSize(size_extract)
                        new_img = reader_im.Execute()

                        reader_seg.SetExtractIndex(index_extract)
                        reader_seg.SetExtractSize(size_extract)
                        new_seg = reader_seg.Execute()

                        seed = (np.array(size_extract)//2).tolist()
                        removed_seg = sf.remove_other_vessels(new_seg, seed)

                        labels, means = sf.connected_comp_info(removed_seg, new_seg)
                        print("\nThe labels are: \n")
                        print(labels)

                        # if len(means)==1 and means[0]!=255.0:
                        #
                        #     new_seg = new_seg//255
                        #     sitk.WriteImage(new_seg, seg_out_dir + case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')
                        #     sitk.WriteImage(new_seg, '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/vtk_train_masks/'+ case_dict['NAME']+'_' +str(N-n_old)+ '.vtk')
                        #
                        #     print("\n****************** We changed this one! ******************\n")
                        #     print("\n****************** " +case_dict['NAME'] +'_'+ str(N-n_old)+ " ******************\n")
                        #     P = P+1
                            # sitk.Show(new_img, title="Image"+str(N), debugOn=True)
                            # sitk.Show(new_seg, title="Seg"+str(N), debugOn=True)
                            # sitk.Show(removed_seg, title="Old Seg"+str(N), debugOn=True)

                        # We want binary 0/1 instead of 0/255
                        removed_seg = removed_seg//255

                        if len(means) != 1:
                            O = O+1
                            mul_l.append(case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')

                        #sitk.WriteImage(new_img, image_out_dir + case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')
                        #sitk.WriteImage(removed_seg, seg_out_dir + case_dict['NAME'] +'_'+ str(N-n_old) +'.nii.gz')

                        #sitk.WriteImage(new_img, '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/vtk_train/' + case_dict['NAME']+'_' +str(N-n_old)+ '.vtk')
                        #sitk.WriteImage(removed_seg, '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/vtk_train_masks/'+ case_dict['NAME']+'_' +str(N-n_old)+ '.vtk')

                        #sitk.Show(new_img, title="Image"+str(N), debugOn=True)
                        #sitk.Show(new_seg, title="Seg"+str(N), debugOn=True)
                        #sitk.Show(removed_seg, title="Seg"+str(N), debugOn=True)
                        N = N+1
                    except:
                        print("\n                               *****************************ERROR: did not save files for " + str(N-n_old))
                        print("Model " + case_dict['NAME'] +"\n")
                        K = K+1

                else:
                    print("\n There's a cap inside! \n")
                    try:
                        reader_seg.SetExtractIndex(index_extract)
                        reader_seg.SetExtractSize(size_extract)
                        new_seg = reader_seg.Execute()
                        #sitk.WriteImage(new_seg, '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/throw_outs/'+'throwout_'+ case_dict['NAME']+'_'+str(M-m_old)+ '.vtk')
                        M=M+1
                    except:
                        print("\n                               *****************************ERROR: did not save throwout for " + str(M-m_old))
                        print("Model " + case_dict['NAME'] +"\n")
                        K = K+1

            lengths = np.cumsum(np.insert(np.linalg.norm(np.diff(locs[count:], axis=0), axis=1), 0, 0))
            move = 1
            count = count+1
            if count == len(locs):
                on_cent = False
                break
            while lengths[move] < global_config['MOVE_DIST']*rads[count] :
                count = count+1
                move = move+1
                if count == len(locs):
                    on_cent = False
                    break

            if on_cent:
                print('Next we move ' + str(lengths[move]))
                print(" ")

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
print("****************** P: " + str(P) +" were changed! ****************** \n")
print("****************** O: " + str(O) +" have more than one label! ****************** \n")
for i in mul_l:
    print(i)

print("\n--- %s seconds ---" % (time.time() - start_time))

import pdb; pdb.set_trace()
# img = sitk.ReadeImage('my_input.png')
# sitk.WriteImage(img, 'my_output.jpg')
