# Script to view troublesome cases

import view_data as vd

if __name__=='__main__':
    #set directories

    directory_trouble = str('/Users/numisveinsson/Documents/Berkeley/Research/BloodVesselML3D/Data/cases/trouble.txt')
    trouble_cases = open(directory_trouble).readlines()
    trouble_cases = [f.replace('\n','') for f in trouble_cases]
    trouble_cases = [f.replace('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train_masks/','') for f in trouble_cases]
    index = 0

    directory = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train/')
    directory_mask = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train_masks/')

    trouble = vd.view_volumes(trouble_cases, index, directory, directory_mask)
    import pdb; pdb.set_trace()
