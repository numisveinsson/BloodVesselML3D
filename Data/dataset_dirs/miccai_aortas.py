import os
def get_miccai_aorta_dataset_names(directory):
    
    """
    Returns a pandas dataframe of the VMR dataset names
    Input: directory (str)
    """
    cases_im = os.listdir(directory+'images/')
    cases_im = [f for f in cases_im if f.endswith('.nrrd')]
    cases_im = [f.replace('.nrrd', '') for f in cases_im]

    cases_cent = os.listdir(directory+'centerlines/')
    cases_cent = [f for f in cases_cent if f.endswith('.vtp')]
    cases_cent = [f.replace('.vtp', '') for f in cases_cent]

    cases_surf = os.listdir(directory+'surfaces/')
    cases_surf = [f for f in cases_surf if f.endswith('.vtp')]
    cases_surf = [f.replace('.vtp', '') for f in cases_surf]
    cases_surf = [f.replace('.seg', '') for f in cases_surf]

    # only keep cases that have all files
    cases = [f for f in cases_im if f in cases_cent and f in cases_surf]

    import pdb; pdb.set_trace()

    return cases