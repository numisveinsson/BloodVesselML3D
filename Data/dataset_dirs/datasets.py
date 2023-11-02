import pandas as pd

class VMR_dataset:
    """
    Class for VMR dataset
    Assumes the data directory is organized as follows:
        Images (raw image data)
            Scaled Images (globally scaled images 0-1)
            Cropped Images (cropped around segmentation mask)
        Truths (segmentation masks)
        Surfaces (surface meshes)
        Centerlines (centerline meshes)
    """
    def __init__(self, directory, modality=None, anatomy=None):
        self.dir = directory
        self.modality = modality
        self.anatomy = anatomy
        try:
            self.df = get_vmr_dataset_names()
        except Exception as e:
            print(f'Cound not get VMR dataset names from google sheet: {e}')
            self.df = None
        # get the cases of specified modality and anatomy
        if modality is not None:
            self.df = get_vmr_names_modality(self.df, modality)
        if anatomy is not None:
            self.df = get_vmr_names_anatomy(self.df, anatomy)

    def sort_cases(self, testing, test_cases):
        """
        Sort the cases into train+val and test
        Input: test_cases (list of str)
        Output: cases (list of str)
        """
        if not testing:
            cases = self.df['Legacy Name'].values
            cases = [f for f in cases if f not in test_cases]
            cases.sort()
        else:
            cases = [f for f in self.df['Legacy Name'].values if f in test_cases]

        self.cases = cases

        return cases
    
    def check_which_cases_in_image_dir(self, cases_list):
        """
        Check which cases are in the data directory (Images)
        """
        import os
        images = os.listdir(self.dir + 'Images/')
        # ignore hidden files
        images = [f for f in images if f[0] != '.']
        # remove the file extension
        images = [f.split('.')[0] for f in images]
        # remove duplicates
        images = list(set(images))
        # sort
        images.sort()

        cases = [f for f in cases_list if f in images]

        # print the cases not in image dir
        print('Cases not in image dir:')
        imgs_mis = [f for f in cases_list if f not in images]
        for img in imgs_mis:
            print(img)

        return cases

    def compare_image_dir_and_csv(self):
        """
        Check which cases are in the data directory (Images)
        """
        # create a list of images in the directory
        import os
        images = os.listdir(self.dir + 'Images/')
        # ignore hidden files
        images = [f for f in images if f[0] != '.']
        # remove the file extension
        images = [f.split('.')[0] for f in images]
        # remove duplicates
        images = list(set(images))
        # sort
        images.sort()

        self.image_cases = images

        # check which cases are in the google sheet
        if self.df is not None:
            print('Cases in google sheet:')
            print(self.df['Legacy Name'].values)
            print(self.df['Name'].values)
        else:
            print('No google sheet dataframe found')
        # print the cases that are in image dir but not in google sheet
        print('Cases in image dir but not in google sheet:')
        print([f for f in images if f not in self.df['Legacy Name'].values and f not in self.df['Name'].values])
        # print the cases that are in google sheet but not in image dir
        print('Cases in google sheet but not in image dir:')
        print([f for f in self.df['Legacy Name'].values if f not in images and f not in self.df['Name'].values])

        # return list of cases in image dir that are in the google sheet
        return images
        
def get_case_dict_dir(directory, model, img_ext):
    """
    Get the directories of the case
    Input: model (str)
    Output: dict of directories
    """
    dir_image, dir_seg, dir_cent, dir_surf = directories( directory, model, img_ext)
    dir_dict = {'NAME': model,
                'IMAGE': dir_image,
                'SEGMENTATION': dir_seg,
                'CENTERLINE': dir_cent,
                'SURFACE': dir_surf}
    return dir_dict

def build_sheet_url():
    doc_id = '1NCyC18ot2Vht5LlFX7Hq-g0jaLLwf5wirBcqOnWWn8s'
    sheet_id = 'svprojects'
    return f'https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv&sheet={sheet_id}'

def write_df_to_local(df):
    file_path = './VMR_dataset_names.csv'
    df.to_csv(file_path)

def get_vmr_dataset_names():
    """
    Returns a pandas dataframe of the VMR dataset names
    """
    url = build_sheet_url()
    df = pd.read_csv(url)
    df = df[df['Legacy Name'].notna()]
    return df

def get_vmr_dataset_names_local():
    """
    Returns a pandas dataframe of the VMR dataset names
    """
    file_path = 'Data/VMR_dataset_names.csv'
    return pd.read_csv(file_path)

def get_vmr_names_modality(df, modality):
    """
    Returns a pandas dataframe of the VMR dataset names
    Input: modality (list of str)
    Possible values: 
    'CT'
    'MR'
    """
    return df[df['Image Modality'].isin(modality)]

def get_vmr_names_anatomy(df, anatomy):
    """
    Returns a pandas dataframe of the VMR dataset names
    Input: anatomy (list of str)
    Possible values: 
    'Aorta'
    'Abdominal Aorta'
    'Coronary'
    'Cerebral'
    'Pulmonary Fontan'
    'Pulmonary'
    'Pulmonary Glenn'
    """
    return df[df['Anatomy'].isin(anatomy)]

def get_vmr_names_disease(df, disease):
    """
    Returns a pandas dataframe of the VMR dataset names
    Input: disease (list of str)
    """
    return df[df['Disease'].isin(disease)]

def get_vmr_names_disease(df, image_mod):
    """
    Returns a pandas dataframe of the VMR dataset names
    Input: image_mod (list of str)
    """
    return df[df['Image Modality'].isin(image_mod)]

def directories(directory, model, img_ext):
    """
    Function to return the directories of
        Image Volume
        Segmentation Volume
        Centerline VTP
        Surface Mesh VTP
    """
    dir_image = directory +'images/'+model+img_ext
    dir_seg = directory +'truths/'+model+img_ext
    dir_cent = directory + 'centerlines/'+model+'.vtp'
    dir_surf = directory + 'surfaces/'+model+'.vtp'

    return dir_image, dir_seg, dir_cent, dir_surf

def vmr_directories_old(directory, model, global_scale, dir_seg_exist, cropped, original):
    """
    Function to return the directories of
        Image Volume
        Segmentation Volume
        Centerline VTP
        Surface Mesh VTP
    for a specific model in the
    Vascular Model Repository
    """
    if not cropped:
        if global_scale:
            dir_image = directory +'scaled_images/'+model.replace('_aorta','')+'.vtk'
            dir_seg = directory +'truths/'+model.replace('_aorta','')+'.vtk'
        else:
            if original:
                dir_image = directory +'images/OSMSC' + model[0:4]+'/OSMSC'+model[0:4]+'-cm.mha'
                dir_seg = directory + 'images/OSMSC'+model[0:4]+'/'+model+'/'+model+'-cm.mha'
            else:
                dir_image = directory +'images/'+model.replace('_aorta','')+'.vtk'
                dir_seg = directory +'truths/'+model.replace('_aorta','')+'.vtk'
    else:
        dir_image = directory +'images/OSMSC' + model[0:4]+'/OSMSC'+model[0:4]+'-cm.mha'

    if not dir_seg_exist:
        dir_seg = None

    dir_cent = directory + 'centerlines/'+model+'.vtp'
    dir_surf = directory + 'surfaces/'+model+'.vtp'

    return dir_image, dir_seg, dir_cent, dir_surf

if __name__ == '__main__':

    df = get_vmr_dataset_names()
    # keep only where 'Legacy Name' is not NaN
    
    dataset = VMR_dataset('/Users/numisveins/Documents/vascular_data_3d/')
    import pdb; pdb.set_trace()
    dataset.check_which_cases_in_image_dir()
