import os
import shutil
import sys

sys.stdout.flush()
sys.path.insert(0, '../..')
sys.path.insert(0, '..')
# from modules import io

# main script
if __name__ == "__main__":
    """
    This script is used to create the nnUNet dataset names
    from a dataset that has the following structure:

    Data
    ├── ct_train
    ├── ct_train_masks
    ├── ct_test
    ├── ct_test_masks

    (or mr instead of ct)
    Into a dataset that has the following structure:
    
    Dataset
    ├── imagesTr
    ├── imagesTs
    └── labelsTr
    """

    # global_config_file = "./config/global.yaml"
    # global_config = io.load_yaml(global_config_file)
    # modalities = global_config['MODALITY']

    # directory = '/Users/numisveins/Documents/Automatic_Tracing_Data/global_nnunet_miccai_aortas/'
    # directory = '/global/scratch/users/numi/vascular_data_3d/extraction_output/aortas_aortofem/'
    # directory = '/global/scratch/users/numi/MICCAI_AVT_Data/extraction_output/'
    # directory_out = '/global/scratch/users/numi/vascular_data_3d/extraction_output/aortas_aortofem/'
    directory = '/Users/numisveins/Documents/PARSE_dataset/'
    directory_out = '/Users/numisveins/Documents/PARSE_dataset/'
    modality = 'ct'

    start_from = 0 #28707 

    new_dir_dataset_name = 'Dataset014_PULMPARSE'+modality.upper()
    append = ('PULMPARSE').lower() + modality.lower()

    also_test = False

    out_data_dir = os.path.join(directory_out, new_dir_dataset_name)

    # create new dataset directory
    try:
        os.mkdir(os.path.join(directory_out, new_dir_dataset_name))
    except FileExistsError:
        print(f'Directory {new_dir_dataset_name} already exists')
    
    fns_in = [modality+'_train',
            modality+'_train_masks',
           ]
    if also_test:
        fns_in.append(modality+'_test')
        fns_in.append(modality+'_test_masks')
    fns_out = ['imagesTr',
            'labelsTr',
           ]
    if also_test:
        fns_out.append('imagesTs')
        fns_out.append('labelsTs')
    
    for fn in fns_out:
        try:
            os.mkdir(os.path.join(directory_out, new_dir_dataset_name, fn))
        except FileExistsError:
            print(f'Directory {fn} already exists')

    for fn in fns_in:
        imgs = os.listdir(os.path.join(directory, fn))
        imgs = [img for img in imgs if img.endswith('.nii.gz')]
        imgs.sort()

        for i, img in enumerate(imgs):
            new_name = f'{append}_{(i+1+start_from):03d}_0000.nii.gz'
            if fns_out[fns_in.index(fn)] == 'labelsTr' or fns_out[fns_in.index(fn)] == 'labelsTs':
                new_name = new_name.replace('_0000', '')
            print(f'Copying {img} to {new_name}')
            if img != new_name:
                # copy with new name
                shutil.copy(os.path.join(directory, fn, img), os.path.join(out_data_dir, fns_out[fns_in.index(fn)], new_name))