import os
import shutil
import sys
import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-outdir', '--outdir',
                        type=str,
                        help='Output directory')
    parser.add_argument('-indir', '--indir',
                        type=str,
                        help='Input directory')
    parser.add_argument('-name', '--name',
                        default='AORTAS',
                        type=str,
                        help='Dataset name')
    parser.add_argument('-dataset_number', '--dataset_number',
                        default=1,
                        type=int,
                        help='Dataset number')
    parser.add_argument('-modality', '--modality',
                        type=str,
                        help='Modality')
    parser.add_argument('-start_from', '--start_from',
                        type=int,
                        default=0,
                        help='Number to start dataset from, use if adding to existing dataset')
    args = parser.parse_args()

    # global_config_file = "./config/global.yaml"
    # global_config = io.load_yaml(global_config_file)
    # modalities = global_config['MODALITY']

    directory = args.indir  # '/global/scratch/users/numi/vascular_data_3d/extraction_output/nnunet_only_one_aorta_vmr/'
    directory_out = args.outdir  # '/global/scratch/users/numi/vascular_data_3d/extraction_output/nnunet_only_one_aorta_vmr/'
    modality = args.modality  # 'mr' or 'ct'

    start_from = args.start_from  # 0 28707
    name = args.name  # 'SEQAORTASONE'
    dataset_number = args.dataset_number  # 1
    # make number 2 digits
    if dataset_number < 10:
        dataset_number = '0' + str(dataset_number)
    else:
        dataset_number = str(dataset_number)

    new_dir_dataset_name = 'Dataset0'+dataset_number+'_'+name+modality.upper()
    append = name.lower() + modality.lower()

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
        # check if exists
        if not os.path.exists(os.path.join(directory, fn)):
            print(f'{fn} does not exist')
            continue
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