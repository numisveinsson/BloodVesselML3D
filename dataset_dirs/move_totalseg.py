if __name__ == "__main__":
    """
    This script is used to move the raw image (ct.nii.gz) and heart seg (heart.nii.gz) files from a folder structure:

    Directory
    ├── directory_name_1
        ├── ct.nii.gz
        ├── segmentations
            ├── heart.nii.gz
    ├── directory_name_2
        ├── ct.nii.gz
        ├── segmentations
            ├── heart.nii.gz

    Into a dataset that has the following structure:
    
    Output_Directory
    ├── imagesTr
        ├── directory_name_1.nii.gz
        ├── directory_name_2.nii.gz
    └── labelsTr
        ├── directory_name_1.nii.gz
        ├── directory_name_2.nii.gz

    Where the directory_name_1.nii.gz and directory_name_2.nii.gz are the raw images and the directory_name_1.nii.gz and directory_name_2.nii.gz are the heart segmentations.

    """

    import os

    directory = '/Users/numisveins/Downloads/Totalsegmentator_dataset_small_v201/'

    out_data_dir = '/Users/numisveins/Documents/Total_Segmentator_Data/'

    # get list of folders in directory

    folders = os.listdir(directory)
    folders = [f for f in folders if os.path.isdir(os.path.join(directory, f))]

    # create new dataset directory

    try:
        os.mkdir(out_data_dir)
    except FileExistsError:
        print(f'Directory {out_data_dir} already exists')

    # create new dataset directory
    try:
        os.mkdir(os.path.join(out_data_dir, 'imagesTr'))
        os.mkdir(os.path.join(out_data_dir, 'labelsTr'))
    except FileExistsError:
        print(f'Directory {out_data_dir} already exists')

    for folder in folders:

        # get the raw image and heart seg files
        raw_img = os.path.join(directory, folder, 'ct.nii.gz')
        heart_seg = os.path.join(directory, folder, 'segmentations', 'heart.nii.gz')

        # copy the files to the new directory
        try:
            os.system(f'cp {raw_img} {os.path.join(out_data_dir, "imagesTr", folder+".nii.gz")}')
            os.system(f'cp {heart_seg} {os.path.join(out_data_dir, "labelsTr", folder+".nii.gz")}')
        except Exception as e:
            print(e)
            print(f'Error copying files from {raw_img} and {heart_seg} to {os.path.join(out_data_dir, "imagesTr", folder+".nii.gz")} and {os.path.join(out_data_dir, "labelsTr", folder+".nii.gz")}')
            continue