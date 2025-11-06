
def move_files(directory, out_dir):
    """
    The directory must have the following structure:
    directory
        - name_1 (folder)
            - image (folder)
                - name_1.nii.gz
            - label (folder)
                - name_1.nii.gz
        - name_2 (folder)
            - image (folder)
                - name_2.nii.gz
            - label (folder)
                - name_2.nii.gz
    The script copies images into new directory, so that the structure is:
    directory
        images
            - name_1.nii.gz
            - name_2.nii.gz
        labels
            - name_1.nii.gz
            - name_2.nii.gz
    """

    import os
    import shutil

    # create new dataset directory
    try:
        os.mkdir(os.path.join(out_dir, 'images'))
    except FileExistsError:
        print(f'Directory images already exists')
    try:
        os.mkdir(os.path.join(out_dir, 'labels'))
    except FileExistsError:
        print(f'Directory labels already exists')

    for case in os.listdir(directory):
        if case.startswith('.'):
            continue
        print(case)
        case_dir = os.path.join(directory, case)
        for folder in os.listdir(case_dir):
            if folder.startswith('.'):
                continue
            print(folder)
            folder_dir = os.path.join(case_dir, folder)
            for fn in os.listdir(folder_dir):
                if fn.startswith('.'):
                    continue
                print(fn)
                fn_dir = os.path.join(folder_dir, fn)
                if folder == 'image':
                    shutil.copy(fn_dir, os.path.join(out_dir, 'images', fn))
                elif folder == 'label':
                    shutil.copy(fn_dir, os.path.join(out_dir, 'labels', fn))

if __name__=='__main__':

    # script to move files from subfolders to one folder

    directory = '/Users/numisveins/Downloads/train/'
    out_dir = '/Users/numisveins/Documents/PARSE_dataset/'

    move_files(directory, out_dir)