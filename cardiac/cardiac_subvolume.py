import os
import SimpleITK as sitk
import numpy as np

def extract_subvolume_folder(folder, subvolume_size_phys=(185, 160, 176)):
    """
    Extract a subvolume around the cardiac region from a 3D image using SimpleITK.
    The folder has the following structure:
    folder
    ├── binary_segs
        ├── case_1.nii.gz
    ├── images
        ├── case_1.nii.gz

    For each case in the folder, the script:
        1. Reads in both the 3D binary segmentation and the 3D image.
        2. Calculates the center of mass of the binary segmentation.
        3. Extracts a subvolume around the center of mass. The size is determined by the subvolume_size parameter.
           To calculate the size of the subvolume, the script uses the spacing of the 3D image.
        4. Saves the subvolume as a new 3D image.
    
    It creates a new folder called 'subvolumes' with the following structure:
    folder
    ├── binary_segs
        ├── case_1.nii.gz
    ├── images
        ├── case_1.nii.gz
    ├── subvolumes
        ├── case_1.nii.gz

    Args:
        folder: str, path to the folder containing the 3D images
        subvolume_size: tuple, size of the subvolume to extract (default is (185, 160, 176)), in mm
    Returns:
        None
    """
    # create new folder for the subvolumes
    subvolume_folder = os.path.join(folder, 'subvolumes')
    try:
        os.mkdir(subvolume_folder)
    except FileExistsError:
        print(f'Directory {subvolume_folder} already exists')

    # get the list of cases in the folder
    cases = os.listdir(os.path.join(folder, 'images'))
    cases = [case.split('.')[0] for case in cases if case.endswith('.nii.gz')]

    for case in cases:
        # read in the binary segmentation and the image
        binary_seg = sitk.ReadImage(os.path.join(folder, 'binary_segs', f'{case}.nii.gz'))
        img = sitk.ReadImage(os.path.join(folder, 'images', f'{case}.nii.gz'))

        # calculate the center of mass of the binary segmentation
        binary_np = sitk.GetArrayFromImage(binary_seg).transpose(2, 1, 0)
        locs_1 = np.where(binary_np == 1)
        center_of_mass = np.mean(locs_1, axis=1).astype(int)

        # calculate the size of the subvolume in pixels
        subvolume_size = np.array(subvolume_size_phys) // np.array(img.GetSpacing())
        subvolume_size = subvolume_size.astype(int).tolist()

        # calculate index of the subvolume
        subvolume_index = center_of_mass - np.array(subvolume_size) // 2
        # make sure the subvolume is within the image
        subvolume_index = np.minimum(subvolume_index, np.array(img.GetSize()) - np.array(subvolume_size))
        subvolume_index = np.maximum(subvolume_index, 0)
        subvolume_index = subvolume_index.astype(int).tolist()
        # make sure the size is within the image
        subvolume_size = np.minimum(subvolume_size, np.array(img.GetSize()) - np.array(subvolume_index))
        subvolume_size = subvolume_size.astype(int).tolist()

        print(f"Subvolume size: {subvolume_size}")
        print(f"Total image size: {img.GetSize()}")
        print(f"Subvolume length: {np.array(subvolume_size) * np.array(img.GetSpacing())}")
        print(f"Total image length: {np.array(img.GetSize()) * np.array(img.GetSpacing())}")

        # extract the subvolume around the center of mass
        subvolume = sitk.RegionOfInterest(img, size=subvolume_size, index=subvolume_index)

        # save the subvolume
        sitk.WriteImage(subvolume, os.path.join(subvolume_folder, f'{case}.nii.gz'))


if __name__ == "__main__":
    """
    Script to extract a subvolume around the cardiac region from a 3D image
    using SimpleITK.
    """
    folder = '/Users/numisveins/Documents/data_combo_paper/mr_data/'
    scale_down = 1
    # size = [185, 160, 176]
    size = [180, 209, 223]
    extract_subvolume_folder(folder, subvolume_size_phys=(size[0]*scale_down, size[1]*scale_down, size[2]*scale_down))