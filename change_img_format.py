import os
import SimpleITK as sitk
from modules import vtk_functions as vf


def change_mha_vti(file_dir, label=False):
    """
    Change the format of a file from .mha to .vti
    SITK does not support .vti format, so we need to use the vtk functions
    Args:
        file_dir: str, path to the file
    Returns:
        None
    """
    img = sitk.ReadImage(file_dir)
    if label:
        img = sitk.Cast(img, sitk.sitkUInt8)
    img = vf.exportSitk2VTK(img)[0]

    return img


def change_vti_sitk(file_dir):
    """
    Change the format of a file from .vti to .mha
    SITK does not support .vti format, so we need to use the vtk functions
    Args:
        file_dir: str, path to the file
    Returns:
        None
    """
    img = vf.read_img(file_dir)
    img = vf.exportVTK2Sitk(img)

    return img


if __name__ == '__main__':

    # import pdb; pdb.set_trace()

    input_format = '.nii.gz'
    output_format = '.mha'
    label = False  # false if raw image
    surface = False  # true if we want to save the surface vtp file

    rem_str = ''  # 'coroasocact_0'

    data_folder = '/Users/numisveins/Documents/datasets/TOPCOW_willis_dataset/topcow_batch-1_40pairMRCT_30062023/imagesTr/'
    out_folder = data_folder+'new_format/'

    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]

    # sort the files
    imgs = sorted(imgs)

    try:
        os.mkdir(out_folder)
        imgs_old = []
    except Exception as e:
        print(e)
        imgs_old = os.listdir(out_folder)

    if input_format == '.dcm':
        # it's a dicom folder with multiple files
        # create a 3D image from the dicom files
        print(f"Reading dicom files from {data_folder}")
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(data_folder)
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        print(f"Saving 3D image to {out_folder+'3D_image'+output_format}")
        sitk.WriteImage(img, out_folder+'3D_image'+output_format)
    else:
        # it's a single file
        for fn in imgs:
            if fn.replace(input_format, output_format) in imgs_old:
                print(f"File {fn} already exists in folder")
                continue
            else:
                print(f"Converting file {fn} to new format {output_format}")
            if input_format != '.vti' and output_format != '.vti':
                img = sitk.ReadImage(data_folder+fn)
                if label:
                    img = sitk.Cast(img, sitk.sitkUInt8)
            elif input_format == '.vti':
                if output_format == '.mha' or output_format == '.nii.gz':
                    img = change_vti_sitk(data_folder+fn)
                else:
                    img = vf.read_img(data_folder+fn).GetOutput()
            elif output_format == '.vti':
                if input_format == '.mha' or input_format == '.nii.gz':
                    img = change_mha_vti(data_folder+fn, label)
            else:
                print('Invalid input/output format')
                break

            if rem_str:
                fn = fn.replace(rem_str, '')

            img_name = fn.replace(input_format, '')
            # make int and remove 1
            # img_name = str(int(img_name)-1)
            # img_name = img_name.zfill(2) + output_format

            print(f"Saving {img_name}")
            if output_format != '.vti':
                sitk.WriteImage(img, out_folder+img_name+output_format)
            else:
                vf.write_img(out_folder+img_name+output_format, img)

            if surface:
                img_vtk = vf.exportSitk2VTK(img)[0]
                poly = vf.vtk_marching_cube(img_vtk, 0, 1)
                vf.write_geo(out_folder+img_name+'.vtp', poly)
