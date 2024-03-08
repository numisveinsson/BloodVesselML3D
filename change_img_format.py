import os
import SimpleITK as sitk
from modules import vtk_functions as vf

if __name__=='__main__':

    # import pdb; pdb.set_trace()

    input_format = '.mha'
    output_format = '.nrrd'
    label = True # false if raw image

    # rem_str = 'coroasocact_0'
    rem_str = '_seg_rem_3d_fullres_0'

    data_folder = '/Users/numisveins/Downloads/output_v1_500_stopmin/segs/'
    out_folder = data_folder+'new_format/'

    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]
    
    try:
        os.mkdir(out_folder)
        imgs_old = []
    except Exception as e: 
        print(e)
        imgs_old = os.listdir(out_folder)

    for fn in imgs:
        if fn.replace(input_format, output_format) in imgs_old:
            print(f"File {fn} already exists in folder")
            continue
        else:
            print(f"Converting file {fn} to new format {output_format}")
        if input_format != '.vti':
            img = sitk.ReadImage(data_folder+fn)
            if label:
                img = sitk.Cast(img, sitk.sitkUInt8)
        else:
            img = vf.read_img(data_folder+fn).GetOutput()
        
        if rem_str:
            fn = fn.replace(rem_str, '')
        
        img_name = fn.replace(input_format, '')
        # make int and remove 1 
        # img_name = str(int(img_name)-1)
        # img_name = img_name.zfill(2) + output_format

        if input_format != '.vti':
            sitk.WriteImage(img, out_folder+img_name+output_format)
        else:
            vf.write_img(out_folder+img_name+output_format, img)
