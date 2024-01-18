import os
import SimpleITK as sitk

if __name__=='__main__':

    # import pdb; pdb.set_trace()

    input_format = '.nii.gz'
    output_format = '.mha'
    label = False # false if raw image

    data_folder = '/Users/numisveins/Downloads/output_2d_aortasmicct/'
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
        img = sitk.ReadImage(data_folder+fn)
        if label:
            img = sitk.Cast(img, sitk.sitkUInt8)

        sitk.WriteImage(img, out_folder+fn.replace(input_format, output_format))
