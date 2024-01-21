import os
import SimpleITK as sitk

if __name__=='__main__':

    # import pdb; pdb.set_trace()

    input_format = '.mha'
    output_format = '.nrrd'
    label = False # false if raw image

    # rem_str = 'coroasocact_0'
    rem_str = '_seg_3d_fullres_0'

    data_folder = '/Users/numisveins/Downloads/segseg_preds_coronaries/'
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
        if rem_str:
            img_name = fn.replace(rem_str, '')
        
        img_name = img_name.replace(input_format, '')
        # make int and remove 1 
        # img_name = str(int(img_name)-1)
        # img_name = img_name.zfill(2) + output_format

        sitk.WriteImage(img, out_folder+img_name+output_format)
