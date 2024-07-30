import os
import SimpleITK as sitk


def change_img_scale(img_path, out_path, scale):
    """
    Change the scale of the image in the path

    Do this by scaling the image spacing
    but keeping everything else the same

    :param img_path: path to the image
    :param out_path: path to save the new image
    :param scale: new scale
    :return: None
    """
    img = sitk.ReadImage(img_path)
    img.SetSpacing((img.GetSpacing()[0]*scale,
                    img.GetSpacing()[1]*scale, img.GetSpacing()[2]*scale))
    sitk.WriteImage(img, out_path)


if __name__ == '__main__':
    """
    Change the scale of the images in a folder

    Do this by scaling the image spacing
    but keeping everything else the same

    """

    input_format = '.mha'
    output_format = '.mha'

    scale = 10

    data_folder = '/Users/numisveins/Downloads/output_2d_aortofemct_mic23/new_format/'
    out_folder = data_folder+'new_format/'

    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]

    try:
        os.mkdir(out_folder)
        imgs_old = []
    except Exception as e:
        print(e)
        imgs_old = os.listdir(out_folder)

    for img in imgs:
        img_path = os.path.join(data_folder, img)
        out_path = os.path.join(out_folder, img.replace(input_format,
                                                        output_format))
        change_img_scale(img_path, out_path, scale)
