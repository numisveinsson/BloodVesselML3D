import os
import SimpleITK as sitk
import math


def change_img_scale(img_path, scale, scale_origin=None):
    """
    Change the scale of the image in the path

    Do this by scaling the image spacing
    but keeping everything else the same

    :param img_path: path to the image
    :param scale: new scale
    :return: sitk image
    """
    img = sitk.ReadImage(img_path)
    img.SetSpacing((img.GetSpacing()[0]*scale,
                    img.GetSpacing()[1]*scale, img.GetSpacing()[2]*scale))

    if scale_origin:
        img.SetOrigin((img.GetOrigin()[0]*scale_origin,
                       img.GetOrigin()[1]*scale_origin, img.GetOrigin()[2]*scale_origin))

    return img


def flip_img(img, flip_ax):
    """
    Flip the image in the axis

    :param img: sitk image
    :param flip_ax: list of axis to flip the image
        [True, False, False] flips the image in the x axis
    :return: sitk image
    """
    return sitk.Flip(img, flip_ax)


if __name__ == '__main__':
    """
    Change the scale of the images in a folder

    Do this by scaling the image spacing
    but keeping everything else the same

    flip_axis = [True, False, False] flips the image in the x axis
    flip_axis = [False, True, False] flips the image in the y axis
    flip_axis = [False, False, True] flips the image in the z axis

    """

    input_format = '.nii.gz'
    output_format = '.mha'

    flip = True
    permute = False
    scale = 1

    scale_origin = 1  # or None

    flip_axis = [True, True, False]

    data_folder = '/Users/numisveins/Documents/data_papers/data_combo_paper/ct_data/Ground truth cardiac segmentations/'
    out_folder = data_folder+'new_format/'

    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]
    imgs = sorted(imgs)

    try:
        os.mkdir(out_folder)
        imgs_old = []
    except Exception as e:
        print(e)
        imgs_old = os.listdir(out_folder)

    for ind, img in enumerate(imgs):
        img_path = os.path.join(data_folder, img)
        out_path = os.path.join(out_folder, img.replace(input_format,
                                                        output_format))
        img = change_img_scale(img_path, scale, scale_origin)

        if flip:
            print(f'Flipping image {img_path} in axis {flip_axis}')
            img = flip_img(img, flip_axis)

        if permute:
            img = sitk.PermuteAxes(img, [0, 1, 2])

        sitk.WriteImage(img, out_path)
