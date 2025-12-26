import os
import SimpleITK as sitk
import math
from datetime import datetime


def change_img_scale(img_path, scale, scale_origin=None, direction_matrix=None, change_lps_to_ras=False, verbose=False, if_spacing_file=False, spacing_value=None):
    """
    Change the scale of the image in the path

    Do this by scaling the image spacing
    but keeping everything else the same

    :param img_path: path to the image
    :param scale: new scale
    :param change_lps_to_ras: whether to change image from LPS to RAS coordinate system
    :return: sitk image
    """
    img = sitk.ReadImage(img_path)

    if if_spacing_file:
        img.SetSpacing(spacing_value)

    if verbose:
        try:
            print(f"[VERBOSE] Reading image: {img_path}")
            print(f"[VERBOSE] Original spacing: {img.GetSpacing()}")
            print(f"[VERBOSE] Original origin: {img.GetOrigin()}")
            print(f"[VERBOSE] Original direction: {img.GetDirection()}")
        except Exception:
            # Some SITK images may not expose these; ignore in that case
            pass

    if scale != 1:
        if verbose:
            print(f"[VERBOSE] Changing image scale of {img_path} by factor {scale}")
        img.SetSpacing((img.GetSpacing()[0]*scale,
                    img.GetSpacing()[1]*scale, img.GetSpacing()[2]*scale))

    if scale_origin:
        if verbose:
            print(f"[VERBOSE] Changing origin by factor {scale_origin}")
        img.SetOrigin((img.GetOrigin()[0]*scale_origin,
                       img.GetOrigin()[1]*scale_origin, img.GetOrigin()[2]*scale_origin))

    # If a direction matrix (3x3) is provided, set it on the image.
    # Expect a flat list/tuple of length 9 (row-major), e.g. [1,0,0,0,1,0,0,0,1]
    if direction_matrix is not None:
        # Basic validation
        if not (hasattr(direction_matrix, '__len__') and len(direction_matrix) == 9):
            raise ValueError("direction_matrix must be an iterable of 9 numbers (3x3 matrix flattened)")
        if verbose:
            print(f"[VERBOSE] Setting direction matrix to: {direction_matrix}")
        img.SetDirection(tuple(direction_matrix))

    if change_lps_to_ras:
        if verbose:
            print(f"[VERBOSE] Changing image from LPS to RAS coordinate system")
        # Flip the first two axes (X and Y)
        direction = list(img.GetDirection())
        direction = [-direction[0], -direction[1], direction[2],
                     -direction[3], -direction[4], direction[5],
                     -direction[6], -direction[7], direction[8]]
        img.SetDirection(tuple(direction))

        origin = list(img.GetOrigin())
        origin = [-origin[0], -origin[1], origin[2]]
        img.SetOrigin(tuple(origin))

    if verbose:
        try:
            print(f"[VERBOSE] Final spacing: {img.GetSpacing()}")
            print(f"[VERBOSE] Final origin: {img.GetOrigin()}")
            print(f"[VERBOSE] Final direction: {img.GetDirection()}")
        except Exception:
            pass

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

    input_format = '.mha'
    output_format = '.mha'

    data_folder = '/Users/nsveinsson/Documents/datasets/CAS_cerebral_dataset/CAS2023_trainingdataset/mm/truths/'
    out_folder = data_folder.replace('mm','cm')

    if_spacing_file = False
    spacing_file = '/Users/nsveinsson/Documents/datasets/CAS_cerebral_dataset/CAS2023_trainingdataset/mm/meta.csv'

    list_names = []  # if empty, process all files in the folder

    flip = False
    permute = False
    scale = 0.1

    change_lps_to_ras = False

    scale_origin = 0.1 # or None

    # Optional: set a direction matrix (3x3) to override image direction
    # Provide as a flat list/tuple of 9 values (row-major), or set to None to keep original
    # Example identity: [1,0,0,0,1,0,0,0,1]
    direction_matrix = [1,0,0,0,1,0,0,0,1] # or None

    # Verbose: when True, print detailed per-file info
    verbose = False

    flip_axis = [True, True, False]

    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]
    imgs = sorted(imgs)

    if if_spacing_file:
        import pandas as pd
        spacing_df = pd.read_csv(spacing_file)
        # only keep 'spacing', they are sorted
        spacing_values = spacing_df['spacing'].values
        # read as tuples
        spacing_values = [tuple(map(float, x[1:-1].split(','))) for x in spacing_values]

    if list_names:
        imgs = [f for f in imgs if any(name in f for name in list_names)]

    try:
        os.mkdir(out_folder)
        imgs_old = []
    except Exception as e:
        print(e)
        imgs_old = os.listdir(out_folder)

    # Prepare changelog file in output folder
    log_path = os.path.join(out_folder, 'change_log.txt')
    def write_log(msg: str):
        with open(log_path, 'a') as lf:
            lf.write(msg + '\n')

    # Write header with parameters and timestamp
    header = (
        f"===== Change Log - {datetime.now().isoformat()} =====\n"
        f"input_format: {input_format}\n"
        f"output_format: {output_format}\n"
        f"scale: {scale}\n"
        f"scale_origin: {scale_origin}\n"
        f"direction_matrix: {direction_matrix}\n"
        f"flip: {flip}\n"
        f"permute: {permute}\n"
        f"verbose: {verbose}\n"
        "--------------------------------------------"
    )
    write_log(header)

    for ind, img in enumerate(imgs):
        img_path = os.path.join(data_folder, img)
        out_path = os.path.join(out_folder, img.replace(input_format,
                                                        output_format))
        if if_spacing_file:
            spacing_value = spacing_values[imgs.index(img)]
        else:
            spacing_value = None

        img = change_img_scale(img_path, scale, scale_origin, direction_matrix, change_lps_to_ras=change_lps_to_ras, verbose=verbose, if_spacing_file=if_spacing_file, spacing_value=spacing_value)

        if flip:
            if verbose:
                print(f'[VERBOSE] Flipping image {img_path} in axis {flip_axis}')
            img = flip_img(img, flip_axis)

        if permute:
            if verbose:
                print(f'[VERBOSE] Permuting axes for image {img_path}')
            img = sitk.PermuteAxes(img, [0, 1, 2])
                
        sitk.WriteImage(img, out_path)
        msg = (
            f"{datetime.now().isoformat()} | Saved {img} -> {out_path} | "
            f"scale={scale if scale!=1 else 'none'} | scale_origin={scale_origin} | "
            f"direction_set={'yes' if direction_matrix is not None else 'no'} | "
            f"flipped={'yes' if flip else 'no'} | permuted={'yes' if permute else 'no'}"
        )
        write_log(msg)
        print(f'Image {ind+1}/{len(imgs)} saved to {out_path}')

    # Write footer summary
    write_log(f"Finished processing {len(imgs)} images at {datetime.now().isoformat()}")
