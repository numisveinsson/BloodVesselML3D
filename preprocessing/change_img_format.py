import os
import SimpleITK as sitk
from modules import vtk_functions as vf


def _compute_bounds_sitk(img):
    """Compute axis-aligned physical bounds of a SITK image using origin, spacing, direction."""
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    # Build 3x3 direction matrix
    dir_mat = (
        (direction[0], direction[1], direction[2]),
        (direction[3], direction[4], direction[5]),
        (direction[6], direction[7], direction[8]),
    )

    # Generate corner indices
    ix = [0, max(size[0] - 1, 0)]
    iy = [0, max(size[1] - 1, 0)]
    iz = [0, max(size[2] - 1, 0)]

    corners = []
    for i in ix:
        for j in iy:
            for k in iz:
                # scaled voxel index
                sx = i * spacing[0]
                sy = j * spacing[1]
                sz = k * spacing[2]
                # apply direction matrix
                px = origin[0] + dir_mat[0][0] * sx + dir_mat[0][1] * sy + dir_mat[0][2] * sz
                py = origin[1] + dir_mat[1][0] * sx + dir_mat[1][1] * sy + dir_mat[1][2] * sz
                pz = origin[2] + dir_mat[2][0] * sx + dir_mat[2][1] * sy + dir_mat[2][2] * sz
                corners.append((px, py, pz))

    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)
    min_z = min(c[2] for c in corners)
    max_z = max(c[2] for c in corners)

    return ((min_x, max_x), (min_y, max_y), (min_z, max_z))


def _compute_bounds_vtk(vtk_img):
    """Compute axis-aligned physical bounds of a VTK image using origin, spacing, extent."""
    img = vtk_img
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    ex = img.GetExtent()  # (xmin, xmax, ymin, ymax, zmin, zmax)

    ix = [ex[0], ex[1]]
    iy = [ex[2], ex[3]]
    iz = [ex[4], ex[5]]

    corners = []
    for i in ix:
        for j in iy:
            for k in iz:
                px = origin[0] + i * spacing[0]
                py = origin[1] + j * spacing[1]
                pz = origin[2] + k * spacing[2]
                corners.append((px, py, pz))

    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)
    min_z = min(c[2] for c in corners)
    max_z = max(c[2] for c in corners)

    return ((min_x, max_x), (min_y, max_y), (min_z, max_z))


def _compare_bounds(bounds_a, bounds_b, tol=1e-4):
    """Compare two sets of bounds per axis with tolerance. Returns (ok, diffs)."""
    diffs = []
    for axis in range(3):
        dmin = abs(bounds_a[axis][0] - bounds_b[axis][0])
        dmax = abs(bounds_a[axis][1] - bounds_b[axis][1])
        diffs.append((dmin, dmax))
    ok = all(dmin <= tol and dmax <= tol for dmin, dmax in diffs)
    return ok, diffs


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


def change_vti_sitk(file_dir, label=False):
    """
    Change the format of a file from .vti to .mha
    SITK does not support .vti format, so we need to use the vtk functions
    Args:
        file_dir: str, path to the file
        label: bool, whether this is a label segmentation (will cast to UInt8)
    Returns:
        None
    """
    img = vf.read_img(file_dir)
    img = vf.exportVTK2Sitk(img)
    
    if label:
        img = sitk.Cast(img, sitk.sitkUInt8)

    return img


if __name__ == '__main__':

    # import pdb; pdb.set_trace()

    input_format = '.nrrd'  # '.dcm' or '.nii.gz' or '.vti'
    output_format = '.mha'
    label = False  # false if raw image
    surface = False  # true if we want to save the surface vtp file
    
    # Only treat files as labels if they contain this string in the filename
    # Set to None or empty string to use the global 'label' setting for all files
    label_if_string = ''  # e.g., 'seg', 'mask', 'label', 'gt'

    rem_str = ''  # 'coroasocact_0'

    data_folder = '/Users/nsveinsson/Documents/datasets/ASOCA_dataset/images/'
    out_folder = data_folder.replace('images', 'images_mha/')

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
            
            # Determine if this file should be treated as a label
            is_label = label  # Default to global setting
            if label_if_string:
                is_label = label_if_string in fn
                print(f"  File contains '{label_if_string}': {is_label} -> treating as {'label' if is_label else 'image'}")
            
            # Compute pre-conversion bounds
            before_bounds = None
            if input_format == '.vti':
                vtk_before = vf.read_img(data_folder+fn).GetOutput()
                before_bounds = _compute_bounds_vtk(vtk_before)
            else:
                sitk_before = sitk.ReadImage(data_folder+fn)
                before_bounds = _compute_bounds_sitk(sitk_before)
            
            if input_format != '.vti' and output_format != '.vti':
                img = sitk.ReadImage(data_folder+fn)
                if is_label:
                    img = sitk.Cast(img, sitk.sitkUInt8)
            elif input_format == '.vti':
                if output_format == '.mha' or output_format == '.nii.gz':
                    img = change_vti_sitk(data_folder+fn, is_label)
                else:
                    img = vf.read_img(data_folder+fn).GetOutput()
            elif output_format == '.vti':
                if input_format in ['.mha', '.nii.gz', '.nii', '.mhd']:
                    img = change_mha_vti(data_folder+fn, is_label)
            else:
                print('Invalid input/output format')
                break

            if rem_str:
                fn = fn.replace(rem_str, '')

            img_name = fn.replace(input_format, '')
            # make int and remove 1
            # img_name = str(int(img_name)-1)
            # img_name = img_name.zfill(2) + output_format

            # Compute post-conversion bounds and compare
            if output_format != '.vti':
                after_bounds = _compute_bounds_sitk(img)
            else:
                after_bounds = _compute_bounds_vtk(img)
            ok, diffs = _compare_bounds(before_bounds, after_bounds)
            print(f"  Bounds before:  X{before_bounds[0]} Y{before_bounds[1]} Z{before_bounds[2]}")
            print(f"  Bounds after:   X{after_bounds[0]} Y{after_bounds[1]} Z{after_bounds[2]}")
            if ok:
                print("  Bounds check: OK (no accidental transform)")
            else:
                print(f"  WARNING: Bounds differ (tol=1e-4). Diffs per axis (min,max): {diffs}")

            print(f"Saving {img_name}")
            if output_format != '.vti':
                sitk.WriteImage(img, out_folder+img_name+output_format)
            else:
                vf.write_img(out_folder+img_name+output_format, img)

            if surface and is_label:  # Only create surfaces for label files
                img_vtk = vf.exportSitk2VTK(img)[0]
                poly = vf.vtk_marching_cube(img_vtk, 0, 1)
                vf.write_geo(out_folder+img_name+'.vtp', poly)
