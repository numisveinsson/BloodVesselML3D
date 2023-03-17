## Functions to bind SITK functionality

import SimpleITK as sitk
import numpy as np

def read_image(file_dir_image):
    """
    Read image from file
    Args:
        file_dir_image: image directory
    Returns:
        SITK image reader
    """
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_dir_image)
    file_reader.ReadImageInformation()
    return file_reader

def read_image_numpy(file_dir_image):
    """
    Read image from file as numpy array
    Args:
        file_dir_image: image directory
    Returns:
        SITK image reader
        numpy array
    """
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_dir_image)
    file_reader.ReadImageInformation()

    file_img = sitk.ReadImage(file_dir_image)
    file_np_array = sitk.GetArrayFromImage(file_img)

    return file_reader, file_np_array

def create_new(file_reader):
    """
    Create new SITK image with same formating as another
    Args:
        file_reader: reader from another image
    Returns:
        SITK image
    """
    result_img = sitk.Image(file_reader.GetSize(), file_reader.GetPixelID(),
                            file_reader.GetNumberOfComponents())
    result_img.SetSpacing(file_reader.GetSpacing())
    result_img.SetOrigin(file_reader.GetOrigin())
    result_img.SetDirection(file_reader.GetDirection())
    return result_img

def create_new_from_numpy(file_reader, np_array):
    """
    Create new SITK image with same formating as another
    And values from an input numpy array
    Args:
        file_reader: reader from another image
        np_array: np array with image values
    Returns:
        SITK image
    """
    result_img = sitk.GetImageFromArray(np_array)
    result_img.SetSpacing(file_reader.GetSpacing())
    result_img.SetOrigin(file_reader.GetOrigin())
    result_img.SetDirection(file_reader.GetDirection())
    
    return result_img

def write_image(image, outputImageFileName):
    """
    Write image to file
    Args:
        SITK image, filename
    Returns:
        image file
    """
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outputImageFileName)
    writer.Execute(image)
    
    return None

def remove_other_vessels(image, seed):
    """
    Remove all labelled vessels except the one of interest
    Args:
        SITK image, seed point pointing to point in vessel of interest
    Returns:
        binary image file (either 0 or 1)
    """
    ccimage = sitk.ConnectedComponent(image)
    label = ccimage[seed]
    #print("The label is " + str(label))
    if label == 0:
        label = 1
    labelImage = sitk.BinaryThreshold(ccimage, lowerThreshold=label, upperThreshold=label)
    labelImage = labelImage
    return labelImage

def connected_comp_info(original_seg, print_condition):
    """
    Print info on the component being kept
    """
    removed_seg = sitk.ConnectedComponent(original_seg)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(removed_seg, original_seg)
    means = []

    for l in stats.GetLabels():
        if print_condition:
            print("Label: {0} -> Mean: {1} Size: {2}".format(l, stats.GetMean(l), stats.GetPhysicalSize(l)))
        means.append(stats.GetMean(l))
    return stats.GetLabels(), means

def extract_volume(reader_im, index_extract, size_extract):
    """
    Function to extract a smaller volume from a larger one using sitk
    args:
        reader_im: sitk image reader
        index_extract: the index of the lower corner for extraction
        size_extract: number of voxels to extract in each direction
    return:
        new_img: sitk image volume
    """
    reader_im.SetExtractIndex(index_extract)
    reader_im.SetExtractSize(size_extract)
    new_img = reader_im.Execute()

    return new_img

def map_to_image(point, radius, size_volume, origin_im, spacing_im, prop=1):
    """
    Function to map a point and radius to volume metrics
    args:
        point: point of volume center
        radius: radius at that point
        size_volume: multiple of radius equal the intended
            volume size
        origin_im: image origin
        spacing_im: image spacing
        prop: proportion of image to be counted for caps contraint
    return:
        size_extract: number of voxels to extract in each dim
        index_extract: index for sitk volume extraction
        voi_min/max: boundaries of volume for caps constraint
    """
    size_extract = np.ceil(size_volume*radius/spacing_im)
    index_extract = np.rint((point-origin_im - (size_volume/2)*radius)/spacing_im)

    voi_min = point - (size_volume/2)*radius*prop
    voi_max = point + (size_volume/2)*radius*prop

    return size_extract, index_extract, voi_min, voi_max

def rotate_volume():

    return rotated_voi

def import_image(image_dir):
    """
    Function to import image via sitk
    args:
        file_dir_image: image directory
    return:
        reader_img: sitk image volume reader
        origin_im: image origin coordinates
        size_im: image size
        spacing_im: image spacing
    """
    reader_im = read_image(image_dir)
    origin_im = np.array(list(reader_im.GetOrigin()))
    size_im = np.array(list(reader_im.GetSize()))
    spacing_im = np.array(list(reader_im.GetSpacing()))

    return reader_im, origin_im, size_im, spacing_im
