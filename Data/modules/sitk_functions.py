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
        
    return stats.GetLabels(), means, removed_seg

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
    # if it is a reader object, do the following
    if type(reader_im) == sitk.ImageFileReader:
        reader_im.SetExtractIndex(index_extract)
        reader_im.SetExtractSize(size_extract)
        new_img = reader_im.Execute()
    # if it is a sitk image, do the following
    elif type(reader_im) == sitk.Image:
        new_img = sitk.RegionOfInterest(reader_im, size_extract, index_extract)
    else:
        print('Error: reader_im must be a sitk image or reader object')
        return None

    return new_img

def rotate_volume_tangent(sitk_img, tangent, point):
    """
    Function to rotate a volume so that the tangent is aligned with the x-axis
    args:
        sitk_img: sitk image volume
        tangent: tangent vector
        point: point to rotate around
    """
    # sitk needs point to be a tuple of floats
    point = tuple([float(i) for i in point])

    # Get the direction of the image
    direction = sitk_img.GetDirection()

    # Get the angle between the tangent and the x-axis
    angle = np.arccos(np.dot(direction[0:3], tangent))

    # Get the axis of rotation
    axis = np.cross(direction[0:3], tangent)

    # Create the rotation matrix
    rotation = sitk.VersorTransform(axis, angle)

    # Create the affine transformation
    affine = sitk.AffineTransform(3)
    affine.SetCenter(point)
    affine.SetMatrix(rotation.GetMatrix())

    # Apply the transformation
    sitk_img = sitk.Resample(sitk_img, sitk_img, affine, sitk.sitkLinear, 0.0, sitk_img.GetPixelID())

    return sitk_img

def map_to_image(point, radius, size_volume, origin_im, spacing_im, size_im, prop=1):
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
    ratio = 1/2 # how much can be outside volume

    size_extract = np.ceil(size_volume*radius/spacing_im)
    index_extract = np.rint((point-origin_im - (size_volume/2)*radius)/spacing_im)
    end_bounds = index_extract+size_extract
    
    voi_min = point - (size_volume/2)*radius*prop
    voi_max = point + (size_volume/2)*radius*prop

    for i, ind in enumerate(np.logical_and(end_bounds > size_im,(end_bounds- size_im) < ratio*size_extract )):
        if ind:
            # print('\nsub-volume outside global volume, correcting\n')
            size_extract[i] = size_im[i] - index_extract[i]

    for i, ind in enumerate(np.logical_and(index_extract < np.zeros(3),(np.zeros(3)-index_extract) < ratio*size_extract )):
        if ind:
            # print('\nsub-volume outside global volume, correcting\n')
            index_extract[i] = 0

    return size_extract, index_extract, voi_min, voi_max

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

def sitk_to_numpy(Image):

    np_array = sitk.GetArrayFromImage(Image)
    return np_array

def numpy_to_sitk(numpy, file_reader = None):

    Image = sitk.GetImageFromArray(numpy)

    if file_reader:

        Image.SetSpacing(file_reader.GetSpacing())
        Image.SetOrigin(file_reader.GetOrigin())
        Image.SetDirection(file_reader.GetDirection())

    return Image