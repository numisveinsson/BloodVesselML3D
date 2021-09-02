## Functions to bind SITK functionality

import SimpleITK as sitk

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
    return file_reader

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
    labelImage = sitk.BinaryThreshold(ccimage, lowerThreshold=label, upperThreshold=label)
    labelImage = labelImage
    return labelImage

def connected_comp_info(removed_seg, original_seg):
    """
    Print info on the component being kept
    """
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(removed_seg, original_seg)
    means = []
    for l in stats.GetLabels():
        print("Label: {0} -> Mean: {1} Size: {2}".format(l, stats.GetMean(l), stats.GetPhysicalSize(l)))
        means.append(stats.GetMean(l))
    return stats.GetLabels(), means
