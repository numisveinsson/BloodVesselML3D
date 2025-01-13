import SimpleITK as sitk
import vtk


def transform_image(vtk_image_data):
    # Create a transform that rotates 90 degrees around the x-axis
    transform = vtk.vtkTransform()
    transform.RotateX(90)

    # Apply the transform to the vtkImageData
    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputData(vtk_image_data)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # Get the transformed output as vtkStructuredGrid
    structured_grid = transform_filter.GetOutput()

    # Convert vtkStructuredGrid back to vtkImageData
    bounds = structured_grid.GetBounds()
    spacing = vtk_image_data.GetSpacing()
    origin = vtk_image_data.GetOrigin()  # Use the original origin
    extent = [0] * 6

    # Calculate new extents based on the transformed bounds and spacing
    for i in range(3):
        extent[2 * i] = 0
        extent[2 * i + 1] = int((bounds[2 * i + 1] - bounds[2 * i]) / spacing[i])

    # Create vtkImageData object for the output
    rotated_image_data = vtk.vtkImageData()
    rotated_image_data.SetSpacing(spacing)
    rotated_image_data.SetExtent(extent)
    rotated_image_data.SetOrigin(origin)  # Use original origin

    # Copy the point data (scalar values, etc.) from the structured grid to the vtkImageData
    rotated_image_data.GetPointData().ShallowCopy(structured_grid.GetPointData())

    return rotated_image_data


# Function to read .vti file using VTK
def load_vti_image(file_path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_path)
    reader.Update()  # Read the file
    return reader.GetOutput()


# Function to get image properties
def get_image_info_vtk(image_data):
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    dimensions = image_data.GetDimensions()

    # For orientation, we use the direction cosines (identity matrix in most cases)
    mat = image_data.GetDirectionMatrix()
    # print(f"Direction matrix:\n{mat}")

    # Compute bounds
    bounds = [origin[i] + spacing[i] * (dimensions[i] - 1) for i in range(3)]
    for i in range(3):
        if bounds[i] < origin[i]:
            origin[i], bounds[i] = bounds[i], origin

    info = {
        "Origin": origin,
        "Spacing": spacing,
        "Dimensions": dimensions,
        "Bounds": bounds}
    return info


# Load images using SimpleITK
def load_image(image_path):
    return sitk.ReadImage(image_path)


# Get relevant properties
def get_image_info(image):
    # Cpmpute bounds
    bounds = [image.TransformIndexToPhysicalPoint((image.GetSize()[i] - 1, 0, 0))[i] for i in range(3)]

    info = {
        "Origin": image.GetOrigin(),
        "Spacing": image.GetSpacing(),
        "Size": image.GetSize(),
        "Direction": image.GetDirection(),
        "Bounds": bounds
    }
    return info


# Compare image properties
def compare_images(image1, image2):
    images_info = [get_image_info(image) for image in [image1, image2]]

    # Function to compare each property
    def compare_property(prop_name):
        prop_values = [info[prop_name] for info in images_info]
        if all(val == prop_values[0] for val in prop_values):
            return f"{prop_name} is the same across all images: {prop_values[0]}"
        else:
            return f"{prop_name} differs: {prop_values}"

    # Compare all properties
    for prop in ["Origin", "Spacing", "Size", "Direction"]:
        print(compare_property(prop))


# Check RAS coordinate system
def check_coordinate_system(image):
    direction = image.GetDirection()
    if direction == (1, 0, 0, 0, 1, 0, 0, 0, 1):
        return "Image is in RAS coordinate system"
    else:
        return "Image is not in RAS coordinate system"


# Main function
if __name__ == "__main__":
    # Provide the paths to your images
    image_path1 = '/Users/numisveins/Documents/vascular_data_3d/mmwhs_mr_train_manual_gt/sv_projects/1001/Images/1001.vti'
    image_path2 = '/Users/numisveins/Documents/vascular_data_3d/mmwhs_mr_train_manual_gt/images/1001.nii.gz'
    image_path3 = '/Users/numisveins/Documents/vascular_data_3d/mmwhs_mr_train_manual_gt/images/new_format/new_format/1001.mha'

    # Provide the path to your .vti image
    vti_file_path = image_path1

    # Load the image data
    image_data = load_vti_image(vti_file_path)

    # Transform the image
    image_data = transform_image(image_data)

    # Write the transformed image to a new .vti file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName('/Users/numisveins/Downloads/transformed_image.vti')
    writer.SetInputData(image_data)
    writer.Write()

    # Get and print image properties
    image_info = get_image_info_vtk(image_data)

    for key, value in image_info.items():
        print(f"{key}: {value}")

    # Load the images
    # image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    image3 = load_image(image_path3)

    # Compare images
    compare_images(image2, image3)

    # # Check if they are in the RAS coordinate system
    # for i, img in enumerate([image2, image3], 1):
    #     print(f"Image {i}: {check_coordinate_system(img)}")

    # Check bounds
    print("Origin of image 1:", image_info["Origin"])
    print("Bounds of image 1:", image_info["Bounds"])
    print("Origin of image 2:", get_image_info(image2)["Origin"])
    print("Bounds of image 2:", get_image_info(image2)["Bounds"])
    print("Origin of image 3:", get_image_info(image3)["Origin"])
    print("Bounds of image 3:", get_image_info(image3)["Bounds"])