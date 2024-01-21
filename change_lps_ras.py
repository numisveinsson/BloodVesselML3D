import vtk

if __name__=='__main__':
    # script to convert vtp with lps coordinates to ras coordinates

    # DIR = '/Users/numisveins/Documents/Automatic_Tracing_Data/train_version_5_aortas/all_train/'
    # Load your VTP file
    vtp_file_path = "/Users/numisveins/Documents/ASOCA_dataset/centerlines/Normal_10.vtp"
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file_path)
    reader.Update()

    # Get the PolyData
    poly_data = reader.GetOutput()

    # Get the points from the PolyData
    points = poly_data.GetPoints()

    # Iterate through each point and flip the sign of X and Y coordinates
    for i in range(points.GetNumberOfPoints()):
        x, y, z = points.GetPoint(i)
        points.SetPoint(i, -x, -y, z)

    # Update the PolyData
    poly_data.Modified()
    # Write the modified PolyData back to a VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("/Users/numisveins/Documents/ASOCA_dataset/centerlines/new/file.vtp")
    writer.SetInputData(poly_data)
    writer.Write()

    import SimpleITK as sitk

    # Load your NRRD image
    nrrd_file_path = "/Users/numisveins/Documents/ASOCA_dataset/images/Normal_10.nrrd"
    image = sitk.ReadImage(nrrd_file_path)

    # Get the original image direction matrix
    original_direction_matrix = image.GetDirection()

    # Create a new direction matrix for LPS coordinate system
    lps_direction_matrix = (-1, 0, 0, 0, -1, 0, 0, 0, 1)

    # Set the new origin to (0, 0, same)
    new_origin = (0, 0, image.GetOrigin()[2])

    # Update the image direction matrix and origin
    image.SetDirection(lps_direction_matrix)
    image.SetOrigin(new_origin)

    # Write the modified image to a new NRRD file
    output_nrrd_file_path = "/Users/numisveins/Documents/ASOCA_dataset/images/new/file.nrrd"
    sitk.WriteImage(image, output_nrrd_file_path)