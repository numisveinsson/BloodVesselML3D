import SimpleITK as sitk
import vtk
import numpy

def binary_thinning(image_path, output_path):
    # Read the input binary image
    image = sitk.ReadImage(image_path)

    # Create the BinaryThinningImageFilter
    thinning_filter = sitk.BinaryThinningImageFilter()
    # thinning_filter.SetForegroundValue(255)
    # thinning_filter.SetBackgroundValue(0)
    # thinning_filter.SetThinBorder(True)
    # thinning_filter.SetUseImageSpacing(True)
    # thinning_filter.SetMaintainTopology(True)
    # thinning_filter.SetFullyConnected(True)
    # Apply thinning filter to the input image
    thin_image = thinning_filter.Execute(image)

    # Write the thinned image to disk
    sitk.WriteImage(thin_image, output_path)

    return thin_image

def create_centerline_mesh(thin_image, output_mesh_path):

    # Convert thin image to numpy array
    thin_array = sitk.GetArrayFromImage(thin_image)

    # Convert thin image points to vtkPoints
    points = vtk.vtkPoints()
    for idx, value in numpy.ndenumerate(thin_array):
        if value != 0:
            points.InsertNextPoint(idx[2], idx[1], idx[0])  # Note: VTK expects z, y, x indices

    # Create polyline from points
    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(points.GetNumberOfPoints())
    for i in range(points.GetNumberOfPoints()):
        polyline.GetPointIds().SetId(i, i)

    # Create vtkCellArray to store polylines
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(polyline)

    # Create vtkPolyData to store points and lines
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    # Write the polydata to a VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_mesh_path)
    writer.SetInputData(polydata)
    writer.Write()

if __name__ == "__main__":
    
    # input_image_path = "/Users/numisveins/Documents/PARSE_dataset/labels/PA000005.nii.gz"
    input_image_path = "/Users/numisveins/Downloads/bryan_out/3d_fullres_0006_0001_0/0006_0001_seg_rem_3d_fullres_0.mha"
    # output_image_path = "/Users/numisveins/Downloads/PA000005_thinned.mha"
    output_image_path = "/Users/numisveins/Downloads/0006_0001_seg_rem_3d_fullres_0_thinned.mha"
    thin_image = binary_thinning(input_image_path, output_image_path)

    # thin_image = sitk.ReadImage("/Users/numisveins/Downloads/PA000005_thinned.mha")

    output_mesh_path = "/Users/numisveins/Downloads/PA000005_thinned.vtp"
    create_centerline_mesh(thin_image, output_mesh_path)
