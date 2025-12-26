import os
import vtk


def scale_polydata(input_file, output_file, scale_factor):
    # Determine file type and use appropriate reader
    if input_file.endswith('.vtp'):
        reader = vtk.vtkXMLPolyDataReader()
    elif input_file.endswith('.stl'):
        reader = vtk.vtkSTLReader()
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    reader.SetFileName(input_file)
    reader.Update()

    # Get the polydata from the reader
    polydata = reader.GetOutput()

    # Get the points of the polydata
    points = polydata.GetPoints()

    # Scale the points by the scale_factor
    for i in range(points.GetNumberOfPoints()):
        x, y, z = points.GetPoint(i)
        points.SetPoint(i, x * scale_factor, y * scale_factor, z * scale_factor)

    # Determine file type and use appropriate writer
    if output_file.endswith('.vtp'):
        writer = vtk.vtkXMLPolyDataWriter()
    elif output_file.endswith('.stl'):
        writer = vtk.vtkSTLWriter()
    else:
        raise ValueError(f"Unsupported file format: {output_file}")
    
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()


def process_folder(input_folder, output_folder, scale_factor):
    # Loop over all .vtp and .stl files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.vtp', '.stl')):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)

            # Scale the polydata and save the new file
            scale_polydata(input_file, output_file, scale_factor)
            print(f"Scaled {file_name} and saved to {output_file}")


if __name__ == "__main__":

    # Example usage:
    input_folder = '/Users/nsveinsson/Documents/datasets/ASOCA_dataset/mm/surfaces/'
    output_folder = '/Users/nsveinsson/Documents/datasets/ASOCA_dataset/cm/surfaces/'
    scale_factor = 0.1  # Change this value to whatever scaling factor you need

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    process_folder(input_folder, output_folder, scale_factor)
