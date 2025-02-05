import os
import vtk


def scale_polydata(input_file, output_file, scale_factor):
    # Read the VTP file
    reader = vtk.vtkXMLPolyDataReader()
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
    
    # Write the scaled polydata to a new file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()


def process_folder(input_folder, output_folder, scale_factor):
    # Loop over all .vtp files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.vtp'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            
            # Scale the polydata and save the new file
            scale_polydata(input_file, output_file, scale_factor)
            print(f"Scaled {file_name} and saved to {output_file}")

# Example usage:
input_folder = 'path/to/your/input_folder'
output_folder = 'path/to/your/output_folder'
scale_factor = 2.0  # Change this value to whatever scaling factor you need

process_folder(input_folder, output_folder, scale_factor)
