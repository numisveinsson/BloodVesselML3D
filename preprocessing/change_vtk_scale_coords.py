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
    from modules.logger import get_logger
    logger = get_logger(__name__)
    
    # Loop over all .vtp and .stl files in the folder
    files = [f for f in os.listdir(input_folder) if f.endswith(('.vtp', '.stl'))]
    logger.info(f"Found {len(files)} surface files to process")
    
    for file_name in files:
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)

        # Scale the polydata and save the new file
        scale_polydata(input_file, output_file, scale_factor)
        logger.info(f"Scaled {file_name} and saved to {output_file}")
    
    logger.info(f"Completed scaling {len(files)} files")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scale VTK polydata files (surfaces) by a factor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python change_vtk_scale_coords.py --input_dir /path/to/surfaces --output_dir /path/to/output --scale_factor 0.1
  
  # Using environment variables:
  export INPUT_DIR=/path/to/surfaces
  python change_vtk_scale_coords.py --scale_factor 0.1
        """
    )
    parser.add_argument('--input_dir', '--input-dir',
                       type=str,
                       default=None,
                       help='Directory containing input surface files (.vtp or .stl). '
                            'Defaults to INPUT_DIR env var or ./data/surfaces/')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Directory to write scaled surface files. '
                            'Defaults to OUTPUT_DIR env var or inferred from input_dir')
    parser.add_argument('--scale_factor', '--scale-factor',
                       type=float,
                       required=True,
                       help='Scale factor to apply to coordinates (e.g., 0.1 to convert mm to cm)')
    
    args = parser.parse_args()
    
    # Priority: command-line arg > environment variable > default
    input_folder = (args.input_dir or 
                   os.getenv('INPUT_DIR') or 
                   './data/surfaces/')
    output_folder = (args.output_dir or 
                    os.getenv('OUTPUT_DIR') or 
                    input_folder.rstrip('/') + '_scaled/')
    scale_factor = args.scale_factor

    # Validate directories
    if not os.path.exists(input_folder):
        raise ValueError(f"Input directory not found: {input_folder}. "
                        f"Provide --input_dir argument or set INPUT_DIR environment variable.")

    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # Initialize logger
    from modules.logger import get_logger
    logger = get_logger(__name__)
    logger.info(f"Scaling surfaces from {input_folder} to {output_folder} with factor {scale_factor}")
    
    process_folder(input_folder, output_folder, scale_factor)
