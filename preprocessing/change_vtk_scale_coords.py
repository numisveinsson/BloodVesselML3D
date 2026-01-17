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

    # Also scale any data arrays named 'MaximumInscribedSphereRadius' in point or cell data
    try:
        from modules.logger import get_logger
        logger = get_logger(__name__)
    except Exception:
        logger = None

    for data_name, data in (('point', polydata.GetPointData()), ('cell', polydata.GetCellData())):
        if not data:
            continue
        arr = data.GetArray('MaximumInscribedSphereRadius')
        if arr is None:
            continue
        nc = arr.GetNumberOfComponents()
        n_tuples = arr.GetNumberOfTuples()
        for tidx in range(n_tuples):
            if nc == 1:
                try:
                    val = arr.GetTuple1(tidx)
                    arr.SetTuple1(tidx, val * scale_factor)
                except AttributeError:
                    tup = arr.GetTuple(tidx)
                    arr.SetTuple(tidx, (tup[0] * scale_factor,))
            else:
                tup = list(arr.GetTuple(tidx))
                for c in range(nc):
                    tup[c] = tup[c] * scale_factor
                arr.SetTuple(tidx, tup)
        if logger:
            logger.info(f"Scaled 'MaximumInscribedSphereRadius' in {data_name} data by {scale_factor}")

    polydata.Modified()

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
    try:
        from modules.logger import get_logger
    except Exception:
        import sys
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
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
  
  # Using default directory:
  python change_vtk_scale_coords.py --scale_factor 0.1
        """
    )
    parser.add_argument('--input_dir', '--input-dir',
                       type=str,
                       default=None,
                       help='Directory containing input surface files (.vtp or .stl). '
                            'Defaults to ./data/surfaces/')
    parser.add_argument('--output_dir', '--output-dir',
                       type=str,
                       default=None,
                       help='Directory to write scaled surface files. '
                            'Defaults to inferred from input_dir')
    parser.add_argument('--scale_factor', '--scale-factor',
                       type=float,
                       required=True,
                       help='Scale factor to apply to coordinates (e.g., 0.1 to convert mm to cm)')
    
    args = parser.parse_args()
    
    # Use command-line arguments (required or default)
    input_folder = args.input_dir or './data/surfaces/'
    output_folder = args.output_dir or input_folder.rstrip('/') + '_scaled/'
    scale_factor = args.scale_factor

    # Validate directories
    if not os.path.exists(input_folder):
        raise ValueError(f"Input directory not found: {input_folder}. "
                        f"Provide --input_dir argument.")

    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # Initialize logger
    try:
        from modules.logger import get_logger
    except Exception:
        import sys
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from modules.logger import get_logger
    logger = get_logger(__name__)
    logger.info(f"Scaling surfaces from {input_folder} to {output_folder} with factor {scale_factor}")
    
    process_folder(input_folder, output_folder, scale_factor)
