"""
Unit tests for preprocessing/change_vtk_scale_coords.py

Tests the scaling of VTK polydata coordinates.
"""
import unittest
import tempfile
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import sys

# Add preprocessing directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessing'))
from change_vtk_scale_coords import scale_polydata, process_folder


class TestScalePolyData(unittest.TestCase):
    """Test the scale_polydata function."""
    
    def setUp(self):
        """Set up temporary directory and test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_scale_vtp_file(self):
        """Test scaling a VTP file."""
        # Create a simple sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(10, 20, 30)
        sphere.SetRadius(5.0)
        sphere.Update()
        
        # Save to VTP
        input_path = os.path.join(self.input_dir, 'test.vtp')
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(input_path)
        writer.SetInputData(sphere.GetOutput())
        writer.Write()
        
        # Scale by factor of 2
        output_path = os.path.join(self.output_dir, 'test.vtp')
        scale_polydata(input_path, output_path, 2.0)
        
        # Read back the scaled file
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(output_path)
        reader.Update()
        scaled_poly = reader.GetOutput()
        
        # Check that coordinates were scaled
        original_center = sphere.GetOutput().GetCenter()
        scaled_center = scaled_poly.GetCenter()
        
        self.assertAlmostEqual(scaled_center[0], original_center[0] * 2.0, places=4)
        self.assertAlmostEqual(scaled_center[1], original_center[1] * 2.0, places=4)
        self.assertAlmostEqual(scaled_center[2], original_center[2] * 2.0, places=4)
    
    def test_scale_stl_file(self):
        """Test scaling an STL file."""
        # Create a simple cube
        cube = vtk.vtkCubeSource()
        cube.SetCenter(5, 5, 5)
        cube.SetXLength(2)
        cube.SetYLength(2)
        cube.SetZLength(2)
        cube.Update()
        
        # Save to STL
        input_path = os.path.join(self.input_dir, 'test.stl')
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(input_path)
        writer.SetInputData(cube.GetOutput())
        writer.Write()
        
        # Scale by factor of 0.5
        output_path = os.path.join(self.output_dir, 'test.stl')
        scale_polydata(input_path, output_path, 0.5)
        
        # Read back
        reader = vtk.vtkSTLReader()
        reader.SetFileName(output_path)
        reader.Update()
        scaled_poly = reader.GetOutput()
        
        # Check scaling
        original_center = cube.GetOutput().GetCenter()
        scaled_center = scaled_poly.GetCenter()
        
        self.assertAlmostEqual(scaled_center[0], original_center[0] * 0.5, places=4)
        self.assertAlmostEqual(scaled_center[1], original_center[1] * 0.5, places=4)
        self.assertAlmostEqual(scaled_center[2], original_center[2] * 0.5, places=4)
    
    def test_scale_by_one(self):
        """Test that scale factor of 1.0 doesn't change geometry."""
        # Create sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(0, 0, 0)
        sphere.SetRadius(10.0)
        sphere.Update()
        
        # Save
        input_path = os.path.join(self.input_dir, 'test.vtp')
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(input_path)
        writer.SetInputData(sphere.GetOutput())
        writer.Write()
        
        # Scale by 1.0
        output_path = os.path.join(self.output_dir, 'test.vtp')
        scale_polydata(input_path, output_path, 1.0)
        
        # Read back
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(output_path)
        reader.Update()
        scaled_poly = reader.GetOutput()
        
        # Compare points
        original_points = vtk_to_numpy(sphere.GetOutput().GetPoints().GetData())
        scaled_points = vtk_to_numpy(scaled_poly.GetPoints().GetData())
        
        np.testing.assert_array_almost_equal(original_points, scaled_points)
    
    def test_scale_with_negative_coordinates(self):
        """Test scaling with negative coordinates."""
        # Create polydata with negative coordinates
        points = vtk.vtkPoints()
        points.InsertNextPoint(-10, -20, -30)
        points.InsertNextPoint(-5, -10, -15)
        points.InsertNextPoint(0, 0, 0)
        
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        
        # Save
        input_path = os.path.join(self.input_dir, 'test.vtp')
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(input_path)
        writer.SetInputData(poly)
        writer.Write()
        
        # Scale by 2.0
        output_path = os.path.join(self.output_dir, 'test.vtp')
        scale_polydata(input_path, output_path, 2.0)
        
        # Read and verify
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(output_path)
        reader.Update()
        scaled_poly = reader.GetOutput()
        
        scaled_points = vtk_to_numpy(scaled_poly.GetPoints().GetData())
        self.assertAlmostEqual(scaled_points[0, 0], -20.0)
        self.assertAlmostEqual(scaled_points[0, 1], -40.0)
        self.assertAlmostEqual(scaled_points[0, 2], -60.0)
    
    def test_unsupported_format_raises_error(self):
        """Test that unsupported file format raises error."""
        input_path = os.path.join(self.input_dir, 'test.obj')
        output_path = os.path.join(self.output_dir, 'test.obj')
        
        with self.assertRaises(ValueError):
            scale_polydata(input_path, output_path, 2.0)
    
    def test_point_count_preserved(self):
        """Test that number of points is preserved after scaling."""
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)
        sphere.Update()
        
        original_count = sphere.GetOutput().GetNumberOfPoints()
        
        # Save and scale
        input_path = os.path.join(self.input_dir, 'test.vtp')
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(input_path)
        writer.SetInputData(sphere.GetOutput())
        writer.Write()
        
        output_path = os.path.join(self.output_dir, 'test.vtp')
        scale_polydata(input_path, output_path, 3.5)
        
        # Read back
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(output_path)
        reader.Update()
        
        self.assertEqual(reader.GetOutput().GetNumberOfPoints(), original_count)


class TestProcessFolder(unittest.TestCase):
    """Test the process_folder function."""
    
    def setUp(self):
        """Set up temporary directories and test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_process_multiple_vtp_files(self):
        """Test processing multiple VTP files in a folder."""
        # Create multiple test files
        for i in range(3):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(i*10, i*10, i*10)
            sphere.SetRadius(5.0)
            sphere.Update()
            
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(os.path.join(self.input_dir, f'test_{i}.vtp'))
            writer.SetInputData(sphere.GetOutput())
            writer.Write()
        
        # Process folder
        process_folder(self.input_dir, self.output_dir, 2.0)
        
        # Check that all files were processed
        output_files = os.listdir(self.output_dir)
        self.assertEqual(len(output_files), 3)
        
        for i in range(3):
            self.assertIn(f'test_{i}.vtp', output_files)
    
    def test_process_multiple_stl_files(self):
        """Test processing multiple STL files."""
        # Create test STL files
        for i in range(2):
            cube = vtk.vtkCubeSource()
            cube.SetCenter(i, i, i)
            cube.Update()
            
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(os.path.join(self.input_dir, f'cube_{i}.stl'))
            writer.SetInputData(cube.GetOutput())
            writer.Write()
        
        # Process folder
        process_folder(self.input_dir, self.output_dir, 0.5)
        
        # Check outputs
        output_files = os.listdir(self.output_dir)
        self.assertEqual(len(output_files), 2)
    
    def test_process_mixed_file_types(self):
        """Test processing folder with both VTP and STL files."""
        # Create VTP file
        sphere = vtk.vtkSphereSource()
        sphere.Update()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(os.path.join(self.input_dir, 'sphere.vtp'))
        writer.SetInputData(sphere.GetOutput())
        writer.Write()
        
        # Create STL file
        cube = vtk.vtkCubeSource()
        cube.Update()
        writer_stl = vtk.vtkSTLWriter()
        writer_stl.SetFileName(os.path.join(self.input_dir, 'cube.stl'))
        writer_stl.SetInputData(cube.GetOutput())
        writer_stl.Write()
        
        # Process folder
        process_folder(self.input_dir, self.output_dir, 2.0)
        
        # Check both files were processed
        output_files = os.listdir(self.output_dir)
        self.assertEqual(len(output_files), 2)
        self.assertIn('sphere.vtp', output_files)
        self.assertIn('cube.stl', output_files)
    
    def test_process_empty_folder(self):
        """Test processing empty folder."""
        # Process empty folder (should not raise error)
        process_folder(self.input_dir, self.output_dir, 1.0)
        
        # Output folder should be empty
        self.assertEqual(len(os.listdir(self.output_dir)), 0)
    
    def test_process_folder_ignores_other_files(self):
        """Test that processing ignores non-VTP/STL files."""
        # Create a VTP file
        sphere = vtk.vtkSphereSource()
        sphere.Update()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(os.path.join(self.input_dir, 'valid.vtp'))
        writer.SetInputData(sphere.GetOutput())
        writer.Write()
        
        # Create a text file (should be ignored)
        with open(os.path.join(self.input_dir, 'readme.txt'), 'w') as f:
            f.write('test')
        
        # Process folder
        process_folder(self.input_dir, self.output_dir, 2.0)
        
        # Only VTP file should be processed
        output_files = os.listdir(self.output_dir)
        self.assertEqual(len(output_files), 1)
        self.assertEqual(output_files[0], 'valid.vtp')


class TestScaleFactors(unittest.TestCase):
    """Test different scale factor scenarios."""
    
    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_scale_up_large_factor(self):
        """Test scaling up by large factor."""
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(1, 1, 1)
        sphere.SetRadius(1.0)
        sphere.Update()
        
        input_path = os.path.join(self.temp_dir, 'input.vtp')
        output_path = os.path.join(self.temp_dir, 'output.vtp')
        
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(input_path)
        writer.SetInputData(sphere.GetOutput())
        writer.Write()
        
        # Scale by 1000
        scale_polydata(input_path, output_path, 1000.0)
        
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(output_path)
        reader.Update()
        
        scaled_center = reader.GetOutput().GetCenter()
        self.assertAlmostEqual(scaled_center[0], 1000.0, places=2)
    
    def test_scale_down_small_factor(self):
        """Test scaling down by small factor."""
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(1000, 1000, 1000)
        sphere.Update()
        
        input_path = os.path.join(self.temp_dir, 'input.vtp')
        output_path = os.path.join(self.temp_dir, 'output.vtp')
        
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(input_path)
        writer.SetInputData(sphere.GetOutput())
        writer.Write()
        
        # Scale by 0.001
        scale_polydata(input_path, output_path, 0.001)
        
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(output_path)
        reader.Update()
        
        scaled_center = reader.GetOutput().GetCenter()
        self.assertAlmostEqual(scaled_center[0], 1.0, places=2)


if __name__ == '__main__':
    unittest.main()
