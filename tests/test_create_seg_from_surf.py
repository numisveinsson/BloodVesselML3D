"""
Unit tests for create_seg_from_surf.py

Tests the conversion of surface meshes to segmentation images.
"""
import unittest
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import sys
import os

# Add global directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'global'))
from create_seg_from_surf import (
    eraseBoundary,
    build_transform_matrix,
    exportSitk2VTK,
    vtkImageResample,
    vtk_marching_cube,
    exportPython2VTK,
    smooth_polydata,
    decimation,
    appendPolyData,
    convertPolyDataToImageData
)


class TestEraseBoundary(unittest.TestCase):
    """Test the eraseBoundary function."""
    
    def test_erase_boundary_3d(self):
        """Test erasing boundary pixels in 3D array."""
        labels = np.ones((10, 10, 10), dtype=np.int32)
        result = eraseBoundary(labels, 1, 0)
        
        # Check that boundary is set to 0
        self.assertEqual(result[0, 5, 5], 0)
        self.assertEqual(result[-1, 5, 5], 0)
        self.assertEqual(result[5, 0, 5], 0)
        self.assertEqual(result[5, -1, 5], 0)
        self.assertEqual(result[5, 5, 0], 0)
        self.assertEqual(result[5, 5, -1], 0)
        
        # Check that interior is still 1
        self.assertEqual(result[5, 5, 5], 1)
    
    def test_erase_boundary_multiple_pixels(self):
        """Test erasing multiple boundary pixels."""
        labels = np.ones((20, 20, 20), dtype=np.int32)
        result = eraseBoundary(labels, 3, 0)
        
        # Check that 3 pixels from boundary are erased
        self.assertEqual(result[2, 10, 10], 0)
        self.assertEqual(result[3, 10, 10], 1)
        self.assertEqual(result[-3, 10, 10], 0)
        self.assertEqual(result[-4, 10, 10], 1)


class TestBuildTransformMatrix(unittest.TestCase):
    """Test the build_transform_matrix function."""
    
    def test_identity_transform(self):
        """Test transform matrix with identity direction."""
        img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        img.SetOrigin([0, 0, 0])
        img.SetSpacing([1.0, 1.0, 1.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        matrix = build_transform_matrix(img)
        
        # Check that it's a 4x4 matrix
        self.assertEqual(matrix.shape, (4, 4))
        
        # Check diagonal (spacing)
        self.assertEqual(matrix[0, 0], 1.0)
        self.assertEqual(matrix[1, 1], 1.0)
        self.assertEqual(matrix[2, 2], 1.0)
        self.assertEqual(matrix[3, 3], 1.0)
        
        # Check origin
        self.assertEqual(matrix[0, 3], 0.0)
        self.assertEqual(matrix[1, 3], 0.0)
        self.assertEqual(matrix[2, 3], 0.0)
    
    def test_transform_with_spacing(self):
        """Test transform matrix with non-unit spacing."""
        img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        img.SetOrigin([10, 20, 30])
        img.SetSpacing([2.0, 3.0, 4.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        matrix = build_transform_matrix(img)
        
        # Check spacing in matrix
        self.assertEqual(matrix[0, 0], 2.0)
        self.assertEqual(matrix[1, 1], 3.0)
        self.assertEqual(matrix[2, 2], 4.0)
        
        # Check origin
        self.assertEqual(matrix[0, 3], 10.0)
        self.assertEqual(matrix[1, 3], 20.0)
        self.assertEqual(matrix[2, 3], 30.0)


class TestExportPython2VTK(unittest.TestCase):
    """Test the exportPython2VTK function."""
    
    def test_export_int_array(self):
        """Test exporting integer numpy array to VTK."""
        arr = np.arange(27, dtype=np.int32).reshape(3, 3, 3)
        vtk_array = exportPython2VTK(arr)
        
        # Check that it's a VTK array
        self.assertIsNotNone(vtk_array)
        self.assertEqual(vtk_array.GetNumberOfTuples(), 27)
    
    def test_export_float_array(self):
        """Test exporting float numpy array to VTK."""
        arr = np.random.rand(5, 5, 5).astype(np.float32)
        vtk_array = exportPython2VTK(arr)
        
        self.assertIsNotNone(vtk_array)
        self.assertEqual(vtk_array.GetNumberOfTuples(), 125)


class TestExportSitk2VTK(unittest.TestCase):
    """Test the exportSitk2VTK function."""
    
    def test_export_simple_image(self):
        """Test exporting SimpleITK image to VTK."""
        img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        img.SetOrigin([0, 0, 0])
        img.SetSpacing([1.0, 1.0, 1.0])
        
        vtk_img, matrix = exportSitk2VTK(img)
        
        # Check VTK image properties
        self.assertIsNotNone(vtk_img)
        self.assertEqual(vtk_img.GetDimensions(), (10, 10, 10))
        self.assertEqual(vtk_img.GetSpacing(), (1.0, 1.0, 1.0))
        
        # Check matrix
        self.assertIsNotNone(matrix)
    
    def test_export_with_custom_spacing(self):
        """Test exporting with custom spacing."""
        img = sitk.Image([5, 5, 5], sitk.sitkFloat32)
        img.SetOrigin([0, 0, 0])
        img.SetSpacing([1.0, 1.0, 1.0])
        
        vtk_img, _ = exportSitk2VTK(img, spacing=[2.0, 2.0, 2.0])
        
        self.assertEqual(vtk_img.GetSpacing(), (2.0, 2.0, 2.0))


class TestVTKMarshingCube(unittest.TestCase):
    """Test the vtk_marching_cube function."""
    
    def test_marching_cube_simple(self):
        """Test marching cubes on simple segmentation."""
        # Create a simple segmentation with a cube
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[5:15, 5:15, 5:15] = 1
        
        # Convert to VTK
        vtk_array = numpy_to_vtk(arr.flatten('F'))
        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(20, 20, 20)
        vtk_img.GetPointData().SetScalars(vtk_array)
        vtk_img.SetSpacing(1, 1, 1)
        vtk_img.SetOrigin(0, 0, 0)
        
        # Run marching cubes
        mesh = vtk_marching_cube(vtk_img, 0, 1)
        
        # Check that mesh was created
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.GetNumberOfPoints(), 0)
        self.assertGreater(mesh.GetNumberOfCells(), 0)


class TestSmoothPolydata(unittest.TestCase):
    """Test the smooth_polydata function."""
    
    def test_smooth_sphere(self):
        """Test smoothing a sphere."""
        # Create a sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(5.0)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        sphere.Update()
        
        poly = sphere.GetOutput()
        initial_points = poly.GetNumberOfPoints()
        
        # Smooth it
        smoothed = smooth_polydata(poly, iteration=10)
        
        # Check that it's still a valid polydata
        self.assertIsNotNone(smoothed)
        self.assertEqual(smoothed.GetNumberOfPoints(), initial_points)
        self.assertGreater(smoothed.GetNumberOfCells(), 0)


class TestDecimation(unittest.TestCase):
    """Test the decimation function."""
    
    def test_decimate_sphere(self):
        """Test decimating a sphere."""
        # Create a sphere with many points
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(5.0)
        sphere.SetThetaResolution(32)
        sphere.SetPhiResolution(32)
        sphere.Update()
        
        poly = sphere.GetOutput()
        initial_points = poly.GetNumberOfPoints()
        
        # Decimate by 50%
        decimated = decimation(poly, 0.5)
        
        # Check that points were reduced
        self.assertIsNotNone(decimated)
        self.assertLess(decimated.GetNumberOfPoints(), initial_points)
        self.assertGreater(decimated.GetNumberOfCells(), 0)


class TestAppendPolyData(unittest.TestCase):
    """Test the appendPolyData function."""
    
    def test_append_two_spheres(self):
        """Test appending two sphere polydata."""
        # Create two spheres
        sphere1 = vtk.vtkSphereSource()
        sphere1.SetCenter(0, 0, 0)
        sphere1.SetRadius(2.0)
        sphere1.Update()
        
        sphere2 = vtk.vtkSphereSource()
        sphere2.SetCenter(5, 0, 0)
        sphere2.SetRadius(2.0)
        sphere2.Update()
        
        poly_list = [sphere1.GetOutput(), sphere2.GetOutput()]
        
        # Append them
        combined = appendPolyData(poly_list)
        
        # Check that combined has more points
        self.assertIsNotNone(combined)
        self.assertGreater(combined.GetNumberOfPoints(), 
                          sphere1.GetOutput().GetNumberOfPoints())


class TestConvertPolyDataToImageData(unittest.TestCase):
    """Test the convertPolyDataToImageData function."""
    
    def test_sphere_to_image(self):
        """Test converting a sphere to image data."""
        # Create a sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(10, 10, 10)
        sphere.SetRadius(5.0)
        sphere.Update()
        
        # Create reference image
        ref_im = vtk.vtkImageData()
        ref_im.SetDimensions(20, 20, 20)
        ref_im.SetSpacing(1.0, 1.0, 1.0)
        ref_im.SetOrigin(0, 0, 0)
        ref_im.AllocateScalars(vtk.VTK_INT, 1)
        ref_im.GetPointData().SetScalars(
            numpy_to_vtk(np.zeros(8000, dtype=np.int32))
        )
        
        # Convert sphere to image
        result = convertPolyDataToImageData(sphere.GetOutput(), ref_im)
        
        # Check that result is valid
        self.assertIsNotNone(result)
        self.assertEqual(result.GetDimensions(), (20, 20, 20))
        
        # Check that some voxels are set to 1 (inside sphere)
        arr = vtk_to_numpy(result.GetPointData().GetScalars())
        self.assertGreater(np.sum(arr == 0), 0)  # Some voxels should be 0 (inside, reversed stencil)


class TestVTKImageResample(unittest.TestCase):
    """Test the vtkImageResample function."""
    
    def test_resample_nearest_neighbor(self):
        """Test resampling with nearest neighbor."""
        # Create simple image
        img = vtk.vtkImageData()
        img.SetDimensions(10, 10, 10)
        img.SetSpacing(1.0, 1.0, 1.0)
        img.SetOrigin(0, 0, 0)
        img.AllocateScalars(vtk.VTK_FLOAT, 1)
        
        # Resample to different spacing
        resampled = vtkImageResample(img, [2.0, 2.0, 2.0], 'NN')
        
        # Check new spacing
        self.assertIsNotNone(resampled)
        self.assertEqual(resampled.GetSpacing(), (2.0, 2.0, 2.0))
    
    def test_resample_linear(self):
        """Test resampling with linear interpolation."""
        img = vtk.vtkImageData()
        img.SetDimensions(10, 10, 10)
        img.SetSpacing(1.0, 1.0, 1.0)
        img.SetOrigin(0, 0, 0)
        img.AllocateScalars(vtk.VTK_FLOAT, 1)
        
        resampled = vtkImageResample(img, [0.5, 0.5, 0.5], 'linear')
        
        self.assertIsNotNone(resampled)
        self.assertEqual(resampled.GetSpacing(), (0.5, 0.5, 0.5))


if __name__ == '__main__':
    unittest.main()
