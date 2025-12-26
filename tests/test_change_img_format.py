"""
Unit tests for preprocessing/change_img_format.py

Tests the conversion between different image formats (MHA, VTI, NIFTI, etc.)
"""
import unittest
import tempfile
import os
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import sys

# Add preprocessing directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessing'))
from change_img_format import (
    _compute_bounds_sitk,
    _compute_bounds_vtk,
    _compare_bounds
)


class TestComputeBoundsSITK(unittest.TestCase):
    """Test the _compute_bounds_sitk function."""
    
    def test_simple_image_bounds(self):
        """Test bounds computation for simple image."""
        img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        img.SetOrigin([0, 0, 0])
        img.SetSpacing([1.0, 1.0, 1.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        bounds = _compute_bounds_sitk(img)
        
        # Expected bounds: (0, 9), (0, 9), (0, 9)
        self.assertAlmostEqual(bounds[0][0], 0.0, places=4)
        self.assertAlmostEqual(bounds[0][1], 9.0, places=4)
        self.assertAlmostEqual(bounds[1][0], 0.0, places=4)
        self.assertAlmostEqual(bounds[1][1], 9.0, places=4)
        self.assertAlmostEqual(bounds[2][0], 0.0, places=4)
        self.assertAlmostEqual(bounds[2][1], 9.0, places=4)
    
    def test_image_with_offset_origin(self):
        """Test bounds with non-zero origin."""
        img = sitk.Image([5, 5, 5], sitk.sitkFloat32)
        img.SetOrigin([10, 20, 30])
        img.SetSpacing([2.0, 2.0, 2.0])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        bounds = _compute_bounds_sitk(img)
        
        # Expected: origin + (size-1)*spacing
        self.assertAlmostEqual(bounds[0][0], 10.0, places=4)
        self.assertAlmostEqual(bounds[0][1], 10.0 + 4*2.0, places=4)
        self.assertAlmostEqual(bounds[1][0], 20.0, places=4)
        self.assertAlmostEqual(bounds[1][1], 20.0 + 4*2.0, places=4)
    
    def test_image_with_non_identity_direction(self):
        """Test bounds with non-identity direction matrix."""
        img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        img.SetOrigin([0, 0, 0])
        img.SetSpacing([1.0, 1.0, 1.0])
        # 90 degree rotation around z-axis
        img.SetDirection([0, -1, 0, 1, 0, 0, 0, 0, 1])
        
        bounds = _compute_bounds_sitk(img)
        
        # Bounds should be computed correctly for rotated image
        self.assertIsInstance(bounds, tuple)
        self.assertEqual(len(bounds), 3)
        for axis_bounds in bounds:
            self.assertEqual(len(axis_bounds), 2)


class TestComputeBoundsVTK(unittest.TestCase):
    """Test the _compute_bounds_vtk function."""
    
    def test_vtk_image_bounds(self):
        """Test bounds computation for VTK image."""
        img = vtk.vtkImageData()
        img.SetDimensions(10, 10, 10)
        img.SetOrigin(0, 0, 0)
        img.SetSpacing(1.0, 1.0, 1.0)
        
        bounds = _compute_bounds_vtk(img)
        
        # VTK extent is 0 to dims-1
        self.assertAlmostEqual(bounds[0][0], 0.0, places=4)
        self.assertAlmostEqual(bounds[0][1], 9.0, places=4)
        self.assertAlmostEqual(bounds[1][0], 0.0, places=4)
        self.assertAlmostEqual(bounds[1][1], 9.0, places=4)
    
    def test_vtk_image_with_spacing(self):
        """Test bounds with custom spacing."""
        img = vtk.vtkImageData()
        img.SetDimensions(5, 5, 5)
        img.SetOrigin(10, 20, 30)
        img.SetSpacing(3.0, 3.0, 3.0)
        
        bounds = _compute_bounds_vtk(img)
        
        # Expected: origin + extent_max * spacing
        self.assertAlmostEqual(bounds[0][0], 10.0, places=4)
        self.assertAlmostEqual(bounds[0][1], 10.0 + 4*3.0, places=4)


class TestCompareBounds(unittest.TestCase):
    """Test the _compare_bounds function."""
    
    def test_identical_bounds(self):
        """Test comparing identical bounds."""
        bounds_a = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
        bounds_b = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
        
        ok, diffs = _compare_bounds(bounds_a, bounds_b)
        
        self.assertTrue(ok)
        for axis_diffs in diffs:
            self.assertAlmostEqual(axis_diffs[0], 0.0)
            self.assertAlmostEqual(axis_diffs[1], 0.0)
    
    def test_within_tolerance(self):
        """Test bounds within tolerance."""
        bounds_a = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
        bounds_b = ((0.00001, 10.00001), (0.0, 10.0), (0.0, 10.0))
        
        ok, diffs = _compare_bounds(bounds_a, bounds_b, tol=0.001)
        
        self.assertTrue(ok)
    
    def test_outside_tolerance(self):
        """Test bounds outside tolerance."""
        bounds_a = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
        bounds_b = ((0.0, 11.0), (0.0, 10.0), (0.0, 10.0))
        
        ok, diffs = _compare_bounds(bounds_a, bounds_b, tol=0.1)
        
        self.assertFalse(ok)
        self.assertGreater(diffs[0][1], 0.1)
    
    def test_negative_bounds(self):
        """Test bounds with negative coordinates."""
        bounds_a = ((-5.0, 5.0), (-10.0, 10.0), (0.0, 20.0))
        bounds_b = ((-5.0, 5.0), (-10.0, 10.0), (0.0, 20.0))
        
        ok, diffs = _compare_bounds(bounds_a, bounds_b)
        
        self.assertTrue(ok)


class TestImageFormatConversion(unittest.TestCase):
    """Integration tests for format conversion."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_sitk_image_creation(self):
        """Test creating and saving SimpleITK image."""
        # Create test image
        arr = np.random.rand(10, 10, 10).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetOrigin([0, 0, 0])
        img.SetSpacing([1.0, 1.0, 1.0])
        
        # Save as MHA
        output_path = os.path.join(self.temp_dir, "test.mha")
        sitk.WriteImage(img, output_path)
        
        # Read back
        img_read = sitk.ReadImage(output_path)
        
        # Verify
        self.assertEqual(img.GetSize(), img_read.GetSize())
        self.assertEqual(img.GetSpacing(), img_read.GetSpacing())
    
    def test_nifti_format(self):
        """Test NIfTI format I/O."""
        # Create test image
        arr = np.ones((5, 5, 5), dtype=np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetOrigin([10, 20, 30])
        img.SetSpacing([2.0, 2.0, 2.0])
        
        # Save as NIfTI
        output_path = os.path.join(self.temp_dir, "test.nii.gz")
        sitk.WriteImage(img, output_path)
        
        # Read back
        img_read = sitk.ReadImage(output_path)
        
        # Verify metadata preserved
        self.assertEqual(img.GetSize(), img_read.GetSize())
        np.testing.assert_array_almost_equal(img.GetOrigin(), img_read.GetOrigin())
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved during format conversion."""
        # Create image with specific metadata
        img = sitk.Image([8, 8, 8], sitk.sitkFloat32)
        img.SetOrigin([5.5, 10.5, 15.5])
        img.SetSpacing([1.5, 2.5, 3.5])
        img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        # Save and reload
        path1 = os.path.join(self.temp_dir, "test.mha")
        sitk.WriteImage(img, path1)
        img_read = sitk.ReadImage(path1)
        
        # Check metadata
        np.testing.assert_array_almost_equal(img.GetOrigin(), img_read.GetOrigin())
        np.testing.assert_array_almost_equal(img.GetSpacing(), img_read.GetSpacing())
        np.testing.assert_array_almost_equal(img.GetDirection(), img_read.GetDirection())


class TestBoundsConsistency(unittest.TestCase):
    """Test bounds consistency between SITK and VTK."""
    
    def test_bounds_match_simple_case(self):
        """Test that SITK and VTK bounds match for simple image."""
        # Create SITK image
        sitk_img = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        sitk_img.SetOrigin([0, 0, 0])
        sitk_img.SetSpacing([1.0, 1.0, 1.0])
        sitk_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        # Create equivalent VTK image
        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(10, 10, 10)
        vtk_img.SetOrigin(0, 0, 0)
        vtk_img.SetSpacing(1.0, 1.0, 1.0)
        
        # Compare bounds
        bounds_sitk = _compute_bounds_sitk(sitk_img)
        bounds_vtk = _compute_bounds_vtk(vtk_img)
        
        ok, diffs = _compare_bounds(bounds_sitk, bounds_vtk)
        self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main()
