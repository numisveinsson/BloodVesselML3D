"""
Unit tests for preprocessing/change_img_scale_coords.py

Tests the scaling of image coordinates, spacing, and coordinate system transformations.
"""
import unittest
import tempfile
import os
import numpy as np
import SimpleITK as sitk
import sys

# Add preprocessing directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preprocessing'))
from change_img_scale_coords import change_img_scale


class TestChangeImageScale(unittest.TestCase):
    """Test the change_img_scale function."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test image
        arr = np.random.rand(10, 10, 10).astype(np.float32)
        self.test_img = sitk.GetImageFromArray(arr)
        self.test_img.SetOrigin([10, 20, 30])
        self.test_img.SetSpacing([1.0, 1.0, 1.0])
        self.test_img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        self.test_path = os.path.join(self.temp_dir, "test.mha")
        sitk.WriteImage(self.test_img, self.test_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_scale_spacing_up(self):
        """Test scaling spacing up by factor of 2."""
        result = change_img_scale(self.test_path, scale=2.0)
        
        # Spacing should be doubled
        expected_spacing = (2.0, 2.0, 2.0)
        self.assertAlmostEqual(result.GetSpacing()[0], expected_spacing[0])
        self.assertAlmostEqual(result.GetSpacing()[1], expected_spacing[1])
        self.assertAlmostEqual(result.GetSpacing()[2], expected_spacing[2])
        
        # Origin should remain unchanged
        self.assertEqual(result.GetOrigin(), self.test_img.GetOrigin())
    
    def test_scale_spacing_down(self):
        """Test scaling spacing down by factor of 0.5."""
        result = change_img_scale(self.test_path, scale=0.5)
        
        # Spacing should be halved
        expected_spacing = (0.5, 0.5, 0.5)
        self.assertAlmostEqual(result.GetSpacing()[0], expected_spacing[0])
        self.assertAlmostEqual(result.GetSpacing()[1], expected_spacing[1])
        self.assertAlmostEqual(result.GetSpacing()[2], expected_spacing[2])
    
    def test_no_scale_change(self):
        """Test with scale=1 (no change)."""
        result = change_img_scale(self.test_path, scale=1.0)
        
        # Everything should remain the same
        self.assertEqual(result.GetSpacing(), self.test_img.GetSpacing())
        self.assertEqual(result.GetOrigin(), self.test_img.GetOrigin())
        self.assertEqual(result.GetDirection(), self.test_img.GetDirection())
    
    def test_scale_origin(self):
        """Test scaling the origin."""
        result = change_img_scale(self.test_path, scale=1.0, scale_origin=2.0)
        
        # Origin should be doubled
        expected_origin = (20.0, 40.0, 60.0)
        self.assertAlmostEqual(result.GetOrigin()[0], expected_origin[0])
        self.assertAlmostEqual(result.GetOrigin()[1], expected_origin[1])
        self.assertAlmostEqual(result.GetOrigin()[2], expected_origin[2])
    
    def test_scale_both_spacing_and_origin(self):
        """Test scaling both spacing and origin."""
        result = change_img_scale(self.test_path, scale=3.0, scale_origin=0.5)
        
        # Check spacing
        expected_spacing = (3.0, 3.0, 3.0)
        self.assertAlmostEqual(result.GetSpacing()[0], expected_spacing[0])
        
        # Check origin
        expected_origin = (5.0, 10.0, 15.0)
        self.assertAlmostEqual(result.GetOrigin()[0], expected_origin[0])
    
    def test_set_direction_matrix(self):
        """Test setting a custom direction matrix."""
        # Identity matrix
        new_direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        result = change_img_scale(self.test_path, scale=1.0, direction_matrix=new_direction)
        
        self.assertEqual(result.GetDirection(), tuple(new_direction))
    
    def test_set_direction_matrix_rotated(self):
        """Test setting a rotated direction matrix."""
        # 90 degree rotation around z-axis
        new_direction = [0, -1, 0, 1, 0, 0, 0, 0, 1]
        result = change_img_scale(self.test_path, scale=1.0, direction_matrix=new_direction)
        
        result_dir = result.GetDirection()
        for i, expected in enumerate(new_direction):
            self.assertAlmostEqual(result_dir[i], expected)
    
    def test_invalid_direction_matrix(self):
        """Test that invalid direction matrix raises error."""
        with self.assertRaises(ValueError):
            change_img_scale(self.test_path, scale=1.0, direction_matrix=[1, 0, 0])
    
    def test_lps_to_ras_conversion(self):
        """Test converting from LPS to RAS coordinate system."""
        result = change_img_scale(self.test_path, scale=1.0, change_lps_to_ras=True)
        
        # First two components of origin should be negated
        expected_origin = (-10.0, -20.0, 30.0)
        self.assertAlmostEqual(result.GetOrigin()[0], expected_origin[0])
        self.assertAlmostEqual(result.GetOrigin()[1], expected_origin[1])
        self.assertAlmostEqual(result.GetOrigin()[2], expected_origin[2])
        
        # First two rows of direction should be negated
        direction = result.GetDirection()
        self.assertAlmostEqual(direction[0], -1.0)
        self.assertAlmostEqual(direction[4], -1.0)
        self.assertAlmostEqual(direction[8], 1.0)
    
    def test_custom_spacing_value(self):
        """Test setting spacing from a custom value."""
        custom_spacing = (2.5, 3.5, 4.5)
        result = change_img_scale(
            self.test_path, 
            scale=1.0, 
            if_spacing_file=True, 
            spacing_value=custom_spacing
        )
        
        self.assertAlmostEqual(result.GetSpacing()[0], custom_spacing[0])
        self.assertAlmostEqual(result.GetSpacing()[1], custom_spacing[1])
        self.assertAlmostEqual(result.GetSpacing()[2], custom_spacing[2])
    
    def test_verbose_mode(self):
        """Test that verbose mode doesn't crash."""
        # Just ensure verbose mode runs without error
        result = change_img_scale(self.test_path, scale=2.0, verbose=True)
        self.assertIsNotNone(result)
    
    def test_combined_transformations(self):
        """Test applying multiple transformations together."""
        new_direction = [0, -1, 0, 1, 0, 0, 0, 0, 1]
        result = change_img_scale(
            self.test_path, 
            scale=2.0,
            scale_origin=0.5,
            direction_matrix=new_direction,
            change_lps_to_ras=True
        )
        
        # Verify all transformations were applied
        self.assertAlmostEqual(result.GetSpacing()[0], 2.0)
        self.assertAlmostEqual(result.GetOrigin()[0], -5.0)  # scaled by 0.5 then negated
        self.assertIsNotNone(result)


class TestScalePreservesData(unittest.TestCase):
    """Test that scaling operations preserve image data."""
    
    def setUp(self):
        """Set up test image with known data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test image with specific values
        arr = np.arange(125).reshape(5, 5, 5).astype(np.float32)
        self.test_img = sitk.GetImageFromArray(arr)
        self.test_img.SetOrigin([0, 0, 0])
        self.test_img.SetSpacing([1.0, 1.0, 1.0])
        
        self.test_path = os.path.join(self.temp_dir, "test.mha")
        sitk.WriteImage(self.test_img, self.test_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_unchanged_after_spacing_scale(self):
        """Test that pixel data is unchanged after spacing scale."""
        result = change_img_scale(self.test_path, scale=5.0)
        
        # Get data arrays
        original_arr = sitk.GetArrayFromImage(self.test_img)
        result_arr = sitk.GetArrayFromImage(result)
        
        # Data should be identical
        np.testing.assert_array_equal(original_arr, result_arr)
    
    def test_data_unchanged_after_origin_scale(self):
        """Test that pixel data is unchanged after origin scale."""
        result = change_img_scale(self.test_path, scale=1.0, scale_origin=10.0)
        
        original_arr = sitk.GetArrayFromImage(self.test_img)
        result_arr = sitk.GetArrayFromImage(result)
        
        np.testing.assert_array_equal(original_arr, result_arr)
    
    def test_data_unchanged_after_lps_ras(self):
        """Test that pixel data is unchanged after LPS/RAS conversion."""
        result = change_img_scale(self.test_path, scale=1.0, change_lps_to_ras=True)
        
        original_arr = sitk.GetArrayFromImage(self.test_img)
        result_arr = sitk.GetArrayFromImage(result)
        
        np.testing.assert_array_equal(original_arr, result_arr)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def setUp(self):
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_very_small_scale(self):
        """Test with very small scale factor."""
        arr = np.ones((5, 5, 5), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([10.0, 10.0, 10.0])
        
        path = os.path.join(self.temp_dir, "test.mha")
        sitk.WriteImage(img, path)
        
        result = change_img_scale(path, scale=0.001)
        
        expected_spacing = (0.01, 0.01, 0.01)
        self.assertAlmostEqual(result.GetSpacing()[0], expected_spacing[0], places=5)
    
    def test_very_large_scale(self):
        """Test with very large scale factor."""
        arr = np.ones((5, 5, 5), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([0.1, 0.1, 0.1])
        
        path = os.path.join(self.temp_dir, "test.mha")
        sitk.WriteImage(img, path)
        
        result = change_img_scale(path, scale=1000.0)
        
        expected_spacing = (100.0, 100.0, 100.0)
        self.assertAlmostEqual(result.GetSpacing()[0], expected_spacing[0])
    
    def test_negative_origin(self):
        """Test with negative origin values."""
        arr = np.ones((5, 5, 5), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetOrigin([-50, -100, -150])
        
        path = os.path.join(self.temp_dir, "test.mha")
        sitk.WriteImage(img, path)
        
        result = change_img_scale(path, scale=1.0, scale_origin=2.0)
        
        expected_origin = (-100.0, -200.0, -300.0)
        self.assertAlmostEqual(result.GetOrigin()[0], expected_origin[0])


if __name__ == '__main__':
    unittest.main()
