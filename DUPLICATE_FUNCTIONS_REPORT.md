# Duplicate Functions Report - BloodVesselML3D Repository

## Critical Duplicates (Exact or Near-Exact Copies)

### 1. **VTK/SITK Conversion Functions**

#### `exportSitk2VTK` - 3 copies:
- `modules/vtk_functions.py:723` (canonical version)
- `global/create_seg_from_surf.py:163` (with minor float→int conversion addition)
- `global/create_surf_from_seg.py:186` (identical to modules version)

**Recommendation**: Keep only in `modules/vtk_functions.py`. Remove from global scripts and import instead.

#### `exportPython2VTK` - 3 copies:
- `modules/vtk_functions.py:788` (canonical)
- `global/create_seg_from_surf.py:254`
- `global/create_surf_from_seg.py:272`

**Recommendation**: Keep only in `modules/vtk_functions.py`. Remove from global scripts.

#### `build_transform_matrix` - 3 copies:
- `modules/vtk_functions.py:781` (canonical)
- `global/create_seg_from_surf.py:156`
- `global/create_surf_from_seg.py:178`

**Recommendation**: Keep only in `modules/vtk_functions.py`. Remove from global scripts.

---

### 2. **VTK Processing Functions**

#### `vtk_marching_cube` - 4 copies:
- `modules/vtk_functions.py:876` (uses `vtkMarchingCubes`)
- `global/create_seg_from_surf.py:235` (uses `vtkDiscreteMarchingCubes`)
- `global/create_surf_from_seg.py:253` (uses `vtkDiscreteMarchingCubes`)
- Note: Different implementations (Discrete vs regular MarchingCubes)

**Recommendation**: Keep both versions in `modules/vtk_functions.py` with clear names:
- `vtk_marching_cube` (existing)
- `vtk_discrete_marching_cube` (new name for discrete version)

#### `vtk_marching_cube_multi` - 3 copies:
- `global/create_surf_from_seg.py:16`
- `preprocessing/combine_segs.py:12` (still exists, not moved!)
- `cardiac/combine_segs.py:16` (just moved)

**Recommendation**: Keep only in `modules/vtk_functions.py` as it's a utility function.

#### `smooth_polydata` - 4 copies:
- `global/create_seg_from_surf.py:267`
- `global/create_surf_from_seg.py:285`
- `preprocessing/combine_segs.py:99` (still exists!)
- `cardiac/combine_segs.py:103`

**Recommendation**: Move to `modules/vtk_functions.py`

#### `decimation` - 2 copies:
- `global/create_seg_from_surf.py:291`
- `global/create_surf_from_seg.py:310`

**Recommendation**: Move to `modules/vtk_functions.py`

#### `appendPolyData` - 2 copies:
- `global/create_seg_from_surf.py:310`
- `global/create_surf_from_seg.py:329`

**Recommendation**: Move to `modules/vtk_functions.py`

#### `bound_polydata_by_image` - 4 copies:
- `modules/vtk_functions.py:690` (basic version)
- `global/create_seg_from_surf.py:326` (basic version)
- `global/create_surf_from_seg.py:345` (basic version)
- `preprocessing/combine_segs.py:530` (extended with name parameter - still exists!)
- `cardiac/combine_segs.py:534` (extended version)

**Recommendation**: Keep both versions in `modules/vtk_functions.py`:
- Basic version (existing)
- Extended version with name parameter

#### `convertPolyDataToImageData` - 4 copies:
- `global/create_seg_from_surf.py:341`
- `global/create_surf_from_seg.py:361`
- `preprocessing/combine_segs.py:124` (still exists!)
- `cardiac/combine_segs.py:128`

**Recommendation**: Move to `modules/vtk_functions.py`

#### `thresholdPolyData` - 2 copies:
- `preprocessing/combine_segs.py:71` (still exists!)
- `cardiac/combine_segs.py:75`

**Recommendation**: Move to `modules/vtk_functions.py`

---

### 3. **Image Processing Functions**

#### `eraseBoundary` - 3 copies:
- `global/create_seg_from_surf.py:69`
- `global/create_surf_from_seg.py:89`
- `pca_segs.py` (possibly)

**Recommendation**: Move to `modules/pre_process.py` or `modules/sitk_functions.py`

#### `vtkImageResample` - 2 copies:
- `global/create_seg_from_surf.py:205`
- `global/create_surf_from_seg.py:223`

**Recommendation**: Move to `modules/vtk_functions.py`

#### `surface_to_image` - 2 copies:
- `global/create_seg_from_surf.py:90`
- `global/create_surf_from_seg.py:110`

**Recommendation**: Move to `modules/vtk_functions.py`

#### `convert_seg_to_surfs` - 2 copies:
- `global/create_seg_from_surf.py:126`
- `global/create_surf_from_seg.py:146`

**Recommendation**: Consolidate into one version in `modules/vtk_functions.py`

---

### 4. **Resample Functions**

#### Multiple `resample` functions with different signatures:
- `modules/pre_process.py:11` - `resample(sitkIm, resolution, order, dim)` (canonical)
- `modules/pre_process.py:95` - `resample_spacing(sitkIm, resolution, dim, template_size, order)`
- `modules/pre_process.py:112` - `resample_scale(sitkIm, ref_img, scale_factor, order)`
- `modules/vascular_data.py:370` - `resample_image(vtk_im, min_)` (VTK specific)
- `preprocessing/change_img_resample.py:10` - `resample_image(img_sitk, target_size, target_spacing, order)` (new wrapper)
- `pca_segs.py:71` - `resample_to_size(img, new_size)`
- `modules/sampling_functions.py:690` - `resample_vol(removed_seg, resample_size)`

**Recommendation**: Keep all in `modules/pre_process.py` as they serve different purposes. The new wrapper in `change_img_resample.py` is fine as it's script-specific.

---

### 5. **I/O Functions**

#### No major duplicates found, but note:
- `modules/vtk_functions.py` has `read_geo`, `write_geo`, `read_img`, `write_img`
- `modules/sitk_functions.py` has `read_image`, `write_image`
- `modules/sampling_functions.py:1821` has a different `write_img` (should be renamed)

**Recommendation**: Rename `sampling_functions.write_img` to `write_subvolume_img` or similar.

---

## Important File Issue

**preprocessing/combine_segs.py still exists!** You moved it to cardiac but didn't delete the original. There are now two copies:
- `preprocessing/combine_segs.py` (old)
- `cardiac/combine_segs.py` (new)

---

## Recommended Actions

### Immediate:
1. **Delete** `preprocessing/combine_segs.py` (duplicate of cardiac version)
2. **Move common functions to modules/vtk_functions.py**:
   - `vtk_marching_cube_multi`
   - `smooth_polydata`
   - `decimation`
   - `appendPolyData`
   - `convertPolyDataToImageData`
   - `thresholdPolyData`
   - `vtkImageResample`
   - `surface_to_image`
   - `bound_polydata_by_image` (both versions)

3. **Update global scripts** to import from modules instead of duplicating:
   - `global/create_seg_from_surf.py`
   - `global/create_surf_from_seg.py`

4. **Move to modules/sitk_functions.py**:
   - `eraseBoundary`
   - `convert_seg_to_surfs`

### Clean up:
5. Rename `sampling_functions.write_img` to avoid confusion
6. Add clear docstrings to distinguish between similar functions
7. Create a `modules/surface_utils.py` for surface-specific operations if vtk_functions.py gets too large

### Testing:
8. Update tests to import from canonical locations
9. Run all tests to ensure imports work correctly

---

## Summary Statistics
- **Critical duplicates requiring immediate action**: ~15 functions
- **Files with most duplicates**: global/create_seg_from_surf.py, global/create_surf_from_seg.py
- **Estimated LOC savings**: ~500-700 lines
- **Maintenance improvement**: High (single source of truth for each function)
