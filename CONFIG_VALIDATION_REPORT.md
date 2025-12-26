# Configuration Files Validation Report

## Summary
Analyzed 13 YAML configuration files in the `config/` directory for parameter consistency.

## Parameter Consistency Issues

### Missing Parameters

#### 1. **WRITE_VOXEL_PYRAMID** 
- **Missing in:** `global_fewer_samples.yaml`, `global_original.yaml`
- **Present in:** All other configs
- **Action Required:** Add this parameter to ensure consistency

#### 2. **NUM_CROSS_SECTIONS**
- **Missing in:** `global.yaml`, `global_fewer_samples.yaml`, `global_original.yaml`, `global_more_samples.yaml`, `global_fewer_samples_savio.yaml`, `global_more_samples_savio.yaml`, `test.yaml`
- **Present in:** `trajectories.yaml`, `asoca_centerlines.yaml`, `global_more_samples_savio_gala.yaml`, `trajectories_savio.yaml`, `trajectories_savio_aortas.yaml`, `test2.yaml`
- **Action Required:** Either add to all configs or remove from some (depends on whether cross-sectional writing is intended)

#### 3. **RESAMPLE_CROSS_IMG**
- **Missing in:** Same files as NUM_CROSS_SECTIONS
- **Present in:** Same files as NUM_CROSS_SECTIONS  
- **Action Required:** Should be paired with NUM_CROSS_SECTIONS

#### 4. **WRITE_TRAJECTORIES**
- **Missing in:** `global.yaml`, `global_fewer_samples.yaml`, `global_original.yaml`, `global_more_samples.yaml`, `global_fewer_samples_savio.yaml`, `global_more_samples_savio.yaml`
- **Present in:** `trajectories.yaml`, `asoca_centerlines.yaml`, `test.yaml`, `global_more_samples_savio_gala.yaml`, `trajectories_savio.yaml`, `trajectories_savio_aortas.yaml`, `test2.yaml`
- **Action Required:** Add to all configs for consistency

#### 5. **N_SLICES**
- **Missing in:** Same 6 files as WRITE_TRAJECTORIES
- **Present in:** Same 7 files as WRITE_TRAJECTORIES
- **Action Required:** Should be paired with WRITE_TRAJECTORIES

### Complete Parameter Set
Based on analysis, a complete config should have these 43 parameters:

1. `DATA_DIR` ✓ (all configs)
2. `DATASET_NAME` ✓ (all configs)
3. `TESTING` ✓ (all configs)
4. `MODALITY` ✓ (all configs)
5. `IMG_EXT` ✓ (all configs)
6. `BINARIZE` ✓ (all configs)
7. `SCALED` ✓ (all configs)
8. `CROPPED` ✓ (all configs)
9. `ANATOMY` ✓ (all configs)
10. `OUTLET_CLASSES` ✓ (all configs)
11. `VALIDATION_PROP` ✓ (all configs)
12. `EXTRACT_VOLUMES` ✓ (all configs)
13. `ROTATE_VOLUMES` ✓ (all configs)
14. `RESAMPLE_VOLUMES` ✓ (all configs)
15. `RESAMPLE_SIZE` ✓ (all configs)
16. `AUGMENT_VOLUMES` ✓ (all configs)
17. `WRITE_SAMPLES` ✓ (all configs)
18. `WRITE_IMG` ✓ (all configs)
19. `REMOVE_OTHER` ✓ (all configs)
20. `WRITE_SURFACE` ✓ (all configs)
21. `WRITE_CENTERLINE` ✓ (all configs)
22. `WRITE_VOXEL_PYRAMID` ⚠️ (missing in 2 configs)
23. `WRITE_DISCRETE_CENTERLINE` ✓ (all configs)
24. `DISCRETE_CENTERLINE_N_POINTS` ✓ (all configs)
25. `WRITE_OUTLET_STATS` ✓ (all configs)
26. `WRITE_OUTLET_IMG` ✓ (all configs)
27. `UPSAMPLE_OUTLET_IMG` ✓ (all configs)
28. `WRITE_CROSS_SECTIONAL` ✓ (all configs)
29. `NUM_CROSS_SECTIONS` ⚠️ (missing in 7 configs)
30. `RESAMPLE_CROSS_IMG` ⚠️ (missing in 7 configs)
31. `WRITE_TRAJECTORIES` ⚠️ (missing in 6 configs)
32. `N_SLICES` ⚠️ (missing in 6 configs)
33. `WRITE_VTK` ✓ (all configs)
34. `WRITE_VTK_THROWOUT` ✓ (all configs)
35. `MOVE_DIST` ✓ (all configs)
36. `SIZE_RADIUS` ✓ (all configs)
37. `RADIUS_ADD` ✓ (all configs)
38. `RADIUS_SCALE` ✓ (all configs)
39. `CAPFREE` ✓ (all configs)
40. `CAPFREE_PROP` ✓ (all configs)
41. `MOVE_SLOWER_LARGE` ✓ (all configs)
42. `MU_SIZE` ✓ (all configs)
43. `SIGMA_SIZE` ✓ (all configs)
44. `MU_SHIFT` ✓ (all configs)
45. `SIGMA_SHIFT` ✓ (all configs)
46. `NUMBER_SAMPLES` ✓ (all configs)
47. `NUMBER_SAMPLES_BIFURC` ✓ (all configs)
48. `NUMBER_SAMPLES_START` ✓ (all configs)
49. `MOVE_SLOWER_BIFURC` ✓ (all configs)
50. `MAX_SAMPLES` ✓ (all configs)
51. `TEST_CASES` ✓ (all configs)

## Detailed File Analysis

### Files Missing Parameters:

#### `global.yaml`
- Missing: `WRITE_VOXEL_PYRAMID`, `NUM_CROSS_SECTIONS`, `RESAMPLE_CROSS_IMG`, `WRITE_TRAJECTORIES`, `N_SLICES`

#### `global_fewer_samples.yaml`
- Missing: `WRITE_VOXEL_PYRAMID`, `NUM_CROSS_SECTIONS`, `RESAMPLE_CROSS_IMG`, `WRITE_TRAJECTORIES`, `N_SLICES`

#### `global_original.yaml`
- Missing: `WRITE_VOXEL_PYRAMID`, `NUM_CROSS_SECTIONS`, `RESAMPLE_CROSS_IMG`, `WRITE_TRAJECTORIES`, `N_SLICES`

#### `global_more_samples.yaml`
- Missing: `NUM_CROSS_SECTIONS`, `RESAMPLE_CROSS_IMG`, `WRITE_TRAJECTORIES`, `N_SLICES`

#### `global_fewer_samples_savio.yaml`
- Missing: `NUM_CROSS_SECTIONS`, `RESAMPLE_CROSS_IMG`, `WRITE_TRAJECTORIES`, `N_SLICES`

#### `global_more_samples_savio.yaml`
- Missing: `NUM_CROSS_SECTIONS`, `RESAMPLE_CROSS_IMG`, `WRITE_TRAJECTORIES`, `N_SLICES`

#### `test.yaml`
- Missing: `NUM_CROSS_SECTIONS`, `RESAMPLE_CROSS_IMG`, `N_SLICES`
- Note: Has `WRITE_CROSS_SECTIONAL: True` and `WRITE_TRAJECTORIES: True` but missing related parameters

#### `test2.yaml`
- Missing: None! This is the most complete config

## Recommendations

### Option 1: Add Missing Parameters (Recommended)
Add the missing parameters to all config files with sensible defaults:
- `WRITE_VOXEL_PYRAMID: False`
- `NUM_CROSS_SECTIONS: 2`
- `RESAMPLE_CROSS_IMG: 400`
- `WRITE_TRAJECTORIES: False`
- `N_SLICES: 20`

### Option 2: Make Parameters Optional
Ensure the code that reads these configs handles missing parameters gracefully with defaults.

### Priority Actions

1. **HIGH PRIORITY:** Add `WRITE_VOXEL_PYRAMID` to:
   - `global_fewer_samples.yaml`
   - `global_original.yaml`

2. **MEDIUM PRIORITY:** Add trajectory-related parameters to the 6 global_*.yaml configs:
   - `NUM_CROSS_SECTIONS`
   - `RESAMPLE_CROSS_IMG`
   - `WRITE_TRAJECTORIES`
   - `N_SLICES`

3. **LOW PRIORITY:** Add `NUM_CROSS_SECTIONS`, `RESAMPLE_CROSS_IMG`, `N_SLICES` to:
   - `test.yaml` (already has `WRITE_TRAJECTORIES: True`)

## Config File Groups

### Group 1: Global Configs (Production)
- `global.yaml`
- `global_fewer_samples.yaml`
- `global_original.yaml`
- `global_more_samples.yaml`

### Group 2: Savio Cluster Configs
- `global_fewer_samples_savio.yaml`
- `global_more_samples_savio.yaml`
- `global_more_samples_savio_gala.yaml` (most complete in this group)

### Group 3: Trajectory Configs
- `trajectories.yaml`
- `trajectories_savio.yaml`
- `trajectories_savio_aortas.yaml`
- `asoca_centerlines.yaml` (technically a trajectory config based on parameters)

### Group 4: Test Configs
- `test.yaml`
- `test2.yaml` (most complete overall)

## Implementation Note
The most complete and consistent config file is **`test2.yaml`** with all parameters present. Consider using it as a template for updating other configs.
