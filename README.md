# Blood Vessel Modeling Data Pre and Post Processing

This repository contains code to process data used to train machine learning methods for geometric modeling of blood vessels using medical image data. The data used is:
    1. Medical image scans
    2. Ground truth segmentations
    3. Centerlines
    4. Surface meshes (if applicable)

The fundamental idea is to 'piece up' vasculature into hundreds/thousands of vascular segments. These segments can be:
    1. Image subvolumes/patches (3D/2D)
    2. Local surface representations
    3. Local centerline segments
    4. Local outlet/bifurcation/size/orientation information
    etc.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Main Scripts](#main-scripts)
- [Preprocessing Scripts](#preprocessing-scripts)
- [Global Processing Scripts](#global-processing-scripts)
- [Cardiac Processing Scripts](#cardiac-processing-scripts)
- [Modules](#modules)
- [Environment Variables](#environment-variables)
- [Testing](#testing)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate bloodvesselml3d
```

## Project Structure

```
BloodVesselML3D/
├── config/              # Configuration YAML files
├── modules/             # Shared utility modules
│   ├── vtk_functions.py # VTK-related functions
│   ├── sitk_functions.py # SimpleITK-related functions
│   └── ...
├── preprocessing/       # Image format conversion and preprocessing
├── global/             # Global processing scripts
├── cardiac/            # Cardiac-specific processing
├── tests/              # Unit tests
└── ...
```

## Configuration

The main configuration file is `config/global.yaml`. This file contains settings for:
- `DATA_DIR`: Directory where input data is stored
- `OUT_DIR`: Directory where output will be written
- `DATASET_NAME`: Name of the dataset
- `MODALITY`: Imaging modality ('CT', 'MR', or both)
- `IMG_EXT`: Image file extension
- Various processing flags (see below)

### Key Configuration Parameters

- `DATA_DIR`: Input data directory path
- `TESTING`: Boolean flag for testing mode
- `MODALITY`: Imaging modality ('CT', 'MR', or ['CT','MR'])
- `IMG_EXT`: File extension for images (e.g., '.mha', '.vti')
- `ANATOMY`: Type of anatomy ('ALL' or specific list)
- `EXTRACT_VOLUMES`: Extract volumes from images
- `ROTATE_VOLUMES`: Rotate volumes during processing
- `RESAMPLE_VOLUMES`: Resample volumes
- `RESAMPLE_SIZE`: Target size for resampling
- `WRITE_SAMPLES`: Write sample data
- `WRITE_IMG`: Write image files
- `WRITE_SURFACE`: Write surface meshes
- `WRITE_CENTERLINE`: Write centerline data
- `WRITE_CROSS_SECTIONAL`: Write cross-sectional data
- `NUM_CROSS_SECTIONS`: Number of cross-sections
- `RESAMPLE_CROSS_IMG`: Cross-sectional image resampling size
- `WRITE_TRAJECTORIES`: Write trajectory data
- `N_SLICES`: Number of slices for processing

## Main Scripts

### gather_sampling_data_parallel.py

Main script for parallel processing of data sampling from multiple cases. Uses multiprocessing to speed up processing.

**Usage:**
```bash
python3 gather_sampling_data_parallel.py \
    --config_name global \
    --outdir ./extracted_data/ \
    --num_cores 4 \
    --perc_dataset 1.0
```

**Arguments:**
- `--config_name` / `-config_name`: Name of configuration file (required, without .yaml extension)
  - Example: `--config_name global` uses `config/global.yaml`
- `--outdir` / `-outdir`: Output directory for extracted data (default: `./extracted_data/`)
- `--perc_dataset` / `-perc_dataset`: Percentage of dataset to use, 0.0 to 1.0 (default: `1.0`)
- `--num_cores` / `-num_cores`: Number of CPU cores to use for parallel processing (default: `1`)
- `--start_from` / `-start_from`: Start processing from case number (default: `0`)
- `--end_at` / `-end_at`: End processing at case number, -1 for all cases (default: `-1`)

**Examples:**
```bash
# Process all cases with 4 cores
python3 gather_sampling_data_parallel.py --config_name global --num_cores 4

# Process 50% of dataset with 8 cores
python3 gather_sampling_data_parallel.py --config_name global --perc_dataset 0.5 --num_cores 8

# Process cases 10 to 20
python3 gather_sampling_data_parallel.py --config_name global --start_from 10 --end_at 20

# Custom output directory
python3 gather_sampling_data_parallel.py --config_name global --outdir /path/to/output
```

The script uses configuration from the specified YAML file in `config/` and processes cases in the dataset according to the provided arguments.

## Preprocessing Scripts

All preprocessing scripts support command-line arguments with environment variable fallbacks.

### change_img_format.py

Convert images between different formats (.mha, .vti, .nrrd, .nii.gz, etc.).

**Usage:**
```bash
python preprocessing/change_img_format.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --input_format .nrrd \
    --output_format .mha \
    --label_if_string seg
```

**Arguments:**
- `--input_dir`: Input directory (or `INPUT_DIR` env var)
- `--output_dir`: Output directory (or `OUTPUT_DIR` env var)
- `--input_format`: Input file extension (default: `.nrrd`)
- `--output_format`: Output file extension (default: `.mha`)
- `--label`: Treat all files as label segmentations
- `--label_if_string`: Auto-detect labels by filename string
- `--rem_str`: String to remove from filenames
- `--surface`: Also create surface meshes for label images

### change_img_resample.py

Resample images to target size or spacing.

**Usage:**
```bash
# Resample to target spacing
python preprocessing/change_img_resample.py \
    --input_dir /path/to/images \
    --target_spacing 0.03 0.03 0.03 \
    --order 1

# Resample to target size
python preprocessing/change_img_resample.py \
    --input_dir /path/to/images \
    --target_size 512 512 512
```

**Arguments:**
- `--input_dir`: Input directory
- `--output_dir`: Output directory
- `--target_size`: Target size [x, y, z] (mutually exclusive with `--target_spacing`)
- `--target_spacing`: Target spacing [x, y, z] in mm
- `--order`: Interpolation order (0=nearest, 1=linear, 2=bspline)
- `--testing_samples`: Filter to specific sample names
- `--no_skip_existing`: Re-process existing files

### change_img_scale_coords.py

Scale image spacing/origin and transform coordinate systems.

**Usage:**
```bash
python preprocessing/change_img_scale_coords.py \
    --input_dir /path/to/images \
    --scale 0.1 \
    --change_lps_to_ras \
    --verbose
```

**Arguments:**
- `--input_dir`: Input directory
- `--output_dir`: Output directory
- `--scale`: Scale factor for spacing (default: 1.0)
- `--scale_origin`: Scale factor for origin
- `--spacing_file`: CSV file with spacing values
- `--direction_matrix`: 3x3 direction matrix (9 values)
- `--flip`: Flip image
- `--flip_axis`: Axis to flip [x, y, z]
- `--permute`: Permute axes
- `--change_lps_to_ras`: Convert LPS to RAS
- `--filter_names`: Filter files by name
- `--verbose`: Detailed output

### change_vtk_scale_coords.py

Scale VTK surface files (.vtp, .stl).

**Usage:**
```bash
python preprocessing/change_vtk_scale_coords.py \
    --input_dir /path/to/surfaces \
    --output_dir /path/to/output \
    --scale_factor 0.1
```

**Arguments:**
- `--input_dir`: Input directory
- `--output_dir`: Output directory
- `--scale_factor`: Scale factor (required)

### change_lps_ras.py

Convert files from LPS to RAS coordinate system.

**Usage:**
```bash
# Single file
python preprocessing/change_lps_ras.py \
    --input_file /path/to/file.vtp \
    --output_file /path/to/output.vtp \
    --file_type vtp

# Batch processing
python preprocessing/change_lps_ras.py \
    --input_dir /path/to/surfaces \
    --output_dir /path/to/output \
    --file_type vtp
```

**Arguments:**
- `--input_file` / `--input_dir`: Input file or directory
- `--output_file` / `--output_dir`: Output file or directory
- `--file_type`: Type of file (`vtp` or `image`)
- `--transform_type`: `lps_to_ras` or `permute_xyz`

### compare_imgs.py

Compare image properties and optionally transform images.

**Usage:**
```bash
python preprocessing/compare_imgs.py \
    --image1 /path/to/img1.mha \
    --image2 /path/to/img2.mha \
    --image3 /path/to/img3.mha
```

**Arguments:**
- `--image1`, `--image2`, `--image3`: Image files to compare
- `--vti_file`: VTI file to transform
- `--output_file`: Output path for transformed file
- `--transform`: Apply rotation transform

## Global Processing Scripts

### create_seg_from_surf.py

Create segmentation images from surface meshes.

**Usage:**
```bash
python global/create_seg_from_surf.py \
    --surfaces_dir /path/to/surfaces \
    --images_dir /path/to/images \
    --output_dir /path/to/output
```

**Arguments:**
- `--surfaces_dir`: Directory with surface mesh files (.vtp or .stl)
- `--images_dir`: Directory with image files
- `--output_dir`: Directory to write output segmentations
- `--img_ext`: Input image file extension (default: `.mha`)
- `--output_ext`: Output file extension (default: `.mha`)

### create_surf_from_seg.py

Create surface meshes from segmentation images.

**Usage:**
```bash
python global/create_surf_from_seg.py \
    --segmentations_dir /path/to/segmentations \
    --output_dir /path/to/output \
    --smooth \
    --keep_largest
```

**Arguments:**
- `--segmentations_dir`: Directory with segmentation images
- `--output_dir`: Directory to write output surfaces
- `--spacing_file`: CSV file with spacing values (optional)
- `--filter_string`: Filter images by string (optional)
- `--smooth`: Apply smoothing to surfaces
- `--keep_largest`: Keep only largest connected component
- `--img_ext`: Input image extension (default: `.mha`)
- `--output_ext`: Output surface extension (default: `.vtp`)

## Cardiac Processing Scripts

### combine_segs.py

Combine cardiac and vascular segmentations.

**Usage:**
```bash
python cardiac/combine_segs.py \
    --meshes_dir /path/to/meshes \
    --images_dir /path/to/images \
    --vascular_dir /path/to/vascular_segs
```

**Arguments:**
- `--meshes_dir`: Directory with cardiac mesh files (.vtp)
- `--images_dir`: Directory with image files (.vti)
- `--vascular_dir`: Directory with vascular segmentation files (.vti)
- `--write_all` / `--no-write-all`: Control writing intermediate files
- `--no_valve` / `--with-valve`: Control valve processing

## Modules

The repository uses a modular structure with shared functions:

### modules/vtk_functions.py

Central repository for VTK-related utility functions including:
- `vtk_marching_cube`, `vtk_marching_cube_multi`
- `smooth_polydata`, `decimation`
- `exportSitk2VTK`, `exportPython2VTK`
- `convertPolyDataToImageData`
- `vtkImageResample`
- And many more...

### modules/sitk_functions.py

SimpleITK-related utility functions including:
- `eraseBoundary`
- `convert_seg_to_surfs`
- `read_image`, `write_image`
- `sitk_to_numpy`, `numpy_to_sitk`
- And more...

## Environment Variables

All scripts support environment variables as fallbacks when command-line arguments are not provided. Priority order:

1. **Command-line arguments** (highest priority)
2. **Environment variables** (fallback)
3. **Default values** (last resort)

**Common Environment Variables:**
- `DATA_DIR`: Base data directory
- `INPUT_DIR`: Input directory for preprocessing
- `OUTPUT_DIR`: Output directory
- `CARDIAC_MESHES_DIR`: Cardiac meshes directory
- `CARDIAC_IMAGES_DIR`: Cardiac images directory
- `VASCULAR_SEGS_DIR`: Vascular segmentations directory
- `SURFACES_DIR`: Surfaces directory
- `IMAGES_DIR`: Images directory
- `SEGMENTATIONS_DIR`: Segmentations directory
- `SPACING_FILE`: Path to spacing CSV file
- `FILTER_STRING`: Filter string for file processing

**Example:**
```bash
export DATA_DIR=/path/to/data
export INPUT_DIR=/path/to/input
python preprocessing/change_img_format.py --input_format .nrrd --output_format .mha
```

## Data Structure

The code expects data to be stored in a particular folder structure:

```
DATA_DIR/
├── images/
│   └── case0.mha
├── centerlines/
│   └── case0.vtp
├── truths/
│   └── case0.mha
└── surfaces/  (if applicable)
    └── case0.vtp
```

## Testing

Run tests with:

```bash
python -m pytest tests/
```

Or run specific tests:

```bash
python -m pytest tests/test_create_seg_from_surf.py
python -m pytest tests/test_create_surf_from_seg.py
```

## Additional Information

### Code Refactoring

The codebase has been refactored to:
- Consolidate duplicate functions into shared modules
- Remove hardcoded paths (now use command-line arguments or environment variables)
- Replace print statements with proper logging
- Remove debug code (`pdb.set_trace()`)
- Ensure consistent configuration files

### Help Documentation

All scripts include comprehensive help documentation:

```bash
python <script_name>.py --help
```

This will display:
- Available arguments
- Default values
- Usage examples
- Argument descriptions

## License

See LICENSE file for details.

## Contributing

When contributing:
1. Use the centralized modules for shared functionality
2. Use command-line arguments instead of hardcoded paths
3. Use the logger module instead of print statements
4. Update tests when adding new functionality
5. Follow the existing code style
