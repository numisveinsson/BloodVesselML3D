# Additional Repository Recommendations

This document provides recommendations for improving code quality, maintainability, and best practices beyond the duplicate functions cleanup.

## 🔴 High Priority Issues

### 1. Remove Debug Code from Production
**Issue**: Found 34 instances of `pdb.set_trace()` and debug breakpoints in production code.

**Files affected**:
- `modules/vtk_functions.py` (2 instances)
- `global/create_seg_from_surf.py` (1 instance)
- `cardiac/combine_segs.py` (1 instance)
- `visualization/view_data.py` (2 instances)
- `tests/test_discr_cent.py` (2 instances)
- And many more...

**Recommendation**:
- Remove all `pdb.set_trace()` calls from production code
- Use proper logging or conditional debugging flags
- Consider using `logging.debug()` with appropriate log levels

**Action**: Create a script to find and remove all `pdb.set_trace()` calls:
```bash
grep -r "pdb.set_trace" --include="*.py" . | grep -v "__pycache__" | grep -v "test"
```

### 2. Replace Hardcoded Paths with Configuration
**Issue**: Found hardcoded user-specific paths in multiple files:
- `/Users/numisveins/...` in `cardiac/combine_segs.py`
- `/Users/nsveinsson/...` in `global/create_seg_from_surf.py`, `global/create_surf_from_seg.py`
- Hardcoded paths in config files

**Files affected**:
- `cardiac/combine_segs.py` (lines 448-468)
- `global/create_seg_from_surf.py` (lines 91-94)
- `global/create_surf_from_seg.py` (lines 79, 86)
- `config/global_fewer_samples.yaml` (line 1)

**Recommendation**:
- Move all paths to configuration files
- Use environment variables for user-specific paths
- Add path validation and error messages if paths don't exist
- Document required directory structure in README

**Example fix**:
```python
# Instead of:
directory = '/Users/numisveins/Documents/data/'

# Use:
directory = os.getenv('DATA_DIR', global_config.get('DATA_DIR', './data/'))
if not os.path.exists(directory):
    raise ValueError(f"Data directory not found: {directory}")
```

### 3. Replace Print Statements with Logging
**Issue**: Found 509 `print()` statements across 40 files. This makes debugging difficult and doesn't allow for log levels.

**Recommendation**:
- Implement proper logging using Python's `logging` module
- Create a centralized logging configuration
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Allow log output to be configured (console, file, both)

**Action**: Create `modules/logger.py`:
```python
import logging
import sys

def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
```

### 4. Fix Configuration Inconsistencies
**Issue**: Already documented in `CONFIG_VALIDATION_REPORT.md`, but needs action.

**Missing parameters in multiple config files**:
- `WRITE_VOXEL_PYRAMID` (missing in 2 files)
- `NUM_CROSS_SECTIONS` (missing in 7 files)
- `RESAMPLE_CROSS_IMG` (missing in 7 files)
- `WRITE_TRAJECTORIES` (missing in 6 files)
- `N_SLICES` (missing in 6 files)

**Recommendation**:
- Add missing parameters to all config files with sensible defaults
- Create a config validation script that runs before processing
- Document all parameters in README or separate CONFIG.md

## 🟡 Medium Priority Issues

### 5. Improve Error Handling
**Issue**: Many file operations and data processing steps lack proper error handling.

**Recommendation**:
- Add try-except blocks around file I/O operations
- Provide meaningful error messages
- Log errors appropriately
- Consider using context managers for file operations

**Example**:
```python
# Instead of:
img = sitk.ReadImage(file_path)

# Use:
try:
    img = sitk.ReadImage(file_path)
except Exception as e:
    logger.error(f"Failed to read image {file_path}: {e}")
    raise
```

### 6. Complete Requirements.txt
**Issue**: `requirements.txt` only has 4 packages, but the codebase uses many more:
- SimpleITK (used extensively)
- numpy (used extensively)
- yaml (for config files)
- pandas (used in some scripts)
- matplotlib (used in visualization)
- scikit-learn (used in pca_segs.py)

**Recommendation**:
- Add all dependencies with version numbers
- Consider using `environment.yml` (already exists) as the primary dependency file
- Add a `setup.py` or `pyproject.toml` for proper package management
- Document installation instructions in README

**Suggested requirements.txt**:
```
numpy>=1.20.0
scipy>=1.7.0
SimpleITK>=2.0.0
vtk==8.1.1
scikit-image>=0.18.0
PyYAML>=5.4.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
imageio>=2.9.0
```

### 7. Add Type Hints
**Issue**: Most functions lack type hints, making code harder to understand and maintain.

**Recommendation**:
- Add type hints to function signatures
- Use `typing` module for complex types
- Consider using `mypy` for type checking

**Example**:
```python
# Instead of:
def exportSitk2VTK(sitkIm, spacing=None):

# Use:
from typing import Optional, Tuple
import SimpleITK as sitk
import vtk

def exportSitk2VTK(
    sitkIm: sitk.Image, 
    spacing: Optional[Tuple[float, float, float]] = None
) -> Tuple[vtk.vtkImageData, vtk.vtkMatrix4x4]:
```

### 8. Improve Documentation
**Issue**: 
- README is basic and doesn't cover all use cases
- Many functions lack docstrings
- No API documentation
- Configuration parameters not fully documented

**Recommendation**:
- Expand README with:
  - Installation instructions
  - Quick start guide
  - Configuration guide
  - Examples for common workflows
  - Troubleshooting section
- Add docstrings to all public functions following Google or NumPy style
- Create `docs/` directory with:
  - API reference
  - Configuration guide
  - Contributing guidelines
- Add docstring examples for complex functions

### 9. Code Organization
**Issue**: Some files in root directory could be better organized:
- `old_extract_3d_data.py` (should be in `archive/` or removed)
- `augmentation_example.py` (could be in `examples/`)
- `pca_segs.py` (could be in `analysis/` or `preprocessing/`)

**Recommendation**:
- Create `archive/` for deprecated code
- Create `examples/` for example scripts
- Move analysis scripts to `analysis/` directory
- Consider creating a `scripts/` directory for standalone utility scripts

### 10. Add Unit Tests
**Issue**: Limited test coverage. Only 9 test files, mostly for specific functions.

**Recommendation**:
- Add tests for core modules:
  - `modules/vtk_functions.py` (test key functions)
  - `modules/sitk_functions.py` (test key functions)
  - `modules/sampling_functions.py` (test key functions)
- Add integration tests for main workflows
- Set up CI/CD to run tests automatically
- Aim for at least 60% code coverage

## 🟢 Low Priority / Nice to Have

### 11. Add Pre-commit Hooks
**Recommendation**:
- Set up pre-commit hooks for:
  - Code formatting (black, autopep8)
  - Linting (pylint, flake8)
  - Type checking (mypy)
  - Remove trailing whitespace
  - Check for debug code

### 12. Add Code Formatting
**Recommendation**:
- Standardize on a code formatter (black or autopep8)
- Add `.editorconfig` or `pyproject.toml` with formatting rules
- Run formatter on entire codebase

### 13. Add CI/CD Pipeline
**Recommendation**:
- Set up GitHub Actions or similar for:
  - Running tests
  - Linting
  - Type checking
  - Building documentation

### 14. Performance Improvements
**Recommendation**:
- Profile code to identify bottlenecks
- Consider parallelizing more operations
- Cache expensive computations where appropriate
- Optimize VTK operations if possible

### 15. Add Data Validation
**Recommendation**:
- Add validation for input data formats
- Check file existence before processing
- Validate configuration parameters
- Add data integrity checks

## Summary Statistics

- **Debug code**: 34 instances of `pdb.set_trace()`
- **Print statements**: 509 across 40 files
- **Hardcoded paths**: 11 instances
- **Missing dependencies**: ~6-8 packages not in requirements.txt
- **Test files**: 9 (limited coverage)
- **Config inconsistencies**: 5 missing parameters across multiple files

## Priority Action Plan

1. **Week 1**: Remove debug code, fix hardcoded paths
2. **Week 2**: Implement logging, update requirements.txt
3. **Week 3**: Fix config inconsistencies, improve error handling
4. **Week 4**: Add type hints to core modules, improve documentation
5. **Ongoing**: Add tests, improve code organization

## Tools to Help

- **black**: Code formatting
- **pylint/flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks
- **sphinx**: Documentation generation

