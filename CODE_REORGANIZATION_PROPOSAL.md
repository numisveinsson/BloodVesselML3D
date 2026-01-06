# Code Reorganization Proposal

## Current Issues

The main script `gather_sampling_data_parallel.py` has several organizational problems:

1. **God Function**: The `sample_case` function is 400+ lines and handles too many responsibilities
2. **Mixed Concerns**: Data loading, processing, writing, and orchestration are all intertwined
3. **Hard to Test**: Large monolithic functions with many dependencies
4. **Hard to Maintain**: Changes require understanding the entire function
5. **Poor Separation**: No clear boundaries between different processing stages

## Proposed Structure

### New Directory Organization

```
sampling/
├── __init__.py
├── pipeline.py              # Main pipeline orchestration
├── case_processor.py        # Case-level processing logic
├── sample_processor.py      # Sample-level processing logic
├── data_loader.py           # Data loading and preparation
├── writers.py               # All writing operations
├── config.py                # Configuration handling
└── processors/              # Specialized processors
    ├── __init__.py
    ├── surface_processor.py
    ├── centerline_processor.py
    ├── outlet_processor.py
    └── trajectory_processor.py
```

### Key Design Principles

1. **Single Responsibility**: Each module/class has one clear purpose
2. **Dependency Injection**: Pass dependencies explicitly rather than accessing globals
3. **Composition over Inheritance**: Build complex behavior from simple components
4. **Testability**: Small, focused functions that are easy to test
5. **Clear Data Flow**: Explicit data structures passed between stages

### Module Responsibilities

#### 1. `pipeline.py` - Main Orchestration
- Entry point and argument parsing
- Multiprocessing setup and coordination
- High-level workflow: load config → process modalities → collect results
- **Should be**: ~150-200 lines

#### 2. `case_processor.py` - Case-Level Processing
- Handles processing of a single case
- Coordinates data loading, centerline iteration, and sample extraction
- Manages case-level state (counters, tracking)
- **Key methods**:
  - `process_case()` - Main entry point for case processing
  - `_setup_case_data()` - Load and prepare case data
  - `_process_centerlines()` - Iterate over centerlines
  - `_finalize_case()` - Write case results and cleanup
- **Should be**: ~200-250 lines

#### 3. `sample_processor.py` - Sample-Level Processing
- Processes individual samples from centerline points
- Handles volume extraction, rotation, and transformations
- Coordinates sample-level feature extraction
- **Key methods**:
  - `process_sample()` - Main sample processing
  - `_extract_volume()` - Extract subvolume
  - `_process_sample_features()` - Extract features (surface, centerline, outlets, etc.)
  - `_prepare_sample_output()` - Prepare data for writing
- **Should be**: ~200-250 lines

#### 4. `data_loader.py` - Data Loading
- Handles all data loading operations
- Provides clean interfaces for loading images, segmentations, centerlines, surfaces
- Manages data caching if needed
- **Key classes/functions**:
  - `CaseDataLoader` - Loads all data for a case
  - `load_image_data()` - Load image and segmentation
  - `load_centerline()` - Load and process centerline
  - `load_surface()` - Load surface mesh
- **Should be**: ~150-200 lines

#### 5. `writers.py` - Output Writing
- All file writing operations
- Organized by output type
- **Key classes/functions**:
  - `SampleWriter` - Writes sample outputs (images, VTK, etc.)
  - `StatsWriter` - Writes CSV/statistics
  - `TrajectoryWriter` - Writes trajectory data
  - `SurfaceWriter` - Writes surface outputs
  - `CenterlineWriter` - Writes centerline outputs
- **Should be**: ~200-300 lines (can be split further if needed)

#### 6. `processors/` - Specialized Processors
- Focused processors for specific feature types
- **surface_processor.py**: Surface extraction and processing
- **centerline_processor.py**: Centerline extraction and discretization
- **outlet_processor.py**: Outlet detection and statistics
- **trajectory_processor.py**: Trajectory calculation and projection

#### 7. `config.py` - Configuration Management
- Configuration loading and validation
- Provides configuration object with typed access
- **Key classes**:
  - `SamplingConfig` - Configuration container class

### Data Structures

Introduce clear data structures to pass between stages:

```python
@dataclass
class CaseData:
    """Container for all case-level data"""
    name: str
    image: sitk.Image
    segmentation: sitk.Image
    centerline: vtkPolyData
    surface: Optional[vtkPolyData] = None
    case_dict: Dict = None

@dataclass
class SampleData:
    """Container for sample-level data"""
    name: str
    image: sitk.Image
    segmentation: sitk.Image
    stats: Dict
    center: np.ndarray
    radius: float
    # ... other sample properties

@dataclass
class ProcessingState:
    """Track processing state across samples"""
    sample_count: int = 0
    throwout_count: int = 0
    error_count: int = 0
    # ... other counters
```

### Benefits of This Structure

1. **Maintainability**: Each module has a clear, focused purpose
2. **Testability**: Small functions are easy to unit test
3. **Reusability**: Components can be reused in different contexts
4. **Readability**: Clear data flow and separation of concerns
5. **Extensibility**: Easy to add new processors or output types
6. **Debugging**: Easier to isolate and fix issues
7. **Parallelization**: Clear boundaries make parallelization safer

### Migration Strategy

1. **Phase 1**: Create new structure alongside existing code
2. **Phase 2**: Move data loading logic to `data_loader.py`
3. **Phase 3**: Extract sample processing to `sample_processor.py`
4. **Phase 4**: Extract case processing to `case_processor.py`
5. **Phase 5**: Move writing logic to `writers.py`
6. **Phase 6**: Refactor `pipeline.py` to use new modules
7. **Phase 7**: Add tests and remove old code

### Example: Refactored `sample_case` Function

**Before**: 400+ line monolithic function

**After**: Clean orchestration in `case_processor.py`:

```python
class CaseProcessor:
    def __init__(self, config, output_dirs, data_loader, sample_processor, writers):
        self.config = config
        self.output_dirs = output_dirs
        self.data_loader = data_loader
        self.sample_processor = sample_processor
        self.writers = writers
    
    def process_case(self, case_fn):
        """Process a single case"""
        # Check if already done
        if self._is_case_done(case_fn):
            return self._empty_result()
        
        # Load case data
        case_data = self.data_loader.load_case(case_fn)
        
        # Process centerlines
        state = ProcessingState()
        results = CaseResults()
        
        for centerline_path in case_data.centerlines:
            centerline_results = self._process_centerline(
                case_data, centerline_path, state
            )
            results.merge(centerline_results)
        
        # Write results
        self.writers.write_case_results(case_data.name, results)
        
        return results
```

### Comparison: Current vs Proposed

#### Current Flow
```
gather_sampling_data_parallel.py (600 lines)
├── sample_case() [400+ lines] - Does everything
│   ├── Data loading (mixed in)
│   ├── Centerline processing (mixed in)
│   ├── Sample extraction (mixed in)
│   ├── Feature extraction (mixed in)
│   ├── Writing (mixed in)
│   └── State management (mixed in)
└── Main script [200 lines] - Setup and multiprocessing
```

#### Proposed Flow
```
sampling/
├── pipeline.py - Main orchestration (~150 lines)
│   └── Coordinates high-level workflow
│
├── case_processor.py - Case-level logic (~200 lines)
│   └── Orchestrates case processing
│       ├── Uses DataLoader
│       ├── Uses SampleProcessor
│       └── Uses Writers
│
├── sample_processor.py - Sample-level logic (~200 lines)
│   └── Processes individual samples
│       └── Uses specialized processors
│
├── data_loader.py - Data loading (~150 lines)
│   └── Clean data loading interface
│
├── writers.py - Output writing (~250 lines)
│   └── All writing operations
│
└── processors/ - Specialized processors
    ├── surface_processor.py
    ├── centerline_processor.py
    ├── outlet_processor.py
    └── trajectory_processor.py
```

### Implementation Considerations

#### Backward Compatibility
- Keep existing `sampling_functions.py` during migration
- New modules import from `sampling_functions.py` initially
- Gradually move functions to new locations
- Update imports once migration is complete

#### Testing Strategy
- Start with integration tests for new structure
- Add unit tests for each new module
- Maintain existing tests during migration
- Test data flow between modules

#### Performance Considerations
- Minimize data copying between modules
- Use lazy loading where possible
- Maintain multiprocessing efficiency
- Profile before/after to ensure no regression

### Alternative: Object-Oriented Approach

Instead of functional approach, could use classes:

```python
class SamplingPipeline:
    """Main pipeline orchestrator"""
    def run(self, config, cases):
        ...

class CaseProcessor:
    """Processes individual cases"""
    def process(self, case_fn):
        ...

class SampleProcessor:
    """Processes individual samples"""
    def process(self, sample_params):
        ...
```

**Recommendation**: Use a hybrid approach - classes for stateful processors (CaseProcessor, SampleProcessor), functions for stateless utilities.

### Next Steps

1. Review and approve this proposal
2. Create the new directory structure
3. Start with `data_loader.py` (least dependencies)
4. Gradually migrate functionality
5. Add unit tests as we go
6. Update documentation
7. Performance testing and optimization

