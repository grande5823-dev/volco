# VOLCO - LLM Reference Document

This document provides a comprehensive overview of the Volco repository structure and functionality to minimize context needed for LLM code editing. It serves as a reference for large language models to understand the codebase without requiring all modules to be uploaded as context.

> **Note**: This reference document is based on examination of key files including volco.py, app/geometry/voxel_space.py, app/instructions/gcode.py, app/reporter/report.py, app/configs/simulation.py, app/configs/printer.py, app/geometry/sphere.py, app/physics/acceleration modules, and app/postprocessing/fea modules. It may not cover aspects contained in files that were not examined. As the repository evolves, this document should be updated accordingly.

## 1. Repository Overview
VOLCO (VOLume COnserving) is a 3D printing simulation model that predicts the final shape of printed materials by simulating the printing process in a voxelized space. It takes G-code as input and produces a voxel 3D-matrix or STL file as output, enabling accurate prediction of material distribution during extrusion-based 3D printing.

## 2. Directory Structure
```
volco/
├── app/                    # Core application modules
│   ├── configs/            # Configuration handling
│   ├── geometry/           # Spatial calculations and voxel operations
│   ├── instructions/       # G-code parsing and instruction handling
│   ├── physics/            # Physics simulation (acceleration, volume)
│   │   └── acceleration/   # Speed profiles and acceleration models
│   ├── postprocessing/     # Post-processing of simulation results
│   │   └── fea/            # Finite Element Analysis for structural simulation
│   ├── reporter/           # Output generation and visualization
│   └── solvers/            # Mathematical solvers
├── examples/               # Example files and usage demonstrations
├── tests/                  # Test suite mirroring app structure
├── trim_top_layers.py      # Preprocessing: trim G-code files to their top N layers
└── volco.py                # Main entry point
```

## 3. Core Components

### Main Entry Point (volco.py)
- Provides `run_simulation()` function that orchestrates the entire simulation process
- Accepts G-code, printer configuration, and simulation configuration as inputs
- Returns a `SimulationOutput` object containing the voxel space and mesh data

### Configuration (app/configs/)
- `Printer` class: Manages printer hardware settings (nozzle diameter, acceleration, etc.)
- `Simulation` class: Manages simulation parameters (voxel size, step size, etc.)
- `Arguments` class: Handles command-line argument parsing

### G-code Processing (app/instructions/)
- `Instruction` class: Abstract base class for all instruction types
- `Gcode` class: Parses G-code files and extracts movement coordinates and extrusion information
- Key methods: `read()`, `_define_movement()`, `_max_min_extru_coordinates()`

### Geometry (app/geometry/)
- `VoxelSpace` class: Manages the 3D voxel grid where the simulation occurs
- `Sphere` class: Models material deposition as spheres in the voxel space
- `GeometryMath` class: Provides utility functions for geometric calculations
- Key methods: `VoxelSpace.initialize_space()`, `VoxelSpace.print()`, `Sphere.deposit_sphere()`

### Physics (app/physics/)
- `Volume` class: Handles volume calculations for material deposition
- `AccelerationManager` class: Calculates speed profiles for nozzle and extruder; owns the discrete time vector shared by both profiles (extended when a motion filter is used, since volume calculation aligns both displacement arrays element-wise)
- `NozzleSpeed` class: Models the nozzle movement speed; optionally wraps the ideal profile in a `FilteredSpeedProfile` (see `motion_filter` parameter)
- `ExtruderSpeed` class: Models the extruder feed rate. Two modes: coupled (default; trapezoid forced to complete within the nozzle's `total_time`) and decoupled (when `printer.extruder_motion_filter` is set; `nozzle_speed` reference required). In decoupled mode, target/threshold/acceleration are the nozzle's scaled by extrusion-length/filament-length, and the profile is a `FilteredSpeedProfile` around a trapezoid with that scaled kinematics
- `SpeedProfile` class: Abstract base class for speed profiles
- `TrapezoidalSpeedProfile` class: Implements a trapezoidal speed profile (`calculate_speed_in_time` exposes the velocity array with an `end_hold` value for times beyond `final_time`)
- `FlatSpeedProfile` class: Implements a constant speed profile
- `FilteredSpeedProfile` class: Wraps an ideal profile and filters its velocity through a transfer function, modelling limited motion-system bandwidth
- `TransferFunction` class: Continuous-time LTI transfer function built from config (`first_order`, `second_order`, or `num_den` types); must be stable, proper, integrator-free, and have unit DC gain

### Reporting (app/reporter/)
- `SimulationOutput` class: Processes and visualizes simulation results
- `mesh` module: Generates and exports 3D meshes
- `heightmap` module: `compute_heightmap()` extracts a 2D top-surface heightmap (highest filled voxel at or above a candidate plane `z >= z_min` per x/y column; empty columns set to a missing-value sentinel, default NaN)
- `visualization` module: Creates visualizations of simulation results
- Key methods: `crop_voxel_space()`, `generate_mesh()`, `export_mesh_to_stl()`, `extract_heightmap()`, `export_heightmap()`, `visualize_mesh()`

### Solvers (app/solvers/)
- `BisectionMethod` class: Implements the bisection method for finding roots
- Key method: `execute()`: Executes the bisection method algorithm

## 4. Key Data Flows

### G-code to Voxel Space
1. `Gcode.read()` → Parses G-code into movement coordinates
2. `VoxelSpace.initialize_space()` → Creates empty voxel grid
3. `VoxelSpace.print()` → For each filament segment:
   - Calculate direction vector and filament length
   - Determine volume distribution with `Volume.get_volumes_for_filament()`
   - Deposit material with `_deposit_filament()`
4. `Sphere.deposit_sphere()` → Models material deposition at each step

### Robustness guards
- Zero-length extrusion moves (prime/unretract) are skipped and their volume is logged (`voxel_space.py`)
- Extrusion moves at or below the bed plane (z <= 0) are skipped and logged; the sphere centre would sit below the space floor and nothing can be deposited below the nozzle then
- Sphere bounding-box indexes are clamped to the voxel-space boundaries on all axes (a negative lower index would make numpy slicing wrap to the far end of the array)
- The bisection solver used for sphere radii has an iteration cap that raises instead of hanging when the target volume is unreachable
- Segments much shorter than the motion filters' settling times use a coarser time grid step instead of an oversized grid

### Voxel Space to Mesh Output
1. `SimulationOutput.crop_voxel_space()` → Crops voxel space to region of interest (crop start indexes are stored so outputs stay registered to G-code coordinates)
2. `SimulationOutput.generate_mesh()` → Converts voxels to 3D mesh using marching cubes
3. `SimulationOutput.export_mesh_to_stl()` → Exports mesh to STL file
4. `SimulationOutput.visualize_mesh()` → Creates visualization with Trimesh or Plotly

### Voxel Space to Heightmap Output (top-surface-only alternative)
- `run_simulation(..., heightmap_z_min=<mm>)` or CLI `--heightmap <mm>` skips mesh generation entirely and produces only the 2D heightmap
- `SimulationOutput.extract_heightmap(z_min, missing_value)` → dict with `heights` (nx×ny, mm), `x`/`y` column-center axes in G-code coordinates, metadata; `export_heightmap(z_min, ...)` writes `<simulation_name>_heightmap.npz`
- Voxel k is counted as model surface only when entirely at/above the plane (k·voxel_size ≥ z_min); reported height is its top face (k+1)·voxel_size; empty columns get `missing_value` (NaN by default)

### Acceleration-Based Volume Distribution
1. `AccelerationManager.calculate_speed_profiles()` → Calculates nozzle and extruder speed profiles
2. `NozzleSpeed.calculate_displacements()` → Computes nozzle position over time
3. `ExtruderSpeed.calculate_displacements()` → Computes extruder position over time
4. `Volume.get_volumes_for_filament()` → Distributes volume based on speed profiles
5. Material deposition adjusted according to calculated volumes

## 5. Configuration Options

See README.md for default values

### Simulation Configuration
- `voxel_size`: Size of each voxel in mm (smaller = higher accuracy but more memory usage)
- `step_size`: Distance between simulation steps in mm (smaller = higher accuracy but slower)
- `x_offset`, `y_offset`, `z_offset`: Offsets for the voxel space boundaries (default: 5 * nozzle_diameter)
- `sphere_z_offset`: Distance to offset the sphere center below the nozzle (default: 0.5 * nozzle_diameter)
- `x_crop`, `y_crop`, `z_crop`: Cropping boundaries for the final output (default: ["all", "all"] for each)
- `radius_increment`: Increment for sphere radius in bisection method
- `solver_tolerance`: Tolerance for volume conservation in bisection method (default: 0.0001)
- `arc_segment_length`: Maximum length of each linear arc-tessellation segment in mm (default: 0.1)
- `consider_acceleration`: Whether to consider acceleration in volume distribution (default: false)
- `stl_ascii`: Whether to export STL in ASCII format (default: false)

### Printer Configuration
- `nozzle_diameter`: Diameter of the printer nozzle in mm
- `feedstock_filament_diameter`: Diameter of the input filament in mm
- `nozzle_jerk_speed`: Maximum instantaneous speed change for nozzle in mm/s
- `extruder_jerk_speed`: Maximum instantaneous speed change for extruder in mm/s
- `nozzle_acceleration`: Acceleration of the nozzle in mm/s²
- `extruder_acceleration`: Acceleration of the extruder in mm/s²
- `nozzle_motion_filter` (optional): Continuous-time transfer function filtering the ideal nozzle velocity profile (`first_order`/`second_order`/`num_den`); only active when `consider_acceleration` is true
- `extruder_motion_filter` (optional): Transfer function decoupling the extruder from the nozzle; the scaled ideal extrusion trajectory is filtered to model extrusion-system lag; only active when `consider_acceleration` is true

## 6. Usage Patterns

### Command Line Usage
```bash
python volco.py --gcode=examples/gcode_example.gcode --sim=examples/simulation_settings.json --printer=examples/printer_settings.json
```

### Preprocessing for Top-Surface-Only Simulations
`trim_top_layers.py` rewrites a G-code file keeping only the top N layers
(`-n`) or top H mm (`--height`): it drops lower layers, start/end G-code
extrusions and travels, and shifts kept layers down near the bed (unless
`--no-shift`) so the voxel space is only as tall as the kept stack. The kept
section is renormalized to G90/G21/M83 with per-move E deltas; X/Y/Z are
rewritten with tracked absolute coordinates. Verified equivalent to the
original's top layers by the simulator's own parser (tests/test_trim_top_layers.py).
Caveat: the first kept layer is then simulated as printed onto the bare bed.

### Python API Usage
```python
from volco import run_simulation

# Using file paths
output = run_simulation(
    gcode_path='examples/gcode_example.gcode',
    printer_config_path='examples/printer_settings.json',
    sim_config_path='examples/simulation_settings.json'
)

# Using Python variables
output = run_simulation(
    gcode=gcode_content,  # G-code as a string
    printer_config=printer_config_dict,  # Printer config as a dictionary
    sim_config=sim_config_dict  # Simulation config as a dictionary
)

# Working with output
output.export_mesh_to_stl()
fig = output.visualize_mesh(visualizer='plotly')
```

## 7. Key Algorithms

- **G-code Parsing**: Converts G-code commands to 3D coordinates and extrusion values, handling different positioning modes and unit conversions. Strict parameter validation applies only to G0/G1/G92; parameters of all other commands are parsed best-effort, tolerating vendor-specific syntax (e.g. `M1002 gcode_claim_action : 29`). G2/G3 arcs in the XY plane (G17) are tessellated into linear segments (`arc_segment_length`, default 0.1 mm); I/J and R forms and helical Z are supported; malformed arcs are skipped with a warning but consume their extrusion reference; each arc sub-segment becomes an independent filament for deposition. A file with no extrusion moves raises a clear error
- **Voxel Space Initialization**: Creates a 3D grid based on printing dimensions and voxel size
- **Material Deposition**: Uses sphere-based model to deposit material in voxel space, with radius determined by volume conservation. Deposition is in-place on the sphere's bounding-box region (reverted after each bisection trial radius, re-applied once accepted); trial growth of the voxel space persists and may leave extra empty boundary slabs
- **Volume Conservation**: Bisection method to determine sphere radius for accurate volume deposition. Filled volume is tracked incrementally (per-sphere delta on a running count carried across segments), not by re-counting the whole voxel space
- **Acceleration Profiles**: Trapezoidal speed profiles for realistic nozzle and extruder movement
- **Motion Filtering**: Optional causal filtering of the ideal nozzle velocity via a continuous-time transfer function (`FilteredSpeedProfile`), with steady-state jerk-speed boundary conditions, an extended time window, and distance conservation enforced by unit DC gain. Filtering is applied with a ZOH-discretized filter via `scipy.signal.lfilter` (matches `lsim` to ~1e-13 for constant/step inputs; sub-micrometer displacement deviation for ramps)
- **Decoupled Extrusion**: Optional extruder lag model — ideal extrusion = scaled ideal motion velocity (constant volume per unit length), actual extrusion = ideal filtered by a transfer function; redistributes volume along segments (under at start, over at end); material extruded after the nozzle leaves the segment is not deposited
- **Mesh Generation**: Marching cubes algorithm to convert voxel space to 3D mesh

## 8. Module Dependencies

- `volco.py` → `app/configs`, `app/instructions`, `app/geometry`, `app/reporter`
- `app/geometry/voxel_space.py` → `app/geometry/sphere.py`, `app/physics/volume.py`
- `app/physics/volume.py` → `app/physics/acceleration/acceleration_profiles.py`
- `app/reporter/report.py` → `app/reporter/mesh.py`, `app/reporter/visualization.py`
- `app/instructions/gcode.py` → `app/instructions/instruction.py`
- `app/geometry/sphere.py` → `app/solvers/bisection_method.py`

## 9. Common Modification Patterns

### Adding New Speed Profiles
1. Create a new class in `app/physics/acceleration/` that extends `SpeedProfile`
2. Implement the `calculate_displacements()` method
3. Update `Volume.get_volumes_for_filament()` to use the new profile

### Supporting Additional G-code Commands
1. Modify `app/instructions/gcode.py` to recognize new commands
2. Update the `read()` method to handle the new commands
3. Implement any necessary processing in `_define_movement()`

### Adding New Visualization Options
1. Add new visualization function in `app/reporter/visualization.py`
2. Update `SimulationOutput.visualize_mesh()` to support the new option
3. Add any necessary parameters to the visualization method

### Modifying Material Deposition Model
1. Update `app/geometry/sphere.py` or create a new deposition model class
2. Modify `VoxelSpace._deposit_filament()` to use the new model
3. Update volume calculation methods if necessary
4. Adjust the `sphere_z_offset` centroid of spheres below nozzle

## 10. Performance Considerations

- **Voxel Size**: Smaller voxel sizes increase accuracy but exponentially increase memory usage and computation time
- **Step Size**: Smaller step sizes increase simulation accuracy but increase computation time
- **Acceleration Consideration**: Enabling acceleration profiles increases accuracy but adds computational overhead
- **Mesh Generation**: Marching cubes algorithm can be memory-intensive for large voxel spaces
- **Cropping**: Proper cropping of the voxel space before mesh generation can significantly improve performance
- **Bisection Method Tolerance**: Lower tolerance values increase volume accuracy but require more iterations
- **Sphere Radius Increment**: Smaller increments increase accuracy but require more iterations in the bisection method

## 11. Finite Element Analysis (FEA) Module

The FEA module provides structural analysis capabilities for voxel models generated by VOLCO simulations. For a comprehensive overview of this module, refer to:

- [llm_ref_fea.md](llm_ref_fea.md) - Detailed documentation of the FEA module structure and functionality

Key components include:
- Mesh generation from voxel matrices
- Linear static structural analysis
- Visualization of displacements, stresses, and strains
- Results storage and retrieval in multiple formats

This module enables users to perform structural analysis on 3D printed parts simulated by VOLCO, providing insights into mechanical behavior under load.