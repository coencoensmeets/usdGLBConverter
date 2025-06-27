<br />
<p align="center">
    <a href="https://github.com/coencoensmeets/usdGLBConverter">
        <img src="https://go.forrester.com/wp-content/uploads/2022/09/USDLogo400x400-267x300.png" alt="USD Logo" width="80" height="80">
    </a>

    <h3 align="center">RobotUSD - USD to glTF Converter for Robotics Applications</h3>

    <p align="center">
        High-performance Python package for converting USD (Universal Scene Description) robot models to glTF format.<br>
        Specialized support for multi-material meshes, joint hierarchies, and robotics workflows.
        <br />
        <a href="https://github.com/coencoensmeets/usdGLBConverter"><strong>Explore the docs »</strong></a>
        <br />
        <br />
        <a href="https://github.com/coencoensmeets/usdGLBConverter">View Demo</a>
        ·
        <a href="https://github.com/coencoensmeets/usdGLBConverter/issues">Report Bug</a>
        ·
        <a href="https://github.com/coencoensmeets/usdGLBConverter/issues">Request Feature</a>
    </p>
</p>

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **🤖 Robot-Aware Conversion**: Understands robot joint hierarchies and link structures
- **🎨 Multi-Material Support**: Handles complex meshes with multiple materials via GeomSubsets
- **⚡ High Performance**: Optimized with numpy, caching, and iterative algorithms
- **🔧 Joint Transform Correction**: Robust handling of joint-corrected transforms for accurate positioning
- **📊 Comprehensive Analysis**: Detailed statistics and validation of robot structures
- **🖥️ CLI Tools**: Command-line interface for batch processing and automation
- **🔍 Debug Support**: Extensive logging and validation capabilities

## Installation

### From Source (Development)

```bash
git clone <repository-url>
cd usdGLBConverter
pip install -e .
```

## Quick Start

### Python API

```python
from robotusd import USDRobot, USDToGLTFConverter
from pxr import Usd

# Load USD stage
stage = Usd.Stage.Open("path/to/robot.usd")

# Create robot structure
robot = USDRobot(stage, name="MyRobot")

# Print robot structure
robot.print_structure()

# Convert to glTF
converter = USDToGLTFConverter(robot)
converter.export("output/robot.glb")
```

### Command Line Interface

```bash
# Convert a single USD file to glTF
robotusd-convert Assets/Robots/Franka/franka.usd Output/franka.glb

# Get detailed information about a robot
robotusd-info Assets/Robots/Franka/franka.usd

# Convert with custom options
robotusd-convert input.usd output.gltf --format gltf --validate

# Batch convert multiple files
robotusd-convert Assets/Robots/ Output/ --recursive --format glb
```

## Supported USD Features

### Robot Structure
- ✅ Joint hierarchies with parent-child relationships
- ✅ Physics joints with `localPos0/localPos1` and `localRot0/localRot1`
- ✅ Fallback transforms using `xformOp:orient` and `xformOp:translate`
- ✅ Link-based mesh organization
- ✅ Automatic base link detection

### Geometry & Materials
- ✅ Triangle meshes with position data
- ✅ Multiple materials per mesh via GeomSubsets
- ✅ Material binding inheritance
- ✅ PBR material properties (diffuse, metallic, roughness, emissive)
- ✅ UsdPreviewSurface and OmniPBR shaders

### Transforms
- ✅ Joint-corrected transforms for accurate positioning
- ✅ Local position and rotation properties
- ✅ Scale inheritance
- ✅ Transform caching for performance

## Output Formats

- **GLB**: Binary glTF format (recommended for web/AR/VR)
- **glTF + .bin**: Separate JSON and binary files

## Project Structure

```
usdGLBConverter/
├── src/robotusd/           # Main package
│   ├── __init__.py         # Package initialization and exports
│   ├── robot_structure.py  # Robot hierarchy classes
│   ├── usd2gltf.py        # Conversion engine
│   ├── math_utils.py      # Math utilities
│   ├── usd_utils.py       # USD utilities
│   └── cli.py             # Command-line interface
├── Tests/                  # Test scripts
├── Assets/                 # Sample robot models
├── Output/                 # Converted glTF files
├── setup.py               # Package configuration
└── README.md              # This file
```

## Advanced Usage

### Custom Material Processing

```python
from robotusd import USDMesh

# Access mesh materials
for link in robot.get_all_links():
    for mesh in link.meshes:
        if mesh.has_multiple_materials():
            print(f"Mesh {mesh.name} has {mesh.get_material_count()} materials")
            for subset_info in mesh.get_geom_subsets_with_materials():
                print(f"  - Subset: {subset_info['name']}")
                print(f"  - Material: {subset_info['material'].GetName()}")
```

### Joint Analysis

```python
# Analyze joint properties
for joint in robot.get_all_joints():
    print(f"Joint: {joint.name}")
    print(f"Type: {joint.joint_type}")
    
    # Get local positions and rotations
    pos0, pos1 = joint.get_local_positions()
    rot0, rot1 = joint.get_local_rotations()
    
    print(f"Local positions: {pos0} -> {pos1}")
    print(f"Local rotations: {rot0} -> {rot1}")
    
    # Get axis and limits
    axis = joint.get_axis()
    limits = joint.get_limits()
    print(f"Axis: {axis}, Limits: {limits}")
```

### Transform Debugging

```python
# Get joint-corrected transforms
for link in robot.get_all_links():
    if link.parent_joint:
        translation, rotation, scale = link.calculate_joint_corrected_transform()
        print(f"Link {link.name}:")
        print(f"  Translation: {translation}")
        print(f"  Rotation: {rotation}")
        print(f"  Scale: {scale}")
```

### Debug Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('robotusd')
logger.setLevel(logging.DEBUG)

# Run conversion with detailed output
robot = USDRobot(stage)
```

## Requirements

- Python 3.8+ (3.13 maximum for usd-core compatibility)
- NumPy >= 1.19.0
- usd-core >= 23.05

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA Omniverse for USD ecosystem
- Pixar for Universal Scene Description

## Changelog

### Version 0.1.0
- Initial release
- Basic USD to glTF conversion
- Multi-material support
- Joint hierarchy processing
- CLI tools
- Performance optimizations

---

For more information, examples, and documentation, visit the [project repository](https://github.com/coencoensmeets/usdGLBConverter).
