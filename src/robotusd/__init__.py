"""
RobotUSD - USD to glTF Converter for Robotics Applications

A Python package for converting Universal Scene Description (USD) robot models
to glTF format, with support for complex robot hierarchies, joints, materials,
and multi-material meshes.

Key Features:
- Convert USD robot models to glTF/GLB format
- Support for complex joint hierarchies
- Multi-material mesh support via GeomSubsets
- Optimized performance for large robot models
- Material binding and texture support
- Physics-aware joint extraction

Example Usage:
    from robotusd import USDRobot, USDToGLTFConverter
    from pxr import Usd
    
    # Load USD stage
    stage = Usd.Stage.Open("robot.usd")
    
    # Create robot structure
    robot = USDRobot(stage, "MyRobot")
    
    # Convert to glTF
    converter = USDToGLTFConverter(robot)
    converter.export("robot.glb")
"""

__version__ = "0.1.0"
__author__ = "Coen Smeets"
__license__ = "MIT"

# Import main classes for easy access
from .robot_structure import USDRobot, USDLink, USDJoint, USDMesh
from .usd2gltf import USDToGLTFConverter
from .math_utils import (
    quat_to_list,
    quaternion_multiply,
    quaternion_inverse,
    euler_to_quat,
    rotate_vector
)
from .usd_utils import get_prim_from_name, get_all_joints, print_data_prim

# Define what gets imported with "from robotusd import *"
__all__ = [
    # Main classes
    "USDRobot",
    "USDLink", 
    "USDJoint",
    "USDMesh",
    "USDToGLTFConverter",
    
    # Utility functions
    "quat_to_list",
    "quaternion_multiply", 
    "quaternion_inverse",
    "euler_to_quat",
    "rotate_vector",
    
    # USD utility functions
    "get_prim_from_name",
    "get_all_joints",
    "print_data_prim",
    
    # Package metadata
    "__version__",
    "__author__",
    "__license__",
]

# Package-level convenience functions
def convert_usd_to_gltf(usd_file_path: str, output_path: str, robot_name: str = "Robot") -> bool:
    """
    Convenience function to convert a USD file directly to glTF.
    
    Args:
        usd_file_path: Path to the input USD file
        output_path: Path for the output glTF/GLB file
        robot_name: Name for the robot (default: "Robot")
        
    Returns:
        bool: True if conversion was successful, False otherwise
        
    Example:
        convert_usd_to_gltf("robot.usd", "robot.glb", "MyRobot")
    """
    try:
        from pxr import Usd
        
        # Load USD stage
        stage = Usd.Stage.Open(usd_file_path)
        if not stage:
            return False
            
        # Create robot structure
        robot = USDRobot(stage, robot_name)
        
        # Convert to glTF
        converter = USDToGLTFConverter(robot)
        converter.export(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error converting {usd_file_path} to {output_path}: {e}")
        return False

def get_robot_info(usd_file_path: str) -> dict:
    """
    Get information about a USD robot file without converting it.
    
    Args:
        usd_file_path: Path to the input USD file
        
    Returns:
        dict: Robot information including links, joints, meshes, and materials
        
    Example:
        info = get_robot_info("robot.usd")
        print(f"Robot has {info['total_links']} links and {info['total_joints']} joints")
    """
    try:
        from pxr import Usd
        
        # Load USD stage
        stage = Usd.Stage.Open(usd_file_path)
        if not stage:
            return {}
            
        # Create robot structure
        robot = USDRobot(stage, "InfoRobot")
        
        # Get statistics
        stats = robot.get_tree_statistics()
        
        return {
            "file_path": usd_file_path,
            "robot_name": robot.name,
            "base_link": stats.get("base_link"),
            "total_links": stats.get("total_links", 0),
            "total_joints": stats.get("total_joints", 0),
            "links_with_meshes": stats.get("links_with_meshes", 0),
            "total_material_assignments": stats.get("total_material_assignments", 0),
            "unique_materials": stats.get("unique_materials_count", 0),
            "joint_types": stats.get("joint_types", {}),
            "max_depth": stats.get("max_depth", 0),
            "leaf_links": stats.get("leaf_links", [])
        }
        
    except Exception as e:
        print(f"Error getting info for {usd_file_path}: {e}")
        return {}

# Version check for dependencies
def check_dependencies() -> bool:
    """
    Check if all required dependencies are available.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from pxr import Usd, UsdGeom, UsdShade, Gf
    except ImportError:
        missing_deps.append("usd-core (or pxr)")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install numpy usd-core")
        return False
    
    return True