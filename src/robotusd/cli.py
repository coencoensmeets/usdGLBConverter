#!/usr/bin/env python3
"""
Command-line interface for RobotUSD - USD to glTF Converter
"""

import argparse
import sys
import os
import logging
import time
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity flags."""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


def validate_input_file(file_path: str) -> bool:
    """Validate that the input USD file exists and is readable."""
    if not os.path.exists(file_path):
        logger.error(f"Input file does not exist: {file_path}")
        return False
    
    if not os.path.isfile(file_path):
        logger.error(f"Input path is not a file: {file_path}")
        return False
    
    if not file_path.lower().endswith(('.usd', '.usda', '.usdc', '.usdz')):
        logger.warning(f"Input file may not be a USD file (unexpected extension): {file_path}")
    
    return True


def validate_output_file(file_path: str) -> bool:
    """Validate that the output path is writable and has correct extension."""
    output_dir = os.path.dirname(os.path.abspath(file_path))
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Cannot create output directory {output_dir}: {e}")
            return False
    
    if not os.access(output_dir, os.W_OK):
        logger.error(f"Output directory is not writable: {output_dir}")
        return False
    
    if not file_path.lower().endswith(('.gltf', '.glb')):
        logger.error(f"Output file must have .gltf or .glb extension: {file_path}")
        return False
    
    return True


def convert_command(args) -> int:
    """Handle the conversion command."""
    try:
        # Import here to avoid dependency issues if USD is not installed
        from robotusd import USDRobot, USDToGLTFConverter, check_dependencies
        from pxr import Usd
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Required dependencies are not available")
            return 1
        
        # Validate input and output files
        if not validate_input_file(args.input):
            return 1
        
        if not validate_output_file(args.output):
            return 1
        
        logger.info(f"Converting {args.input} to {args.output}")
        start_time = time.perf_counter()
        
        # Load USD stage
        logger.info("Loading USD stage...")
        stage = Usd.Stage.Open(args.input)
        if not stage:
            logger.error(f"Failed to load USD file: {args.input}")
            return 1
        
        # Create robot structure
        logger.info("Building robot structure...")
        robot = USDRobot(stage, args.robot_name)
        
        if not robot.links:
            logger.error("No robot links found in USD file")
            return 1
        
        logger.info(f"Found {len(robot.links)} links and {len(robot.joints)} joints")
        
        # Create converter
        logger.info("Creating glTF converter...")
        converter = USDToGLTFConverter(robot)
        
        # Perform conversion
        logger.info("Converting to glTF...")
        converter.convert_robot_to_gltf()
        
        # Export to file
        logger.info(f"Exporting to {args.output}...")
        converter.export(args.output)
        
        end_time = time.perf_counter()
        logger.info(f"Conversion completed successfully in {end_time - start_time:.2f} seconds")
        
        # Print summary statistics
        if not args.quiet:
            logger.info("=== Conversion Summary ===")
            logger.info(f"Input file: {args.input}")
            logger.info(f"Output file: {args.output}")
            logger.info(f"Robot name: {args.robot_name}")
            logger.info(f"glTF nodes: {len(converter.gltf_nodes)}")
            logger.info(f"glTF meshes: {len(converter.meshes_gltf)}")
            logger.info(f"glTF materials: {len(converter.materials_gltf)}")
            logger.info(f"Binary data size: {len(converter.bin_data):,} bytes")
        
        return 0
        
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        logger.error("Please install USD library (usd-core) and other dependencies")
        return 1
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def info_command(args) -> int:
    """Handle the info command to display robot information."""
    try:
        # Import here to avoid dependency issues if USD is not installed
        from robotusd import get_robot_info, check_dependencies
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Required dependencies are not available")
            return 1
        
        # Validate input file
        if not validate_input_file(args.input):
            return 1
        
        logger.info(f"Analyzing USD file: {args.input}")
        
        # Get robot information
        info = get_robot_info(args.input)
        
        if not info:
            logger.error("Failed to analyze USD file")
            return 1
        
        # Print information
        print("=" * 60)
        print("USD Robot Information")
        print("=" * 60)
        print(f"File: {info.get('file_path', 'Unknown')}")
        print(f"Robot Name: {info.get('robot_name', 'Unknown')}")
        print(f"Base Link: {info.get('base_link', 'Not found')}")
        print(f"Total Links: {info.get('total_links', 0)}")
        print(f"Total Joints: {info.get('total_joints', 0)}")
        print(f"Links with Meshes: {info.get('links_with_meshes', 0)}")
        print(f"Total Material Assignments: {info.get('total_material_assignments', 0)}")
        print(f"Unique Materials: {info.get('unique_materials', 0)}")
        print(f"Tree Depth: {info.get('max_depth', 0)}")
        
        # Joint types
        joint_types = info.get('joint_types', {})
        if joint_types:
            print("\nJoint Types:")
            for joint_type, count in joint_types.items():
                print(f"  {joint_type}: {count}")
        
        # Leaf links
        leaf_links = info.get('leaf_links', [])
        if leaf_links:
            print(f"\nLeaf Links ({len(leaf_links)}):")
            for link in leaf_links[:10]:  # Show first 10
                print(f"  {link}")
            if len(leaf_links) > 10:
                print(f"  ... and {len(leaf_links) - 10} more")
        
        print("=" * 60)
        
        return 0
        
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        logger.error("Please install USD library (usd-core) and other dependencies")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='robotusd',
        description='USD to glTF Converter for Robotics Applications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert USD file to glTF
  robotusd-convert robot.usd robot.glb
  
  # Convert with custom robot name
  robotusd-convert robot.usd robot.glb --robot-name "MyRobot"
  
  # Get information about a USD robot
  robotusd-info robot.usd
  
  # Verbose output for debugging
  robotusd-convert robot.usd robot.glb --verbose
        """
    )
    
    # Global options
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output for debugging'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert USD file to glTF format',
        description='Convert a USD robot file to glTF/GLB format'
    )
    convert_parser.add_argument(
        'input',
        help='Input USD file path'
    )
    convert_parser.add_argument(
        'output',
        help='Output glTF/GLB file path'
    )
    convert_parser.add_argument(
        '--robot-name',
        default='Robot',
        help='Name for the robot (default: Robot)'
    )
    convert_parser.set_defaults(func=convert_command)
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Display information about a USD robot file',
        description='Analyze and display information about a USD robot file'
    )
    info_parser.add_argument(
        'input',
        help='Input USD file path'
    )
    info_parser.set_defaults(func=info_command)
    
    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    # Execute the appropriate command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


# Entry points for console scripts
def main_convert():
    """Entry point for robotusd-convert command."""
    # Simulate convert command
    sys.argv = ['robotusd-convert'] + sys.argv[1:] if len(sys.argv) > 1 else ['robotusd-convert', '--help']
    
    # Create parser and parse only convert-related args
    parser = argparse.ArgumentParser(prog='robotusd-convert')
    parser.add_argument('input', help='Input USD file path')
    parser.add_argument('output', help='Output glTF/GLB file path')
    parser.add_argument('--robot-name', default='Robot', help='Name for the robot')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output except errors')
    
    args = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    return convert_command(args)


def main_info():
    """Entry point for robotusd-info command."""
    # Create parser for info command
    parser = argparse.ArgumentParser(prog='robotusd-info')
    parser.add_argument('input', help='Input USD file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output except errors')
    
    args = parser.parse_args()
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    return info_command(args)


if __name__ == '__main__':
    sys.exit(main())
