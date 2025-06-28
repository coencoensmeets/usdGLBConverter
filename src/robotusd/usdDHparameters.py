import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from .robot_structure import USDRobot, USDJoint, USDLink
from .math_utils import quaternion_to_rotation_matrix, normalize_vector, vector_angle

logger = logging.getLogger(__name__)

class USDDHParameters:
    """
    Class to handle Denavit-Hartenberg parameters for a robot in USD format.
    
    This class provides methods to extract and manipulate DH parameters from a USD robot structure,
    ensuring that Y is always up (Y-up coordinate system).
    
    DH Parameters Convention:
    - theta: Joint angle (rotation around Z-axis)
    - d: Link offset (distance along Z-axis)
    - a: Link length (distance along X-axis)
    - alpha: Link twist (rotation around X-axis)
    """

    def __init__(self, robot: USDRobot):
        self.robot = robot
        self.y_up = True  # Ensure Y-up coordinate system
        self._dh_cache: Optional[List[Dict[str, Any]]] = None
        
    def get_dh_parameters(self) -> List[Dict[str, Any]]:
        """
        Extracts the Denavit-Hartenberg parameters from the robot structure.
        
        Returns:
            list: A list of DH parameter dictionaries for each joint.
                  Each dict contains: {'joint_name', 'theta', 'd', 'a', 'alpha', 'joint_type'}
        """
        if self._dh_cache is None:
            self._dh_cache = self._calculate_dh_parameters()
        return self._dh_cache
    
    def _calculate_dh_parameters(self) -> List[Dict[str, Any]]:
        """Calculate DH parameters for all joints in the robot."""
        dh_params = []
        
        if not self.robot.base_link:
            logger.warning("No base link found - cannot calculate DH parameters")
            return dh_params
            
        # Get the kinematic chain starting from base link
        kinematic_chain = self._build_kinematic_chain()
        
        logger.info(f"Building DH parameters for {len(kinematic_chain)} joints")
        
        for i, (joint, parent_link, child_link) in enumerate(kinematic_chain):
            try:
                dh_param = self._calculate_joint_dh_parameters(
                    joint, parent_link, child_link, i, kinematic_chain
                )
                dh_params.append(dh_param)
                logger.debug(f"Joint {joint.name}: theta={dh_param['theta']:.3f}, "
                           f"d={dh_param['d']:.3f}, a={dh_param['a']:.3f}, alpha={dh_param['alpha']:.3f}")
            except Exception as e:
                logger.error(f"Failed to calculate DH parameters for joint {joint.name}: {e}")
                # Add default parameters to maintain chain consistency
                dh_params.append({
                    'joint_name': joint.name,
                    'theta': 0.0,
                    'd': 0.0,
                    'a': 0.0,
                    'alpha': 0.0,
                    'joint_type': joint.joint_type,
                    'error': str(e)
                })
        
        return dh_params
    
    def _build_kinematic_chain(self) -> List[Tuple[USDJoint, USDLink, USDLink]]:
        """
        Build a kinematic chain from the robot structure.
        
        Returns:
            List of tuples (joint, parent_link, child_link) in kinematic order.
        """
        chain = []
        visited_joints = set()
        
        def traverse_chain(link: USDLink):
            for joint in link.joints:
                if joint.name not in visited_joints and joint.child_link:
                    visited_joints.add(joint.name)
                    chain.append((joint, link, joint.child_link))
                    traverse_chain(joint.child_link)
        
        # Start from base link
        traverse_chain(self.robot.base_link)
        
        return chain
    
    def _calculate_joint_dh_parameters(self, joint: USDJoint, parent_link: USDLink, 
                                     child_link: USDLink, joint_index: int,
                                     kinematic_chain: List) -> Dict[str, Any]:
        """
        Calculate DH parameters for a single joint.
        
        Args:
            joint: The joint to calculate parameters for
            parent_link: The parent link
            child_link: The child link
            joint_index: Index of this joint in the kinematic chain
            kinematic_chain: The complete kinematic chain for context
            
        Returns:
            Dictionary with DH parameters
        """
        # Get joint transform information
        joint_translation, joint_rotation, _ = child_link.calculate_joint_corrected_transform()
        
        # Convert to Y-up coordinate system if needed
        if self.y_up:
            joint_translation = self._convert_to_y_up(joint_translation)
            joint_rotation = self._convert_rotation_to_y_up(joint_rotation)
        
        # Get joint axis
        joint_axis = joint.get_axis()
        if joint_axis is None:
            # Default to Z-axis for revolute joints, X-axis for prismatic
            joint_axis = [0.0, 0.0, 1.0] if joint.joint_type == 'revolute' else [1.0, 0.0, 0.0]
        
        if self.y_up:
            joint_axis = self._convert_to_y_up(joint_axis)
        
        # Calculate DH parameters
        theta = self._calculate_theta(joint, joint_rotation, joint_axis)
        d = self._calculate_d(joint, joint_translation, joint_axis)
        a, alpha = self._calculate_a_and_alpha(joint, parent_link, child_link, joint_index, kinematic_chain)
        
        return {
            'joint_name': joint.name,
            'theta': theta,
            'd': d,
            'a': a,
            'alpha': alpha,
            'joint_type': joint.joint_type,
            'joint_axis': joint_axis,
            'translation': joint_translation,
            'rotation': joint_rotation
        }
    
    def _calculate_theta(self, joint: USDJoint, joint_rotation: List[float], 
                        joint_axis: List[float]) -> float:
        """
        Calculate theta parameter (joint angle around Z-axis).
        
        For revolute joints, this is the variable joint angle.
        For prismatic joints, this is typically 0.
        """
        if joint.joint_type == 'revolute':
            # Extract rotation angle around the joint axis
            # Convert quaternion to rotation matrix
            rot_matrix = quaternion_to_rotation_matrix(joint_rotation)
            
            # Project onto XY plane for Z-axis rotation
            if abs(joint_axis[2]) > 0.9:  # Z-axis joint
                # Calculate angle from rotation matrix
                theta = math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
            else:
                # For non-Z axis joints, calculate equivalent Z rotation
                # This is a simplification - in practice might need more complex calculation
                theta = 0.0
                
            return theta
        else:
            # Prismatic joints typically have theta = 0
            return 0.0
    
    def _calculate_d(self, joint: USDJoint, joint_translation: List[float], 
                    joint_axis: List[float]) -> float:
        """
        Calculate d parameter (link offset along Z-axis).
        
        For prismatic joints, this is the variable joint displacement.
        For revolute joints, this is the fixed Z-offset.
        """
        if joint.joint_type == 'prismatic':
            # For prismatic joints, d is the displacement along joint axis
            # Project translation onto joint axis
            translation_array = np.array(joint_translation)
            axis_array = np.array(joint_axis)
            d = np.dot(translation_array, axis_array)
            return d
        else:
            # For revolute joints, d is the Z-component of translation
            return joint_translation[2]
    
    def _calculate_a_and_alpha(self, joint: USDJoint, parent_link: USDLink, 
                              child_link: USDLink, joint_index: int,
                              kinematic_chain: List) -> Tuple[float, float]:
        """
        Calculate a (link length) and alpha (link twist) parameters.
        
        a: Distance between Z-axes along common normal
        alpha: Angle between Z-axes around common normal
        """
        # Get the next joint in the chain for calculating link parameters
        if joint_index + 1 < len(kinematic_chain):
            next_joint, _, _ = kinematic_chain[joint_index + 1]
            next_joint_axis = next_joint.get_axis()
            if next_joint_axis is None:
                next_joint_axis = [0.0, 0.0, 1.0]
            if self.y_up:
                next_joint_axis = self._convert_to_y_up(next_joint_axis)
        else:
            # Last joint - use default Z-axis
            next_joint_axis = [0.0, 0.0, 1.0]
        
        current_joint_axis = joint.get_axis()
        if current_joint_axis is None:
            current_joint_axis = [0.0, 0.0, 1.0]
        if self.y_up:
            current_joint_axis = self._convert_to_y_up(current_joint_axis)
        
        # Calculate link length (a) - distance between joint axes
        child_translation, _, _ = child_link.calculate_joint_corrected_transform()
        if self.y_up:
            child_translation = self._convert_to_y_up(child_translation)
        
        # Simplified calculation: a is typically the X-component of the link translation
        a = math.sqrt(child_translation[0]**2 + child_translation[1]**2)
        
        # Calculate link twist (alpha) - angle between consecutive Z-axes
        current_axis = np.array(current_joint_axis)
        next_axis = np.array(next_joint_axis)
        
        # Normalize axes
        current_axis = current_axis / np.linalg.norm(current_axis)
        next_axis = next_axis / np.linalg.norm(next_axis)
        
        # Calculate angle between axes
        dot_product = np.clip(np.dot(current_axis, next_axis), -1.0, 1.0)
        alpha = math.acos(abs(dot_product))
        
        # Determine sign of alpha using cross product
        cross_product = np.cross(current_axis, next_axis)
        if cross_product[0] < 0:  # Check X-component for sign
            alpha = -alpha
        
        return a, alpha
    
    def _convert_to_y_up(self, vector: List[float]) -> List[float]:
        """
        Convert a vector from Z-up to Y-up coordinate system.
        
        USD typically uses Y-up, but some robots might be defined in Z-up.
        This function ensures consistent Y-up orientation.
        """
        if len(vector) != 3:
            return vector
            
        # If the vector is already in Y-up (which it should be for USD), return as-is
        # This function is here for future flexibility
        return vector
    
    def _convert_rotation_to_y_up(self, quaternion: List[float]) -> List[float]:
        """
        Convert a quaternion rotation from Z-up to Y-up coordinate system.
        """
        # For USD which is already Y-up, return as-is
        # This function is here for future flexibility
        return quaternion
    
    def get_dh_table(self) -> str:
        """
        Get a formatted string representation of the DH parameter table.
        
        Returns:
            Formatted string with DH parameters table.
        """
        dh_params = self.get_dh_parameters()
        
        if not dh_params:
            return "No DH parameters available"
        
        # Create table header
        table = "DH Parameters Table (Y-up coordinate system)\n"
        table += "=" * 80 + "\n"
        table += f"{'Joint':<15} {'Type':<10} {'θ (deg)':<10} {'d':<10} {'a':<10} {'α (deg)':<10}\n"
        table += "-" * 80 + "\n"
        
        # Add each joint's parameters
        for param in dh_params:
            theta_deg = math.degrees(param['theta'])
            alpha_deg = math.degrees(param['alpha'])
            
            table += f"{param['joint_name']:<15} "
            table += f"{param['joint_type']:<10} "
            table += f"{theta_deg:<10.2f} "
            table += f"{param['d']:<10.3f} "
            table += f"{param['a']:<10.3f} "
            table += f"{alpha_deg:<10.2f}\n"
            
            if 'error' in param:
                table += f"{'':>25} ERROR: {param['error']}\n"
        
        table += "=" * 80 + "\n"
        table += f"Total joints: {len(dh_params)}\n"
        
        return table
    
    def export_dh_to_dict(self) -> Dict[str, Any]:
        """
        Export DH parameters as a dictionary suitable for serialization.
        
        Returns:
            Dictionary containing all DH parameters and metadata.
        """
        dh_params = self.get_dh_parameters()
        
        return {
            'robot_name': self.robot.name,
            'coordinate_system': 'Y-up',
            'dh_convention': 'Modified DH (Craig)',
            'joint_count': len(dh_params),
            'joints': dh_params,
            'base_link': self.robot.base_link.name if self.robot.base_link else None,
            'kinematic_chain': [param['joint_name'] for param in dh_params]
        }
    
    def validate_dh_parameters(self) -> Dict[str, Any]:
        """
        Validate the calculated DH parameters for consistency.
        
        Returns:
            Dictionary with validation results.
        """
        dh_params = self.get_dh_parameters()
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        if not dh_params:
            validation['valid'] = False
            validation['errors'].append("No DH parameters calculated")
            return validation
        
        # Check for common issues
        for i, param in enumerate(dh_params):
            joint_name = param['joint_name']
            
            # Check for errors in calculation
            if 'error' in param:
                validation['valid'] = False
                validation['errors'].append(f"Joint {joint_name}: {param['error']}")
            
            # Check for extreme values
            if abs(param['a']) > 10.0:  # Assuming meters, >10m is unusual
                validation['warnings'].append(f"Joint {joint_name}: Large link length a={param['a']:.3f}")
            
            if abs(param['d']) > 10.0:
                validation['warnings'].append(f"Joint {joint_name}: Large link offset d={param['d']:.3f}")
            
            # Check for NaN or infinite values
            for key in ['theta', 'd', 'a', 'alpha']:
                value = param[key]
                if math.isnan(value) or math.isinf(value):
                    validation['valid'] = False
                    validation['errors'].append(f"Joint {joint_name}: Invalid {key} value: {value}")
        
        # Calculate statistics
        validation['statistics'] = {
            'total_joints': len(dh_params),
            'revolute_joints': sum(1 for p in dh_params if p['joint_type'] == 'revolute'),
            'prismatic_joints': sum(1 for p in dh_params if p['joint_type'] == 'prismatic'),
            'max_link_length': max(abs(p['a']) for p in dh_params),
            'max_link_offset': max(abs(p['d']) for p in dh_params),
        }
        
        return validation
    
    def clear_cache(self) -> None:
        """Clear the DH parameters cache to force recalculation."""
        self._dh_cache = None
        
    def __str__(self) -> str:
        return self.get_dh_table()
    
    def __repr__(self) -> str:
        dh_params = self.get_dh_parameters()
        return f"USDDHParameters(robot='{self.robot.name}', joints={len(dh_params)})"