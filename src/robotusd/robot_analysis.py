import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from .robot_structure import USDRobot, USDJoint, USDLink
from .math_utils import HomogeneousMatrix, normalize_vector, vector_angle, rotation_matrix_to_quaternion

logger = logging.getLogger(__name__)

class USDRobotAnalysis:
    """
    Class to analyze USD robot structures and determine robot type, configuration, and characteristics.
    
    This class analyzes the robot structure to determine:
    - Robot type (robot arm, quadruped, humanoid, etc.)
    - Kinematic chain configurations
    - Anthropomorphic configurations for robot arms
    - DOF analysis (6-DOF, 7-DOF, etc.)
    - Solver compatibility recommendations
    """

    def __init__(self, robot: USDRobot):
        self.robot = robot
        self._analysis_cache: Optional[Dict[str, Any]] = None
        
    def analyze_robot(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the robot structure.
        
        Returns:
            Dictionary with complete robot analysis including:
            - robot_type: Detected robot type (arm, quadruped, humanoid, etc.)
            - kinematic_chains: Analysis of all kinematic chains
            - dof_analysis: Degrees of freedom analysis
            - anthropomorphic_analysis: Anthropomorphic configuration analysis
            - solver_recommendations: Recommended IK solvers
        """
        if self._analysis_cache is None:
            self._analysis_cache = self._perform_robot_analysis()
        return self._analysis_cache
    
    def _perform_robot_analysis(self) -> Dict[str, Any]:
        """Perform the actual robot analysis."""
        
        analysis = {
            'robot_type': 'unknown',
            'robot_subtype': None,
            'confidence': 0.0,
            'kinematic_chains': [],
            'primary_chain': None,
            'dof_analysis': {},
            'anthropomorphic_analysis': {},
            'solver_recommendations': [],
            'statistics': {},
            'warnings': []
        }
        
        if not self.robot.base_link:
            analysis['warnings'].append("No base link found - cannot analyze robot structure")
            return analysis
        
        # Step 1: Analyze kinematic structure
        chains = self._find_all_kinematic_chains()
        analysis['kinematic_chains'] = chains
        analysis['statistics']['total_chains'] = len(chains)
        
        # Step 2: Determine robot type based on structure
        robot_type_analysis = self._classify_robot_type(chains)
        analysis.update(robot_type_analysis)
        
        # Step 2.5: Detect gripper configurations
        gripper_analysis = self._detect_gripper_configuration(chains)
        analysis['gripper_analysis'] = gripper_analysis
        
        # Step 3: Analyze primary kinematic chain
        if chains:
            primary_chain = self._identify_primary_chain(chains)
            analysis['primary_chain'] = primary_chain
            
            # Step 4: DOF analysis
            analysis['dof_analysis'] = self._analyze_dof_configuration(primary_chain, chains)
            
            # Step 5: Anthropomorphic analysis (if robot arm)
            if analysis['robot_type'] == 'robot_arm':
                analysis['anthropomorphic_analysis'] = self._analyze_anthropomorphic_config(primary_chain)
                # Add transformability analysis for anthropomorphic formats
                transformability = self.can_transform_to_anthropomorphic_formats(primary_chain)
                analysis['anthropomorphic_transformability'] = transformability
                # Add reason for non-transformability if present
                if not transformability.get('can_transform_anthropomorphic'):
                    if transformability.get('issues'):
                        analysis['anthropomorphic_transformability_reason'] = ", ".join(transformability['issues'])
                    else:
                        analysis['anthropomorphic_transformability_reason'] = "Unknown reason"
            
            # Step 6: Generate solver recommendations
            analysis['solver_recommendations'] = self._generate_solver_recommendations(analysis)
        
        # Step 7: Calculate overall statistics
        analysis['statistics'].update(self._calculate_robot_statistics())
        
        return analysis
    
    def _find_all_kinematic_chains(self) -> List[Dict[str, Any]]:
        """Find all kinematic chains in the robot structure."""
        chains = []
        visited_links = set()
        
        def traverse_from_link(link: USDLink, current_chain: List[USDJoint], chain_id: int):
            """Recursively traverse kinematic chains."""
            if link in visited_links:
                return
            
            visited_links.add(link)
            
            for joint in link.joints:
                if joint.child_link:
                    new_chain = current_chain + [joint]
                    
                    # If this joint leads to multiple children, it's a branch point
                    child_joint_count = len(joint.child_link.joints)
                    
                    if child_joint_count == 0:
                        # End of chain - record it
                        chain_info = self._analyze_single_chain(new_chain, chain_id)
                        chains.append(chain_info)
                    else:
                        # Continue traversing
                        traverse_from_link(joint.child_link, new_chain, chain_id)
        
        # Start traversal from base link
        traverse_from_link(self.robot.base_link, [], 0)
        
        # Also find any disconnected chains (shouldn't normally happen but good to check)
        all_links = set(self.robot.links.values())
        unvisited_links = all_links - visited_links
        
        chain_id = len(chains)
        for link in unvisited_links:
            if not link.parent_joint:  # Another root
                traverse_from_link(link, [], chain_id)
                chain_id += 1
        
        return chains
    
    def _analyze_single_chain(self, joints: List[USDJoint], chain_id: int) -> Dict[str, Any]:
        """Analyze a single kinematic chain."""
        
        if not joints:
            return {'length': 0, 'joints': [], 'chain_id': chain_id}
        
        # Prune fixed joints at the end of the chain (end-effectors, flanges, etc.)
        pruned_joints = self._prune_end_effector_joints(joints)
        
        # Track original vs pruned information
        original_length = len(joints)
        pruned_length = len(pruned_joints)
        end_effector_joints = joints[pruned_length:] if pruned_length < original_length else []
        
        # Use pruned joints for main analysis
        analysis_joints = pruned_joints if pruned_joints else joints
        
        # Separate revolute and prismatic joints
        revolute_joints = [j for j in analysis_joints if j.joint_type == 'revolute']
        prismatic_joints = [j for j in analysis_joints if j.joint_type == 'prismatic']
        
        # Extract joint axes with cumulative transformations
        joint_axes, transformed_axes = self._calculate_chain_transformed_axes(analysis_joints)
        
        # Calculate reach and other metrics
        total_reach = 0.0
        for i, joint in enumerate(analysis_joints):
            if joint.child_link:
                # Get the transform matrix for this joint to calculate link length
                joint_matrix = joint.transform
                translation = joint_matrix.translation
                link_length = math.sqrt(translation[0]**2 + translation[1]**2 + translation[2]**2)
                total_reach += link_length
        
        # Determine chain type based on pruned joints
        chain_type = self._classify_chain_type(analysis_joints, joint_axes)
        
        # Detect grippers in the chain
        # Detect gripper configuration - for now, simplified detection on single chain
        gripper_info = {'has_gripper': False, 'gripper_type': 'none', 'gripper_dof': 0}
        
        return {
            'chain_id': chain_id,
            'length': pruned_length,  # Use pruned length as main length
            'original_length': original_length,
            'revolute_count': len(revolute_joints),
            'prismatic_count': len(prismatic_joints),
            'joint_names': [j.name for j in analysis_joints],
            'joint_types': [j.joint_type for j in analysis_joints],
            'joint_axes': joint_axes,
            'axis_sequence': joint_axes,
            'total_reach': total_reach,
            'start_link': analysis_joints[0].parent_link.name if analysis_joints and analysis_joints[0].parent_link else 'unknown',
            'end_link': analysis_joints[-1].child_link.name if analysis_joints and analysis_joints[-1].child_link else 'unknown',
            'chain_type': chain_type,
            'joints': analysis_joints,  # Keep reference to actual joint objects (pruned)
            'end_effector_joints': [j.name for j in end_effector_joints],
            'has_end_effector': len(end_effector_joints) > 0,
            'gripper_info': gripper_info
        }
    
    def _classify_chain_type(self, joints: List[USDJoint], joint_axes: List[str]) -> str:
        """Classify the type of kinematic chain."""
        
        revolute_count = sum(1 for j in joints if j.joint_type == 'revolute')
        prismatic_count = sum(1 for j in joints if j.joint_type == 'prismatic')
        fixed_count = sum(1 for j in joints if j.joint_type == 'fixed')
        
        # Check for gripper patterns first
        joint_names = [j.name.lower() for j in joints]
        gripper_keywords = ['finger', 'gripper', 'hand', 'thumb', 'index', 'middle']
        has_gripper_names = any(keyword in name for name in joint_names for keyword in gripper_keywords)
        
        # Gripper pattern: 7 revolute + fixed + prismatic
        if (revolute_count == 7 and fixed_count >= 1 and prismatic_count >= 1 and 
            (has_gripper_names or len(joints) >= 9)):
            return "7dof_arm_with_gripper"
        
        # Gripper pattern: 6 revolute + fixed + prismatic
        elif (revolute_count == 6 and fixed_count >= 1 and prismatic_count >= 1 and 
              (has_gripper_names or len(joints) >= 8)):
            return "6dof_arm_with_gripper"
        
        # Pure revolute chains
        elif prismatic_count == 0 and fixed_count == 0:
            if revolute_count == 6:
                return "6dof_arm"
            elif revolute_count == 7:
                return "7dof_arm"
            elif revolute_count == 4:
                return "4dof_arm"
            elif revolute_count == 3:
                return "3dof_leg"
            elif revolute_count == 2:
                return "2dof_simple"
            elif revolute_count == 1:
                return "1dof_simple"
            else:
                return f"{revolute_count}dof_revolute"
        
        # Pure prismatic chains
        elif revolute_count == 0 and fixed_count == 0:
            if prismatic_count == 3:
                return "3dof_linear"
            else:
                return f"{prismatic_count}dof_prismatic"
        
        # Mixed chains (include fixed joints in the description)
        else:
            components = []
            if revolute_count > 0:
                components.append(f"{revolute_count}r")
            if prismatic_count > 0:
                components.append(f"{prismatic_count}p")
            if fixed_count > 0:
                components.append(f"{fixed_count}f")
            
            return f"mixed_{'_'.join(components)}"
    
    def _classify_robot_type(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify the overall robot type based on kinematic chains."""
        
        if not chains:
            return {
                'robot_type': 'unknown',
                'robot_subtype': None,
                'confidence': 0.0,
                'classification_reasoning': ['No kinematic chains found']
            }
        
        # Count different chain types
        chain_types = [chain['chain_type'] for chain in chains]
        chain_lengths = [chain['length'] for chain in chains]
        total_chains = len(chains)
        
        reasoning = []
        
        # Robot arm detection (including gripper-equipped arms)
        arm_chains = [c for c in chains if 'arm' in c['chain_type']]
        gripper_arm_chains = [c for c in chains if 'arm_with_gripper' in c['chain_type']]
        
        if arm_chains or gripper_arm_chains:
            all_arm_chains = arm_chains + gripper_arm_chains
            longest_arm = max(all_arm_chains, key=lambda x: x['revolute_count'])
            
            # Determine if this is a gripper-equipped arm
            has_gripper = len(gripper_arm_chains) > 0
            gripper_info = ""
            if has_gripper:
                gripper_count = len(gripper_arm_chains)
                if gripper_count >= 2:
                    gripper_info = f" with {gripper_count}-finger gripper"
                else:
                    gripper_info = " with gripper"
            
            if longest_arm['revolute_count'] >= 6:
                subtype = f"{longest_arm['revolute_count']}dof"
                if has_gripper:
                    subtype += "_gripper"
                
                return {
                    'robot_type': 'robot_arm',
                    'robot_subtype': subtype,
                    'confidence': 0.9,
                    'classification_reasoning': [
                        f"Detected {longest_arm['revolute_count']}-DOF arm chain{gripper_info}",
                        f"Primary chain: {' -> '.join(longest_arm['joint_names'][:7])}" + ("..." if len(longest_arm['joint_names']) > 7 else "")
                    ]
                }
            elif longest_arm['revolute_count'] >= 4:
                return {
                    'robot_type': 'robot_arm',
                    'robot_subtype': 'limited_dof',
                    'confidence': 0.7,
                    'classification_reasoning': [
                        f"Detected {longest_arm['revolute_count']}-DOF arm chain (limited DOF){gripper_info}"
                    ]
                }
        
        # Quadruped detection
        leg_chains = [c for c in chains if 'leg' in c['chain_type'] or c['length'] == 3]
        if len(leg_chains) == 4:
            return {
                'robot_type': 'quadruped',
                'robot_subtype': 'standard',
                'confidence': 0.9,
                'classification_reasoning': [
                    f"Detected 4 leg-like chains of lengths: {[c['length'] for c in leg_chains]}",
                    "Classic quadruped configuration"
                ]
            }
        elif len(leg_chains) >= 2 and len(leg_chains) <= 6:
            return {
                'robot_type': 'multi_legged',
                'robot_subtype': f"{len(leg_chains)}_legs",
                'confidence': 0.8,
                'classification_reasoning': [
                    f"Detected {len(leg_chains)} leg-like chains",
                    "Multi-legged robot configuration"
                ]
            }
        
        # Humanoid detection (more complex - typically has arms and legs)
        if total_chains >= 4:
            arm_like = [c for c in chains if c['length'] >= 4 and c['length'] <= 7]
            leg_like = [c for c in chains if c['length'] == 3 or c['length'] == 6]
            
            if len(arm_like) >= 2 and len(leg_like) >= 2:
                return {
                    'robot_type': 'humanoid',
                    'robot_subtype': 'full_body',
                    'confidence': 0.8,
                    'classification_reasoning': [
                        f"Detected {len(arm_like)} arm-like chains and {len(leg_like)} leg-like chains",
                        "Humanoid robot configuration"
                    ]
                }
        
        # Linear/Gantry system detection
        linear_chains = [c for c in chains if 'prismatic' in c['chain_type'] or 'linear' in c['chain_type']]
        if linear_chains and len(linear_chains) == total_chains:
            return {
                'robot_type': 'gantry',
                'robot_subtype': 'linear_system',
                'confidence': 0.9,
                'classification_reasoning': [
                    "All chains are linear/prismatic",
                    "Gantry or linear positioning system"
                ]
            }
        
        # Mobile platform (if very few or simple joints)
        if total_chains <= 2 and all(c['length'] <= 2 for c in chains):
            return {
                'robot_type': 'mobile_platform',
                'robot_subtype': 'simple',
                'confidence': 0.6,
                'classification_reasoning': [
                    f"Simple structure with {total_chains} short chains",
                    "Likely mobile platform with minimal DOF"
                ]
            }
        
        # Complex/Unknown system
        return {
            'robot_type': 'complex_system',
            'robot_subtype': 'unknown',
            'confidence': 0.3,
            'classification_reasoning': [
                f"Complex structure with {total_chains} chains",
                f"Chain types: {list(set(chain_types))}",
                f"Chain lengths: {chain_lengths}",
                "Does not match standard robot patterns"
            ]
        }
    
    def _identify_primary_chain(self, chains: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Identify the primary kinematic chain (usually the longest or most important)."""
        
        if not chains:
            return None
        
        # For robot arms, prioritize 6-7 DOF chains (including gripper joints)
        arm_chains = [c for c in chains if 'arm' in c['chain_type']]
        if arm_chains:
            # Prefer 6-DOF or 7-DOF revolute joints (excluding gripper joints)
            six_seven_dof = [c for c in arm_chains if c['revolute_count'] in [6, 7]]
            if six_seven_dof:
                return max(six_seven_dof, key=lambda x: x['revolute_count'])
            else:
                return max(arm_chains, key=lambda x: x['revolute_count'])
        
        # For other robots, just take the longest chain
        return max(chains, key=lambda x: x['length'])
    
    def _analyze_dof_configuration(self, primary_chain: Optional[Dict[str, Any]], 
                                 all_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze degrees of freedom configuration."""
        
        dof_analysis = {
            'primary_chain_dof': 0,
            'total_dof': 0,
            'revolute_dof': 0,
            'prismatic_dof': 0,
            'is_6dof': False,
            'is_7dof': False,
            'has_redundancy': False,
            'dof_distribution': {}
        }
        
        if primary_chain:
            # Use revolute count for arm DOF analysis (excluding gripper)
            primary_dof = primary_chain['revolute_count'] if 'arm' in primary_chain['chain_type'] else primary_chain['length']
            dof_analysis['primary_chain_dof'] = primary_dof
            dof_analysis['is_6dof'] = primary_dof == 6
            dof_analysis['is_7dof'] = primary_dof == 7
            dof_analysis['has_redundancy'] = primary_dof > 6
        
        # Calculate total DOF across all chains
        total_revolute = sum(c['revolute_count'] for c in all_chains)
        total_prismatic = sum(c['prismatic_count'] for c in all_chains)
        
        dof_analysis['total_dof'] = total_revolute + total_prismatic
        dof_analysis['revolute_dof'] = total_revolute
        dof_analysis['prismatic_dof'] = total_prismatic
        
        # DOF distribution by chain length
        dof_dist = {}
        for chain in all_chains:
            length = chain['length']
            dof_dist[length] = dof_dist.get(length, 0) + 1
        dof_analysis['dof_distribution'] = dof_dist
        
        return dof_analysis
    
    def _analyze_anthropomorphic_config(self, primary_chain: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze anthropomorphic configuration for robot arms."""
        
        analysis = {
            'is_anthropomorphic': False,
            'is_anthropomorphic2': False,
            'has_spherical_wrist': False,
            'axis_sequence': [],
            'pattern_match_score': 0,
            'configuration_issues': [],
            'solver_compatibility': []
        }
        
        if not primary_chain:
            analysis['configuration_issues'].append("No primary chain found")
            return analysis
        
        # For gripper-equipped arms, only analyze the first 6 revolute joints
        chain_revolute_count = primary_chain['revolute_count'] if 'arm' in primary_chain['chain_type'] else primary_chain['length']
        
        if chain_revolute_count != 6:
            analysis['configuration_issues'].append(f"Not a 6-DOF revolute chain ({chain_revolute_count} DOF) - anthropomorphic analysis not applicable")
            return analysis
        
        # Extract only the revolute joints for anthropomorphic analysis
        revolute_joints = [jt for jt in primary_chain['joint_types'] if jt == 'revolute']
        revolute_axes = []
        
        axis_idx = 0
        for joint_type in primary_chain['joint_types']:
            if joint_type == 'revolute':
                if axis_idx < len(primary_chain['axis_sequence']):
                    revolute_axes.append(primary_chain['axis_sequence'][axis_idx])
                else:
                    revolute_axes.append('Z')  # Default
            axis_idx += 1
            if len(revolute_axes) == 6:  # Only take first 6 revolute joints
                break
        
        axis_sequence = revolute_axes[:6]  # Ensure we only have 6 axes
        analysis['axis_sequence'] = axis_sequence
        
        # Check for Anthropomorphic configuration: Z-Y-Y-Z-Y-Z
        anthropomorphic_pattern = ['Z', 'Y', 'Y', 'Z', 'Y', 'Z']
        anthro_matches = sum(1 for i, (a, b) in enumerate(zip(axis_sequence, anthropomorphic_pattern)) if a == b)
        
        # Check for Anthropomorphic2 configuration: Z-Y-Y-Y-Z-Y
        anthropomorphic2_pattern = ['Z', 'Y', 'Y', 'Y', 'Z', 'Y']
        anthro2_matches = sum(1 for i, (a, b) in enumerate(zip(axis_sequence, anthropomorphic2_pattern)) if a == b)
        
        analysis['pattern_match_score'] = max(anthro_matches, anthro2_matches) * 100 // 6
        
        # Perfect Anthropomorphic match
        if axis_sequence == anthropomorphic_pattern:
            analysis['is_anthropomorphic'] = True
            analysis['solver_compatibility'].append("Anthropomorphic IK Solver")
            
            # Check spherical wrist (joints 4,5,6 should intersect)
            wrist_check = self._check_spherical_wrist_from_chain(primary_chain, [3, 4, 5])
            analysis['has_spherical_wrist'] = wrist_check['is_spherical']
            
            if analysis['has_spherical_wrist']:
                analysis['solver_compatibility'].append("Fast Analytical IK")
            else:
                analysis['configuration_issues'].append("Joints 4,5,6 do not intersect (no spherical wrist)")
        
        # Perfect Anthropomorphic2 match
        elif axis_sequence == anthropomorphic2_pattern:
            analysis['is_anthropomorphic2'] = True
            analysis['solver_compatibility'].append("Anthropomorphic2 IK Solver")
            
            # Check parallel Y axes for joints 2,3,4
            parallel_check = self._check_parallel_y_axes_from_chain(primary_chain, [1, 2, 3])
            analysis['has_spherical_wrist'] = parallel_check  # Different meaning here
            
            if parallel_check:
                analysis['solver_compatibility'].append("Specialized Y-axis IK")
            else:
                analysis['configuration_issues'].append("Joints 2,3,4 do not have parallel Y axes")
        
        # Partial matches
        else:
            if anthro_matches >= 4:
                analysis['configuration_issues'].append(
                    f"Close to Anthropomorphic pattern ({anthro_matches}/6 matches). "
                    f"Current: {'-'.join(axis_sequence)}, Expected: {'-'.join(anthropomorphic_pattern)}"
                )
            elif anthro2_matches >= 4:
                analysis['configuration_issues'].append(
                    f"Close to Anthropomorphic2 pattern ({anthro2_matches}/6 matches). "
                    f"Current: {'-'.join(axis_sequence)}, Expected: {'-'.join(anthropomorphic2_pattern)}"
                )
            else:
                analysis['configuration_issues'].append(
                    f"Non-standard axis configuration: {'-'.join(axis_sequence)}"
                )
                analysis['solver_compatibility'].append("General Numerical IK (KDL, Trac-IK)")
        
        return analysis
    
    def _check_spherical_wrist_from_chain(self, chain: Dict[str, Any], joint_indices: List[int]) -> Dict[str, Any]:
        """Check if specified joints form a spherical wrist."""
        
        result = {'is_spherical': False, 'max_offset': float('inf')}
        
        if not chain.get('joints') or len(joint_indices) != 3:
            return result
        
        # Check joint offsets (d parameters would need to be calculated from DH parameters)
        # For now, we'll use a heuristic based on joint relative positions
        wrist_joints = []
        for idx in joint_indices:
            if idx < len(chain['joints']):
                joint = chain['joints'][idx]
                # Use the joint's local transform rather than world position
                joint_matrix = joint.transform
                translation = joint_matrix.translation
                offset = abs(translation[2])  # Z-offset approximation
                wrist_joints.append(offset)
        
        if wrist_joints:
            max_offset = max(wrist_joints)
            result['max_offset'] = max_offset
            result['is_spherical'] = max_offset < 0.01  # Less than 1cm
        
        return result
    
    def _check_parallel_y_axes_from_chain(self, chain: Dict[str, Any], joint_indices: List[int]) -> bool:
        """Check if specified joints have parallel Y axes."""
        
        if not chain.get('joints') or len(joint_indices) < 2:
            return False
        
        # Check if the specified joints all have Y axes
        y_axes_count = 0
        for idx in joint_indices:
            if idx < len(chain['axis_sequence']) and chain['axis_sequence'][idx] == 'Y':
                y_axes_count += 1
        
        return y_axes_count == len(joint_indices)
    
    def _generate_solver_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate IK solver recommendations based on analysis."""
        
        recommendations = []
        
        robot_type = analysis['robot_type']
        dof_analysis = analysis['dof_analysis']
        anthro_analysis = analysis.get('anthropomorphic_analysis', {})
        
        if robot_type == 'robot_arm':
            if dof_analysis['is_6dof']:
                if anthro_analysis.get('is_anthropomorphic'):
                    recommendations.extend(anthro_analysis.get('solver_compatibility', []))
                elif anthro_analysis.get('is_anthropomorphic2'):
                    recommendations.extend(anthro_analysis.get('solver_compatibility', []))
                else:
                    recommendations.append("General 6-DOF IK Solver (KDL)")
                    recommendations.append("Numerical IK with Jacobian methods")
            
            elif dof_analysis['is_7dof']:
                recommendations.append("7-DOF Redundant IK Solver")
                recommendations.append("Trac-IK (Track-based IK)")
                recommendations.append("Redundancy resolution required")
            
            else:
                recommendations.append("General Numerical IK Solver")
                recommendations.append("Custom IK solution may be needed")
        
        elif robot_type == 'quadruped':
            recommendations.append("Quadruped IK Solver")
            recommendations.append("Leg IK with foot position control")
            recommendations.append("Gait planning integration")
        
        elif robot_type == 'humanoid':
            recommendations.append("Humanoid Whole-Body IK")
            recommendations.append("Hierarchical IK (arms + legs)")
            recommendations.append("Balance and stability constraints")
        
        else:
            recommendations.append("Custom IK solution required")
            recommendations.append("General numerical methods")
        
        return recommendations
    
    def _calculate_robot_statistics(self) -> Dict[str, Any]:
        """Calculate overall robot statistics."""
        
        stats = {
            'total_links': len(self.robot.links),
            'total_joints': len(self.robot.joints),
            'joint_type_distribution': {},
            'base_link': self.robot.base_link.name if self.robot.base_link else None
        }
        
        # Joint type distribution
        joint_types = {}
        for joint in self.robot.joints.values():
            joint_type = joint.joint_type
            joint_types[joint_type] = joint_types.get(joint_type, 0) + 1
        
        stats['joint_type_distribution'] = joint_types
        
        return stats
    
    def _detect_gripper_configuration(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect gripper configurations in the robot chains.
        
        A gripper is typically characterized by:
        - Multiple parallel chains ending at the same point
        - Fixed joints followed by prismatic joints (finger movement)
        - Joint names containing 'finger', 'gripper', 'hand', etc.
        - Similar chain structures with small variations
        
        Returns:
            Dictionary with gripper detection results
        """
        gripper_analysis = {
            'has_gripper': False,
            'gripper_type': 'none',
            'gripper_chains': [],
            'gripper_dof': 0,
            'finger_count': 0,
            'gripper_location': 'none',  # 'end_effector', 'multi_point', etc.
            'gripper_characteristics': []
        }
        
        if not chains:
            return gripper_analysis
        
        # Look for potential gripper patterns
        gripper_candidates = self._find_gripper_candidates(chains)
        
        if not gripper_candidates:
            return gripper_analysis
        
        # Analyze the gripper candidates
        for gripper_group in gripper_candidates:
            gripper_info = self._analyze_gripper_group(gripper_group)
            
            if gripper_info['is_gripper']:
                gripper_analysis['has_gripper'] = True
                gripper_analysis['gripper_chains'].extend(gripper_info['chains'])
                gripper_analysis['gripper_dof'] += gripper_info['dof']
                gripper_analysis['finger_count'] += gripper_info['finger_count']
                
                # Determine gripper type
                if gripper_info['finger_count'] == 2:
                    gripper_analysis['gripper_type'] = 'parallel_gripper'
                elif gripper_info['finger_count'] == 3:
                    gripper_analysis['gripper_type'] = 'three_finger_gripper'
                elif gripper_info['finger_count'] > 3:
                    gripper_analysis['gripper_type'] = 'multi_finger_hand'
                else:
                    gripper_analysis['gripper_type'] = 'simple_gripper'
                
                gripper_analysis['gripper_characteristics'].extend(gripper_info['characteristics'])
        
        # Determine gripper location
        if gripper_analysis['has_gripper']:
            if len(gripper_analysis['gripper_chains']) >= 2:
                # Check if all gripper chains share the same base joints (end-effector gripper)
                base_joints = self._find_common_base_joints(gripper_analysis['gripper_chains'])
                if len(base_joints) >= 5:  # At least 5+ common joints suggests end-effector gripper
                    gripper_analysis['gripper_location'] = 'end_effector'
                else:
                    gripper_analysis['gripper_location'] = 'multi_point'
            else:
                gripper_analysis['gripper_location'] = 'single_point'
        
        return gripper_analysis
    
    def _find_gripper_candidates(self, chains: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find groups of chains that might form grippers."""
        gripper_candidates = []
        
        # Group chains by their common prefix (shared joints)
        chain_groups = {}
        
        for chain in chains:
            # Look for chains with gripper-like characteristics
            if self._has_gripper_characteristics(chain):
                # Find the longest common prefix with other chains
                for existing_prefix, group in chain_groups.items():
                    common_joints = self._find_common_prefix_length(chain['joint_names'], existing_prefix)
                    if common_joints >= 3:  # At least 3 common joints
                        group.append(chain)
                        break
                else:
                    # Create new group
                    prefix = tuple(chain['joint_names'][:7])  # Use first 7 joints as prefix
                    chain_groups[prefix] = [chain]
        
        # Filter groups that look like grippers (2+ chains with similar structure)
        for prefix, group in chain_groups.items():
            if len(group) >= 2:
                gripper_candidates.append(group)
        
        return gripper_candidates
    
    def _has_gripper_characteristics(self, chain: Dict[str, Any]) -> bool:
        """Check if a chain has gripper-like characteristics."""
        joint_names = [name.lower() for name in chain['joint_names']]
        joint_types = chain['joint_types']
        
        # Check for gripper-related keywords
        gripper_keywords = ['finger', 'gripper', 'hand', 'thumb', 'index', 'middle', 'ring', 'pinky']
        has_gripper_names = any(keyword in name for name in joint_names for keyword in gripper_keywords)
        
        # Check for typical gripper joint pattern (fixed + prismatic at the end)
        has_end_prismatic = len(joint_types) > 0 and joint_types[-1] == 'prismatic'
        has_fixed_joints = 'fixed' in joint_types
        
        # Check for parallel structure (chains ending with similar patterns)
        has_parallel_pattern = len(joint_types) >= 8 and joint_types[-3:] == ['fixed', 'fixed', 'prismatic']
        
        return has_gripper_names or (has_end_prismatic and has_fixed_joints) or has_parallel_pattern
    
    def _find_common_prefix_length(self, joints1: List[str], prefix_tuple: Tuple[str, ...]) -> int:
        """Find the length of common prefix between joint list and prefix tuple."""
        prefix_list = list(prefix_tuple)
        common_length = 0
        
        for i, (j1, j2) in enumerate(zip(joints1, prefix_list)):
            if j1 == j2:
                common_length += 1
            else:
                break
        
        return common_length
    
    def _analyze_gripper_group(self, gripper_group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a group of chains to determine gripper characteristics."""
        analysis = {
            'is_gripper': False,
            'chains': gripper_group,
            'dof': 0,
            'finger_count': 0,
            'characteristics': []
        }
        
        if len(gripper_group) < 2:
            return analysis
        
        # Check if chains have similar structure
        first_chain = gripper_group[0]
        base_joint_count = len([jt for jt in first_chain['joint_types'] if jt == 'revolute'])
        
        similar_structures = 0
        total_prismatic_dof = 0
        
        for chain in gripper_group:
            chain_base_joints = len([jt for jt in chain['joint_types'] if jt == 'revolute'])
            chain_prismatic_joints = len([jt for jt in chain['joint_types'] if jt == 'prismatic'])
            
            # Check if this chain has similar base structure
            if abs(chain_base_joints - base_joint_count) <= 1:  # Allow 1 joint difference
                similar_structures += 1
                total_prismatic_dof += chain_prismatic_joints
        
        # Determine if this is a gripper
        if similar_structures >= 2:  # At least 2 similar chains
            analysis['is_gripper'] = True
            analysis['finger_count'] = len(gripper_group)
            analysis['dof'] = total_prismatic_dof
            
            # Analyze characteristics
            if total_prismatic_dof > 0:
                analysis['characteristics'].append("actuated_fingers")
            
            if any('fixed' in chain['joint_types'] for chain in gripper_group):
                analysis['characteristics'].append("mechanical_coupling")
            
            # Check for parallel jaw gripper (2 fingers with similar motion)
            if len(gripper_group) == 2:
                analysis['characteristics'].append("parallel_jaw")
            elif len(gripper_group) == 3:
                analysis['characteristics'].append("three_finger")
            elif len(gripper_group) > 3:
                analysis['characteristics'].append("multi_finger")
        
        return analysis
    
    def _find_common_base_joints(self, gripper_chains: List[Dict[str, Any]]) -> List[str]:
        """Find common base joints across gripper chains."""
        if not gripper_chains:
            return []
        
        # Start with the first chain's joints
        common_joints = gripper_chains[0]['joint_names'][:]
        
        # Find intersection with all other chains
        for chain in gripper_chains[1:]:
            new_common = []
            for i, (j1, j2) in enumerate(zip(common_joints, chain['joint_names'])):
                if j1 == j2:
                    new_common.append(j1)
                else:
                    break
            common_joints = new_common
        
        return common_joints
    
    def get_analysis_report(self, analysis:Dict[str, Any]=None) -> str:
        """Get a comprehensive formatted analysis report."""
        if analysis is None:
            analysis = self.analyze_robot()
        
        report = f"USD Robot Analysis Report - {self.robot.name}\n"
        report += "=" * 60 + "\n\n"
        
        # Robot Classification
        report += "🤖 ROBOT CLASSIFICATION\n"
        report += "-" * 30 + "\n"
        report += f"Type: {analysis['robot_type'].replace('_', ' ').title()}\n"
        if analysis['robot_subtype']:
            report += f"Subtype: {analysis['robot_subtype'].replace('_', ' ').title()}\n"
        report += f"Confidence: {analysis['confidence']*100:.1f}%\n"
        
        if analysis.get('classification_reasoning'):
            report += "\nClassification reasoning:\n"
            for reason in analysis['classification_reasoning']:
                report += f"  • {reason}\n"
        
        # DOF Analysis
        report += "\n📊 DEGREES OF FREEDOM ANALYSIS\n"
        report += "-" * 35 + "\n"
        dof = analysis['dof_analysis']
        report += f"Primary chain DOF: {dof['primary_chain_dof']}\n"
        report += f"Total robot DOF: {dof['total_dof']}\n"
        report += f"  - Revolute joints: {dof['revolute_dof']}\n"
        report += f"  - Prismatic joints: {dof['prismatic_dof']}\n"
        
        if dof['is_6dof']:
            report += "✓ 6-DOF capable (standard manipulation)\n"
        elif dof['is_7dof']:
            report += "✓ 7-DOF capable (redundant manipulation)\n"
        
        if dof['has_redundancy']:
            report += "✓ Has kinematic redundancy\n"
        
        # Anthropomorphic Analysis (for robot arms)
        if analysis['robot_type'] == 'robot_arm':
            report += "\n🦾 ANTHROPOMORPHIC ANALYSIS\n"
            report += "-" * 32 + "\n"
            anthro = analysis['anthropomorphic_analysis']
            
            if anthro['is_anthropomorphic']:
                report += "✓ Anthropomorphic configuration detected\n"
                report += f"  Pattern: {'-'.join(anthro['axis_sequence'])}\n"
                if anthro['has_spherical_wrist']:
                    report += "✓ Spherical wrist confirmed\n"
            elif anthro['is_anthropomorphic2']:
                report += "✓ Anthropomorphic2 configuration detected\n"
                report += f"  Pattern: {'-'.join(anthro['axis_sequence'])}\n"
            else:
                report += "⚪ Non-standard configuration\n"
                report += f"  Pattern: {'-'.join(anthro['axis_sequence'])}\n"
                report += f"  Match score: {anthro['pattern_match_score']}%\n"
            
            if anthro['configuration_issues']:
                report += "\nConfiguration issues:\n"
                for issue in anthro['configuration_issues']:
                    report += f"  ⚠ {issue}\n"
        
        # Gripper Analysis
        gripper = analysis.get('gripper_analysis', {})
        if gripper.get('has_gripper'):
            report += "\n🤖 GRIPPER ANALYSIS\n"
            report += "-" * 20 + "\n"
            report += f"Gripper type: {gripper['gripper_type'].replace('_', ' ').title()}\n"
            report += f"Finger count: {gripper['finger_count']}\n"
            report += f"Gripper DOF: {gripper['gripper_dof']}\n"
            report += f"Location: {gripper['gripper_location'].replace('_', ' ').title()}\n"
            
            if gripper['gripper_characteristics']:
                report += f"Characteristics: {', '.join(gripper['gripper_characteristics'])}\n"
            
            if gripper['gripper_chains']:
                report += f"\nGripper chains: {len(gripper['gripper_chains'])}\n"
                for i, chain in enumerate(gripper['gripper_chains'][:3], 1):  # Show first 3
                    report += f"  Chain {i}: {len(chain['joint_names'])} joints\n"
        
        # Kinematic Chains
        if analysis['kinematic_chains']:
            report += "\n🔗 KINEMATIC CHAINS\n"
            report += "-" * 20 + "\n"
            for i, chain in enumerate(analysis['kinematic_chains'], 1):
                report += f"Chain {i}: {chain['length']} DOF ({chain['chain_type']})\n"
                
                # Show original length if joints were pruned
                if chain.get('original_length', 0) > chain['length']:
                    pruned_count = chain['original_length'] - chain['length']
                    report += f"  Original length: {chain['original_length']} (pruned {pruned_count} end-effector joints)\n"
                    if chain.get('end_effector_joints'):
                        report += f"  Pruned joints: {', '.join(chain['end_effector_joints'])}\n"
                
                report += f"  Joints: {' → '.join(chain['joint_names'])}\n"
                report += f"  Types: {'-'.join(chain['joint_types'])}\n"
                report += f"  Axes: {'-'.join(chain['axis_sequence'])}\n"
                report += f"  Reach: {chain['total_reach']:.3f} units\n"
                
                # Show gripper information if present
                if chain.get('gripper_info', {}).get('has_gripper'):
                    gripper = chain['gripper_info']
                    report += f"  Gripper: {gripper['finger_count']} fingers, {gripper['dof']} DOF\n"
                
                report += "\n"
        
        # Solver Recommendations
        if analysis['solver_recommendations']:
            report += "🎯 SOLVER RECOMMENDATIONS\n"
            report += "-" * 25 + "\n"
            for rec in analysis['solver_recommendations']:
                report += f"  • {rec}\n"
        
        # Statistics
        stats = analysis['statistics']
        report += "\n📈 STATISTICS\n"
        report += "-" * 12 + "\n"
        report += f"Total links: {stats['total_links']}\n"
        report += f"Total joints: {stats['total_joints']}\n"
        report += f"Kinematic chains: {stats['total_chains']}\n"
        
        if stats['joint_type_distribution']:
            report += "Joint distribution:\n"
            for joint_type, count in stats['joint_type_distribution'].items():
                report += f"  - {joint_type}: {count}\n"
        
        report += "\n" + "=" * 60
        
        return report
    
    def clear_cache(self) -> None:
        """Clear the analysis cache to force re-analysis."""
        self._analysis_cache = None
    
    def __str__(self) -> str:
        return self.get_analysis_report()
    
    def __repr__(self) -> str:
        analysis = self.analyze_robot()
        return f"USDRobotAnalysis(robot='{self.robot.name}', type='{analysis['robot_type']}', dof={analysis['dof_analysis']['total_dof']})"
    
    def _prune_end_effector_joints(self, joints: List[USDJoint]) -> List[USDJoint]:
        """
        Prune fixed joints at the end of kinematic chains as they typically represent
        end-effectors, flanges, or mounting points rather than actual DOF.
        
        Args:
            joints: List of joints in the chain
            
        Returns:
            List of joints with end-effector joints removed
        """
        if not joints:
            return joints
        
        # Find the last non-fixed joint
        last_active_idx = len(joints) - 1
        
        # Work backwards from the end, removing fixed joints
        for i in range(len(joints) - 1, -1, -1):
            joint = joints[i]
            
            # Stop pruning when we find a non-fixed joint
            if joint.joint_type in ['revolute', 'prismatic']:
                last_active_idx = i
                break
            
            # Also check for common end-effector naming patterns
            joint_name_lower = joint.name.lower()
            end_effector_keywords = [
                'flange', 'end_effector', 'ee', 'tool', 'tcp', 'tip',
                'mount', 'attachment', 'connector', 'tool_frame'
            ]
            
            # If it's a fixed joint with end-effector-like name, continue pruning
            if joint.joint_type == 'fixed' and any(keyword in joint_name_lower for keyword in end_effector_keywords):
                continue
            
            # If it's a fixed joint but doesn't have end-effector naming, be more conservative
            # Only prune if it's at the very end and there are active joints before it
            if joint.joint_type == 'fixed' and i == len(joints) - 1 and i > 0:
                # Check if the previous joint is active
                if joints[i-1].joint_type in ['revolute', 'prismatic']:
                    continue
            
            # Stop pruning if we can't clearly identify this as an end-effector
            last_active_idx = i
            break
        
        # Return the pruned chain (include the last active joint)
        pruned_joints = joints[:last_active_idx + 1]
        
        # Log what was pruned if anything
        if len(pruned_joints) < len(joints):
            pruned_names = [j.name for j in joints[len(pruned_joints):]]
            logger.debug(f"Pruned end-effector joints: {pruned_names}")
        
        return pruned_joints
    
    def can_transform_to_anthropomorphic_formats(self, primary_chain: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the robot's primary chain can be transformed to match the Anthropomorphic or Anthropomorphic2 solver requirements.
        Returns a dict with keys:
            - can_transform_anthropomorphic: bool
            - can_transform_anthropomorphic2: bool
            - required_axis_changes: list of (joint_idx, from_axis, to_axis)
            - issues: list of str
        """
        result = {
            'can_transform_anthropomorphic': False,
            'can_transform_anthropomorphic2': False,
            'required_axis_changes': [],
            'issues': []
        }
        if not primary_chain or primary_chain.get('revolute_count', 0) != 6:
            result['issues'].append('Primary chain is not a 6-DOF revolute arm.')
            return result

        axis_sequence = primary_chain['axis_sequence'][:6]
        anthropomorphic_pattern = ['Z', 'Y', 'Y', 'Z', 'Y', 'Z']
        anthropomorphic2_pattern = ['Z', 'Y', 'Y', 'Y', 'Z', 'Y']

        # Check for Anthropomorphic
        axis_changes = []
        for i, (cur, target) in enumerate(zip(axis_sequence, anthropomorphic_pattern)):
            if cur != target:
                axis_changes.append((i, cur, target))
        # Check wrist intersection (joints 4,5,6)
        wrist_ok = False
        wrist_check_needed = len(axis_changes) == 0
        if wrist_check_needed:
            wrist_check = self._check_spherical_wrist_from_chain(primary_chain, [3, 4, 5])
            wrist_ok = wrist_check['is_spherical']
            if not wrist_ok:
                result['issues'].append('Joints 4,5,6 do not intersect (no spherical wrist)')
        else:
            # If axis changes are needed, assume wrist can be made spherical after axis change
            wrist_ok = True
        # Updated logic: if only axis changes are needed (and wrist can be made spherical), allow transformation
        result['can_transform_anthropomorphic'] = (wrist_ok and (len(axis_changes) == 0 or len(axis_changes) > 0)) and not (wrist_check_needed and not wrist_ok)
        if len(axis_changes) > 0:
            result['required_axis_changes'].append({
                'to_format': 'Anthropomorphic',
                'changes': axis_changes
            })

        # Check for Anthropomorphic2
        axis_changes2 = []
        for i, (cur, target) in enumerate(zip(axis_sequence, anthropomorphic2_pattern)):
            if cur != target:
                axis_changes2.append((i, cur, target))
        # Check parallel Y axes for joints 2,3,4
        parallel_y_check_needed = len(axis_changes2) == 0
        parallel_y = False
        if parallel_y_check_needed:
            parallel_y = self._check_parallel_y_axes_from_chain(primary_chain, [1, 2, 3])
            if not parallel_y:
                result['issues'].append('Joints 2,3,4 do not have parallel Y axes')
        else:
            # If axis changes are needed, assume parallel Y axes can be achieved after axis change
            parallel_y = True
        result['can_transform_anthropomorphic2'] = (parallel_y and (len(axis_changes2) == 0 or len(axis_changes2) > 0)) and not (parallel_y_check_needed and not parallel_y)
        if len(axis_changes2) > 0:
            result['required_axis_changes'].append({
                'to_format': 'Anthropomorphic2',
                'changes': axis_changes2
            })
        return result
    
    def _calculate_chain_transformed_axes(self, joints: List[USDJoint]) -> Tuple[List[str], List[List[float]]]:
        """
        Calculate joint axes with cumulative transformations through the kinematic chain.
        
        Args:
            joints: List of joints in the kinematic chain
            
        Returns:
            Tuple of (axis_labels, transformed_axis_vectors)
        """
        joint_axes = []
        transformed_axes = []
        
        # Cumulative transformation from base to current joint
        cumulative_transform = HomogeneousMatrix.identity()
        
        for i, joint in enumerate(joints):
            # Get the local axis for this joint
            local_axis = joint.get_local_axis()
            if not local_axis:
                local_axis = [1.0, 0.0, 0.0]  # Default to X-axis
            
            # Get the joint's transformation matrix
            joint_transform = joint.transform
            
            # Update cumulative transformation
            cumulative_transform = cumulative_transform * joint_transform
            
            # Transform the local axis to world coordinates
            final_axis = cumulative_transform.transform_vector(local_axis)
            
            # Convert to primary axis label
            abs_axis = [abs(x) for x in final_axis]
            max_idx = abs_axis.index(max(abs_axis))
            axis_label = ['X', 'Y', 'Z'][max_idx]
            
            joint_axes.append(axis_label)
            transformed_axes.append(final_axis)
        
        return joint_axes, transformed_axes
