import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from .robot_structure import USDRobot, USDJoint, USDLink
from .math_utils import HomogeneousMatrix, XYZList_to_string

logger = logging.getLogger(__name__)

SOLVERS = {
	'anthromorphic': {
		'sequence': ['Z', 'Y', 'Y', 'Z', 'Y', 'Z'],
		'translate': [[True, True, True],
					  [True, False, True],
					  [True, False, True],
					  [False, False, True],
					  [False, False, True],
					  [False, False, True]]
	},
	'anthromorphic2': {
		'sequence': ['Z', 'Y', 'Z', 'Y', 'Z', 'Y'],
		'translate': [[False, False, True],
					  [False, False, True],
					  [False, False, True],
					  [False, True, False],
					  [False, False, True],
					  [False, True, False]]
	}
}

class BaseRobotAnalysis:
	"""
	Base class for robot analysis with common functionality.
	
	This class provides the foundation for analyzing USD robot structures and determining
	robot type, configuration, and characteristics.
	"""

	def __init__(self, robot: USDRobot):
		self.robot = robot
		self._analysis_cache: Optional[Dict[str, Any]] = None

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
		if self.robot.base_link:
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
		joint_axes = [joint.get_axis() for joint in analysis_joints]
		
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
		
		# All joints align
		aligned_joints = self._validate_aligned_joints({'joints': analysis_joints})
		
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
			'aligned_joints': aligned_joints,
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
		
		# Robot arm patterns (prioritize revolute joints, fixed joints are structural)
		elif revolute_count >= 6 and prismatic_count == 0:
			if revolute_count == 6:
				return "6dof_arm"
			elif revolute_count == 7:
				return "7dof_arm"
			else:
				return f"{revolute_count}dof_arm"
		
		# Lower DOF robot arm patterns
		elif revolute_count >= 4 and revolute_count < 6 and prismatic_count == 0:
			if revolute_count == 4:
				return "4dof_arm"
			elif revolute_count == 5:
				return "5dof_arm"
			else:
				return f"{revolute_count}dof_arm"
		
		# Leg patterns (typically 3 DOF)
		elif revolute_count == 3 and prismatic_count == 0 and fixed_count <= 1:
			return "3dof_leg"
		
		# Simple patterns with low DOF
		elif revolute_count == 2 and prismatic_count == 0:
			return "2dof_simple"
		elif revolute_count == 1 and prismatic_count == 0:
			return "1dof_simple"
		
		# Pure prismatic chains
		elif revolute_count == 0 and fixed_count == 0:
			if prismatic_count == 3:
				return "3dof_linear"
			else:
				return f"{prismatic_count}dof_prismatic"
		
		# Mixed chains with prismatic joints
		elif prismatic_count > 0 and revolute_count > 0:
			components = []
			if revolute_count > 0:
				components.append(f"{revolute_count}r")
			if prismatic_count > 0:
				components.append(f"{prismatic_count}p")
			if fixed_count > 0:
				components.append(f"{fixed_count}f")
			
			return f"mixed_{'_'.join(components)}"
		
		# Fallback for other patterns
		else:
			components = []
			if revolute_count > 0:
				components.append(f"{revolute_count}r")
			if prismatic_count > 0:
				components.append(f"{prismatic_count}p")
			if fixed_count > 0:
				components.append(f"{fixed_count}f")
			
			return f"mixed_{'_'.join(components)}"

	def _validate_aligned_joints(self, chain: Dict[str, Any]) -> List[bool]:
		"""
		Validates which joint frames in the chain have the same axis direction as the reference (first) joint.
		
		This function checks each joint in a kinematic chain to see if it has a parallel axis
		to the reference joint (first joint), which is important for determining solver 
		compatibility and configuration validity.
		
		Args:
			chain: Dictionary containing chain information with 'joints' key
			
		Returns:
			List[bool]: List of booleans, one for each joint, indicating if that joint's 
					   axis is aligned with the reference joint's axis
		"""
		if not chain or 'joints' not in chain:
			return []
		
		joints = chain['joints']
		if len(joints) == 0:
			return []
		
		if len(joints) == 1:
			return [True]  # Single joint is considered aligned with itself
		
		alignment_joints = []
		from .math_utils import quaternion_to_rotation_matrix, normalize_vector
		for joint in joints:
			world_translate, world_quat = joint.get_world_pose()
			world_quat = normalize_vector(world_quat)
			if np.allclose(world_quat, [0, 0, 0, 1], atol=1e-6):
				alignment_joints.append(True)
			else:
				alignment_joints.append(False)
				
		return alignment_joints

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

	def clear_cache(self) -> None:
		"""Clear the analysis cache to force re-analysis."""
		self._analysis_cache = None

	def _find_joint_by_name(self, joint_name: str) -> Optional[USDJoint]:
		"""Find a joint by name in the robot structure."""
		return self.robot.joints.get(joint_name)

	def _has_tool_characteristics(self, joint: USDJoint) -> bool:
		"""Check if a joint has tool-like characteristics."""
		joint_name_lower = joint.name.lower()
		tool_keywords = [
			'tool', 'tcp', 'tip', 'end_effector', 'ee', 'flange',
			'mount', 'attachment', 'connector', 'tool_frame'
		]
		
		return any(keyword in joint_name_lower for keyword in tool_keywords)

	def _analyze_tool_characteristics(self, joint: USDJoint) -> List[str]:
		"""Analyze tool characteristics based on joint properties."""
		characteristics = []
		joint_name_lower = joint.name.lower()
		
		# Tool type detection based on naming
		if 'tcp' in joint_name_lower or 'tool_center_point' in joint_name_lower:
			characteristics.append('tool_center_point')
		elif 'flange' in joint_name_lower:
			characteristics.append('mounting_flange')
		elif 'gripper' in joint_name_lower:
			characteristics.append('gripper_mount')
		elif 'welder' in joint_name_lower or 'welding' in joint_name_lower:
			characteristics.append('welding_tool')
		elif 'camera' in joint_name_lower:
			characteristics.append('vision_system')
		elif 'sensor' in joint_name_lower:
			characteristics.append('sensor_mount')
		
		# Joint type characteristics
		if joint.joint_type == 'fixed':
			characteristics.append('fixed_mount')
		elif joint.joint_type == 'revolute':
			characteristics.append('rotational_tool')
		elif joint.joint_type == 'prismatic':
			characteristics.append('linear_tool')
		
		# Transform characteristics
		local_transform = joint.transform
		translation_magnitude = (local_transform.translation[0]**2 + 
								local_transform.translation[1]**2 + 
								local_transform.translation[2]**2)**0.5
		
		if translation_magnitude > 0.001:  # More than 1mm offset
			characteristics.append('offset_tool')
		else:
			characteristics.append('centered_tool')
		
		return characteristics

class USDRobotAnalysis:
	"""
	Factory class to create appropriate robot analysis instances based on robot type.
	
	This class analyzes the robot structure to determine robot type and creates
	the appropriate specialized analysis class.
	"""

	def __init__(self, robot: USDRobot):
		self.robot = robot
		self._analysis_cache: Optional[Dict[str, Any]] = None
		self._specialized_analyzer: Optional[BaseRobotAnalysis] = None
		
	def analyze_robot(self) -> Dict[str, Any]:
		"""
		Perform comprehensive analysis of the robot structure.
		
		Returns:
			Dictionary with complete robot analysis including:
			- robot_type: Detected robot type (arm, quadruped, humanoid, etc.)
			- kinematic_chains: Analysis of all kinematic chains
			- dof_analysis: Degrees of freedom analysis
		"""
		if self._analysis_cache is None:
			# First determine robot type using basic analysis
			base_analyzer = BaseRobotAnalysis(self.robot)
			chains = base_analyzer._find_all_kinematic_chains()
			robot_type_analysis = self._classify_robot_type(chains)
			
			# Create specialized analyzer based on robot type
			if robot_type_analysis['robot_type'] == 'robot_arm':
				self._specialized_analyzer = RobotArmAnalysis(self.robot)
			elif robot_type_analysis['robot_type'] == 'quadruped':
				self._specialized_analyzer = QuadrupedAnalysis(self.robot)
			else:
				self._specialized_analyzer = UnsupportedRobotAnalysis(self.robot)
			
			# Perform specialized analysis
			self._analysis_cache = self._specialized_analyzer.analyze_robot()
		
		return self._analysis_cache

	def _classify_robot_type(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Classify the overall robot type based on kinematic chains."""
		
		if not chains:
			return {
				'robot_type': 'unknown',
				'robot_subtype': None,
				'confidence': 0.0,
				'classification_reasoning': ['No kinematic chains found']
			}
   
		print(chains)
		
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

	def get_analysis_report(self, analysis: Dict[str, Any] = None) -> str:
		"""Get a comprehensive formatted analysis report."""
		if analysis is None:
			analysis = self.analyze_robot()
		
		if self._specialized_analyzer:
			return self._specialized_analyzer.get_analysis_report(analysis)
		else:
			return "No specialized analyzer available for this robot type."

	def clear_cache(self) -> None:
		"""Clear the analysis cache to force re-analysis."""
		self._analysis_cache = None
		if self._specialized_analyzer:
			self._specialized_analyzer.clear_cache()

	def __str__(self) -> str:
		return self.get_analysis_report()

	def __repr__(self) -> str:
		analysis = self.analyze_robot()
		return f"USDRobotAnalysis(robot='{self.robot.name}', type='{analysis['robot_type']}', dof={analysis['dof_analysis']['total_dof']})"


class RobotArmAnalysis(BaseRobotAnalysis):
	"""
	Specialized analysis class for robot arms.
	
	This class provides detailed analysis specific to robot arms including:
	- Anthropomorphic configurations
	- DOF analysis (6-DOF, 7-DOF, etc.)
	- Solver compatibility recommendations
	- Gripper detection and analysis
	- Tool offset analysis
	"""

	def analyze_robot(self) -> Dict[str, Any]:
		"""Perform comprehensive analysis of the robot arm structure."""
		if self._analysis_cache is None:
			self._analysis_cache = self._perform_robot_analysis()
		return self._analysis_cache

	def _perform_robot_analysis(self) -> Dict[str, Any]:
		"""Perform the actual robot arm analysis."""
		
		analysis = {
			'robot_type': 'robot_arm',
			'robot_subtype': None,
			'confidence': 0.0,
			'kinematic_chains': [],
			'primary_chain': None,
			'dof_analysis': {},
			'solver_analysis': {},
			'gripper_analysis': {},
			'tool_analysis': {},
			'statistics': {},
			'warnings': []
		}
		
		if not self.robot.base_link:
			analysis['warnings'].append("No base link found - cannot analyze robot structure")
			return analysis
		
		# Step 1: Analyze kinematic structure
		chains = self._find_all_kinematic_chains()
		analysis['kinematic_chains'] = chains
		
		# Step 2: Determine robot arm subtype based on structure
		robot_type_analysis = self._classify_robot_arm_type(chains)
		analysis.update(robot_type_analysis)
		
		# Step 3: Detect gripper configurations
		gripper_analysis = self._detect_gripper_configuration(chains)
		analysis['gripper_analysis'] = gripper_analysis
		
		# Step 4: Analyze tool offsets in world frame
		tool_analysis = self._analyze_tool_offsets(chains)
		analysis['tool_analysis'] = tool_analysis
		
		# Step 5: Analyze primary kinematic chain
		if chains:
			primary_chain = self._identify_primary_chain(chains)
			analysis['primary_chain'] = primary_chain
			
			# Step 6: DOF analysis
			analysis['dof_analysis'] = self._analyze_dof_configuration(primary_chain, chains)
			
			# Step 7: Anthropomorphic analysis
			analysis['solver_analysis'] = self._analyze_solvers_config(primary_chain)
		
		# Step 8: Calculate overall statistics
		analysis['statistics'] = self._calculate_robot_statistics()
		
		return analysis

	def _classify_robot_arm_type(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Classify the robot arm type based on kinematic chains."""
		
		if not chains:
			return {
				'robot_subtype': None,
				'confidence': 0.0,
				'classification_reasoning': ['No kinematic chains found']
			}
		
		# Find arm chains (including gripper-equipped arms)
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
					'robot_subtype': subtype,
					'confidence': 0.9,
					'classification_reasoning': [
						f"Detected {longest_arm['revolute_count']}-DOF arm chain{gripper_info}",
						f"Primary chain: {' -> '.join(longest_arm['joint_names'][:7])}" + ("..." if len(longest_arm['joint_names']) > 7 else "")
					]
				}
			elif longest_arm['revolute_count'] >= 4:
				return {
					'robot_subtype': 'limited_dof',
					'confidence': 0.7,
					'classification_reasoning': [
						f"Detected {longest_arm['revolute_count']}-DOF arm chain (limited DOF){gripper_info}"
					]
				}
		
		return {
			'robot_subtype': 'unknown',
			'confidence': 0.3,
			'classification_reasoning': ['No clear arm patterns detected']
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

	def _analyze_solvers_config(self, primary_chain: Dict[str, Any]) -> Dict[str, Any]:
		"""Analyze solver configuration for robot arms."""
		
		analysis = {
			'solvers': {},
			'axis_sequence': [],
			'joint_local_transforms': [],
			'joint_world_transforms': [],
			'configuration_issues': []
		}
		
		if not primary_chain:
			analysis['configuration_issues'].append("No primary chain found")
			return analysis
		
		# For gripper-equipped arms, only analyze the first 6 revolute joints
		chain_revolute_count = primary_chain['revolute_count'] if 'arm' in primary_chain['chain_type'] else primary_chain['length']
		
		# Extract only the revolute joints for anthropomorphic analysis
		revolute_joints = [joint for joint in primary_chain['joints'] if joint.joint_type == 'revolute']
		
		aligned_joints = self._validate_aligned_joints({'joints': revolute_joints})
		if False in aligned_joints:
			analysis['configuration_issues'].append("One or more joints are not aligned - cannot be a solved")
			return analysis
		
		joint_axes = [XYZList_to_string(joint.get_axis()) for joint in revolute_joints]
		joint_world_transforms = [np.round(joint.get_world_pose()[0], 3).tolist() for joint in revolute_joints]
		joint_world_transforms = [[0, 0, 0]] + joint_world_transforms
		joint_local_transforms = [[None, None, None] for joint in revolute_joints]  # Placeholder for local transforms
		
		for i in range(0, len(joint_axes)):
			joint_local_transforms[i] = [joint_world_transforms[i+1][j]-joint_world_transforms[i][j] for j in range(3)]  # Placeholder for local transforms
		
		analysis['axis_sequence'] = joint_axes
		analysis['joint_local_transforms'] = joint_local_transforms
		analysis['joint_world_transforms'] = joint_world_transforms
		
		for solver in SOLVERS:
			sequence = SOLVERS[solver]['sequence']
			translate = SOLVERS[solver]['translate']
			
			if len(sequence) != len(joint_axes):
				analysis['configuration_issues'].append(f"Solver '{solver}' sequence length mismatch ({len(sequence)} vs {len(joint_axes)})")
				continue
			
			# Check if the joint axes match the expected sequence
			analysis['solvers'][solver] = {}
			sequence_match = all(joint_axes[i] == sequence[i] for i in range(len(sequence)))
			analysis['solvers'][solver]['sequence'] = sequence_match
			
			# Check translation compatibility
			translate_compatible = True
			for i in range(len(translate)):
				for j in range(3):
					translate_compatible &= not (translate[i][j] == False and joint_local_transforms[i][j]!=0)
		
			analysis['solvers'][solver]['translate'] = translate_compatible
			
			# If sequence doesn't match, check if it can be transformed via base pose orientation
			if not sequence_match:
				transform_result = self._check_base_pose_transformability(joint_axes, sequence, joint_local_transforms, translate)
				analysis['solvers'][solver]['transformable'] = transform_result['transformable']
				if transform_result['transformable']:
					analysis['solvers'][solver]['required_orientation'] = transform_result['orientation']
					analysis['solvers'][solver]['transform_translate_compatible'] = transform_result['translate_compatible']
				else:
					analysis['solvers'][solver]['transformable'] = False
				analysis['configuration_issues'].append(f"Joint axes do not match solver '{solver}' sequence")
			else:
				analysis['solvers'][solver]['transformable'] = False  # No transformation needed
			
			analysis['solvers'][solver]['compatible'] = sequence_match and translate_compatible
		return analysis

	def _check_base_pose_transformability(self, joint_axes: List[str], target_sequence: List[str], 
										  joint_local_transforms: List[List[float]], translate: List[List[bool]]) -> Dict[str, Any]:
		"""
		Check if joint axes can be transformed to match target sequence by changing base pose orientation.
		
		Args:
			joint_axes: Current joint axis sequence
			target_sequence: Target solver sequence
			joint_local_transforms: Current joint local transforms
			translate: Translation compatibility matrix from solver
			
		Returns:
			Dictionary with transformability information
		"""
		result = {
			'transformable': False,
			'orientation': None,
			'translate_compatible': False
		}
		
		if len(joint_axes) != len(target_sequence):
			return result
		
		# Define possible base pose orientations and their axis transformations
		# These represent 90-degree rotations around X, Y, Z axes
		orientation_transforms = {
			'identity': {'X': 'X', 'Y': 'Y', 'Z': 'Z'},
			'rot_x_90': {'X': 'X', 'Y': 'Z', 'Z': 'Y'},
			'rot_x_180': {'X': 'X', 'Y': 'Y', 'Z': 'Z'},
			'rot_x_270': {'X': 'X', 'Y': 'Z', 'Z': 'Y'},
			'rot_y_90': {'X': 'Z', 'Y': 'Y', 'Z': 'X'},
			'rot_y_180': {'X': 'X', 'Y': 'Y', 'Z': 'Z'},
			'rot_y_270': {'X': 'Z', 'Y': 'Y', 'Z': 'X'},
			'rot_z_90': {'X': 'Y', 'Y': 'X', 'Z': 'Z'},
			'rot_z_180': {'X': 'X', 'Y': 'Y', 'Z': 'Z'},
			'rot_z_270': {'X': 'Y', 'Y': 'X', 'Z': 'Z'},
		}
		
		# Try each orientation transformation
		for orientation_name, transform_map in orientation_transforms.items():
			# Transform the current joint axes
			transformed_axes = []
			for axis in joint_axes:
				if axis in transform_map:
					transformed_axes.append(transform_map[axis])
				else:
					transformed_axes.append(axis)  # Fallback
			
			# Check if transformed axes match target sequence
			if transformed_axes == target_sequence:
				# Check if the transformed local transforms are still compatible
				translate_compatible = self._check_transformed_translate_compatibility(
					joint_local_transforms, translate, orientation_name
				)
				
				result['transformable'] = True
				result['orientation'] = orientation_name
				result['translate_compatible'] = translate_compatible
				break
		
		return result

	def _check_transformed_translate_compatibility(self, joint_local_transforms: List[List[float]], 
												   translate: List[List[bool]], orientation: str) -> bool:
		"""
		Check if local transforms are still compatible after base pose transformation.
		
		Args:
			joint_local_transforms: Current joint local transforms
			translate: Translation compatibility matrix
			orientation: Applied orientation transformation
			
		Returns:
			True if transforms are compatible after transformation
		"""
		if orientation == 'identity':
			# No transformation needed, use original compatibility
			translate_compatible = True
			for i in range(len(translate)):
				for j in range(3):
					translate_compatible &= not (translate[i][j] == False and abs(joint_local_transforms[i][j]) > 1e-6)
			return translate_compatible
		
		# Define transformation matrices for each orientation
		transform_matrices = {
			'rot_x_90': [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
			'rot_x_180': [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
			'rot_x_270': [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
			'rot_y_90': [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
			'rot_y_180': [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
			'rot_y_270': [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
			'rot_z_90': [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
			'rot_z_180': [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
			'rot_z_270': [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
		}
		
		if orientation not in transform_matrices:
			return False
		
		transform_matrix = transform_matrices[orientation]
		
		# Apply transformation to each joint's local transform
		translate_compatible = True
		for i in range(len(translate)):
			if i >= len(joint_local_transforms):
				continue
				
			# Transform the local translation vector
			original_translation = joint_local_transforms[i]
			transformed_translation = [
				sum(transform_matrix[j][k] * original_translation[k] for k in range(3))
				for j in range(3)
			]
			
			# Check compatibility with transformed translation
			for j in range(3):
				translate_compatible &= not (translate[i][j] == False and abs(transformed_translation[j]) > 1e-6)
		
		return translate_compatible

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
	
	def _analyze_tool_offsets(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""
		Analyze tool offsets in world frame for robot chains.
		
		This method identifies tool attachment points and calculates their offsets
		from the end of the kinematic chain in world coordinates.
		
		Args:
			chains: List of kinematic chains
			
		Returns:
			Dictionary with tool analysis results
		"""
		tool_analysis = {
			'has_tools': False,
			'tool_count': 0,
			'tools': [],
			'primary_tool': None,
			'tool_characteristics': []
		}
		
		if not chains:
			return tool_analysis
		
		# Analyze each chain for potential tool offsets
		for chain in chains:
			tool_info = self._analyze_chain_tool_offset(chain)
			if tool_info['has_tool']:
				tool_analysis['has_tools'] = True
				tool_analysis['tool_count'] += 1
				tool_analysis['tools'].append(tool_info)
				
				# Set primary tool (first one found, or longest chain)
				if (tool_analysis['primary_tool'] is None or 
					tool_info['chain_length'] > tool_analysis['primary_tool']['chain_length']):
					tool_analysis['primary_tool'] = tool_info
				
				# Collect characteristics
				tool_analysis['tool_characteristics'].extend(tool_info['characteristics'])
		
		# Remove duplicate characteristics
		tool_analysis['tool_characteristics'] = list(set(tool_analysis['tool_characteristics']))
		
		return tool_analysis

	def _analyze_chain_tool_offset(self, chain: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Analyze a single chain for tool offset information.
		
		Args:
			chain: Kinematic chain dictionary
			
		Returns:
			Dictionary with tool offset analysis for this chain
		"""
		tool_info = {
			'has_tool': False,
			'chain_id': chain.get('chain_id', 0),
			'chain_length': chain.get('length', 0),
			'tool_offset_world': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0, 1.0]},
			'tool_offset_local': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.0, 1.0]},
			'tool_frame_name': None,
			'end_effector_joint': None,
			'characteristics': []
		}
		
		# Get the chain's joints
		joints = chain.get('joints', [])
		if not joints:
			return tool_info
		
		# Check for end-effector joints (these were pruned but contain tool info)
		end_effector_joints = chain.get('end_effector_joints', [])
		has_end_effector = chain.get('has_end_effector', False)
		
		# If we have end-effector joints, analyze them for tool offset
		if has_end_effector and end_effector_joints:
			tool_info['has_tool'] = True
			tool_info['end_effector_joint'] = end_effector_joints[-1]  # Last pruned joint
			tool_info['characteristics'].append('end_effector_mounted')
			
			# Find the actual joint object for the end effector
			end_effector_joint = self._find_joint_by_name(end_effector_joints[-1])
			if end_effector_joint:
				tool_info['tool_frame_name'] = end_effector_joint.name
				
				# Get tool offset in world frame
				world_transform = end_effector_joint.get_world_transform()
				tool_info['tool_offset_world']['position'] = world_transform.translation
				tool_info['tool_offset_world']['orientation'] = world_transform.euler_angles
				
				# Get tool offset in local frame (relative to parent)
				local_transform = end_effector_joint.transform
				tool_info['tool_offset_local']['position'] = local_transform.translation
				tool_info['tool_offset_local']['orientation'] = local_transform.euler_angles
				
				# Analyze tool characteristics based on naming and structure
				tool_info['characteristics'].extend(self._analyze_tool_characteristics(end_effector_joint))
		
		# If no end-effector joints, check if the last joint has tool-like characteristics
		elif joints:
			last_joint = joints[-1]
			if self._has_tool_characteristics(last_joint):
				tool_info['has_tool'] = True
				tool_info['tool_frame_name'] = last_joint.name
				tool_info['characteristics'].append('integrated_tool')
				
				# Get tool offset in world frame
				world_transform = last_joint.get_world_transform()
				tool_info['tool_offset_world']['position'] = world_transform.translation
				tool_info['tool_offset_world']['orientation'] = world_transform.euler_angles
				
				# Get tool offset in local frame
				local_transform = last_joint.transform
				tool_info['tool_offset_local']['position'] = local_transform.translation
				tool_info['tool_offset_local']['orientation'] = local_transform.euler_angles
				
				# Analyze tool characteristics
				tool_info['characteristics'].extend(self._analyze_tool_characteristics(last_joint))
		
		return tool_info

	def get_analysis_report(self, analysis:Dict[str, Any]=None) -> str:
		"""Get a comprehensive formatted analysis report."""
		if analysis is None:
			analysis = self.analyze_robot()
		
		report = f"Robot Arm Analysis Report - {self.robot.name}\n"
		report += "=" * 60 + "\n\n"
		
		# Robot Classification
		report += "🤖 ROBOT ARM CLASSIFICATION\n"
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
		
		# Tool Analysis
		tool_analysis = analysis.get('tool_analysis', {})
		if tool_analysis.get('has_tools'):
			report += "\n🔧 TOOL ANALYSIS\n"
			report += "-" * 15 + "\n"
			report += f"Tool count: {tool_analysis['tool_count']}\n"
			
			if tool_analysis['primary_tool']:
				primary = tool_analysis['primary_tool']
				report += f"Primary tool: {primary['tool_frame_name']}\n"
				report += f"  Chain length: {primary['chain_length']} DOF\n"
				
				# World frame position and orientation
				world_pos = primary['tool_offset_world']['position']
				world_rot = primary['tool_offset_world']['orientation']
				report += f"  World position: [{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}]\n"
				report += f"  World orientation: [{world_rot[0]:.3f}, {world_rot[1]:.3f}, {world_rot[2]:.3f}]\n"
				
				# Local frame position and orientation
				local_pos = primary['tool_offset_local']['position']
				local_rot = primary['tool_offset_local']['orientation']
				report += f"  Local position: [{local_pos[0]:.3f}, {local_pos[1]:.3f}, {local_pos[2]:.3f}]\n"
				report += f"  Local orientation: [{local_rot[0]:.3f}, {local_rot[1]:.3f}, {local_rot[2]:.3f}]\n"
				
				if primary['characteristics']:
					report += f"  Characteristics: {', '.join(primary['characteristics'])}\n"
		
		# Solver Analysis
		report += "\n🔧 SOLVER ANALYSIS\n"
		report += "-" * 30 + "\n"
		solver_analysis = analysis.get('solver_analysis', {})
		if solver_analysis and solver_analysis.get('solvers'):
			report += "Solver compatibility:\n"
			for solver, details in solver_analysis['solvers'].items():
				status = "✔" if details.get('compatible') else "✘"
				report += f"  {status} {solver}\n"
				report += f"    Sequence match: {details.get('sequence')}\n"
				report += f"    Translate compatible: {details.get('translate', 'N/A')}\n"
				
				# Add transformability information
				if details.get('transformable'):
					report += f"    🔄 Transformable via base pose orientation: {details.get('required_orientation')}\n"
					if details.get('transform_translate_compatible'):
						report += f"    ✓ Transform translate compatible\n"
					else:
						report += f"    ✗ Transform translate incompatible\n"
				elif not details.get('sequence'):
					report += f"    ✗ Not transformable via base pose orientation\n"
					
			report += f"Axis sequence: {solver_analysis.get('axis_sequence', [])}\n"
		
		if solver_analysis.get('configuration_issues'):
			report += "\nConfiguration issues:\n"
			for issue in solver_analysis['configuration_issues']:
				report += f"  ⚠ {issue}\n"
		
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
				report += f"  Axes: {'-'.join(str(axis) for axis in chain['axis_sequence'])}\n"
				report += f"  Reach: {chain['total_reach']:.3f} units\n"

				# Add joint alignment to worldframe info
				if 'aligned_joints' in chain and chain['aligned_joints']:
					aligned_count = sum(chain['aligned_joints'])
					total_joints = len(chain['aligned_joints'])
					report += f"  Joints aligned to worldframe: {aligned_count}/{total_joints}\n"
					if aligned_count == total_joints:
						report += "    ✓ All joints aligned to worldframe\n"
					elif aligned_count == 0:
						report += "    ✗ No joints aligned to worldframe\n"
					else:
						report += "    ⚠ Some joints not aligned to worldframe\n"
				
				# Show gripper information if present
				if chain.get('gripper_info', {}).get('has_gripper'):
					gripper = chain['gripper_info']
					report += f"  Gripper: {gripper['finger_count']} fingers, {gripper['dof']} DOF\n"
				
				report += "\n"
		
		# Statistics
		stats = analysis['statistics']
		report += "\n📈 STATISTICS\n"
		report += "-" * 12 + "\n"
		report += f"Total links: {stats['total_links']}\n"
		report += f"Total joints: {stats['total_joints']}\n"
		report += f"Kinematic chains: {len(analysis['kinematic_chains'])}\n"
		report += f"Base link: {stats['base_link']}\n"
		
		if analysis.get('tool_analysis', {}).get('has_tools'):
			report += f"Has tool(s): Yes\n"
		else:
			report += f"Has tool(s): No\n"
		
		if analysis.get('gripper_analysis', {}).get('has_gripper'):
			report += f"Has gripper: Yes\n"
		else:
			report += f"Has gripper: No\n"
		
		if stats['joint_type_distribution']:
			report += "Joint distribution:\n"
			for joint_type, count in stats['joint_type_distribution'].items():
				report += f"  - {joint_type}: {count}\n"
		
		report += "\n" + "=" * 60
		
		return report


class QuadrupedAnalysis(BaseRobotAnalysis):
	"""
	Specialized analysis class for quadruped robots.
	
	This class provides detailed analysis specific to quadruped robots including:
	- Leg configuration analysis
	- Gait analysis capabilities
	- Stability analysis
	- Locomotion DOF analysis
	"""

	def analyze_robot(self) -> Dict[str, Any]:
		"""Perform comprehensive analysis of the quadruped robot structure."""
		if self._analysis_cache is None:
			self._analysis_cache = self._perform_quadruped_analysis()
		return self._analysis_cache

	def _perform_quadruped_analysis(self) -> Dict[str, Any]:
		"""Perform the actual quadruped analysis."""
		
		analysis = {
			'robot_type': 'quadruped',
			'robot_subtype': 'standard',
			'confidence': 0.9,
			'kinematic_chains': [],
			'primary_chain': None,
			'dof_analysis': {},
			'leg_analysis': {},
			'gait_analysis': {},
			'statistics': {},
			'warnings': []
		}
		
		if not self.robot.base_link:
			analysis['warnings'].append("No base link found - cannot analyze robot structure")
			return analysis
		
		# Step 1: Analyze kinematic structure
		chains = self._find_all_kinematic_chains()
		analysis['kinematic_chains'] = chains
		
		# Step 2: Classify quadruped subtype
		quadruped_analysis = self._classify_quadruped_type(chains)
		analysis.update(quadruped_analysis)
		
		# Step 3: Analyze leg configurations
		leg_analysis = self._analyze_leg_configurations(chains)
		analysis['leg_analysis'] = leg_analysis
		
		# Step 4: DOF analysis
		analysis['dof_analysis'] = self._analyze_quadruped_dof(chains)
		
		# Step 5: Gait analysis capabilities
		analysis['gait_analysis'] = self._analyze_gait_capabilities(chains)
		
		# Step 6: Calculate statistics
		analysis['statistics'] = self._calculate_robot_statistics()
		
		return analysis

	def _classify_quadruped_type(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Classify the quadruped type based on kinematic chains."""
		
		leg_chains = [c for c in chains if 'leg' in c['chain_type'] or c['length'] == 3]
		
		if len(leg_chains) == 4:
			return {
				'robot_subtype': 'standard',
				'confidence': 0.9,
				'classification_reasoning': [
					f"Detected 4 leg-like chains of lengths: {[c['length'] for c in leg_chains]}",
					"Classic quadruped configuration"
				]
			}
		elif len(leg_chains) >= 2 and len(leg_chains) <= 6:
			return {
				'robot_subtype': f"{len(leg_chains)}_legs",
				'confidence': 0.8,
				'classification_reasoning': [
					f"Detected {len(leg_chains)} leg-like chains",
					"Multi-legged robot configuration"
				]
			}
		else:
			return {
				'robot_subtype': 'unknown',
				'confidence': 0.3,
				'classification_reasoning': [
					f"Detected {len(leg_chains)} leg-like chains",
					"Does not match standard quadruped patterns"
				]
			}

	def _analyze_leg_configurations(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Analyze leg configurations for quadruped robots."""
		
		leg_analysis = {
			'leg_count': 0,
			'leg_dof': [],
			'leg_symmetry': 'unknown',
			'leg_types': [],
			'leg_characteristics': []
		}
		
		leg_chains = [c for c in chains if 'leg' in c['chain_type'] or c['length'] == 3]
		leg_analysis['leg_count'] = len(leg_chains)
		
		if leg_chains:
			# Analyze each leg
			for leg_chain in leg_chains:
				leg_analysis['leg_dof'].append(leg_chain['length'])
				leg_analysis['leg_types'].append(leg_chain['chain_type'])
			
			# Check for symmetry
			unique_dofs = list(set(leg_analysis['leg_dof']))
			if len(unique_dofs) == 1:
				leg_analysis['leg_symmetry'] = 'symmetric'
				leg_analysis['leg_characteristics'].append('uniform_dof')
			else:
				leg_analysis['leg_symmetry'] = 'asymmetric'
				leg_analysis['leg_characteristics'].append('varied_dof')
			
			# Analyze leg characteristics
			if all(dof == 3 for dof in leg_analysis['leg_dof']):
				leg_analysis['leg_characteristics'].append('standard_3dof_legs')
			elif all(dof >= 2 for dof in leg_analysis['leg_dof']):
				leg_analysis['leg_characteristics'].append('multi_dof_legs')
		
		return leg_analysis

	def _analyze_quadruped_dof(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Analyze degrees of freedom for quadruped robots."""
		
		dof_analysis = {
			'total_dof': 0,
			'locomotion_dof': 0,
			'leg_dof': 0,
			'body_dof': 0,
			'revolute_dof': 0,
			'prismatic_dof': 0,
			'mobility_analysis': {}
		}
		
		# Calculate total DOF
		total_revolute = sum(c['revolute_count'] for c in chains)
		total_prismatic = sum(c['prismatic_count'] for c in chains)
		
		dof_analysis['total_dof'] = total_revolute + total_prismatic
		dof_analysis['revolute_dof'] = total_revolute
		dof_analysis['prismatic_dof'] = total_prismatic
		
		# Analyze leg DOF
		leg_chains = [c for c in chains if 'leg' in c['chain_type'] or c['length'] == 3]
		leg_dof = sum(c['length'] for c in leg_chains)
		dof_analysis['leg_dof'] = leg_dof
		dof_analysis['locomotion_dof'] = leg_dof  # For quadrupeds, locomotion DOF = leg DOF
		
		# Body DOF (non-leg chains)
		body_chains = [c for c in chains if c not in leg_chains]
		body_dof = sum(c['length'] for c in body_chains)
		dof_analysis['body_dof'] = body_dof
		
		# Mobility analysis
		dof_analysis['mobility_analysis'] = {
			'can_walk': leg_dof >= 8,  # Need at least 2 DOF per leg for basic walking
			'can_trot': leg_dof >= 12,  # Need at least 3 DOF per leg for trotting
			'has_body_articulation': body_dof > 0,
			'estimated_speed_capability': 'high' if leg_dof >= 12 else 'medium' if leg_dof >= 8 else 'low'
		}
		
		return dof_analysis

	def _analyze_gait_capabilities(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Analyze gait capabilities for quadruped robots."""
		
		gait_analysis = {
			'supported_gaits': [],
			'gait_characteristics': [],
			'stability_analysis': {}
		}
		
		leg_chains = [c for c in chains if 'leg' in c['chain_type'] or c['length'] == 3]
		
		if len(leg_chains) == 4:
			# Standard quadruped gaits
			avg_leg_dof = sum(c['length'] for c in leg_chains) / 4
			
			if avg_leg_dof >= 3:
				gait_analysis['supported_gaits'].extend(['walk', 'trot', 'pace', 'bound'])
				gait_analysis['gait_characteristics'].append('full_gait_capability')
			elif avg_leg_dof >= 2:
				gait_analysis['supported_gaits'].extend(['walk', 'trot'])
				gait_analysis['gait_characteristics'].append('basic_gait_capability')
			else:
				gait_analysis['supported_gaits'].extend(['walk'])
				gait_analysis['gait_characteristics'].append('limited_gait_capability')
			
			# Stability analysis
			gait_analysis['stability_analysis'] = {
				'static_stability': True,  # 4 legs provide static stability
				'dynamic_stability_capable': avg_leg_dof >= 3,
				'turning_capability': avg_leg_dof >= 2
			}
		
		return gait_analysis

	def get_analysis_report(self, analysis: Dict[str, Any] = None) -> str:
		"""Get a comprehensive formatted analysis report for quadruped robots."""
		if analysis is None:
			analysis = self.analyze_robot()
		
		report = f"Quadruped Robot Analysis Report - {self.robot.name}\n"
		report += "=" * 60 + "\n\n"
		
		# Robot Classification
		report += "🐕 QUADRUPED CLASSIFICATION\n"
		report += "-" * 30 + "\n"
		report += f"Type: {analysis['robot_type'].replace('_', ' ').title()}\n"
		if analysis['robot_subtype']:
			report += f"Subtype: {analysis['robot_subtype'].replace('_', ' ').title()}\n"
		report += f"Confidence: {analysis['confidence']*100:.1f}%\n"
		
		if analysis.get('classification_reasoning'):
			report += "\nClassification reasoning:\n"
			for reason in analysis['classification_reasoning']:
				report += f"  • {reason}\n"
		
		# Leg Analysis
		leg_analysis = analysis.get('leg_analysis', {})
		if leg_analysis:
			report += "\n🦵 LEG ANALYSIS\n"
			report += "-" * 15 + "\n"
			report += f"Leg count: {leg_analysis['leg_count']}\n"
			report += f"Leg DOF: {leg_analysis['leg_dof']}\n"
			report += f"Leg symmetry: {leg_analysis['leg_symmetry']}\n"
			
			if leg_analysis['leg_characteristics']:
				report += f"Characteristics: {', '.join(leg_analysis['leg_characteristics'])}\n"
		
		# DOF Analysis
		report += "\n📊 DEGREES OF FREEDOM ANALYSIS\n"
		report += "-" * 35 + "\n"
		dof = analysis['dof_analysis']
		report += f"Total DOF: {dof['total_dof']}\n"
		report += f"Locomotion DOF: {dof['locomotion_dof']}\n"
		report += f"Leg DOF: {dof['leg_dof']}\n"
		report += f"Body DOF: {dof['body_dof']}\n"
		report += f"  - Revolute joints: {dof['revolute_dof']}\n"
		report += f"  - Prismatic joints: {dof['prismatic_dof']}\n"
		
		# Mobility Analysis
		mobility = dof.get('mobility_analysis', {})
		if mobility:
			report += "\nMobility capabilities:\n"
			report += f"  Can walk: {'✓' if mobility.get('can_walk') else '✗'}\n"
			report += f"  Can trot: {'✓' if mobility.get('can_trot') else '✗'}\n"
			report += f"  Has body articulation: {'✓' if mobility.get('has_body_articulation') else '✗'}\n"
			report += f"  Speed capability: {mobility.get('estimated_speed_capability', 'unknown')}\n"
		
		# Gait Analysis
		gait_analysis = analysis.get('gait_analysis', {})
		if gait_analysis:
			report += "\n🏃 GAIT ANALYSIS\n"
			report += "-" * 15 + "\n"
			
			if gait_analysis['supported_gaits']:
				report += f"Supported gaits: {', '.join(gait_analysis['supported_gaits'])}\n"
			
			if gait_analysis['gait_characteristics']:
				report += f"Gait characteristics: {', '.join(gait_analysis['gait_characteristics'])}\n"
			
			# Stability Analysis
			stability = gait_analysis.get('stability_analysis', {})
			if stability:
				report += "\nStability analysis:\n"
				report += f"  Static stability: {'✓' if stability.get('static_stability') else '✗'}\n"
				report += f"  Dynamic stability capable: {'✓' if stability.get('dynamic_stability_capable') else '✗'}\n"
				report += f"  Turning capability: {'✓' if stability.get('turning_capability') else '✗'}\n"
		
		# Statistics
		stats = analysis['statistics']
		report += "\n📈 STATISTICS\n"
		report += "-" * 12 + "\n"
		report += f"Total links: {stats['total_links']}\n"
		report += f"Total joints: {stats['total_joints']}\n"
		report += f"Kinematic chains: {len(analysis['kinematic_chains'])}\n"
		report += f"Base link: {stats['base_link']}\n"
		
		if stats['joint_type_distribution']:
			report += "Joint distribution:\n"
			for joint_type, count in stats['joint_type_distribution'].items():
				report += f"  - {joint_type}: {count}\n"
		
		report += "\n" + "=" * 60
		
		return report


class UnsupportedRobotAnalysis(BaseRobotAnalysis):
	"""
	Analysis class for unsupported robot types.
	
	This class provides basic analysis for robot types that don't have
	specialized analysis implementations yet.
	"""

	def analyze_robot(self) -> Dict[str, Any]:
		"""Perform basic analysis of unsupported robot types."""
		if self._analysis_cache is None:
			self._analysis_cache = self._perform_basic_analysis()
		return self._analysis_cache

	def _perform_basic_analysis(self) -> Dict[str, Any]:
		"""Perform basic analysis for unsupported robot types."""
		
		analysis = {
			'robot_type': 'unsupported',
			'robot_subtype': 'unknown',
			'confidence': 0.3,
			'kinematic_chains': [],
			'primary_chain': None,
			'dof_analysis': {},
			'statistics': {},
			'warnings': ['This robot type is not yet supported for detailed analysis']
		}
		
		if not self.robot.base_link:
			analysis['warnings'].append("No base link found - cannot analyze robot structure")
			return analysis
		
		# Step 1: Basic kinematic structure analysis
		chains = self._find_all_kinematic_chains()
		analysis['kinematic_chains'] = chains
		
		# Step 2: Basic DOF analysis
		analysis['dof_analysis'] = self._analyze_basic_dof(chains)
		
		# Step 3: Basic statistics
		analysis['statistics'] = self._calculate_robot_statistics()
		
		return analysis

	def _analyze_basic_dof(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Perform basic DOF analysis."""
		
		dof_analysis = {
			'total_dof': 0,
			'revolute_dof': 0,
			'prismatic_dof': 0,
			'chain_count': len(chains),
			'dof_distribution': {}
		}
		
		# Calculate total DOF
		total_revolute = sum(c['revolute_count'] for c in chains)
		total_prismatic = sum(c['prismatic_count'] for c in chains)
		
		dof_analysis['total_dof'] = total_revolute + total_prismatic
		dof_analysis['revolute_dof'] = total_revolute
		dof_analysis['prismatic_dof'] = total_prismatic
		
		# DOF distribution by chain length
		dof_dist = {}
		for chain in chains:
			length = chain['length']
			dof_dist[length] = dof_dist.get(length, 0) + 1
		dof_analysis['dof_distribution'] = dof_dist
		
		return dof_analysis

	def get_analysis_report(self, analysis: Dict[str, Any] = None) -> str:
		"""Get a basic analysis report for unsupported robot types."""
		if analysis is None:
			analysis = self.analyze_robot()
		
		report = f"Basic Robot Analysis Report - {self.robot.name}\n"
		report += "=" * 60 + "\n\n"
		
		# Warning about unsupported type
		report += "⚠️  UNSUPPORTED ROBOT TYPE\n"
		report += "-" * 30 + "\n"
		report += "This robot type is not yet supported for detailed analysis.\n"
		report += "Only basic structural information is available.\n\n"
		
		# Basic DOF Analysis
		report += "📊 BASIC DEGREES OF FREEDOM ANALYSIS\n"
		report += "-" * 35 + "\n"
		dof = analysis['dof_analysis']
		report += f"Total DOF: {dof['total_dof']}\n"
		report += f"  - Revolute joints: {dof['revolute_dof']}\n"
		report += f"  - Prismatic joints: {dof['prismatic_dof']}\n"
		report += f"Chain count: {dof['chain_count']}\n"
		
		if dof['dof_distribution']:
			report += "\nDOF distribution:\n"
			for length, count in dof['dof_distribution'].items():
				report += f"  - {length} DOF chains: {count}\n"
		
		# Basic Statistics
		stats = analysis['statistics']
		report += "\n📈 BASIC STATISTICS\n"
		report += "-" * 18 + "\n"
		report += f"Total links: {stats['total_links']}\n"
		report += f"Total joints: {stats['total_joints']}\n"
		report += f"Base link: {stats['base_link']}\n"
		
		if stats['joint_type_distribution']:
			report += "Joint distribution:\n"
			for joint_type, count in stats['joint_type_distribution'].items():
				report += f"  - {joint_type}: {count}\n"
		
		# Kinematic Chains (basic info)
		if analysis['kinematic_chains']:
			report += "\n🔗 KINEMATIC CHAINS (BASIC INFO)\n"
			report += "-" * 30 + "\n"
			for i, chain in enumerate(analysis['kinematic_chains'], 1):
				report += f"Chain {i}: {chain['length']} DOF ({chain['chain_type']})\n"
				report += f"  Joints: {len(chain['joint_names'])}\n"
				report += f"  Types: {'-'.join(chain['joint_types'])}\n"
				report += f"  Reach: {chain['total_reach']:.3f} units\n\n"
		
		# Request for support
		report += "\n💡 REQUEST SUPPORT\n"
		report += "-" * 15 + "\n"
		report += "To request detailed analysis support for this robot type,\n"
		report += "please contact the development team with robot specifications.\n"
		
		report += "\n" + "=" * 60
		
		return report
