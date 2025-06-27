import json
import struct
import logging
from pxr import Usd, UsdGeom, UsdShade, Gf
import math
import numpy as np
from .math_utils import quat_to_list, quaternion_multiply, quaternion_inverse, euler_to_quat, rotate_vector
from typing import List, Tuple, Optional, Dict, Any, Union, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class USDLink:
    """
    Represents a robot link with its geometry and transformation data.
    """
    def __init__(self, prim: Any, name: str = None) -> None:
        self.prim: Any = prim
        self.name: str = name or prim.GetName()
        logger.debug(f"Creating USDLink: {self.name}, Type: {prim.GetTypeName()}, Path: {prim.GetPath()}")
        self.translation, self.rotation, self.scale = self._extract_transform()
        logger.debug(f"  Transform - Translation: {self.translation}, Rotation: {self.rotation}")
        self.meshes: List['USDMesh'] = []  # Changed from separate mesh_prims and material_prims
        self.joints: List['USDJoint'] = []
        self.parent_joint: Optional['USDJoint'] = None
        
        # Cache for expensive operations
        self._all_child_links_cache: Optional[List['USDLink']] = None
        self._all_materials_cache: Optional[List[Any]] = None
        self._material_summary_cache: Optional[Dict[str, Any]] = None
    
    def _extract_transform(self) -> Tuple[List[float], List[float], List[float]]:
        """Extract translation, rotation, and scale from the USD prim."""
        translation: List[float] = [0.0, 0.0, 0.0]
        rotation: List[float] = [0.0, 0.0, 0.0, 1.0]  # Quaternion (x, y, z, w)
        scale: List[float] = [1.0, 1.0, 1.0]
        if self.prim:
            if self.prim.GetTypeName() == "Xform":
                xform = UsdGeom.Xform(self.prim)
                ops = xform.GetOrderedXformOps()
                for op in ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        val = op.Get()
                        if hasattr(val, "__len__") and len(val) == 3:
                            translation = list(val)
                            logger.debug(f"  Found translation in {self.name}: {translation}")
                    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        val = op.Get()
                        val = quat_to_list(val)
                        if hasattr(val, "__len__") and len(val) == 4:
                            rotation = [val[1], val[2], val[3], val[0]]
                            logger.debug(f"  Found orientation in {self.name}: {rotation}")
                    elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                        val = op.Get()
                        if hasattr(val, "__len__") and len(val) == 3:
                            rotation = euler_to_quat(*val)
                            logger.debug(f"  Found rotation in {self.name}: {val} -> {rotation}")
                    elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        val = op.Get()
                        if hasattr(val, "__len__") and len(val) == 3:
                            scale = list(val)
                            logger.debug(f"  Found scale in {self.name}: {scale}")
            else:
                transform_prim = self._find_transform_prim_for_body(self.prim)
                if transform_prim:
                    logger.debug(f"  Found transform prim for {self.name}: {transform_prim.GetPath()}")
                    xform = UsdGeom.Xform(transform_prim)
                    ops = xform.GetOrderedXformOps()
                    for op in ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            val = op.Get()
                            if hasattr(val, "__len__") and len(val) == 3:
                                translation = list(val)
                                logger.debug(f"  Found translation in transform prim: {translation}")
                        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                            val = op.Get()
                            if hasattr(val, "__len__") and len(val) == 4:
                                rotation = [val[1], val[2], val[3], val[0]]
                        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                            val = op.Get()
                            if hasattr(val, "__len__") and len(val) == 3:
                                rotation = euler_to_quat(*val)
                        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                            val = op.Get()
                            if hasattr(val, "__len__") and len(val) == 3:
                                scale = list(val)
                                
        return translation, rotation, scale

    def add_mesh(self, mesh_prim: Any, material_prim: Any = None) -> None:
        """Add a mesh primitive with its optional material to this link.
        The USDMesh will automatically discover all materials including GeomSubset materials."""
        usd_mesh = USDMesh(mesh_prim, material_prim)
        self.meshes.append(usd_mesh)
        # Invalidate caches
        self._invalidate_caches()

    def add_material(self, material_prim: Any) -> None:
        """Add a material primitive to this link."""
        # This method is now deprecated since materials are handled per mesh
        logger.warning(f"add_material() is deprecated. Materials should be assigned per mesh.")

    def get_meshes_with_materials(self) -> List['USDMesh']:
        """Get all meshes that have materials assigned."""
        return [mesh for mesh in self.meshes if mesh.has_material()]
    
    def get_meshes_without_materials(self) -> List['USDMesh']:
        """Get all meshes that don't have materials assigned."""
        return [mesh for mesh in self.meshes if not mesh.has_material()]
    
    def get_meshes_with_multiple_materials(self) -> List['USDMesh']:
        """Get all meshes that have multiple materials assigned."""
        return [mesh for mesh in self.meshes if mesh.has_multiple_materials()]
    
    def get_total_material_count(self) -> int:
        """Get the total count of all materials across all meshes in this link."""
        return sum(mesh.get_material_count() for mesh in self.meshes)
    
    def get_material_summary(self) -> Dict[str, Any]:
        """Get a summary of materials used by this link's meshes (cached)."""
        if self._material_summary_cache is None:
            summary = {
                'total_meshes': len(self.meshes),
                'meshes_with_materials': len(self.get_meshes_with_materials()),
                'meshes_without_materials': len(self.get_meshes_without_materials()),
                'meshes_with_multiple_materials': len(self.get_meshes_with_multiple_materials()),
                'total_material_count': self.get_total_material_count(),
                'unique_materials': len(self.get_all_materials())
            }
            self._material_summary_cache = summary
        return self._material_summary_cache

    def add_joint(self, joint: 'USDJoint') -> None:
        """Add a child joint to this link."""
        self.joints.append(joint)
        joint.parent_link = self
        # Invalidate caches
        self._all_child_links_cache = None

    def get_all_child_links(self) -> List['USDLink']:
        """Get all child links recursively (cached)."""
        if self._all_child_links_cache is None:
            child_links = []
            for joint in self.joints:
                if joint.child_link:
                    child_links.append(joint.child_link)
                    child_links.extend(joint.child_link.get_all_child_links())
            self._all_child_links_cache = child_links
        return self._all_child_links_cache

    def get_position(self) -> List[float]:
        """Get the position of this link."""
        return self.translation

    def calculate_joint_corrected_transform(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate transform from parent link to this link using joint local positions/rotations.
        This composes the parent link's transform, the joint's localPos0/localRot0 (in parent),
        and the inverse of localPos1/localRot1 (in child).
        The output rotation is the composed relative rotation between parent and child at the joint,
        further composed with the link's own rotation.
        """
        if not self.parent_joint or not self.parent_joint.parent_link:
            logger.debug(f"Link {self.name} is root - using world transform")
            return self.translation, self.rotation, self.scale

        parent_link = self.parent_joint.parent_link
        joint = self.parent_joint
        local_pos0, local_pos1 = joint.get_local_positions()
        local_rot0, local_rot1 = joint.get_local_rotations()

        # Default to identity if not present
        if local_pos0 is None:
            local_pos0 = [0.0, 0.0, 0.0]
        if local_pos1 is None:
            local_pos1 = [0.0, 0.0, 0.0]
        if local_rot0 is None:
            local_rot0 = [0.0, 0.0, 0.0, 1.0]
        if local_rot1 is None:
            local_rot1 = [0.0, 0.0, 0.0, 1.0]

        # Compute inverse of local_rot1
        inv_local_rot1 = quaternion_inverse(local_rot1)
        # Compose relative rotation between parent and child at the joint
        relative_rotation = quaternion_multiply(local_rot0, inv_local_rot1)
        # Compose with the link's own rotation
        final_rotation = quaternion_multiply(relative_rotation, self.rotation)

        rotated_local_pos1 = rotate_vector(relative_rotation, local_pos1)
        relative_translation = [
            local_pos0[0] - rotated_local_pos1[0],
            local_pos0[1] - rotated_local_pos1[1],
            local_pos0[2] - rotated_local_pos1[2]
        ]

        logger.debug(f"  Final joint-corrected transform: pos={relative_translation}, rot={final_rotation}")
        return relative_translation, final_rotation, self.scale.copy()

    def _find_transform_prim_for_body(self, body_prim: Any) -> Optional[Any]:
        """Find the corresponding transform prim for a physics body prim."""
        stage = body_prim.GetStage()
        body_name = body_prim.GetName()
        body_path = str(body_prim.GetPath())
        logger.debug(f"Looking for transform prim for body: {body_name} at {body_path}")
        clean_name = body_name.replace('_physics', '').replace('_body', '').replace('Physics', '').replace('Body', '')
        logger.debug(f"  Clean name: {clean_name}")
        search_paths = []
        parent_path = str(body_prim.GetParent().GetPath()) if body_prim.GetParent() else ""
        if parent_path:
            search_paths.extend([
                f"{parent_path}/{clean_name}",
                f"{parent_path}/{body_name}",
                parent_path
            ])
        search_paths.extend([
            body_path.replace(body_name, clean_name),
            body_path.replace('_physics', '').replace('_body', ''),
        ])
        root_path = "/" + clean_name
        search_paths.append(root_path)
        logger.debug(f"  Searching paths: {search_paths}")
        for path in search_paths:
            try:
                candidate = stage.GetPrimAtPath(path)
                if candidate and candidate.GetTypeName() == "Xform":
                    xform = UsdGeom.Xform(candidate)
                    ops = xform.GetOrderedXformOps()
                    if ops:
                        logger.debug(f"  Found transform prim at: {path}")
                        return candidate
                    else:
                        logger.debug(f"  Xform at {path} has no transform ops")
                else:
                    logger.debug(f"  No valid Xform at {path}")
            except Exception as e:
                logger.debug(f"  Error checking {path}: {e}")
                continue
        logger.debug(f"  No transform prim found for {body_name}")
        return None

    def __str__(self) -> str:
        return f"USDLink({self.name})"
    def __repr__(self) -> str:
        return self.__str__()

    def get_all_materials(self) -> List[Any]:
        """Get all unique material prims used by this link's meshes (cached)."""
        if self._all_materials_cache is None:
            materials = []
            seen_materials = set()
            for mesh in self.meshes:
                # Get all materials from each mesh (handles multiple materials per mesh)
                for material in mesh.get_materials():
                    if material not in seen_materials:
                        materials.append(material)
                        seen_materials.add(material)
            self._all_materials_cache = materials
        return self._all_materials_cache
    
    def _invalidate_caches(self) -> None:
        """Invalidate all cached data when meshes are modified."""
        self._all_materials_cache = None
        self._material_summary_cache = None

class USDJoint:
    """
    Represents a robot joint connecting two links.
    """
    def __init__(self, joint_prim: Any, parent_link: USDLink = None, child_link: USDLink = None) -> None:
        self.joint_prim: Any = joint_prim
        self.name: str = joint_prim.GetName()
        self.parent_link: Optional[USDLink] = parent_link
        self.child_link: Optional[USDLink] = child_link
        self.joint_type: str = self._determine_joint_type()
        self.properties: Dict[str, Any] = self._extract_joint_properties()
    
    def _determine_joint_type(self) -> str:
        """Determine the type of joint from the prim."""
        # Try to get joint type from attributes or name
        if hasattr(self.joint_prim, 'GetAttribute'):
            joint_type_attr = self.joint_prim.GetAttribute('physics:type')
            if joint_type_attr and joint_type_attr.IsValid():
                return joint_type_attr.Get()
        
        # Fallback to name analysis
        name_lower = self.name.lower()
        if 'revolute' in name_lower or 'hinge' in name_lower:
            return 'revolute'
        elif 'prismatic' in name_lower or 'slider' in name_lower:
            return 'prismatic'
        elif 'fixed' in name_lower:
            return 'fixed'
        else:
            return 'revolute'  # Default assumption
    
    def get_connected_body_prims(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Get the two body prims connected by this joint from the joint's physics relationships."""
        body0_prim = body1_prim = None
        for rel in self.joint_prim.GetRelationships():
            if rel.GetName() == "physics:body0":
                targets = rel.GetTargets()
                if targets:
                    body0_prim = self.joint_prim.GetStage().GetPrimAtPath(str(targets[0]))
            elif rel.GetName() == "physics:body1":
                targets = rel.GetTargets()
                if targets:
                    body1_prim = self.joint_prim.GetStage().GetPrimAtPath(str(targets[0]))
        return body0_prim, body1_prim
    
    def _extract_joint_properties(self) -> Dict[str, Any]:
        """Extract all joint properties from the joint prim."""
        properties = {}
        for prop in self.joint_prim.GetProperties():
            prop_name = prop.GetName()
            try:
                prop_value = prop.Get()
                properties[prop_name] = prop_value
            except:
                # Some properties might not have values
                properties[prop_name] = None
        return properties
    
    def get_property(self, property_name: str) -> Any:
        """Get a specific joint property by name."""
        return self.properties.get(property_name)
    
    def has_property(self, property_name: str) -> bool:
        """Check if joint has a specific property."""
        return property_name in self.properties
    
    def get_axis(self) -> Optional[List[float]]:
        """Get the joint axis vector."""
        axis = self.get_property('physics:axis')
        if axis and hasattr(axis, '__len__') and len(axis) == 3:
            return list(axis)
        return None
    
    def get_limits(self) -> Tuple[Optional[float], Optional[float]]:
        """Get joint limits (lower, upper)."""
        lower = self.get_property('physics:lowerLimit')
        upper = self.get_property('physics:upperLimit')
        return (lower, upper)
    
    def get_drive_properties(self) -> Dict[str, Any]:
        """Get all drive-related properties."""
        drive_props = {}
        for key, value in self.properties.items():
            if key.startswith('drive:'):
                drive_props[key] = value
        return drive_props
    
    def get_local_positions(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """Get local positions of the joint attachment points on body0 and body1."""
        pos0 = pos1 = None
        
        if 'physics:localPos0' in self.properties and 'physics:localPos1' in self.properties:
            local_pos0 = self.get_property('physics:localPos0')
            local_pos1 = self.get_property('physics:localPos1')
            
            pos0 = list(local_pos0) if local_pos0 and hasattr(local_pos0, '__len__') and len(local_pos0) == 3 else None
            pos1 = list(local_pos1) if local_pos1 and hasattr(local_pos1, '__len__') and len(local_pos1) == 3 else None
        elif 'xformOp:translate' in self.properties:
            # Fallback to xformOp:translate if localPos0/localPos1 are not available
            local_pos0 = self.get_property('xformOp:translate')
        
        return pos0, pos1
    
    def get_local_rotations(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """Get local rotations of the joint attachment points on body0 and body1."""
        local_rot0 = local_rot1 = None
        if 'physics:localRot0' in self.properties and 'physics:localRot1' in self.properties:
            print(f"[Test] Joint {self.name} has localRot0 and localRot1 properties")
            local_rot0 = self.get_property('physics:localRot0')
            local_rot1 = self.get_property('physics:localRot1')
        elif 'xformOp:orient' in self.properties:
            # Fallback to xformOp:orient if localRot0/localRot1 are not available
            local_rot0 = self.get_property('xformOp:orient')
            
        print(f"[Test] Joint {self.name} local rotations: {local_rot0}, {local_rot1}")
        
        # USD rotations are quaternions in (w, x, y, z) format, convert to (x, y, z, w)
        rot0 = None
        rot1 = None
        
        local_rot0_list = quat_to_list(local_rot0)
        local_rot1_list = quat_to_list(local_rot1)

        if local_rot0_list and len(local_rot0_list) == 4:
            rot0 = [local_rot0_list[1], local_rot0_list[2], local_rot0_list[3], local_rot0_list[0]]  # w,x,y,z -> x,y,z,w

        if local_rot1_list and len(local_rot1_list) == 4:
            rot1 = [local_rot1_list[1], local_rot1_list[2], local_rot1_list[3], local_rot1_list[0]]  # w,x,y,z -> x,y,z,w

        # If no rotations found at all, return (None, None)
        if rot0 is None and rot1 is None:
            return None, None

        return rot0, rot1
    
    def print_properties(self) -> None:
        """Print all joint properties for debugging."""
        print(f"Joint {self.name} properties:")
        if not self.properties:
            print("  No properties found")
            return
        
        for key, value in sorted(self.properties.items()):
            print(f"  {key}: {value}")
        
        # Also print some derived information
        axis = self.get_axis()
        if axis:
            print(f"  Derived axis: {axis}")
        
        limits = self.get_limits()
        if any(limits):
            print(f"  Derived limits: lower={limits[0]}, upper={limits[1]}")
    
    def __str__(self) -> str:
        parent_name = self.parent_link.name if self.parent_link else "None"
        child_name = self.child_link.name if self.child_link else "None"
        axis = self.get_axis()
        limits = self.get_limits()
        local_pos0, local_pos1 = self.get_local_positions()
        
        axis_str = f", axis={axis}" if axis else ""
        limits_str = f", limits={limits}" if any(limits) else ""
        local_pos_str = f", localPos0={local_pos0}, localPos1={local_pos1}" if local_pos0 or local_pos1 else ""
        
        return f"USDJoint({self.name}: {parent_name} -> {child_name}, type={self.joint_type}{axis_str}{limits_str}{local_pos_str})"
    
    def __repr__(self) -> str:
        return self.__str__()

class USDRobot:
    """
    Represents a complete robot with hierarchical link and joint structure.
    """
    def __init__(self, stage: Usd.Stage, name: str = "Robot") -> None:
        self.stage: Usd.Stage = stage
        self.name: str = name
        self.base_link: Optional[USDLink] = None
        self.links: Dict[str, USDLink] = {}
        self.joints: Dict[str, USDJoint] = {}
        self.link_prims: Dict[Any, USDLink] = {}  # Map prim to link
        
        # Performance optimization: caches
        self._prim_cache: Dict[str, Any] = {}
        self._material_cache: Dict[Any, Any] = {}
        self._mesh_to_link_cache: Dict[str, USDLink] = {}
        
        self._build_robot_structure()
    
    def _build_robot_structure(self) -> None:
        """Build the robot structure from USD stage."""
        root = self.stage.GetDefaultPrim() or self.stage.GetPseudoRoot()
        
        logger.info("Building robot structure from USD stage")
        
        # First pass: collect all joints and their physics relationships
        joint_data = []
        self._collect_joints(root, joint_data)
        logger.info(f"Found {len(joint_data)} joints")
        
        # Second pass: create links from physics bodies
        self._create_links_from_joints(joint_data)
        logger.info(f"Created {len(self.links)} links")
        
        # Third pass: build joint connections
        self._build_joint_connections(joint_data)
        logger.info(f"Built joint connections")
        
        # Fourth pass: find and assign meshes with materials to links
        self._assign_meshes_to_links(root)
        total_meshes = sum(len(link.meshes) for link in self.links.values())
        total_materials = sum(len([m for m in link.meshes if m.has_material()]) for link in self.links.values())
        total_all_materials = sum(sum(len(mesh.get_materials()) for mesh in link.meshes) for link in self.links.values())
        logger.info(f"Assigned {total_meshes} meshes to links")
        logger.info(f"Found {total_materials} meshes with materials")
        logger.info(f"Total material assignments: {total_all_materials}")
        
        # Debug: Show mesh and material assignment details
        if logger.isEnabledFor(logging.DEBUG):
            for link_name, link in self.links.items():
                if link.meshes:
                    mesh_info = []
                    for mesh in link.meshes:
                        mesh_str = f"{mesh.name}"
                        if mesh.has_multiple_materials():
                            material_names = mesh.get_all_material_names()
                            mesh_str += f" ({len(material_names)} materials: {', '.join(material_names)})"
                        elif mesh.has_material():
                            mesh_str += f" (material: {mesh.get_material_name()})"
                        else:
                            mesh_str += " (no materials)"
                        mesh_info.append(mesh_str)
                    logger.debug(f"  Link '{link_name}' has meshes: {mesh_info}")
        
        # Find base link (usually the one with no parent joint or named 'base')
        self._identify_base_link()
        if self.base_link:
            logger.info(f"Identified base link: {self.base_link.name}")
        else:
            logger.warning("No base link identified")
        
        # Validate the joint tree structure
        self.validate_joint_tree()
        
        # Print statistics
        self.print_statistics()
        
        # Print joint properties for debugging
        if logger.isEnabledFor(logging.DEBUG):
            self.print_joint_properties()
    
    def _collect_joints(self, prim: Any, joint_data: List) -> None:
        """Recursively collect joint prims and their relationships - optimized version."""
        # Use a stack-based approach instead of recursion for better performance
        stack = [prim]
        
        while stack:
            current_prim = stack.pop()
            
            if 'joint' in current_prim.GetTypeName().lower():
                body0 = body1 = None
                # Use list comprehension for faster relationship processing
                relationships = [(rel.GetName(), rel.GetTargets()) for rel in current_prim.GetRelationships()]
                
                for rel_name, targets in relationships:
                    if rel_name == "physics:body0" and targets:
                        body0 = self._get_prim_cached(str(targets[0]))
                    elif rel_name == "physics:body1" and targets:
                        body1 = self._get_prim_cached(str(targets[0]))
                
                if body0 and body1:
                    joint_data.append((current_prim, body0, body1))
            
            # Add children to stack
            stack.extend(current_prim.GetChildren())
    
    def _get_prim_cached(self, path: str) -> Any:
        """Get prim with caching for better performance."""
        if path not in self._prim_cache:
            self._prim_cache[path] = self.stage.GetPrimAtPath(path)
        return self._prim_cache[path]
    
    def _create_links_from_joints(self, joint_data: List) -> None:
        """Create USDLink objects from joint body relationships."""
        all_body_prims = set()
        for _, body0, body1 in joint_data:
            all_body_prims.add(body0)
            all_body_prims.add(body1)
        
        # Create USDLink for each unique body
        for body_prim in all_body_prims:
            link = USDLink(body_prim)
            self.links[link.name] = link
            self.link_prims[body_prim] = link
    
    def _build_joint_connections(self, joint_data: List) -> None:
        """Build joint connections between links using robust tree-building logic."""
        logger.info("Building joint tree structure...")
        
        # First, create all joints
        for joint_prim, body0, body1 in joint_data:
            joint = USDJoint(joint_prim)
            self.joints[joint.name] = joint
            logger.debug(f"Created joint: {joint.name}")
        
        # Build the tree structure to determine parent-child relationships
        tree_structure = self._build_joint_tree(joint_data)
        
        # Now assign parent-child relationships based on tree structure
        self._assign_joint_relationships(tree_structure, joint_data)
        
        logger.info(f"Successfully built {len(self.joints)} joint connections")
    
    def _build_joint_tree(self, joint_data: List) -> Dict[Any, List[Any]]:
        """Build a tree structure from joint data to determine parent-child relationships."""
        # Collect all bodies and create edges based on joint connections
        all_bodies = set()
        edges = {}  # parent_body -> [child_bodies]
        joint_map = {}  # (parent_body, child_body) -> joint_prim
        
        for joint_prim, body0, body1 in joint_data:
            all_bodies.add(body0)
            all_bodies.add(body1)
            
            parent_body, child_body = body0, body1
            
            # Build the edge map
            if parent_body not in edges:
                edges[parent_body] = []
            edges[parent_body].append(child_body)
            joint_map[(parent_body, child_body)] = joint_prim
            
            logger.debug(f"Joint {joint_prim.GetName()}: {parent_body.GetName()} -> {child_body.GetName()}")
        
        # Find root bodies (those that are never children)
        child_bodies = set()
        for children in edges.values():
            child_bodies.update(children)
        
        root_bodies = all_bodies - child_bodies
        logger.info(f"Found {len(root_bodies)} root bodies: {[b.GetName() for b in root_bodies]}")
        
        if not root_bodies:
            # No clear root - choose based on naming heuristics
            logger.warning("No clear root bodies found, using heuristic selection")
            for body in all_bodies:
                if 'base' in body.GetName().lower():
                    root_bodies = {body}
                    break
            if not root_bodies:
                # Still no root, use the first body
                root_bodies = {next(iter(all_bodies))}
                logger.warning(f"Using arbitrary root: {next(iter(root_bodies)).GetName()}")
        
        return {'edges': edges, 'joint_map': joint_map, 'roots': root_bodies}
    
    def _assign_joint_relationships(self, tree_structure: Dict, joint_data: List) -> None:
        """Assign parent-child relationships to joints based on tree structure."""
        edges = tree_structure['edges']
        joint_map = tree_structure['joint_map']
        
        for parent_body, child_bodies in edges.items():
            parent_link = self.link_prims[parent_body]
            
            for child_body in child_bodies:
                child_link = self.link_prims[child_body]
                joint_prim = joint_map[(parent_body, child_body)]
                joint = self.joints[joint_prim.GetName()]
                
                # Set up the joint connections
                joint.parent_link = parent_link
                joint.child_link = child_link
                child_link.parent_joint = joint
                
                # Add joint to parent link
                parent_link.add_joint(joint)
                
                logger.debug(f"Connected joint {joint.name}: {parent_link.name} -> {child_link.name}")
    
    def _assign_meshes_to_links(self, prim: Any) -> None:
        """Find and assign mesh prims with their materials to their corresponding links - optimized version."""
        # Use breadth-first search for better cache locality
        from collections import deque
        queue = deque([prim])
        meshes_found = []
        
        # First pass: collect all meshes
        while queue:
            current_prim = queue.popleft()
            
            if current_prim.GetTypeName() == "Mesh":
                meshes_found.append(current_prim)
            
            queue.extend(current_prim.GetChildren())
        
        # Second pass: process meshes in batch
        for mesh_prim in meshes_found:
            self._process_single_mesh_assignment(mesh_prim)
    
    def _process_single_mesh_assignment(self, mesh_prim: Any) -> None:
        """Process a single mesh assignment - optimized version."""
        mesh_name = mesh_prim.GetName()
        mesh_path = str(mesh_prim.GetPath())
        
        logger.debug(f"Found mesh: {mesh_name} at path: {mesh_path}")
        
        # Check cache first
        if mesh_path in self._mesh_to_link_cache:
            link = self._mesh_to_link_cache[mesh_path]
            material_prim = self._find_material_for_mesh_prim_cached(mesh_prim)
            link.add_mesh(mesh_prim, material_prim)
            return
        
        # Find the material for this mesh (cached)
        material_prim = self._find_material_for_mesh_prim_cached(mesh_prim)
        if material_prim:
            logger.debug(f"  -> Material: {material_prim.GetName()}")
        else:
            logger.debug(f"  -> No material found")
        
        # Optimized link matching using pre-computed scores
        best_match = self._find_best_link_match(mesh_name, mesh_path)
        
        if best_match:
            logger.debug(f"  -> Assigned mesh '{mesh_name}' to link '{best_match.name}'")
            best_match.add_mesh(mesh_prim, material_prim)
            # Cache the result
            self._mesh_to_link_cache[mesh_path] = best_match
        else:
            logger.warning(f"  -> No matching link found for mesh: {mesh_name}")
    
    def _find_best_link_match(self, mesh_name: str, mesh_path: str) -> Optional[USDLink]:
        """Find the best matching link for a mesh using optimized scoring."""
        best_match = None
        best_score = 0
        
        mesh_name_clean = mesh_name.lower().replace('_', '').replace('-', '')
        mesh_name_words = set(mesh_name_clean.split())
        
        for link in self.links.values():
            score = 0
            
            # Path-based matching (highest priority)
            link_path = str(link.prim.GetPath())
            if mesh_path.startswith(link_path):
                score = 100 + len(link_path)  # Prefer deeper matches
            else:
                # Name-based matching
                link_name_clean = link.name.lower().replace('_', '').replace('-', '')
                link_name_words = set(link_name_clean.split())
                
                # Word overlap scoring
                common_words = mesh_name_words & link_name_words
                if common_words:
                    score = len(common_words) * 10
                
                # Substring matching
                if link_name_clean in mesh_name_clean or mesh_name_clean in link_name_clean:
                    score += len(set(link_name_clean) & set(mesh_name_clean))
            
            if score > best_score:
                best_score = score
                best_match = link
        
        return best_match
    
    def _find_material_for_mesh_prim_cached(self, mesh_prim: Any) -> Optional[Any]:
        """Find the primary material bound to a specific mesh prim - cached version."""
        if mesh_prim in self._material_cache:
            return self._material_cache[mesh_prim]
        
        material_prim = self._find_material_for_mesh_prim_internal(mesh_prim)
        self._material_cache[mesh_prim] = material_prim
        return material_prim
    
    def _find_material_for_mesh_prim_internal(self, mesh_prim: Any) -> Optional[Any]:
        """Internal method to find material for mesh prim."""
        # Check if the mesh has a direct material binding
        material_api = UsdShade.MaterialBindingAPI(mesh_prim)
        if material_api:
            direct_binding = material_api.GetDirectBinding()
            if direct_binding and direct_binding.GetMaterial():
                logger.debug(f"  -> Found direct material binding: {direct_binding.GetMaterial().GetPrim().GetName()}")
                return direct_binding.GetMaterial().GetPrim()
            
            # Check for inherited material binding
            inherited_binding = material_api.ComputeBoundMaterial()
            if inherited_binding[0] and inherited_binding[0].GetMaterial():
                logger.debug(f"  -> Found inherited material binding: {inherited_binding[0].GetMaterial().GetPrim().GetName()}")
                return inherited_binding[0].GetMaterial().GetPrim()
        
        # Method 2: Check for material:binding relationships (more direct approach)
        for rel in mesh_prim.GetRelationships():
            if rel.GetName() == "material:binding":
                targets = rel.GetTargets()
                for target in targets:
                    target_prim = self._get_prim_cached(str(target))
                    if target_prim and target_prim.GetTypeName() == "Material":
                        logger.debug(f"  -> Found material via relationship: {target_prim.GetName()}")
                        return target_prim
        
        # Check parent_prim for material bindings (walk up the hierarchy)
        parent_prim = mesh_prim.GetParent()
        while parent_prim:
            # Check parent with USD Shade API
            parent_material_api = UsdShade.MaterialBindingAPI(parent_prim)
            if parent_material_api:
                parent_binding = parent_material_api.GetDirectBinding()
                if parent_binding and parent_binding.GetMaterial():
                    logger.debug(f"  -> Found material via parent USD Shade API: {parent_binding.GetMaterial().GetPrim().GetName()}")
                    return parent_binding.GetMaterial().GetPrim()
            
            # Also check parent with relationship method
            for rel in parent_prim.GetRelationships():
                if rel.GetName() == "material:binding":
                    targets = rel.GetTargets()
                    for target in targets:
                        target_prim = self._get_prim_cached(str(target))
                        if target_prim and target_prim.GetTypeName() == "Material":
                            logger.debug(f"  -> Found material via parent relationship: {target_prim.GetName()}")
                            return target_prim
            
            parent_prim = parent_prim.GetParent()
        
        return None
    
    def _identify_base_link(self) -> None:
        """Identify the base link of the robot."""
        # Look for link with 'base' in name
        for link in self.links.values():
            if 'base' in link.name.lower() and not link.parent_joint:
                self.base_link = link
                return
        
        # Fallback: find link with no parent joint
        for link in self.links.values():
            if not link.parent_joint:
                self.base_link = link
                return
        
        # Last resort: use first link
        if self.links:
            self.base_link = list(self.links.values())[0]
    
    def get_all_links(self) -> List[USDLink]:
        """Get all links in the robot."""
        return list(self.links.values())
    
    def get_all_joints(self) -> List[USDJoint]:
        """Get all joints in the robot."""
        return list(self.joints.values())
    
    def print_structure(self, link: USDLink = None, prefix: str = "", is_last: bool = True) -> None:
        """Print the robot structure as a tree."""
        if link is None:
            link = self.base_link
            print(f"Robot: {self.name}")
            print(f"Total Links: {len(self.links)}, Total Joints: {len(self.joints)}")
            print("=" * 50)
        
        if not link:
            print("No base link found!")
            return
        
        connector = "└── " if is_last else "├── "
        mesh_count = len(link.meshes)
        material_count = len([m for m in link.meshes if m.has_material()])
        total_material_count = sum(m.get_material_count() for m in link.meshes)
        
        # Show mesh info
        if mesh_count > 0:
            mesh_names = []
            for mesh in link.meshes:
                if mesh.has_multiple_materials():
                    mesh_names.append(f"{mesh.name}({mesh.get_material_count()}mats)")
                else:
                    mesh_names.append(mesh.name)
            mesh_info = f" [{mesh_count} meshes: {', '.join(mesh_names)}]"
        else:
            mesh_info = " [no meshes]"
        
        # Show material info
        if total_material_count > 0:
            if total_material_count > material_count:
                material_info = f" [{material_count} meshes with {total_material_count} materials total]"
            else:
                material_info = f" [{material_count} with materials]"
        else:
            material_info = " [no materials]"
        
        print(f"{prefix}{connector}{link.name} (Link){mesh_info}{material_info}")
        
        # Print joints and their child links
        for i, joint in enumerate(link.joints):
            is_joint_last = (i == len(link.joints) - 1)
            joint_prefix = prefix + ("    " if is_last else "│   ")
            joint_connector = "└── " if is_joint_last else "├── "
            print(f"{joint_prefix}{joint_connector}[{joint.name}] ({joint.joint_type})")
            
            if joint.child_link:
                child_prefix = joint_prefix + ("    " if is_joint_last else "│   ")
                self.print_structure(joint.child_link, child_prefix, True)
    
    def validate_joint_tree(self) -> bool:
        """Validate the joint tree structure for consistency - optimized version."""
        logger.info("Validating joint tree structure...")
        
        issues = []
        
        # Batch validation for better performance
        
        # Check 1 & 2: Joint and link validation in one pass
        for joint in self.joints.values():
            if not joint.parent_link:
                issues.append(f"Joint '{joint.name}' has no parent link")
            if not joint.child_link:
                issues.append(f"Joint '{joint.name}' has no child link")
        
        for link in self.links.values():
            if link != self.base_link and not link.parent_joint:
                issues.append(f"Link '{link.name}' has no parent joint (and is not base link)")
        
        # Check 3: Base link validation
        if self.base_link and self.base_link.parent_joint:
            issues.append(f"Base link '{self.base_link.name}' has a parent joint")
        
        # Check 4: Circular references - optimized with iterative approach
        if self.base_link:
            visited = set()
            stack = [(self.base_link, [self.base_link])]
            
            while stack:
                current_link, path = stack.pop()
                
                if current_link in visited:
                    continue
                    
                visited.add(current_link)
                
                for joint in current_link.joints:
                    if joint.child_link:
                        if joint.child_link in path:
                            issues.append(f"Circular reference detected: {' -> '.join([l.name for l in path] + [joint.child_link.name])}")
                        else:
                            stack.append((joint.child_link, path + [joint.child_link]))
        
        # Check 5: Reachability - optimized with set operations
        if self.base_link:
            reachable = set()
            stack = [self.base_link]
            
            while stack:
                current_link = stack.pop()
                if current_link in reachable:
                    continue
                    
                reachable.add(current_link)
                
                for joint in current_link.joints:
                    if joint.child_link and joint.child_link not in reachable:
                        stack.append(joint.child_link)
            
            unreachable = set(self.links.values()) - reachable
            for link in unreachable:
                issues.append(f"Link '{link.name}' is not reachable from base link")
        
        # Report results
        if issues:
            logger.warning(f"Found {len(issues)} joint tree validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
        else:
            logger.info("Joint tree structure validation passed!")
            return True
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get statistics about the joint tree structure - optimized version."""
        stats = {
            'total_links': len(self.links),
            'total_joints': len(self.joints),
            'base_link': self.base_link.name if self.base_link else None,
            'max_depth': 0,
            'leaf_links': [],
            'links_with_meshes': 0,
            'links_without_meshes': 0,
            'links_with_materials': 0,
            'links_without_materials': 0,
            'meshes_with_multiple_materials': 0,
            'total_material_assignments': 0,
            'unique_materials_count': 0,
            'joint_types': defaultdict(int)
        }
        
        # Batch process links for better performance
        all_materials = set()
        for link in self.links.values():
            # Count links with/without meshes
            if link.meshes:
                stats['links_with_meshes'] += 1
            else:
                stats['links_without_meshes'] += 1
            
            # Check if any mesh has a material
            has_materials = any(mesh.has_material() for mesh in link.meshes)
            if has_materials:
                stats['links_with_materials'] += 1
            else:
                stats['links_without_materials'] += 1
            
            # Count meshes with multiple materials
            stats['meshes_with_multiple_materials'] += sum(1 for mesh in link.meshes if mesh.has_multiple_materials())
            
            # Count total material assignments
            stats['total_material_assignments'] += sum(mesh.get_material_count() for mesh in link.meshes)
            
            # Collect unique materials
            all_materials.update(link.get_all_materials())
        
        stats['unique_materials_count'] = len(all_materials)
        
        # Count joint types using defaultdict for better performance
        for joint in self.joints.values():
            stats['joint_types'][joint.joint_type] += 1
        
        # Convert defaultdict back to regular dict
        stats['joint_types'] = dict(stats['joint_types'])
        
        # Calculate tree depth and find leaf links
        if self.base_link:
            self._calculate_depth_optimized(self.base_link, 0, stats)
        
        return stats
    
    def _calculate_depth_optimized(self, link: USDLink, depth: int, stats: Dict[str, Any]) -> None:
        """Calculate tree depth and leaf links - optimized iterative version."""
        # Use stack-based approach to avoid recursion overhead
        stack = [(link, depth)]
        
        while stack:
            current_link, current_depth = stack.pop()
            stats['max_depth'] = max(stats['max_depth'], current_depth)
            
            if not current_link.joints:  # Leaf link
                stats['leaf_links'].append(current_link.name)
            else:
                # Add children to stack
                for joint in current_link.joints:
                    if joint.child_link:
                        stack.append((joint.child_link, current_depth + 1))
    
    def print_statistics(self) -> None:
        """Print detailed statistics about the robot structure."""
        stats = self.get_tree_statistics()
        
        logger.info("=== Robot Structure Statistics ===")
        logger.info(f"Total Links: {stats['total_links']}")
        logger.info(f"Total Joints: {stats['total_joints']}")
        logger.info(f"Base Link: {stats['base_link']}")
        logger.info(f"Tree Depth: {stats['max_depth']}")
        logger.info(f"Leaf Links: {len(stats['leaf_links'])} ({', '.join(stats['leaf_links'])})")
        logger.info(f"Links with Meshes: {stats['links_with_meshes']}")
        logger.info(f"Links without Meshes: {stats['links_without_meshes']}")
        logger.info(f"Links with Materials: {stats['links_with_materials']}")
        logger.info(f"Links without Materials: {stats['links_without_materials']}")
        logger.info(f"Meshes with Multiple Materials: {stats['meshes_with_multiple_materials']}")
        logger.info(f"Total Material Assignments: {stats['total_material_assignments']}")
        logger.info(f"Unique Materials: {stats['unique_materials_count']}")
        logger.info(f"Joint Types: {stats['joint_types']}")

    def print_joint_properties(self) -> None:
        """Print properties for all joints."""
        logger.info("=== Joint Properties ===")
        for joint_name, joint in self.joints.items():
            logger.info(f"Joint: {joint_name}")
            if joint.properties:
                for key, value in sorted(joint.properties.items()):
                    logger.info(f"  {key}: {value}")
            else:
                logger.info("  No properties found")
            
            # Print derived information
            axis = joint.get_axis()
            if axis:
                logger.info(f"  Derived axis: {axis}")
            
            limits = joint.get_limits()
            if any(limits):
                logger.info(f"  Derived limits: lower={limits[0]}, upper={limits[1]}")
            
            # Print local positions and rotations
            local_pos0, local_pos1 = joint.get_local_positions()
            local_rot0, local_rot1 = joint.get_local_rotations()
            
            if local_pos0:
                logger.info(f"  Local Position 0 (body0): {local_pos0}")
            if local_pos1:
                logger.info(f"  Local Position 1 (body1): {local_pos1}")
            if local_rot0:
                logger.info(f"  Local Rotation 0 (body0): {local_rot0}")
            if local_rot1:
                logger.info(f"  Local Rotation 1 (body1): {local_rot1}")
            
            logger.info("")  # Empty line for readability
    
    def clear_caches(self) -> None:
        """Clear all performance caches - useful when robot structure changes."""
        self._prim_cache.clear()
        self._material_cache.clear()
        self._mesh_to_link_cache.clear()
        
        # Clear link caches
        for link in self.links.values():
            link._invalidate_caches()
            link._all_child_links_cache = None
        
        # Clear mesh caches
        for link in self.links.values():
            for mesh in link.meshes:
                mesh._material_names_cache = None
                mesh._has_material_cache = None
                mesh._has_multiple_materials_cache = None
    
    def __str__(self) -> str:
        return f"USDRobot({self.name}, {len(self.links)} links, {len(self.joints)} joints)"
    
    def __repr__(self) -> str:
        return self.__str__()

class USDMesh:
    """
    Represents a mesh with its associated materials.
    Can handle both single material assignment and multiple materials via GeomSubsets.
    """
    def __init__(self, mesh_prim: Any, material_prim: Any = None) -> None:
        self.mesh_prim: Any = mesh_prim
        self.name: str = mesh_prim.GetName()
        self.path: str = str(mesh_prim.GetPath())
        
        # Handle both single material and multiple materials
        self.material_prim: Optional[Any] = material_prim  # For backward compatibility
        self.materials: List[Any] = []  # List of all materials
        self.geom_subsets: List[Dict[str, Any]] = []  # List of GeomSubset info with materials
        
        # Performance optimization: pre-compute commonly used values
        self._material_names_cache: Optional[List[str]] = None
        self._has_material_cache: Optional[bool] = None
        self._has_multiple_materials_cache: Optional[bool] = None
        
        # Initialize materials
        if material_prim:
            self.materials.append(material_prim)
        
        # Find all GeomSubset materials
        self._find_geom_subset_materials_optimized()
        
        logger.debug(f"Creating USDMesh: {self.name} at {self.path}")
        if self.materials:
            material_names = self.get_all_material_names()
            logger.debug(f"  -> Materials: {material_names}")
    
    def _find_geom_subset_materials_optimized(self) -> None:
        """Find all GeomSubset children with material bindings - optimized version."""
        # Batch process children for better performance
        geom_subset_children = [child for child in self.mesh_prim.GetChildren() 
                               if child.GetTypeName() == "GeomSubset"]
        
        for child in geom_subset_children:
            # Check if this GeomSubset has a material binding
            material_binding_api = UsdShade.MaterialBindingAPI(child)
            if material_binding_api:
                direct_binding = material_binding_api.GetDirectBinding()
                if direct_binding and direct_binding.GetMaterial():
                    material = direct_binding.GetMaterial().GetPrim()
                    
                    # Store GeomSubset info
                    subset_info = {
                        'geom_subset': child,
                        'material': material,
                        'name': child.GetName()
                    }
                    self.geom_subsets.append(subset_info)
                    
                    # Add to materials list if not already present
                    if material not in self.materials:
                        self.materials.append(material)
                    
                    logger.debug(f"  -> Found GeomSubset material: {child.GetName()} -> {material.GetName()}")
    
    def has_material(self) -> bool:
        """Check if this mesh has any materials assigned (cached)."""
        if self._has_material_cache is None:
            self._has_material_cache = len(self.materials) > 0
        return self._has_material_cache
    
    def has_multiple_materials(self) -> bool:
        """Check if this mesh has multiple materials (cached)."""
        if self._has_multiple_materials_cache is None:
            self._has_multiple_materials_cache = len(self.materials) > 1
        return self._has_multiple_materials_cache
    
    def get_all_material_names(self) -> List[str]:
        """Get the names of all assigned materials (cached)."""
        if self._material_names_cache is None:
            self._material_names_cache = [mat.GetName() for mat in self.materials]
        return self._material_names_cache
    
    def get_materials(self) -> List[Any]:
        """Get all material prims."""
        return self.materials.copy()
    
    def get_geom_subsets_with_materials(self) -> List[Dict[str, Any]]:
        """Get all GeomSubsets with their associated materials."""
        return self.geom_subsets.copy()
    
    def get_primary_material(self) -> Optional[Any]:
        """Get the primary material (first one or the main material binding)."""
        if self.material_prim:
            return self.material_prim
        elif self.materials:
            return self.materials[0]
        return None
    
    def get_material_count(self) -> int:
        """Get the number of materials assigned to this mesh."""
        return len(self.materials)
    
    def get_material_name(self) -> Optional[str]:
        """Get the name of the first assigned material (for backward compatibility)."""
        return self.materials[0].GetName() if self.materials else None
    
    def __str__(self) -> str:
        if self.has_multiple_materials():
            material_names = self.get_all_material_names()
            material_info = f" ({len(material_names)} materials: {', '.join(material_names)})"
        elif self.has_material():
            material_info = f" (material: {self.get_material_name()})"
        else:
            material_info = " (no materials)"
        
        return f"USDMesh({self.name}{material_info})"
    
    def __repr__(self) -> str:
        return self.__str__()