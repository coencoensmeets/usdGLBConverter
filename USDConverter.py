import json
import struct
import logging
from pxr import Usd, UsdGeom, Gf
import math
from src.math_utils import quat_to_list, quaternion_multiply, quaternion_inverse, euler_to_quat
from typing import List, Tuple, Optional, Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set logger to INFO level (no DEBUG messages)
logger.setLevel(logging.INFO)

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
        self.mesh_prims: List[Any] = []
        self.joints: List['USDJoint'] = []
        self.parent_joint: Optional['USDJoint'] = None
    
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

    def add_mesh(self, mesh_prim: Any) -> None:
        """Add a mesh primitive to this link."""
        self.mesh_prims.append(mesh_prim)

    def add_joint(self, joint: 'USDJoint') -> None:
        """Add a child joint to this link."""
        self.joints.append(joint)
        joint.parent_link = self

    def get_all_child_links(self) -> List['USDLink']:
        """Get all child links recursively."""
        child_links = []
        for joint in self.joints:
            if joint.child_link:
                child_links.append(joint.child_link)
                child_links.extend(joint.child_link.get_all_child_links())
        return child_links

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

        # Rotate local_pos1 by the full relative_rotation
        def rotate_vector(q, v):
            qx, qy, qz, qw = q
            vx, vy, vz = v
            vq = [vx, vy, vz, 0.0]
            q_inv = quaternion_inverse(q)
            t = quaternion_multiply(q, vq)
            t = quaternion_multiply(t, q_inv)
            return [t[0], t[1], t[2]]

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
        local_pos0 = self.get_property('physics:localPos0')
        local_pos1 = self.get_property('physics:localPos1')
        
        pos0 = list(local_pos0) if local_pos0 and hasattr(local_pos0, '__len__') and len(local_pos0) == 3 else None
        pos1 = list(local_pos1) if local_pos1 and hasattr(local_pos1, '__len__') and len(local_pos1) == 3 else None
        
        return pos0, pos1
    
    def get_local_rotations(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """Get local rotations of the joint attachment points on body0 and body1."""
        local_rot0 = self.get_property('physics:localRot0')
        local_rot1 = self.get_property('physics:localRot1')
        
        # USD rotations are quaternions in (w, x, y, z) format, convert to (x, y, z, w)
        rot0 = None
        rot1 = None
        
        local_rot0_list = quat_to_list(local_rot0)
        local_rot1_list = quat_to_list(local_rot1)

        if local_rot0_list and len(local_rot0_list) == 4:
            rot0 = [local_rot0_list[1], local_rot0_list[2], local_rot0_list[3], local_rot0_list[0]]  # w,x,y,z -> x,y,z,w
        
        if local_rot1_list and len(local_rot1_list) == 4:
            rot1 = [local_rot1_list[1], local_rot1_list[2], local_rot1_list[3], local_rot1_list[0]]  # w,x,y,z -> x,y,z,w
        
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
        
        # Fourth pass: find and assign meshes to links
        self._assign_meshes_to_links(root)
        total_meshes = sum(len(link.mesh_prims) for link in self.links.values())
        logger.info(f"Assigned {total_meshes} meshes to links")
        
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
        """Recursively collect joint prims and their relationships."""
        if 'joint' in prim.GetName().lower():
            body0 = body1 = None
            for rel in prim.GetRelationships():
                if rel.GetName() == "physics:body0":
                    targets = rel.GetTargets()
                    if targets:
                        body0 = self.stage.GetPrimAtPath(str(targets[0]))
                elif rel.GetName() == "physics:body1":
                    targets = rel.GetTargets()
                    if targets:
                        body1 = self.stage.GetPrimAtPath(str(targets[0]))
            
            if body0 and body1:
                joint_data.append((prim, body0, body1))
        
        for child in prim.GetChildren():
            self._collect_joints(child, joint_data)
    
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
        """Find and assign mesh prims to their corresponding links."""
        if prim.GetTypeName() == "Mesh":
            # Find which link this mesh belongs to based on hierarchy or naming
            mesh_name = prim.GetName()
            mesh_path = str(prim.GetPath())
            
            logger.debug(f"Found mesh: {mesh_name} at path: {mesh_path}")
            
            # Try to find a matching link by name similarity
            best_match = None
            best_score = 0
            
            for link in self.links.values():
                # Check if mesh name contains link name or vice versa
                link_name_clean = link.name.lower().replace('_', '').replace('-', '')
                mesh_name_clean = mesh_name.lower().replace('_', '').replace('-', '')
                
                # Score based on string similarity
                if link_name_clean in mesh_name_clean or mesh_name_clean in link_name_clean:
                    score = len(set(link_name_clean) & set(mesh_name_clean))
                    if score > best_score:
                        best_score = score
                        best_match = link
                
                # Also try path-based matching - check if mesh is under link's hierarchy
                link_path = str(link.prim.GetPath())
                if mesh_path.startswith(link_path):
                    # Mesh is under this link's hierarchy - high score
                    score = 100 + len(link_path)  # Prefer deeper matches
                    if score > best_score:
                        best_score = score
                        best_match = link
            
            if best_match:
                logger.debug(f"  -> Assigned mesh '{mesh_name}' to link '{best_match.name}' (score: {best_score})")
                best_match.add_mesh(prim)
            else:
                logger.warning(f"  -> No matching link found for mesh: {mesh_name}")
                # Don't assign to any link - some meshes might not belong to robot links
    
        for child in prim.GetChildren():
            self._assign_meshes_to_links(child)
    
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
        mesh_count = len(link.mesh_prims)
        
        # Show mesh info or indicate no meshes
        if mesh_count > 0:
            mesh_names = [mesh.GetName() for mesh in link.mesh_prims]
            mesh_info = f" [{mesh_count} meshes: {', '.join(mesh_names)}]"
        else:
            mesh_info = " [no meshes]"
        
        print(f"{prefix}{connector}{link.name} (Link){mesh_info}")
        
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
        """Validate the joint tree structure for consistency."""
        logger.info("Validating joint tree structure...")
        
        issues = []
        
        # Check 1: Every joint should have both parent and child links
        for joint in self.joints.values():
            if not joint.parent_link:
                issues.append(f"Joint '{joint.name}' has no parent link")
            if not joint.child_link:
                issues.append(f"Joint '{joint.name}' has no child link")
        
        # Check 2: Every link (except base) should have a parent joint
        for link in self.links.values():
            if link != self.base_link and not link.parent_joint:
                issues.append(f"Link '{link.name}' has no parent joint (and is not base link)")
        
        # Check 3: Base link should have no parent joint
        if self.base_link and self.base_link.parent_joint:
            issues.append(f"Base link '{self.base_link.name}' has a parent joint")
        
        # Check 4: No circular references
        visited = set()
        def check_cycles(link, path):
            if link in path:
                issues.append(f"Circular reference detected: {' -> '.join([l.name for l in path] + [link.name])}")
                return
            if link in visited:
                return
            visited.add(link)
            new_path = path + [link]
            for joint in link.joints:
                if joint.child_link:
                    check_cycles(joint.child_link, new_path)
        
        if self.base_link:
            check_cycles(self.base_link, [])
        
        # Check 5: All links should be reachable from base
        if self.base_link:
            reachable = set()
            def mark_reachable(link):
                reachable.add(link)
                for joint in link.joints:
                    if joint.child_link and joint.child_link not in reachable:
                        mark_reachable(joint.child_link)
            
            mark_reachable(self.base_link)
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
        """Get statistics about the joint tree structure."""
        stats = {
            'total_links': len(self.links),
            'total_joints': len(self.joints),
            'base_link': self.base_link.name if self.base_link else None,
            'max_depth': 0,
            'leaf_links': [],
            'links_with_meshes': 0,
            'links_without_meshes': 0,
            'joint_types': {}
        }
        
        # Count links with/without meshes
        for link in self.links.values():
            if link.mesh_prims:
                stats['links_with_meshes'] += 1
            else:
                stats['links_without_meshes'] += 1
        
        # Count joint types
        for joint in self.joints.values():
            joint_type = joint.joint_type
            stats['joint_types'][joint_type] = stats['joint_types'].get(joint_type, 0) + 1
        
        # Calculate tree depth and find leaf links
        def calculate_depth(link, depth=0):
            stats['max_depth'] = max(stats['max_depth'], depth)
            if not link.joints:  # Leaf link
                stats['leaf_links'].append(link.name)
            for joint in link.joints:
                if joint.child_link:
                    calculate_depth(joint.child_link, depth + 1)
        
        if self.base_link:
            calculate_depth(self.base_link)
        
        return stats
    
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
    
    def __str__(self) -> str:
        return f"USDRobot({self.name}, {len(self.links)} links, {len(self.joints)} joints)"
    
    def __repr__(self) -> str:
        return self.__str__()

class USDToGLTFConverter:
    """
    Converts a USD robot to glTF format using the robot hierarchy directly.
    """
    def __init__(self, usd_robot: USDRobot) -> None:
        self.robot: USDRobot = usd_robot
        
        # glTF export data
        self.gltf_nodes: List[Dict[str, Any]] = []
        self.meshes_gltf: List[Dict[str, Any]] = []
        self.accessors: List[Dict[str, Any]] = []
        self.bufferViews: List[Dict[str, Any]] = []
        self.bin_data: bytes = b''
        
        # Mapping from USDLink to glTF node index
        self.link_to_node_idx: Dict[USDLink, int] = {}

    def convert_robot_to_gltf(self) -> None:
        """Convert the USDRobot structure directly to glTF format."""
        if not self.robot.base_link:
            logger.error("No base link found in robot!")
            return
        
        logger.info(f"Converting robot structure starting from base link: {self.robot.base_link.name}")
        
        # Process the robot hierarchy starting from base link
        self._process_link_recursive(self.robot.base_link, parent_node_idx=None)
        
        logger.info(f"Created {len(self.gltf_nodes)} glTF nodes")

    def _process_link_recursive(self, link: USDLink, parent_node_idx: Optional[int] = None) -> int:
        """Recursively process links and create glTF nodes directly."""
        # Create glTF node for this link
        node_idx = len(self.gltf_nodes)
        
        # Create the glTF node dictionary
        gltf_node = self._create_gltf_node_from_link(link)
        
        # Process meshes for this link
        mesh_indices = self._process_link_meshes(link)
        if mesh_indices:
            # For now, just use the first mesh (glTF nodes can only reference one mesh)
            gltf_node["mesh"] = mesh_indices[0]
        
        # Add to nodes list
        self.gltf_nodes.append(gltf_node)
        self.link_to_node_idx[link] = node_idx
        
        logger.debug(f"Created glTF node {node_idx} for link: {link.name}")
        
        # Add to parent's children if it has a parent
        if parent_node_idx is not None:
            if "children" not in self.gltf_nodes[parent_node_idx]:
                self.gltf_nodes[parent_node_idx]["children"] = []
            self.gltf_nodes[parent_node_idx]["children"].append(node_idx)
        
        # Process child joints and links
        for joint in link.joints:
            if joint.child_link:
                child_node_idx = self._process_link_recursive(joint.child_link, node_idx)
        
        return node_idx

    def _create_gltf_node_from_link(self, link: USDLink) -> Dict[str, Any]:
        """Create a glTF node dictionary from a USDLink."""
        node_dict = {"name": link.name}
        
        # Use joint-corrected transform for proper positioning
        relative_translation, relative_rotation, relative_scale = link.calculate_joint_corrected_transform()
        
        # Add transformation data if not default
        if relative_translation != [0.0, 0.0, 0.0]:
            node_dict["translation"] = relative_translation
        if relative_rotation != [0.0, 0.0, 0.0, 1.0]:
            node_dict["rotation"] = relative_rotation
        if relative_scale != [1.0, 1.0, 1.0]:
            node_dict["scale"] = relative_scale
            
        logger.debug(f"glTF node for {link.name}: translation={relative_translation}, rotation={relative_rotation}")
        return node_dict

    def _process_link_meshes(self, link: USDLink) -> List[int]:
        """Process all meshes for a link and return their glTF mesh indices."""
        mesh_indices = []
        
        for mesh_prim in link.mesh_prims:
            logger.debug(f"Processing mesh '{mesh_prim.GetName()}' for link '{link.name}'")
            mesh_idx = self._process_single_mesh(mesh_prim, link)
            if mesh_idx is not None:
                mesh_indices.append(mesh_idx)
        
        return mesh_indices

    def _process_single_mesh(self, mesh_prim: Any, link: USDLink) -> Optional[int]:
        """Process a single mesh primitive and return its glTF mesh index, applying local translation and scale."""
        def get_local_transform_xform(prim: Usd.Prim) -> Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
            """
            Get the local transformation of a prim using Xformable.
            See https://openusd.org/release/api/class_usd_geom_xformable.html
            Args:
                prim: The prim to calculate the local transformation.
            Returns:
                A tuple of:
                - Translation vector.
                - Rotation quaternion, i.e. 3d vector plus angle.
                - Scale vector.
            """
            xform = UsdGeom.Xformable(prim)
            local_transformation: Gf.Matrix4d = xform.GetLocalTransformation()
            translation: Gf.Vec3d = local_transformation.ExtractTranslation()
            rotation: Gf.Rotation = local_transformation.ExtractRotation()
            scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in local_transformation.ExtractRotationMatrix()))
            return translation, rotation, scale
        
        logger.debug(f"Processing single mesh: {mesh_prim.GetName()}")

        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        faceVertexIndices = mesh.GetFaceVertexIndicesAttr().Get()
        faceVertexCounts = mesh.GetFaceVertexCountsAttr().Get()

        if not points:
            logger.warning(f"Mesh '{mesh_prim.GetName()}' has no points - skipping")
            return None

        logger.debug(f"Mesh has {len(points)} vertices and {len(faceVertexCounts)} faces")

        # Apply local translation, rotation, and scale to points using get_local_transform_xform
        translation, rotation, scale = get_local_transform_xform(mesh_prim)

        # Convert rotation (Gf.Rotation) to a rotation matrix
        rot_matrix = Gf.Matrix3d(rotation.GetQuat())

        transformed_points = []
        transformed_points = [
            tuple(
                (rot_matrix * Gf.Vec3d(pt[0] * scale[0], pt[1] * scale[1], pt[2] * scale[2])) + translation
            )
            for pt in points
        ]

        # Convert mesh data to glTF format
        vertices = [coord for pt in transformed_points for coord in pt]
        indices: List[int] = []
        offset = 0

        for count in faceVertexCounts:
            if count == 3:
                indices.extend([
                    faceVertexIndices[offset],
                    faceVertexIndices[offset+1],
                    faceVertexIndices[offset+2]
                ])
            offset += count

        # Pack binary data
        v_bin = struct.pack('<' + 'f'*len(vertices), *vertices)
        i_bin = struct.pack('<' + 'I'*len(indices), *indices)

        v_offset = len(self.bin_data)
        self.bin_data += v_bin
        i_offset = len(self.bin_data)
        self.bin_data += i_bin

        # Create buffer views
        vertex_buffer_view_idx = len(self.bufferViews)
        self.bufferViews.append({
            "buffer": 0,
            "byteOffset": v_offset,
            "byteLength": len(v_bin),
            "target": 34962  # ARRAY_BUFFER
        })

        index_buffer_view_idx = len(self.bufferViews)
        self.bufferViews.append({
            "buffer": 0,
            "byteOffset": i_offset,
            "byteLength": len(i_bin),
            "target": 34963  # ELEMENT_ARRAY_BUFFER
        })

        # Create accessors
        position_accessor_idx = len(self.accessors)
        points_list = [[pt[0], pt[1], pt[2]] for pt in transformed_points]
        self.accessors.append({
            "bufferView": vertex_buffer_view_idx,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": len(points_list),
            "type": "VEC3",
            "max": [float(max(coord)) for coord in zip(*points_list)],
            "min": [float(min(coord)) for coord in zip(*points_list)]
        })

        index_accessor_idx = len(self.accessors)
        self.accessors.append({
            "bufferView": index_buffer_view_idx,
            "byteOffset": 0,
            "componentType": 5125,  # UNSIGNED_INT
            "count": len(indices),
            "type": "SCALAR"
        })

        # Create glTF mesh
        mesh_name = f"{link.name}_{mesh_prim.GetName()}"
        mesh_idx = len(self.meshes_gltf)
        self.meshes_gltf.append({
            "primitives": [{
                "attributes": {"POSITION": position_accessor_idx},
                "indices": index_accessor_idx
            }],
            "name": mesh_name
        })

        logger.debug(f"Created glTF mesh {mesh_idx}: {mesh_name}")
        return mesh_idx

    def export(self, path:str) -> None:
        """Export the robot to glTF format."""
        logger.info(f"Starting USD to glTF conversion for: {self.robot}")
        if logger.isEnabledFor(logging.INFO):
            self.robot.print_structure()
        
        if not self.robot.links:
            logger.error("No links found in USD robot.")
            return
        
        # Convert robot structure directly to glTF
        logger.info("Converting robot structure to glTF")
        self.convert_robot_to_gltf()
        
        # Count total meshes
        total_meshes = sum(len(link.mesh_prims) for link in self.robot.links.values())
        logger.info(f"Processed {len(self.meshes_gltf)} glTF meshes from {total_meshes} USD meshes")
        
        if not self.meshes_gltf:
            logger.warning("No meshes found to export - creating structure-only glTF.")
        
        # Create final glTF structure
        if path.endswith('.gltf'):
            self._create_final_gltf(path)
        elif path.endswith('.glb'):
            self._create_final_glb(path)
        else:
            logger.error("Unsupported file format. Please use .gltf or .glb")
            return
        
        logger.info("Export completed successfully")

    def _create_final_gltf(self, path:str) -> None:
        """Create and write the final glTF file."""
        gltf_path = path
        bin_path = gltf_path.replace('.gltf', '.bin')
        # Find root nodes (nodes with no parent)
        root_nodes: List[int] = []
        if self.robot.base_link and self.robot.base_link in self.link_to_node_idx:
            root_nodes = [self.link_to_node_idx[self.robot.base_link]]
        
        # Create final glTF structure
        gltf: Dict[str, Any] = {
            "asset": {"version": "2.0"},
            "buffers": [{"uri": bin_path, "byteLength": len(self.bin_data)}],
            "bufferViews": self.bufferViews,
            "accessors": self.accessors,
            "meshes": self.meshes_gltf,
            "nodes": self.gltf_nodes,
            "scenes": [{"nodes": root_nodes}],
            "scene": 0
        }
        
        # Write binary file
        with open(bin_path, "wb") as f:
            f.write(self.bin_data)
        
        # Write glTF file
        with open(gltf_path, "w") as f:
            json.dump(gltf, f, indent=2)
        
        logger.info(f"Exported robot to {gltf_path} and {bin_path}")
        logger.info(f"  - {len(self.gltf_nodes)} nodes")
        logger.info(f"  - {len(self.meshes_gltf)} meshes")
        logger.info(f"  - {len(self.accessors)} accessors")
        logger.info(f"  - {len(self.bufferViews)} buffer views")
    
    def _create_final_glb(self, path: str) -> None:
        """Create and write the final glTF binary (GLB) file."""
        import struct
        gltf_path = path
        bin_path = gltf_path.replace('.glb', '.bin')
        # Find root nodes (nodes with no parent)
        root_nodes: List[int] = []
        if self.robot.base_link and self.robot.base_link in self.link_to_node_idx:
            root_nodes = [self.link_to_node_idx[self.robot.base_link]]
        # Create final glTF structure (JSON chunk)
        gltf: Dict[str, Any] = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(self.bin_data)}],
            "bufferViews": self.bufferViews,
            "accessors": self.accessors,
            "meshes": self.meshes_gltf,
            "nodes": self.gltf_nodes,
            "scenes": [{"nodes": root_nodes}],
            "scene": 0
        }
        gltf_json = json.dumps(gltf, separators=(",", ":"))
        gltf_json_bytes = gltf_json.encode('utf-8')
        # Pad JSON chunk to 4 bytes
        while len(gltf_json_bytes) % 4 != 0:
            gltf_json_bytes += b' '
        # Pad BIN chunk to 4 bytes
        bin_data = self.bin_data
        while len(bin_data) % 4 != 0:
            bin_data += b'\x00'
        # GLB header
        magic = b'glTF'
        version = 2
        length = 12 + 8 + len(gltf_json_bytes) + 8 + len(bin_data)
        # Write GLB
        with open(gltf_path, 'wb') as f:
            # Header
            f.write(struct.pack('<4sII', magic, version, length))
            # JSON chunk
            f.write(struct.pack('<I4s', len(gltf_json_bytes), b'JSON'))
            f.write(gltf_json_bytes)
            # BIN chunk
            f.write(struct.pack('<I4s', len(bin_data), b'BIN\x00'))
            f.write(bin_data)
        logger.info(f"Exported robot to {gltf_path} (GLB format)")
        logger.info(f"  - {len(self.gltf_nodes)} nodes")
        logger.info(f"  - {len(self.meshes_gltf)} meshes")
        logger.info(f"  - {len(self.accessors)} accessors")
        logger.info(f"  - {len(self.bufferViews)} buffer views")


if __name__ == "__main__":
    # Enable detailed debug logging to see transform extraction
    # logging.getLogger().setLevel(logging.DEBUG)
    # logger.setLevel(logging.DEBUG)
    
    stage: Usd.Stage = Usd.Stage.Open("Assets/Go2.usd")
    robot: USDRobot = USDRobot(stage, "Go2Robot")
    converter = USDToGLTFConverter(robot)
    converter.export("Output/Go2.glb")