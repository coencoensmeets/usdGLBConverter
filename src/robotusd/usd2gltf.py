import json
import struct
import logging
from pxr import UsdGeom, UsdShade, Gf
import numpy as np
from .robot_structure import USDRobot, USDLink, USDMesh
from .math_utils import HomogeneousMatrix, quat_to_list, quaternion_multiply, quaternion_inverse, rotate_vector,quat_to_euler
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

class USDToGLTFConverter:
    """
    Converts a USD robot to glTF format using the robot hierarchy directly.
    Refactored for clarity and consistent transform handling.
    """
    def __init__(self, usd_robot: USDRobot) -> None:
        self.robot: USDRobot = usd_robot
        
        # glTF export data
        self.gltf_nodes: List[Dict[str, Any]] = []
        self.meshes_gltf: List[Dict[str, Any]] = []
        self.materials_gltf: List[Dict[str, Any]] = []
        self.accessors: List[Dict[str, Any]] = []
        self.bufferViews: List[Dict[str, Any]] = []
        self.bin_data: bytearray = bytearray()  # Use bytearray for better performance
        
        # Mapping from USDLink to glTF node index
        self.link_to_node_idx: Dict[USDLink, int] = {}
        # Mapping from USD material prim to glTF material index
        self.material_to_idx: Dict[Any, int] = {}
        
        # Cache for transform calculations
        self._transform_cache: Dict[Any, Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]] = {}

    def convert_robot_to_gltf(self) -> None:
        """Convert the USDRobot structure to glTF format with joint-centric hierarchy."""
        if not self.robot.base_link:
            logger.error("No base link found in robot!")
            return
        
        logger.info(f"Converting robot structure to joint-centric hierarchy starting from base link: {self.robot.base_link.name}")
        
        # Create root joint node for the base link
        self._process_joint_centric_hierarchy(self.robot.base_link, parent_node_idx=None)
        
        logger.info(f"Created {len(self.gltf_nodes)} glTF nodes in joint-centric hierarchy")

    def _process_joint_centric_hierarchy(self, link: USDLink, parent_node_idx: Optional[int] = None) -> int:
        """Process robot hierarchy in joint-centric format: Joint -> Link (with meshes) -> Child Joints."""
        if link.parent:
            # This link has a parent joint, so create the joint node first
            joint = link.parent
            joint_node_idx = self._create_joint_node(joint, parent_node_idx)
            
            # Create link node as child of joint node
            link_node_idx = self._create_link_node(link, joint_node_idx)
            
            # Process child joints recursively
            for child_joint in link.joints:
                if child_joint.child_link:
                    self._process_joint_centric_hierarchy(child_joint.child_link, joint_node_idx)
            
            return joint_node_idx
        else:
            # This is the base link - create it directly and then process its joints
            link_node_idx = self._create_link_node(link, parent_node_idx)
            
            # Process child joints recursively  
            for child_joint in link.joints:
                if child_joint.child_link:
                    self._process_joint_centric_hierarchy(child_joint.child_link, link_node_idx)
            
            return link_node_idx

    def _create_joint_node(self, joint, parent_node_idx: Optional[int]) -> int:
        """Create a glTF node for a joint using relative transform to parent."""
        joint_node_idx = len(self.gltf_nodes)
        joint_node = {"name": joint.name}

        # Use relative transform between parent and this joint
        print(f"World_self: {joint.transform_world_to_self}")
        parent_joint = joint.parent_joint
        if parent_joint:
            print(f"world_parent_inverse: {parent_joint.transform_world_to_self.inverse()}")
            relative_transform = parent_joint.transform_world_to_self.inverse() * joint.transform_world_to_self
        else:
            # Root joint - use world transform
            relative_transform = joint.transform_world_to_self

        print(f"From {parent_joint.name if parent_joint else 'World'} to {joint.name} - Relative Transform: {relative_transform}")
        translation, quaternion = self._extract_pose_from_matrix(relative_transform)
        
        # Only add non-identity transforms to reduce file size
        if translation != [0.0, 0.0, 0.0]:
            joint_node["translation"] = translation
        if quaternion != [0.0, 0.0, 0.0, 1.0]:
            joint_node["rotation"] = quaternion

        self.gltf_nodes.append(joint_node)
        
        # Add to parent's children if it has a parent
        if parent_node_idx is not None:
            if "children" not in self.gltf_nodes[parent_node_idx]:
                self.gltf_nodes[parent_node_idx]["children"] = []
            self.gltf_nodes[parent_node_idx]["children"].append(joint_node_idx)
        
        logger.debug(f"Created joint node {joint_node_idx} for joint: {joint.name}")
        logger.debug(f"  Translation: {translation}")
        logger.debug(f"  Rotation: {quaternion}")
        return joint_node_idx

    def _create_link_node(self, link: USDLink, parent_node_idx: Optional[int]) -> int:
        """Create a glTF node for a link with its meshes."""
        link_node_idx = len(self.gltf_nodes)
        link_node = {"name": f"{link.name}_link"}

        # For joint-centric hierarchy, links should have the transform from joint to link geometry
        if link.parent:
            # Link has a parent joint - use transform_self_to_child from the joint
            joint = link.parent
            relative_transform = joint.transform_self_to_child
            translation, quaternion = self._extract_pose_from_matrix(relative_transform)
            
            # Only add non-identity transforms to reduce file size
            if translation != [0.0, 0.0, 0.0]:
                link_node["translation"] = translation
            if quaternion != [0.0, 0.0, 0.0, 1.0]:
                link_node["rotation"] = quaternion
        else:
            # Base link - use its local transform
            translation, quaternion = self._extract_pose_from_matrix(link.local_transform)
            if translation != [0.0, 0.0, 0.0]:
                link_node["translation"] = translation
            if quaternion != [0.0, 0.0, 0.0, 1.0]:
                link_node["rotation"] = quaternion

        # Process materials for this link first
        self._process_link_materials(link)
        
        # Process meshes for this link
        mesh_indices = self._process_link_meshes(link)
        if mesh_indices:
            # For now, just use the first mesh (glTF nodes can only reference one mesh)
            link_node["mesh"] = mesh_indices[0]

        self.gltf_nodes.append(link_node)
        self.link_to_node_idx[link] = link_node_idx
        
        # Add to parent's children if it has a parent
        if parent_node_idx is not None:
            if "children" not in self.gltf_nodes[parent_node_idx]:
                self.gltf_nodes[parent_node_idx]["children"] = []
            self.gltf_nodes[parent_node_idx]["children"].append(link_node_idx)
        
        logger.debug(f"Created link node {link_node_idx} for link: {link.name}")
        return link_node_idx

    def _process_link_recursive(self, link: USDLink, parent_node_idx: Optional[int] = None) -> int:
        """Recursively process links and create glTF nodes with intermediate frame nodes."""
        # Create glTF node for this link's geometry
        geometry_node_idx = len(self.gltf_nodes)
        
        # Create the glTF node dictionary for the link geometry
        gltf_node = self._create_gltf_node_from_link(link)
        
        # Process materials for this link first
        self._process_link_materials(link)
        
        # Process meshes for this link
        mesh_indices = self._process_link_meshes(link)
        if mesh_indices:
            # For now, just use the first mesh (glTF nodes can only reference one mesh)
            gltf_node["mesh"] = mesh_indices[0]
        
        # Add to nodes list
        self.gltf_nodes.append(gltf_node)
        self.link_to_node_idx[link] = geometry_node_idx
        
        logger.debug(f"Created glTF geometry node {geometry_node_idx} for link: {link.name}")
        
        # Add to parent's children if it has a parent
        if parent_node_idx is not None:
            if "children" not in self.gltf_nodes[parent_node_idx]:
                self.gltf_nodes[parent_node_idx]["children"] = []
            self.gltf_nodes[parent_node_idx]["children"].append(geometry_node_idx)
        
        # Process child joints and links
        for joint in link.joints:
            if joint.child_link:
                # Create intermediate frame node for this joint
                frame_node_idx = self._create_joint_frame_node(joint, geometry_node_idx)
                child_node_idx = self._process_link_recursive(joint.child_link, frame_node_idx)
        
        return geometry_node_idx

    def _create_gltf_node_from_link(self, link: USDLink) -> Dict[str, Any]:
        """Create a glTF node dictionary from a USDLink with proper relative transforms."""
        node_dict = {"name": link.name}

        if link.parent:
            # Use the joint's transform property for the link's transform relative to the joint frame
            local_matrix = link.transform_parent_to_self
            print(f"From {link.parent.name} to {link.name} - Local Matrix: {local_matrix}")
            translation, quaternion = self._extract_pose_from_matrix(local_matrix)
            node_dict["translation"] = translation
            node_dict["rotation"] = quaternion
            logger.debug(f"glTF link node for {link.name} (child of joint {link.parent.name}):")
            logger.debug(f"  Final translation: {translation}")
            logger.debug(f"  Final quaternion: {quaternion}")
        else:
            # Base link - use its local transform (relative to world origin)
            local_matrix = link.local_transform
            translation, quaternion = self._extract_pose_from_matrix(local_matrix)
            node_dict["translation"] = translation
            node_dict["rotation"] = quaternion
            logger.debug(f"glTF base link node for {link.name}: translation={translation}, rotation={quaternion}")
        return node_dict
    
    def _create_joint_frame_node(self, joint, parent_node_idx: int) -> int:
        """Create an intermediate frame node for a joint using its transform_parent_to_joint property."""
        frame_node_idx = len(self.gltf_nodes)
        frame_node = {"name": f"{joint.name}_frame"}

        # Use the joint's transform_parent_to_joint property for the frame node
        joint_frame_matrix = joint.transform_parent_to_self
        print(f"from {joint.parent.name} to {joint.name} - Joint Frame Matrix: {joint_frame_matrix}")
        translation, quaternion = self._extract_pose_from_matrix(joint_frame_matrix)
        if translation != [0.0, 0.0, 0.0]:
            frame_node["translation"] = translation
        if quaternion != [0.0, 0.0, 0.0, 1.0]:
            frame_node["rotation"] = quaternion

        self.gltf_nodes.append(frame_node)
        if "children" not in self.gltf_nodes[parent_node_idx]:
            self.gltf_nodes[parent_node_idx]["children"] = []
        self.gltf_nodes[parent_node_idx]["children"].append(frame_node_idx)
        logger.debug(f"Created joint frame node {frame_node_idx} for joint: {joint.name}")
        logger.debug(f"  Translation: {translation}")
        logger.debug(f"  Rotation: {quaternion}")
        return frame_node_idx

    def _process_link_meshes(self, link: USDLink) -> List[int]:
        """Process all meshes for a link and return their glTF mesh indices."""
        mesh_indices = []
        
        for usd_mesh in link.meshes:
            logger.debug(f"Processing mesh '{usd_mesh.name}' for link '{link.name}'")
            mesh_idx = self._process_single_mesh(usd_mesh, link)
            if mesh_idx is not None:
                mesh_indices.append(mesh_idx)
        
        return mesh_indices

    def _process_single_mesh(self, usd_mesh: USDMesh, link: USDLink) -> Optional[int]:
        """Process a single mesh primitive and return its glTF mesh index, applying local translation and scale."""
        logger.debug(f"Processing single mesh: {usd_mesh.name}")

        mesh = UsdGeom.Mesh(usd_mesh.mesh_prim)
        points = mesh.GetPointsAttr().Get()
        faceVertexIndices = mesh.GetFaceVertexIndicesAttr().Get()
        faceVertexCounts = mesh.GetFaceVertexCountsAttr().Get()

        if not points:
            logger.warning(f"Mesh '{usd_mesh.name}' has no points - skipping")
            return None

        logger.debug(f"Mesh has {len(points)} vertices and {len(faceVertexCounts)} faces")
        if usd_mesh.has_multiple_materials():
            logger.debug(f"Mesh has multiple materials: {usd_mesh.get_all_material_names()}")

        # Get the 4x4 transformation matrix from the link
        # if self.robot.get_object_from_primName(link.prim.GetName()) is None:
        #     logger.warning(f"Link '{link.name}' not found in robot structure - Using no additional transformation")
        #     translation = [0.0, 0.0, 0.0]
        #     quaternion = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion (x, y, z, w)
        #     scale = [1.0, 1.0, 1.0]
        # else:
        #     transform_world_to_parent = self.robot.get_object_from_primName(link.prim.GetParent().GetName()).transform_world_to_self
        #     transform_parent_to_mesh = link.local_transform
        #     transform_world_to_link = link.transform_world_to_self
        #     transform_link_to_mesh = transform_world_to_link.inverse() * transform_world_to_parent * transform_parent_to_mesh
        #     translation, quaternion = transform_link_to_mesh.pose
        #     print(f"Test: {quaternion}")
        #     if hasattr(link.scale, '__len__') and len(link.scale) == 3:
        #         scale = [float(link.scale[0]), float(link.scale[1]), float(link.scale[2])]
        #     else:
        #         scale = [1.0, 1.0, 1.0]  # Default scale
                
        translation = [0,0,0]
        quaternion = [0,0,0,1]  # Default to identity if no link transform
        scale = [1.0, 1.0, 1.0]  # Default scale if no link transform

        # Get mesh local transformation
        xform = UsdGeom.Xformable(usd_mesh.mesh_prim)
        local_transformation = xform.GetLocalTransformation()
        mesh_translation = local_transformation.ExtractTranslation()
        mesh_rotation = local_transformation.ExtractRotation()
        mesh_scale_matrix = local_transformation.ExtractRotationMatrix()
        mesh_scale = [mesh_scale_matrix[0].GetLength(), mesh_scale_matrix[1].GetLength(), mesh_scale_matrix[2].GetLength()]
        
        # Convert mesh transformation to lists
        mesh_translation = [float(mesh_translation[0]), float(mesh_translation[1]), float(mesh_translation[2])]
        quat = mesh_rotation.GetQuat()
        mesh_quaternion = [float(quat.GetImaginary()[0]), float(quat.GetImaginary()[1]), float(quat.GetImaginary()[2]), float(quat.GetReal())]  # [x, y, z, w]
        mesh_scale = [float(s) for s in mesh_scale]
        
        # mesh_translation = [0,0,0]
        # mesh_quaternion = [0,0,0,1]  # Default to identity if no mesh transform
        # mesh_scale = [1.0, 1.0, 1.0]  # Default scale if no mesh transform
            
        # Optimize point transformation using numpy
        points_array = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float32)
        
        # Combine all transformations first
        # Combine scales (multiply)
        final_scale = np.array(mesh_scale, dtype=np.float32) * np.array(scale, dtype=np.float32)
        
        # Combine rotations (quaternion multiplication: link * mesh)
        # First normalize quaternions to ensure they're unit quaternions
        mesh_quat_norm = np.linalg.norm(mesh_quaternion)
        link_quat_norm = np.linalg.norm(quaternion)
        
        if mesh_quat_norm > 0:
            mesh_q = np.array(mesh_quaternion, dtype=np.float32) / mesh_quat_norm
        else:
            mesh_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            
        if link_quat_norm > 0:
            link_q = np.array(quaternion, dtype=np.float32) / link_quat_norm
        else:
            link_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        
        # Quaternion multiplication: q1 * q2 (link * mesh)
        x1, y1, z1, w1 = link_q
        x2, y2, z2, w2 = mesh_q
        final_quaternion = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
            w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
        ], dtype=np.float32)
        
        # Combine translations (apply mesh translation first, then transform by link rotation, then add link translation)
        mesh_trans = np.array(mesh_translation, dtype=np.float32)
        link_trans = np.array(translation, dtype=np.float32)
        
        # Rotate mesh translation by link rotation
        if not np.allclose(link_q, [0.0, 0.0, 0.0, 1.0]):
            x, y, z, w = link_q
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            
            link_rot_matrix = np.array([
                [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
                [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
                [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
            ], dtype=np.float32)
            
            rotated_mesh_trans = np.dot(mesh_trans, link_rot_matrix.T)
        else:
            rotated_mesh_trans = mesh_trans
            
        final_translation = rotated_mesh_trans + link_trans
        
        # Apply combined transformations
        # Apply final scale
        if not np.allclose(final_scale, [1.0, 1.0, 1.0]):
            points_array *= final_scale
        
        # Apply final rotation
        if not np.allclose(final_quaternion, [0.0, 0.0, 0.0, 1.0]):
            x, y, z, w = final_quaternion
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z

            final_rot_matrix = np.array([
                [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
                [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
                [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
            ], dtype=np.float32)
            points_array = np.dot(points_array, final_rot_matrix.T)
        
        # Apply final translation
        if not np.allclose(final_translation, [0.0, 0.0, 0.0]):
            points_array += final_translation

        # Convert to list for glTF
        vertices = points_array.flatten().tolist()
        
        # Pack binary data for vertices (shared by all primitives)
        v_bin = struct.pack('<' + 'f'*len(vertices), *vertices)
        v_offset = len(self.bin_data)
        self.bin_data.extend(v_bin)

        # Create vertex buffer view
        vertex_buffer_view_idx = len(self.bufferViews)
        self.bufferViews.append({
            "buffer": 0,
            "byteOffset": v_offset,
            "byteLength": len(v_bin),
            "target": 34962  # ARRAY_BUFFER
        })

        # Create position accessor with optimized min/max calculation
        position_accessor_idx = len(self.accessors)
        min_coords = points_array.min(axis=0).tolist()
        max_coords = points_array.max(axis=0).tolist()
        
        self.accessors.append({
            "bufferView": vertex_buffer_view_idx,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": len(points_array),
            "type": "VEC3",
            "max": max_coords,
            "min": min_coords
        })

        # Create glTF mesh
        mesh_name = f"{link.name}_{usd_mesh.name}"
        mesh_idx = len(self.meshes_gltf)
        
        # Create primitives - handle multiple materials
        primitives = []
        
        if usd_mesh.has_multiple_materials() and usd_mesh.get_geom_subsets_with_materials():
            # Handle GeomSubsets with different materials
            for subset_info in usd_mesh.get_geom_subsets_with_materials():
                geom_subset = subset_info['geom_subset']
                material_prim = subset_info['material']
                
                # Get indices for this subset
                subset_indices = self._get_geom_subset_indices_optimized(geom_subset, faceVertexIndices, faceVertexCounts)
                if not subset_indices:
                    continue
                
                # Create index buffer for this subset
                i_bin = struct.pack('<' + 'I'*len(subset_indices), *subset_indices)
                i_offset = len(self.bin_data)
                self.bin_data.extend(i_bin)

                # Create index buffer view
                index_buffer_view_idx = len(self.bufferViews)
                self.bufferViews.append({
                    "buffer": 0,
                    "byteOffset": i_offset,
                    "byteLength": len(i_bin),
                    "target": 34963  # ELEMENT_ARRAY_BUFFER
                })

                # Create index accessor
                index_accessor_idx = len(self.accessors)
                self.accessors.append({
                    "bufferView": index_buffer_view_idx,
                    "byteOffset": 0,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": len(subset_indices),
                    "type": "SCALAR"
                })

                # Create primitive for this subset
                primitive = {
                    "attributes": {"POSITION": position_accessor_idx},
                    "indices": index_accessor_idx
                }
                
                # Process material for this subset
                material_idx = self._process_single_material(material_prim, link)
                if material_idx is not None:
                    primitive["material"] = material_idx
                    logger.debug(f"Assigned material {material_idx} ({material_prim.GetName()}) to subset {subset_info['name']}")
                
                primitives.append(primitive)
        else:
            # Handle single material or no material (traditional approach) - optimized
            indices = self._triangulate_faces_optimized(faceVertexIndices, faceVertexCounts)

            # Pack binary data for indices
            i_bin = struct.pack('<' + 'I'*len(indices), *indices)
            i_offset = len(self.bin_data)
            self.bin_data.extend(i_bin)

            # Create index buffer view
            index_buffer_view_idx = len(self.bufferViews)
            self.bufferViews.append({
                "buffer": 0,
                "byteOffset": i_offset,
                "byteLength": len(i_bin),
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            })

            # Create index accessor
            index_accessor_idx = len(self.accessors)
            self.accessors.append({
                "bufferView": index_buffer_view_idx,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(indices),
                "type": "SCALAR"
            })

            # Create primitive with attributes and indices
            primitive = {
                "attributes": {"POSITION": position_accessor_idx},
                "indices": index_accessor_idx
            }
            
            # Use the primary material from the USDMesh if available
            if usd_mesh.has_material():
                primary_material = usd_mesh.get_primary_material()
                material_idx = self._process_single_material(primary_material, link)
                if material_idx is not None:
                    primitive["material"] = material_idx
                    logger.debug(f"Assigned material {material_idx} to mesh {mesh_name}")
            else:
                logger.debug(f"Mesh {mesh_name} has no material")
            
            primitives.append(primitive)
        
        self.meshes_gltf.append({
            "primitives": primitives,
            "name": mesh_name
        })

        logger.debug(f"Created glTF mesh {mesh_idx}: {mesh_name} with {len(primitives)} primitives")
        return mesh_idx

    def _process_link_materials(self, link: USDLink) -> None:
        """Process all materials for a link (now handled per-mesh, so this is deprecated)."""
        # Materials are now processed per-mesh in _process_single_mesh
        # This method is kept for backwards compatibility but does nothing
        logger.debug(f"Material processing for link '{link.name}' is now handled per-mesh")

    def _process_single_material(self, material_prim: Any, link: USDLink) -> Optional[int]:
        """Process a single material primitive and return its glTF material index."""
        # Check if we've already processed this material
        if material_prim in self.material_to_idx:
            return self.material_to_idx[material_prim]
        
        logger.debug(f"Processing single material: {material_prim.GetName()}")
        
        material = UsdShade.Material(material_prim)
        if not material:
            logger.warning(f"Material '{material_prim.GetName()}' is not a valid UsdShade.Material - skipping")
            return None
        
        material_name = f"{link.name}_{material_prim.GetName()}"
        material_idx = len(self.materials_gltf)
        
        # Create glTF material dictionary
        gltf_material = {
            "name": material_name,
            "pbrMetallicRoughness": {}
        }
        
        # Extract material properties
        surface_attr = material.GetSurfaceAttr()
        shader_prim = None
        if surface_attr:
            connected_source = UsdShade.ConnectableAPI.GetConnectedSource(surface_attr)
            if connected_source and connected_source[0]:
                shader_prim = connected_source[0]
        if not shader_prim:
            # Try to find a shader in the material's children
            for child in material_prim.GetChildren():
                if child.GetTypeName() in ["Shader", "UsdPreviewSurface", "MaterialX"]:
                    shader_prim = child
                    logger.debug(f"Found shader as child: {shader_prim.GetName()}")
                    break
        if shader_prim:
            # Try to extract common PBR properties
            base_color = self._extract_shader_property(shader_prim, "diffuse_color_constant", [1.0, 1.0, 1.0])
            if base_color != [1.0, 1.0, 1.0]:
                gltf_material["pbrMetallicRoughness"]["baseColorFactor"] = list(base_color) + [1.0]  # Add alpha

            # Try to extract metallic and roughness (OmniPBR may use different names, so fallback to defaults)
            metallic = self._extract_shader_property(shader_prim, "metallic_constant", 0.0)
            if metallic != 0.0:
                gltf_material["pbrMetallicRoughness"]["metallicFactor"] = metallic

            roughness = self._extract_shader_property(shader_prim, "roughness_constant", 0.9)
            if roughness != 0.9:
                gltf_material["pbrMetallicRoughness"]["roughnessFactor"] = roughness

            # Try to extract emissive (OmniPBR may use emissive_color_constant)
            emissive = self._extract_shader_property(shader_prim, "emissive_color_constant", [0.0, 0.0, 0.0])
            if emissive != [0.0, 0.0, 0.0]:
                gltf_material["emissiveFactor"] = list(emissive)
        else:
            # Fallback: try to extract basic material properties from the material prim itself
            logger.debug(f"No surface shader found for material {material_name}, using fallback properties")

            # Set default PBR values
            gltf_material["pbrMetallicRoughness"]["baseColorFactor"] = [0.8, 0.8, 0.8, 1.0]
            gltf_material["pbrMetallicRoughness"]["metallicFactor"] = 0.0
            gltf_material["pbrMetallicRoughness"]["roughnessFactor"] = 0.9
        
        # Add the material to the list
        self.materials_gltf.append(gltf_material)
        self.material_to_idx[material_prim] = material_idx
        
        logger.debug(f"Created glTF material {material_idx}: {material_name}")
        return material_idx
    
    def _extract_shader_property(self, shader_prim: Any, property_name: str, default_value: Any) -> Any:
        """Extract a property value from a shader prim."""
        try:
            attr = shader_prim.GetAttribute(f"inputs:{property_name}")
            if attr and attr.IsValid():
                value = attr.Get()
                if value is not None:
                    # Convert USD types to Python types
                    if hasattr(value, '__len__') and len(value) == 3:
                        return list(value)
                    elif isinstance(value, (int, float)):
                        return float(value)
                    else:
                        return value
        except Exception as e:
            logger.debug(f"Failed to extract property {property_name} from shader: {e}")
        
        return default_value
    
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
        
        # Count total meshes and materials
        total_meshes = sum(len(link.meshes) for link in self.robot.links.values())
        total_materials = sum(mesh.get_material_count() for link in self.robot.links.values() for mesh in link.meshes)
        meshes_with_multiple_materials = sum(1 for link in self.robot.links.values() for mesh in link.meshes if mesh.has_multiple_materials())
        
        logger.info(f"Processed {len(self.meshes_gltf)} glTF meshes from {total_meshes} USD meshes")
        logger.info(f"Processed {len(self.materials_gltf)} glTF materials from {total_materials} USD materials")
        if meshes_with_multiple_materials > 0:
            logger.info(f"Found {meshes_with_multiple_materials} meshes with multiple materials")
        
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
        # Find root nodes (in joint-centric hierarchy, this is the base link or first joint)
        root_nodes: List[int] = []
        
        # Find the first node that's not a child of any other node
        children_set = set()
        for node in self.gltf_nodes:
            if "children" in node:
                children_set.update(node["children"])
        
        for i, node in enumerate(self.gltf_nodes):
            if i not in children_set:
                root_nodes.append(i)
                break  # We expect only one root in a robot hierarchy
        
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
        
        # Add materials if any exist
        if self.materials_gltf:
            gltf["materials"] = self.materials_gltf
        
        # Write binary file
        with open(bin_path, "wb") as f:
            f.write(self.bin_data)
        
        # Write glTF file
        with open(gltf_path, "w") as f:
            json.dump(gltf, f, indent=2)
        
        logger.info(f"Exported robot to {gltf_path} and {bin_path}")
        logger.info(f"  - {len(self.gltf_nodes)} nodes")
        logger.info(f"  - {len(self.meshes_gltf)} meshes")
        logger.info(f"  - {len(self.materials_gltf)} materials")
        logger.info(f"  - {len(self.accessors)} accessors")
        logger.info(f"  - {len(self.bufferViews)} buffer views")
    
    def _create_final_glb(self, path: str) -> None:
        """Create and write the final glTF binary (GLB) file."""
        gltf_path = path
        # Find root nodes (in joint-centric hierarchy, this is the base link or first joint)
        root_nodes: List[int] = []
        
        # Find the first node that's not a child of any other node
        children_set = set()
        for node in self.gltf_nodes:
            if "children" in node:
                children_set.update(node["children"])
        
        for i, node in enumerate(self.gltf_nodes):
            if i not in children_set:
                root_nodes.append(i)
                break  # We expect only one root in a robot hierarchy
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
        
        # Add materials if any exist
        if self.materials_gltf:
            gltf["materials"] = self.materials_gltf
        
        # Optimize JSON serialization
        gltf_json = json.dumps(gltf, separators=(",", ":"))
        gltf_json_bytes = gltf_json.encode('utf-8')
        
        # Pad JSON chunk to 4 bytes
        json_padding = (4 - len(gltf_json_bytes) % 4) % 4
        if json_padding:
            gltf_json_bytes += b' ' * json_padding
        
        # Pad BIN chunk to 4 bytes
        bin_data = bytes(self.bin_data)
        bin_padding = (4 - len(bin_data) % 4) % 4
        if bin_padding:
            bin_data += b'\x00' * bin_padding
        
        # GLB header
        magic = b'glTF'
        version = 2
        length = 12 + 8 + len(gltf_json_bytes) + 8 + len(bin_data)
        
        # Write GLB in one go for better performance
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
        logger.info(f"  - {len(self.materials_gltf)} materials")
        logger.info(f"  - {len(self.accessors)} accessors")
        logger.info(f"  - {len(self.bufferViews)} buffer views")

    def _triangulate_faces_optimized(self, face_vertex_indices: List[int], face_vertex_counts: List[int]) -> List[int]:
        """Optimized triangulation of faces using list comprehension."""
        indices = []
        offset = 0
        
        # Pre-allocate list size for better performance
        triangle_count = sum(1 for count in face_vertex_counts if count == 3)
        if triangle_count > 0:
            indices = [0] * (triangle_count * 3)
            idx_pos = 0
            
            for count in face_vertex_counts:
                if count == 3:
                    indices[idx_pos] = face_vertex_indices[offset]
                    indices[idx_pos + 1] = face_vertex_indices[offset + 1]
                    indices[idx_pos + 2] = face_vertex_indices[offset + 2]
                    idx_pos += 3
                offset += count
        
        return indices
    
    def _get_geom_subset_indices_optimized(self, geom_subset_prim: Any, face_vertex_indices: List[int], face_vertex_counts: List[int]) -> List[int]:
        """Optimized version of GeomSubset index extraction."""
        try:
            geom_subset = UsdGeom.Subset(geom_subset_prim)
            subset_indices_attr = geom_subset.GetIndicesAttr()
            if not subset_indices_attr:
                logger.warning(f"GeomSubset {geom_subset_prim.GetName()} has no indices attribute")
                return []
            
            subset_face_indices = subset_indices_attr.Get()
            if not subset_face_indices:
                logger.warning(f"GeomSubset {geom_subset_prim.GetName()} has empty indices")
                return []
            
            # Convert to set for O(1) lookup
            subset_face_set = set(subset_face_indices)
            
            # Pre-calculate triangle count
            triangle_count = sum(1 for face_idx, face_count in enumerate(face_vertex_counts) 
                               if face_idx in subset_face_set and face_count == 3)
            
            # Pre-allocate list
            triangle_indices = [0] * (triangle_count * 3)
            vertex_offset = 0
            output_idx = 0
            
            for face_idx, face_count in enumerate(face_vertex_counts):
                if face_idx in subset_face_set and face_count == 3:
                    triangle_indices[output_idx] = face_vertex_indices[vertex_offset]
                    triangle_indices[output_idx + 1] = face_vertex_indices[vertex_offset + 1]
                    triangle_indices[output_idx + 2] = face_vertex_indices[vertex_offset + 2]
                    output_idx += 3
                vertex_offset += face_count
            
            logger.debug(f"GeomSubset {geom_subset_prim.GetName()}: {len(subset_face_indices)} faces -> {len(triangle_indices)} triangle indices")
            return triangle_indices
            
        except Exception as e:
            logger.error(f"Error processing GeomSubset {geom_subset_prim.GetName()}: {e}")
            return []

    def _extract_pose_from_matrix(self, matrix: HomogeneousMatrix) -> Tuple[List[float], List[float]]:
        """Helper to extract translation and quaternion from a HomogeneousMatrix."""
        translation, quaternion = matrix.pose
        # Ensure lists for JSON serialization
        translation = list(translation)
        quaternion = list(quaternion)
        return translation, quaternion