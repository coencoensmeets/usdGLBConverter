import json
import struct
import logging
from pxr import Usd, UsdGeom, UsdShade, Gf
import math
import numpy as np
from src.math_utils import quat_to_list, quaternion_multiply, quaternion_inverse, euler_to_quat
from src.robot_structure import USDRobot, USDLink, USDJoint, USDMesh
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

class USDToGLTFConverter:
    """
    Converts a USD robot to glTF format using the robot hierarchy directly.
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
        
        # Process materials for this link first
        self._process_link_materials(link)
        
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

        # Get cached transform or calculate it
        mesh_path = str(usd_mesh.mesh_prim.GetPath())
        if mesh_path not in self._transform_cache:
            xform = UsdGeom.Xformable(usd_mesh.mesh_prim)
            local_transformation: Gf.Matrix4d = xform.GetLocalTransformation()
            translation: Gf.Vec3d = local_transformation.ExtractTranslation()
            rotation: Gf.Rotation = local_transformation.ExtractRotation()
            scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in local_transformation.ExtractRotationMatrix()))
            self._transform_cache[mesh_path] = (translation, rotation, scale)
        
        translation, rotation, scale = self._transform_cache[mesh_path]

        # Optimize point transformation using numpy
        points_array = np.array([(p[0], p[1], p[2]) for p in points], dtype=np.float32)
        
        # Apply scale
        if scale != Gf.Vec3d(1.0, 1.0, 1.0):
            scale_array = np.array([scale[0], scale[1], scale[2]], dtype=np.float32)
            points_array *= scale_array
        
        # Apply rotation if not identity
        if rotation.GetQuat() != Gf.Quatd(1.0, 0.0, 0.0, 0.0):
            rot_matrix = np.array(Gf.Matrix3d(rotation.GetQuat()), dtype=np.float32)
            points_array = np.dot(points_array, rot_matrix.T)
        
        # Apply translation
        if translation != Gf.Vec3d(0.0, 0.0, 0.0):
            translation_array = np.array([translation[0], translation[1], translation[2]], dtype=np.float32)
            points_array += translation_array
        
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


if __name__ == "__main__":
    # Enable detailed debug logging to see transform extraction
    # logging.getLogger().setLevel(logging.DEBUG)
    # logger.setLevel(logging.DEBUG)
    
    stage: Usd.Stage = Usd.Stage.Open("Assets/Go2.usd")
    robot: USDRobot = USDRobot(stage, "Go2Robot")
    converter = USDToGLTFConverter(robot)
    converter.export("Output/Go2.glb")
    
    # stage: Usd.Stage = Usd.Stage.Open("Assets/Robots/Unitree/G1/g1.usd")
    # robot: USDRobot = USDRobot(stage, "G1")
    # converter = USDToGLTFConverter(robot)
    # converter.export("Output/G1.glb")
    
    # stage: Usd.Stage = Usd.Stage.Open("Assets/Robots/Franka/franka.usd")
    # robot: USDRobot = USDRobot(stage, "Franka")
    # converter = USDToGLTFConverter(robot)
    # converter.export("Output/Franka.glb")
    
    # stage: Usd.Stage = Usd.Stage.Open("Assets/Robots/Festo/FestoCobot/festo_cobot.usd")
    # robot: USDRobot = USDRobot(stage, "Festo")
    # converter = USDToGLTFConverter(robot)
    # converter.export("Output/Festo.glb")