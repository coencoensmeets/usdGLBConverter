from pxr import Usd, Sdf
from pxr import UsdGeom, UsdSkel

def print_tree(prim, prefix="", is_last=True, parent_prim=None):
    """
    Print a tree structure of the USD scene hierarchy with joint information.
    
    Args:
        prim: The USD prim to print
        prefix: String prefix for indentation
        is_last: Whether this is the last child at this level
        parent_prim: The parent prim for joint relationship tracking
    """
    # Get the prim type and name
    prim_type = prim.GetTypeName()
    prim_name = prim.GetName()
    
    # Create the tree connector
    connector = "└── " if is_last else "├── "
    
    # Create type description with joint information
    type_desc = ""
    joint_info = ""
    
    # Check if this is a joint (contains 'joint' in name or is a joint type)
    is_joint = ('joint' in prim_name.lower() or 
               prim_type in ["Joint", "SkelRoot"] or
               has_joint_properties(prim))
    
    if prim_type == "Xform":
        if is_joint:
            type_desc = " (Xform/Joint) - joint positioning"
        else:
            type_desc = " (Xform) - positioning"
    elif prim_type == "Mesh":
        type_desc = " (Mesh) - actual geometry"
    elif prim_type == "Scope":
        type_desc = " (Scope) - organization"
    elif prim_type == "SkelRoot":
        type_desc = " (SkelRoot) - skeleton root"
    elif prim_type == "Skeleton":
        type_desc = " (Skeleton) - skeleton definition"
    elif prim_type:
        type_desc = f" ({prim_type})"
    
    # Add joint relationship information
    if is_joint:
        # Check for physics relationships
        body0 = body1 = None
        
        for rel in prim.GetRelationships():
            if rel.GetName() == "physics:body0":
                targets = rel.GetTargets()
                if targets:
                    body0_prim = prim.GetStage().GetPrimAtPath(str(targets[0]))
                    body0 = body0_prim.GetName() if body0_prim else str(targets[0])
            if rel.GetName() == "physics:body1":
                targets = rel.GetTargets()
                if targets:
                    body1_prim = prim.GetStage().GetPrimAtPath(str(targets[0]))
                    body1 = body1_prim.GetName() if body1_prim else str(targets[0])
        
        if body0 and body1:
            joint_info += f" [Connects: {body0} ↔ {body1}]"
        else:
            # Fallback to hierarchical relationships
            joint_children = get_joint_children(prim)
            if parent_prim and is_joint_parent(parent_prim):
                joint_info += f" [Parent: {parent_prim.GetName()}]"
            if joint_children:
                child_names = [child.GetName() for child in joint_children]
                joint_info += f" [Children: {', '.join(child_names)}]"
    
    # Print the current prim
    print(f"{prefix}{connector}{prim_name}{type_desc}{joint_info}")
    
    # Get children
    children = list(prim.GetChildren())
    
    # Print children
    for i, child in enumerate(children):
        is_child_last = (i == len(children) - 1)
        child_prefix = prefix + ("    " if is_last else "│   ")
        print_tree(child, child_prefix, is_child_last, prim)
        
def has_joint_properties(prim):
    """Check if a prim has joint-like properties."""
    # Check for common joint attributes
    if hasattr(prim, 'GetAttribute'):
        joint_attrs = ['xformOp:rotateX', 'xformOp:rotateY', 'xformOp:rotateZ', 
                      'xformOp:rotateXYZ', 'joints', 'bindTransforms']
        for attr_name in joint_attrs:
            if prim.GetAttribute(attr_name).IsValid():
                return True
    return False

def is_joint_parent(prim):
    """Check if a prim is a parent joint."""
    return ('joint' in prim.GetName().lower() or 
            prim.GetTypeName() in ["Joint", "SkelRoot"] or
            has_joint_properties(prim))

def get_joint_children(prim):
    """Get child prims that are joints."""
    joint_children = []
    for child in prim.GetChildren():
        if ('joint' in child.GetName().lower() or 
            child.GetTypeName() in ["Joint", "SkelRoot"] or
            has_joint_properties(child)):
            joint_children.append(child)
    return joint_children

def get_joint_tree(prim):
    """Get a tree structure of prims based on the joint relationships."""
    joint_tree = {}
    
    joints = []
    
    all_bodies = set()
    
    def collect_joints(current_prim):
        if 'joint' in current_prim.GetTypeName().lower():
            # print(f"Found joint: {current_prim}")
            # print(f"  Name: {current_prim.GetName()}")
            # print(f"  Path: {current_prim.GetPath()}")
            # print(f"  Type: {current_prim.GetTypeName()}")
            # print(f"  Properties: {[prop.GetName() for prop in current_prim.GetProperties()]}")
            for prop_name in ['physics:localPos0', 'physics:localPos1', 'physics:localRot0', 'physics:localRot1']:
                attr = current_prim.GetAttribute(prop_name)
            body0 = body1 = None
            for rel in current_prim.GetRelationships():
                if rel.GetName() == "physics:body0":
                    targets = rel.GetTargets()
                    if targets:
                        body0 = current_prim.GetStage().GetPrimAtPath(str(targets[0]))
                if rel.GetName() == "physics:body1":
                    targets = rel.GetTargets()
                    if targets:
                        body1 = current_prim.GetStage().GetPrimAtPath(str(targets[0]))
            if body0 and body1:
                all_bodies.add(body0)
                all_bodies.add(body1)
                joints.append((current_prim, body0, body1))
        
        for child in current_prim.GetChildren():
            collect_joints(child)
    
    collect_joints(prim)
    
    if not joints:
        return {}
    
    edges = {}
    base_bodies = all_bodies.copy()
    for joint_prim, parentBody, childBody in joints:
        if parentBody not in edges:
            edges[parentBody] = []
        if childBody in base_bodies:
            base_bodies.remove(childBody)
        edges[parentBody].append(childBody)
    
    def build_tree(parent):
        """Recursively build the joint tree structure."""
        if parent not in edges:
            return {'name': parent.GetName(), 'prim': parent, 'children': []}
        
        node = {'name': parent.GetName(), 'prim': parent, 'children': []}
        for child in edges[parent]:
            child_node = build_tree(child)
            node['children'].append(child_node)
        
        return node
    
    # Build the joint tree starting from base bodies
    for base_body in base_bodies:
        joint_tree[base_body.GetName()] = build_tree(base_body)
        
    # If no base bodies, use the first joint as root
    if not joint_tree:
        print("[Warning] No base bodies found, using first joint as root.")
        first_joint = joints[0][0]  # Get the first joint prim
        joint_tree[first_joint.GetName()] = build_tree(first_joint)
    
    # Return the joint tree structure
    if not joint_tree:
        return {}
    
    return joint_tree

def print_joint_tree(joint_tree, prefix="", prim_name="", is_last=True):
    """Print the joint tree structure."""
    if not joint_tree:
        return
    
    if isinstance(joint_tree, dict) and 'prim' in joint_tree:
        # This is a single tree node
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{joint_tree['name']}")
        
        children = joint_tree['children']
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            print_joint_tree(child, child_prefix, "", is_child_last)
    else:
        # This is the root dictionary
        root_keys = list(joint_tree.keys())
        for i, key in enumerate(root_keys):
            is_root_last = (i == len(root_keys) - 1)
            print_joint_tree(joint_tree[key], prefix, key, is_root_last)

def get_prim_position(prim):
    """Get the position of a prim."""
    if not prim:
        return [0.0, 0.0, 0.0]
    
    if prim.GetTypeName() == "Xform":
        xform = UsdGeom.Xform(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                val = op.Get()
                if hasattr(val, "__len__") and len(val) == 3:
                    return list(val)
    return [0.0, 0.0, 0.0]

if __name__ == "__main__":
    # Open the USD stage
    stage = Usd.Stage.Open("Assets/Robots/Franka/franka.usd")
    # stage = Usd.Stage.Open("Assets/Robots/BostonDynamics/spot/spot.usd")

    # Get the root prim
    root = stage.GetDefaultPrim()
    if not root:
        root = stage.GetPseudoRoot()

    print("USD Scene Hierarchy:")
    print("=" * 40)

    # Print the tree starting from root
    if root.GetName():
        print_tree(root)
    else:
        # If root has no name, print its children
        children = list(root.GetChildren())
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            print_tree(child, "", is_last)
    
    # Print joint-specific hierarchy
    print("\nJoint Hierarchy Tree:")
    print("=" * 40)
    
    joint_tree = get_joint_tree(root)
    if joint_tree:
        print_joint_tree(joint_tree)
        
    def get_all_joints(prim):
        """Get all joints in the prim hierarchy (typeName == 'Joint')."""
        joints = []
        for child in prim.GetChildren():
            if "Joint" in child.GetTypeName():
                joints.append(child)
            joints.extend(get_all_joints(child))
        return joints
    
    all_joints = get_all_joints(root)
    if all_joints:
        print("\nAll Joints Found:")
        for joint in all_joints:
            print(f"  - {joint.GetName()} (Type: {joint.GetTypeName()})")

