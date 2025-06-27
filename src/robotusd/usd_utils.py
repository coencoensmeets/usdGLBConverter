from pxr import Usd, Sdf
from pxr import UsdGeom, UsdSkel

def get_prim_from_name(stage: Usd.Stage, prim_name: str) -> Usd.Prim:
    """
    Get a USD prim by its name from the stage.
    
    Args:
        stage (Usd.Stage): The USD stage to search in.
        prim_name (str): The name of the prim to find.
    
    Returns:
        Usd.Prim: The found prim, or None if not found.
    """
    for prim in stage.Traverse():
        if prim.GetName() == prim_name:
            return prim
    return None

def get_all_joints(prim: Usd.Prim) -> list:
    """
    Get all joints in the prim hierarchy (typeName == 'Joint').
    
    Args:
        prim (Usd.Prim): The USD prim to search in.
    
    Returns:
        list: A list of joint prims.
    """
    joints = []
    for child in prim.GetChildren():
        if "Joint" in child.GetTypeName():
            joints.append(child)
        joints.extend(get_all_joints(child))
    return joints

def print_data_prim(prim_name:str):
    """
    Print the data prim for a given prim name.
    
    Args:
        prim_name (str): The name of the prim to find.
    """
    stage = Usd.Stage.Open("Assets/Robots/Franka/franka.usd")
    if not stage:
        print(f"Could not open USD file.")
        return
    
    prim = get_prim_from_name(stage, prim_name)
    if prim:
        print(f"Data Prim for {prim_name}: {prim.GetPath()}")
        print(f"Type: {prim.GetTypeName()}")
        print(f"Properties:")
        for prop in prim.GetProperties():
            print(f"  - {prop.GetName()}: {prop.Get()}") if hasattr(prop, "Get") else print(f"  - {prop.GetName()}: No value")
        for ref in prim.GetRelationships():
            print(f"  - Relationship: {ref.GetName()}")
            targets = ref.GetTargets()
            if targets:
                print(f"    Targets: {', '.join(str(t) for t in targets)}")
            else:
                print("    No targets")
    else:
        print(f"Prim with name {prim_name} not found.")