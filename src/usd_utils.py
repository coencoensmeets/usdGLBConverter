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