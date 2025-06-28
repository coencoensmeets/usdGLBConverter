import math
from typing import List
from pxr import Gf
import numpy as np

def quaternion_multiply(q1: List[float], q2: List[float]) -> List[float]:
    """Multiply two quaternions (x, y, z, w format)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return [x, y, z, w]

def quaternion_inverse(q: List[float]) -> List[float]:
    """Calculate the inverse of a quaternion (x, y, z, w format)."""
    x, y, z, w = q
    norm_sq = x*x + y*y + z*z + w*w
    if norm_sq == 0:
        return [0, 0, 0, 1]  # Identity quaternion
    return [-x/norm_sq, -y/norm_sq, -z/norm_sq, w/norm_sq]

def euler_to_quat(x: float, y: float, z: float) -> List[float]:
    """Convert Euler angles to quaternion (x, y, z, w)."""
    x, y, z = math.radians(x), math.radians(y), math.radians(z)
    cx, cy, cz = math.cos(x/2), math.cos(y/2), math.cos(z/2)
    sx, sy, sz = math.sin(x/2), math.sin(y/2), math.sin(z/2)
    qw = cx*cy*cz - sx*sy*sz
    qx = sx*cy*cz + cx*sy*sz
    qy = cx*sy*cz - sx*cy*sz
    qz = cx*cy*sz + sx*sy*cz
    return [qx, qy, qz, qw]

def quat_to_list(q: Gf.Quatf) -> List[float]:
    """
    Convert a usd-core quaternion (with .real and .imaginary attributes) to [w, x, y, z] list.
    """
    if q is None:
        return None
    if isinstance(q, (list, tuple)):
        return list(q)
    try:
        return [q.real, q.imaginary[0], q.imaginary[1], q.imaginary[2]]
    except AttributeError:
        # Fallback: try to cast to list
        return list(q)
    
def rotate_vector(q: List[float], v: List[float]) -> List[float]:
    """
    Rotate a 3D vector v by a quaternion q.

    Args:
        q: Quaternion in [x, y, z, w] format.
        v: 3D vector as [x, y, z].

    Returns:
        Rotated 3D vector as [x, y, z].
    """
    qx, qy, qz, qw = q
    vx, vy, vz = v
    vq = [vx, vy, vz, 0.0]
    q_inv = quaternion_inverse(q)
    t = quaternion_multiply(q, vq)
    t = quaternion_multiply(t, q_inv)
    return [t[0], t[1], t[2]]

def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w format) to 3x3 rotation matrix.
    
    Args:
        q: Quaternion in [x, y, z, w] format
        
    Returns:
        3x3 rotation matrix as numpy array
    """
    x, y, z, w = q
    
    # Normalize quaternion
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0:
        return np.eye(3)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Calculate rotation matrix elements
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])

def normalize_vector(v: List[float]) -> List[float]:
    """
    Normalize a 3D vector.
    
    Args:
        v: 3D vector as [x, y, z]
        
    Returns:
        Normalized 3D vector as [x, y, z]
    """
    length = math.sqrt(sum(x*x for x in v))
    if length == 0:
        return [0.0, 0.0, 0.0]
    return [x/length for x in v]

def vector_angle(v1: List[float], v2: List[float]) -> float:
    """
    Calculate the angle between two 3D vectors.
    
    Args:
        v1: First 3D vector as [x, y, z]
        v2: Second 3D vector as [x, y, z]
        
    Returns:
        Angle in radians
    """
    # Normalize vectors
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    
    # Calculate dot product
    dot_product = sum(a*b for a, b in zip(v1_norm, v2_norm))
    
    # Clamp to avoid numerical errors
    dot_product = max(-1.0, min(1.0, dot_product))
    
    return math.acos(dot_product)

def cross_product(v1: List[float], v2: List[float]) -> List[float]:
    """
    Calculate the cross product of two 3D vectors.
    
    Args:
        v1: First 3D vector as [x, y, z]
        v2: Second 3D vector as [x, y, z]
        
    Returns:
        Cross product as [x, y, z]
    """
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    
    return [
        y1*z2 - z1*y2,
        z1*x2 - x1*z2,
        x1*y2 - y1*x2
    ]

def dot_product(v1: List[float], v2: List[float]) -> float:
    """
    Calculate the dot product of two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Dot product as float
    """
    return sum(a*b for a, b in zip(v1, v2))

def axis_to_quat(from_axis: str, to_axis: str) -> list:
    """
    Return a quaternion (x, y, z, w) that rotates from one axis (as 'X', 'Y', 'Z') to another.
    """
    import numpy as np
    axis_map = {'X': np.array([1,0,0]), 'Y': np.array([0,1,0]), 'Z': np.array([0,0,1])}
    v1 = axis_map[from_axis.upper()]
    v2 = axis_map[to_axis.upper()]
    if np.allclose(v1, v2):
        return [0,0,0,1]
    if np.allclose(v1, -v2):
        # 180 degree rotation around any perpendicular axis
        perp = np.cross(v1, [1,0,0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(v1, [0,1,0])
        perp = perp / np.linalg.norm(perp)
        return [float(perp[0]), float(perp[1]), float(perp[2]), 0.0]
    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    s = np.sin(angle/2.0)
    w = np.cos(angle/2.0)
    return [float(axis[0]*s), float(axis[1]*s), float(axis[2]*s), float(w)]