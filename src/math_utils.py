import math
from typing import List
from pxr import Gf

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