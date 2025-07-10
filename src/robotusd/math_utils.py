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

class HomogeneousMatrix:
    """
    A class for working with 4x4 homogeneous transformation matrices.
    Provides comprehensive functionality for robotics transformations.
    """
    
    def __init__(self, matrix: np.ndarray = None):
        """
        Initialize a HomogeneousMatrix.
        
        Args:
            matrix: 4x4 numpy array. If None, creates identity matrix.
        """
        if matrix is None:
            self._matrix = np.eye(4)
        else:
            if matrix.shape != (4, 4):
                raise ValueError(f"Matrix must be 4x4, got {matrix.shape}")
            self._matrix = matrix.copy()
    
    @property
    def matrix(self) -> np.ndarray:
        """Get the underlying 4x4 matrix."""
        return self._matrix.copy()
    
    @property
    def translation(self) -> List[float]:
        """Get the translation component as [x, y, z]."""
        return self._matrix[0:3, 3].tolist()
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get the 3x3 rotation matrix component."""
        return self._matrix[0:3, 0:3].copy()
    
    @property
    def quaternion(self) -> List[float]:
        """Get the rotation as quaternion (x, y, z, w)."""
        return self._rotation_matrix_to_quaternion(self.rotation_matrix)
    
    @property
    def pose(self) -> tuple[List[float], List[float]]:
        """Get pose as (translation, quaternion)."""
        return self.translation, self.quaternion
    
    @classmethod
    def identity(cls) -> 'HomogeneousMatrix':
        """Create identity transformation matrix."""
        return cls()
    
    @classmethod
    def from_translation(cls, translation: List[float]) -> 'HomogeneousMatrix':
        """Create transformation matrix from translation vector."""
        matrix = np.eye(4)
        matrix[0:3, 3] = translation
        return cls(matrix)
    
    @classmethod
    def from_quaternion(cls, quaternion: List[float]) -> 'HomogeneousMatrix':
        """Create transformation matrix from quaternion (x, y, z, w)."""
        x, y, z, w = quaternion
        
        # Normalize quaternion
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm == 0:
            return cls.identity()
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Calculate rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        rotation_matrix = np.array([
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
        ])
        
        # Create 4x4 homogeneous matrix
        matrix = np.eye(4)
        matrix[0:3, 0:3] = rotation_matrix
        return cls(matrix)
    
    @classmethod
    def from_pose(cls, translation: List[float], quaternion: List[float]) -> 'HomogeneousMatrix':
        """Create transformation matrix from pose (translation + quaternion)."""
        transform = cls.from_quaternion(quaternion)
        transform._matrix[0:3, 3] = translation
        return transform
    
    @classmethod
    def from_rotation_matrix(cls, rotation: np.ndarray, translation: List[float] = None) -> 'HomogeneousMatrix':
        """Create transformation matrix from 3x3 rotation matrix and optional translation."""
        if rotation.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {rotation.shape}")
        
        matrix = np.eye(4)
        matrix[0:3, 0:3] = rotation
        if translation is not None:
            matrix[0:3, 3] = translation
        return cls(matrix)
    
    @classmethod
    def from_axis_angle(cls, axis: List[float], angle: float, translation: List[float] = None) -> 'HomogeneousMatrix':
        """Create transformation matrix from axis-angle rotation and optional translation."""
        # Normalize axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            return cls.identity()
        
        normalized_axis = np.array(axis) / axis_norm
        
        # Rodriguez rotation formula
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Skew-symmetric matrix for cross product
        K = np.array([
            [0, -normalized_axis[2], normalized_axis[1]],
            [normalized_axis[2], 0, -normalized_axis[0]],
            [-normalized_axis[1], normalized_axis[0], 0]
        ])
        
        # Rotation matrix: R = I + sin(θ)*K + (1-cos(θ))*K²
        rotation = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        
        return cls.from_rotation_matrix(rotation, translation)
    
    def multiply(self, other: 'HomogeneousMatrix') -> 'HomogeneousMatrix':
        """Multiply this transformation with another."""
        result_matrix = np.dot(self._matrix, other._matrix)
        return HomogeneousMatrix(result_matrix)
    
    def __mul__(self, other: 'HomogeneousMatrix') -> 'HomogeneousMatrix':
        """Operator overloading for matrix multiplication."""
        return self.multiply(other)
    
    def inverse(self) -> 'HomogeneousMatrix':
        """Calculate the inverse transformation."""
        # For homogeneous transformation matrices, the inverse is:
        # [R^T  -R^T*t]
        # [0    1     ]
        rotation = self._matrix[0:3, 0:3]
        translation = self._matrix[0:3, 3]
        
        rotation_inv = rotation.T
        translation_inv = -np.dot(rotation_inv, translation)
        
        matrix_inv = np.eye(4)
        matrix_inv[0:3, 0:3] = rotation_inv
        matrix_inv[0:3, 3] = translation_inv
        
        return HomogeneousMatrix(matrix_inv)
    
    def transform_point(self, point: List[float]) -> List[float]:
        """Transform a 3D point."""
        point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
        transformed = np.dot(self._matrix, point_homogeneous)
        return transformed[0:3].tolist()
    
    def transform_vector(self, vector: List[float]) -> List[float]:
        """Transform a 3D vector (only rotation, no translation)."""
        rotation = self._matrix[0:3, 0:3]
        vector_array = np.array(vector)
        transformed = np.dot(rotation, vector_array)
        return transformed.tolist()
    
    def transform_points(self, points: List[List[float]]) -> List[List[float]]:
        """Transform multiple 3D points efficiently."""
        points_array = np.array(points)
        ones = np.ones((len(points), 1))
        points_homogeneous = np.hstack([points_array, ones])
        transformed = np.dot(self._matrix, points_homogeneous.T)
        return transformed[0:3, :].T.tolist()
    
    def transform_vectors(self, vectors: List[List[float]]) -> List[List[float]]:
        """Transform multiple 3D vectors efficiently (only rotation)."""
        vectors_array = np.array(vectors)
        rotation = self._matrix[0:3, 0:3]
        transformed = np.dot(rotation, vectors_array.T)
        return transformed.T.tolist()
    
    def set_translation(self, translation: List[float]) -> None:
        """Set the translation component."""
        self._matrix[0:3, 3] = translation
    
    def set_rotation_from_quaternion(self, quaternion: List[float]) -> None:
        """Set the rotation component from quaternion."""
        rotation_transform = self.from_quaternion(quaternion)
        self._matrix[0:3, 0:3] = rotation_transform._matrix[0:3, 0:3]
    
    def set_pose(self, translation: List[float], quaternion: List[float]) -> None:
        """Set both translation and rotation components."""
        self.set_translation(translation)
        self.set_rotation_from_quaternion(quaternion)
    
    def is_valid(self) -> bool:
        """Check if the matrix is a valid transformation matrix."""
        # Check if it's 4x4
        if self._matrix.shape != (4, 4):
            return False
        
        # Check if bottom row is [0, 0, 0, 1]
        expected_bottom = np.array([0, 0, 0, 1])
        if not np.allclose(self._matrix[3, :], expected_bottom):
            return False
        
        # Check if rotation part is orthogonal
        rotation = self._matrix[0:3, 0:3]
        should_be_identity = np.dot(rotation, rotation.T)
        if not np.allclose(should_be_identity, np.eye(3), atol=1e-6):
            return False
        
        # Check if determinant of rotation is 1 (proper rotation, not reflection)
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(self._matrix)) or np.any(np.isinf(self._matrix)):
            return False
        
        return True
    
    def copy(self) -> 'HomogeneousMatrix':
        """Create a copy of this transformation."""
        return HomogeneousMatrix(self._matrix.copy())
    
    def __str__(self) -> str:
        """String representation."""
        trans = self.translation
        quat = self.quaternion
        return f"HomogeneousMatrix(translation=[{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}], quaternion=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}])"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"HomogeneousMatrix(\n{self._matrix}\n)"
    
    @staticmethod
    def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> List[float]:
        """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
        trace = np.trace(rotation)
        
        if trace > 0:
            s = math.sqrt(trace + 1) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (rotation[2, 1] - rotation[1, 2]) / s
            y = (rotation[0, 2] - rotation[2, 0]) / s
            z = (rotation[1, 0] - rotation[0, 1]) / s
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2  # s = 4 * qx
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2  # s = 4 * qy
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2  # s = 4 * qz
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
        
        return [x, y, z, w]

# Backward compatibility functions (using the new HomogeneousMatrix class)
def translation_to_homogeneous_matrix(translation: List[float]) -> np.ndarray:
    """Convert translation vector to 4x4 homogeneous transformation matrix."""
    return HomogeneousMatrix.from_translation(translation).matrix

def quaternion_to_homogeneous_matrix(quaternion: List[float]) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 4x4 homogeneous transformation matrix."""
    return HomogeneousMatrix.from_quaternion(quaternion).matrix

def pose_to_homogeneous_matrix(translation: List[float], quaternion: List[float]) -> np.ndarray:
    """Convert pose (translation + quaternion) to 4x4 homogeneous transformation matrix."""
    return HomogeneousMatrix.from_pose(translation, quaternion).matrix

def homogeneous_matrix_to_pose(matrix: np.ndarray) -> tuple[List[float], List[float]]:
    """Convert 4x4 homogeneous transformation matrix to pose (translation, quaternion)."""
    transform = HomogeneousMatrix(matrix)
    return transform.pose

def multiply_homogeneous_matrices(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """Multiply two 4x4 homogeneous transformation matrices."""
    transform1 = HomogeneousMatrix(matrix1)
    transform2 = HomogeneousMatrix(matrix2)
    return transform1.multiply(transform2).matrix

def inverse_homogeneous_matrix(matrix: np.ndarray) -> np.ndarray:
    """Calculate the inverse of a 4x4 homogeneous transformation matrix."""
    transform = HomogeneousMatrix(matrix)
    return transform.inverse().matrix

def transform_point(matrix: np.ndarray, point: List[float]) -> List[float]:
    """Transform a 3D point using a 4x4 homogeneous transformation matrix."""
    transform = HomogeneousMatrix(matrix)
    return transform.transform_point(point)

def transform_vector(matrix: np.ndarray, vector: List[float]) -> List[float]:
    """Transform a 3D vector using only the rotation part of a 4x4 homogeneous transformation matrix."""
    transform = HomogeneousMatrix(matrix)
    return transform.transform_vector(vector)

def rotation_matrix_to_quaternion(rotation: np.ndarray) -> List[float]:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w) - backward compatibility."""
    return HomogeneousMatrix._rotation_matrix_to_quaternion(rotation)