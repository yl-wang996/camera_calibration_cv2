from diana_control import DianaControl
import numpy as np
import math

def rxryrz2rmat(rxryrz):
    """
    Convert axis with angle magnitude (radians) to rotation matrix
    Args:
        rxryrz: (x,y,z), the axis angle

    Returns:
        matrix: 3x3 rotation matrix
    """
    rxryrz = np.asarray(rxryrz)
    # if is all zero, return identity matrix
    if np.all(rxryrz == 0):
        return np.eye(3)
    x, y, z = rxryrz
    n = math.sqrt(x * x + y * y + z * z)
    angle = n
    axis = rxryrz / n
    return axangle2rmat(axis, angle)

def axangle2rmat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x * x + y * y + z * z)
        x = x / n
        y = y / n
        z = z / n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    return np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])


class RobotState:
    """
    Class to get the TCP pose of the robot
    """
    def __init__(self, ip_addr = "192.168.10.75"):
        self.diana = DianaControl(ip_address=ip_addr)

    def get_tcp_pos(self):
        """
        Get the TCP position of the robot
        :return: (x, y, z, rx, ry, rz), the position and orientation of the TCP, orientation is in axis-value with angle magnitude
        """
        return self.diana.get_tcp_pos()

    def get_tcp_mat(self):
        """
        Get the TCP position of the robot in matrix form (4*4).
        :return: 4*4 matrix, the position and orientation of the TCP
        """
        pose = self.get_tcp_pos()
        mat = np.eye(4)
        mat[:3, :3] = TransformUtil.rxryrz2rmat(pose[3:6])  # convert the axis-angle with value magnitude to rotation matrix
        mat[:3, 3] = pose[:3]
        return mat

    def __del__(self):
        self.diana.close()

def test():
    robot_state = RobotState()
    print(robot_state.get_tcp_pos())
    print(robot_state.get_tcp_mat())