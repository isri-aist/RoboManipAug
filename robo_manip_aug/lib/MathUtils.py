import numpy as np


def get_trans_from_rot_pos(rot, pos):
    """Get transformation (4D square matrix) from rotation (3D square matrix) and position (3D vector)."""
    trans = np.eye(4)
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = pos
    return trans
