"""A rough head pose from YuNet's five landmarks, for describing the datasets (#17).

Five points pin a pose only through a generic face, so the angles are indicative: good enough
to say one dataset has more turned faces than another, not to measure one face. YuNet also pulls
its landmarks towards upright on rotated faces (at 15 degrees of roll it reports about a tenth of
the true eye tilt), so the angles are biased towards zero, roll most of all.
"""

import math
from collections.abc import Sequence
from typing import Final, NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

from ryuk.detector import Landmarks

# A generic adult face in millimetres, in `Landmarks` order, in a frame that matches the camera's
# for a frontal face: x towards the image's right, y down, z away from the camera. The nose tip
# is the origin. The values are rounded adult means from anthropometry (Farkas' norms):
# - eyes 63 mm apart (interpupillary distance), 35.5 mm above the nose tip;
# - mouth corners 53 mm apart, 26.5 mm below the nose tip, so 62 mm below the eyes;
# - the nose tip about 20 mm proud of the eyes and 25 mm of the mouth corners (coarse).
# In-plane they agree with YuNet's median landmarks on LFW and CelebA, in interpupillary
# distances: eyes to nose tip 0.57 (model 0.56), eyes to mouth 0.99 (0.98), mouth width 0.85
# (0.84). The ArcFace alignment template puts the mouth 1.16 below the eyes; fitted to YuNet's
# landmarks that proportion read the typical portrait as pitched 17 degrees down. Zero pitch
# therefore means the typical portrait's pose, not a measured level head.
FACE_MODEL: Final[NDArray[np.float64]] = np.array(
    [
        [-31.5, -35.5, 20.0],  # right eye, on the image's left
        [31.5, -35.5, 20.0],  # left eye
        [0.0, 0.0, 0.0],  # nose tip
        [-26.5, 26.5, 25.0],  # right mouth corner
        [26.5, 26.5, 25.0],  # left mouth corner
    ]
)


class Pose(NamedTuple):
    """Head rotation in degrees, zero for a face looking straight into the camera.

    `yaw` > 0 when the face turns towards the image's right, `pitch` > 0 when it tilts up, and
    `roll` > 0 when it rotates clockwise as the image is viewed. The rotation from the generic
    face to the camera is Rz(roll) Ry(-yaw) Rx(-pitch), with R* the right-handed rotations about
    the camera's axes (x right, y down, z forward).
    """

    yaw: float
    pitch: float
    roll: float


def head_pose(landmarks: Landmarks, image_shape: Sequence[int]) -> Pose | None:
    """Fit the generic face to `landmarks`, or None if no pose in front of the camera fits.

    The camera is a pinhole with no distortion: focal length the image's longer side, principal
    point the image centre. `image_shape` is the image array's shape, (height, width) first.
    """
    height, width = image_shape[0], image_shape[1]
    focal = float(max(height, width))
    camera = np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    points = np.array(landmarks, dtype=np.float64)
    # SQPnP finds the global least-squares pose from as few as three points; the default
    # iterative solver needs six for a non-planar model. It fails an assertion, rather than
    # returning False, on landmarks too close together to carry a pose.
    try:
        found, rvec, tvec = cv2.solvePnP(FACE_MODEL, points, camera, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not found or float(tvec[2, 0]) <= 0:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return _angles(np.asarray(rotation, dtype=np.float64))


def _angles(rotation: NDArray[np.float64]) -> Pose:
    """Pose angles from R = Rz(roll) Ry(b) Rx(a), where b = -yaw and a = -pitch."""
    b = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
    a = math.atan2(rotation[2, 1], rotation[2, 2])
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return Pose(yaw=-math.degrees(b), pitch=-math.degrees(a), roll=math.degrees(roll))
