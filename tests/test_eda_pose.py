"""Head pose from YuNet's five landmarks: known rotations of the generic face come back out."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from ryuk.detector import Detector, Landmarks
from ryuk.eda.pose import FACE_MODEL, Pose, head_pose

FIXTURES = Path(__file__).parent / "fixtures"
SHAPE = (480, 640, 3)


def rotation(yaw: float, pitch: float, roll: float) -> NDArray[np.float64]:
    """R = Rz(roll) Ry(-yaw) Rx(-pitch), the convention `head_pose` documents, built by hand."""
    y, p, r = (math.radians(angle) for angle in (-yaw, -pitch, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    composed: NDArray[np.float64] = rz @ ry @ rx
    return composed


def project(
    yaw: float, pitch: float, roll: float, shape: tuple[int, ...] = SHAPE, distance: float = 600.0
) -> Landmarks:
    """The generic face's landmarks, centred in an image of `shape`, through the pinhole camera
    `head_pose` assumes: focal length the longer side, principal point the image centre.

    At 600 mm and a 640 px focal length the eyes are about 67 px apart, a typical LFW face.
    """
    height, width = shape[:2]
    focal = max(height, width)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    rvec, _ = cv2.Rodrigues(rotation(yaw, pitch, roll))
    points, _ = cv2.projectPoints(
        FACE_MODEL, rvec, np.array([0.0, 0.0, distance]), camera, np.zeros(4)
    )
    return Landmarks(*((float(x), float(y)) for x, y in points.reshape(-1, 2)))


def assert_pose(pose: Pose | None, yaw: float, pitch: float, roll: float) -> None:
    assert pose is not None
    assert pose.yaw == pytest.approx(yaw, abs=1.0)
    assert pose.pitch == pytest.approx(pitch, abs=1.0)
    assert pose.roll == pytest.approx(roll, abs=1.0)


def test_a_frontal_face_has_no_rotation() -> None:
    assert_pose(head_pose(project(0, 0, 0), SHAPE), 0, 0, 0)


@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"),
    [
        (20, 0, 0),
        (-35, 0, 0),
        (0, 15, 0),
        (0, -20, 0),
        (0, 0, 25),
        (0, 0, -10),
        (30, -10, 5),
        (-15, 20, -20),
    ],
)
def test_known_rotations_are_recovered(yaw: float, pitch: float, roll: float) -> None:
    assert_pose(head_pose(project(yaw, pitch, roll), SHAPE), yaw, pitch, roll)


def test_a_face_turned_to_the_image_right_has_positive_yaw() -> None:
    landmarks = project(20, 0, 0)
    pose = head_pose(landmarks, SHAPE)

    # The nose tip, nearest the camera, swings right of the eyes' midpoint.
    assert landmarks.nose_tip[0] > (landmarks.right_eye[0] + landmarks.left_eye[0]) / 2
    assert pose is not None
    assert pose.yaw > 0


def test_a_face_turned_up_has_positive_pitch() -> None:
    landmarks = project(0, 15, 0)
    pose = head_pose(landmarks, SHAPE)

    # The nose tip swings up the image (smaller y) relative to the mouth corners.
    frontal = project(0, 0, 0)
    assert (
        landmarks.right_mouth_corner[1] - landmarks.nose_tip[1]
        > frontal.right_mouth_corner[1] - frontal.nose_tip[1]
    )
    assert pose is not None
    assert pose.pitch > 0


def test_landmarks_rotated_clockwise_in_the_image_give_positive_roll() -> None:
    frontal = project(0, 0, 0)
    angle = math.radians(12)
    # In image coordinates (y down) this matrix turns the x axis towards +y: clockwise on screen.
    turn = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    centre = np.array(frontal.nose_tip)
    turned = [(np.array(point) - centre) @ turn.T + centre for point in frontal]

    pose = head_pose(Landmarks(*((float(x), float(y)) for x, y in turned)), SHAPE)

    assert_pose(pose, 0, 0, 12)


@pytest.mark.parametrize("shape", [(250, 250, 3), (218, 178, 3), (720, 1280, 3)])
def test_the_camera_follows_the_image_shape(shape: tuple[int, ...]) -> None:
    landmarks = project(25, -10, 5, shape, distance=300.0)

    assert_pose(head_pose(landmarks, shape), 25, -10, 5)


def test_landmarks_that_fit_no_face_give_no_pose() -> None:
    point = (100.0, 100.0)

    assert head_pose(Landmarks(point, point, point, point, point), SHAPE) is None


def test_the_astronaut_faces_the_camera() -> None:
    image = cv2.imread(str(FIXTURES / "astronaut.jpg"), cv2.IMREAD_COLOR)
    assert image is not None
    [face] = Detector(FIXTURES / "face_detection_yunet_2026may.onnx").detect(
        np.asarray(image, np.uint8)
    )

    pose = head_pose(face.landmarks, image.shape)

    # A studio portrait facing the camera: every angle within a few degrees of zero.
    assert pose is not None
    assert max(abs(pose.yaw), abs(pose.pitch), abs(pose.roll)) < 10
