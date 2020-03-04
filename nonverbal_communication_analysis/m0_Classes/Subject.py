import json

import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from nonverbal_communication_analysis.environment import (
    CAMERA_ROOM_GEOMETRY, PEOPLE_FIELDS, RELEVANT_FACE_KEYPOINTS,
    RELEVANT_POSE_KEYPOINTS, ROOM_GEOMETRY_REFERENCE)
from nonverbal_communication_analysis.utils import log


def is_relevant_pose_keypoint(entry):
    if entry[0] in RELEVANT_POSE_KEYPOINTS:
        return True
    return False


def is_relevant_face_keypoint(entry):
    if entry[0] in RELEVANT_FACE_KEYPOINTS:
        return True
    return False


def parse_keypoints(_type: str, keypoints: list):
    # [x, y, c]
    keypoints = [keypoints[x:x+3] for x in range(0, len(keypoints), 3)]
    if _type.upper() == 'POSE':
        keypoints_filtered = dict(
            filter(is_relevant_pose_keypoint, enumerate(keypoints)))
    elif _type.upper() == 'FACE':
        keypoints_filtered = dict(
            filter(is_relevant_face_keypoint, enumerate(keypoints)))
    else:
        log("ERROR", "Invalid keypoint type")
        return
    # keypoints_filtered = [keypoint for keypoint in enumerate(keypoints) if keypoint[0] in relevant_keypoints]
    return keypoints_filtered


class Subject(object):
    def __init__(self, camera: str, face_features: list, pose_features: list):
        self.camera = camera
        self.pose = {
            "openpose": self.parse_pose_features(pose_features),
            "densepose": list()
        }
        self.face = {
            "openpose": self.parse_face_features(face_features),
            "openface": list()
        }
        self._id = -1  # self.set_subject_id(camera,
        #    self.pose_features, self.face_features)  # Self assigned attribute

    def set_subject_id(self, camera, pose_features, face_features):
        id_weighing = dict().fromkeys(ROOM_GEOMETRY_REFERENCE.keys(), 0)

        for keypoint in pose_features.values():
            keypoint_x, keypoint_y, keypoint_confidence = keypoint[0], keypoint[1], keypoint[2]
            point = Point(keypoint_x, keypoint_y)
            for quadrant in CAMERA_ROOM_GEOMETRY[camera]:
                if point.intersects(CAMERA_ROOM_GEOMETRY[camera][quadrant]):
                    id_weighing[quadrant] += keypoint_confidence

        return max(id_weighing, key=id_weighing.get)

    def parse_face_features(self, face_features):
        keypoints = parse_keypoints('FACE', face_features)
        return keypoints

    def parse_pose_features(self, pose_features):
        # TODO: Integrate densepose features too
        keypoints = parse_keypoints('POSE', pose_features)
        return keypoints

    def to_json(self):

        obj = {
            "id": self._id,
            "pose": self.pose,
            "face": self.face
        }

        return obj

    def from_json(self):
        return None

    def __str__(self):
        return "Subject { id: %s, pose: %s }" % (self._id, self.pose)
