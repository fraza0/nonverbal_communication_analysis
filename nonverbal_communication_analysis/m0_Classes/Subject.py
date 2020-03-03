import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from nonverbal_communication_analysis.environment import (CAMERA_ROOM_GEOMETRY, ROOM_GEOMETRY_REFERENCE,
                                                          PEOPLE_FIELDS,
                                                          RELEVANT_KEYPOINTS)


def is_relevant_keypoint(entry):
    if entry[0] in RELEVANT_KEYPOINTS:
        return True

    return False


class Subject(object):
    def __init__(self, camera: str, face_features: list, pose_features: list):
        self.camera = camera
        self.face_features = self.parse_face_features(face_features)
        self.pose_features = self.parse_pose_features(pose_features)
        self._id = self.set_subject_id(camera,
                                       self.pose_features, self.face_features)  # Self assigned attribute

    def set_subject_id(self, camera, pose_features, face_features):
        id_weighing = dict().fromkeys(ROOM_GEOMETRY_REFERENCE.keys(), 0)

        for keypoint in pose_features.values():
            keypoint_x, keypoint_y, keypoint_confidence = keypoint[0], keypoint[1], keypoint[2]
            point = Point(keypoint_x, keypoint_y)
            for quadrant in CAMERA_ROOM_GEOMETRY[camera]:
                if point.intersects(CAMERA_ROOM_GEOMETRY[camera][quadrant]):
                    id_weighing[quadrant] += keypoint_confidence

        print(id_weighing)
        return max(id_weighing, key=id_weighing.get)

    def parse_face_features(self, face_features):
        # TODO: Integrate all facial features in the future (Openpose + Openface)
        return face_features

    def parse_pose_features(self, pose_features):
        # [x, y, c]
        keypoints = [pose_features[x:x+3]
                     for x in range(0, len(pose_features), 3)]
        keypoints_filtered = dict(
            filter(is_relevant_keypoint, enumerate(keypoints)))

        return keypoints_filtered

    def __str__(self):
        return "Subject { id: %s, face_features: %s, pose_features: %s }" % (self._id, "[...]", "[...]")
