import json
import operator

import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from nonverbal_communication_analysis.environment import (
    CAMERA_ROOM_GEOMETRY, PEOPLE_FIELDS, RELEVANT_FACE_KEYPOINTS,
    RELEVANT_POSE_KEYPOINTS, SUBJECT_IDENTIFICATION_GRID,
    VALID_SUBJECT_POSE_KEYPOINTS)
from nonverbal_communication_analysis.m6_Visualization.simple_openpose_visualization import \
    Visualizer
from nonverbal_communication_analysis.utils import log

matplotlib.use('QT5Agg')


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
        self.quadrant = -1
        self.confidence = 0
        self.identification_confidence = dict()

    @property
    def quadrant(self):
        return self.__quadrant

    @quadrant.setter
    def quadrant(self, value):
        assert value < 5, "Invalid quadrant property value"
        self.__quadrant = value

    def allocate_subjects(self, allocated_subjects: dict, frame: int, vis: Visualizer = None):
        unallocated_subject = self
        quadrant = unallocated_subject.quadrant

        # if vis is not None:
        #     vis.show(self.camera, frame, unallocated_subject)

        if quadrant not in allocated_subjects:
            print("Assign Subject to Quadrant")
            allocated_subjects[quadrant] = unallocated_subject
            return allocated_subjects
        elif unallocated_subject.is_person():
            print("Is Person!")
            print("Allocated Sub confidence:",
                  allocated_subjects[quadrant].confidence, "Unallocated Sub confidence:", unallocated_subject.confidence)
            if unallocated_subject.confidence > allocated_subjects[quadrant].confidence:
                print("Need to replace unidentified subject with subject in quadrant")
                allocated_subjects[quadrant] = self
                return self.allocate_subjects(allocated_subjects, frame)
            else:
                print("Next quadrant with most confidence value")
                quadrant_confidence = unallocated_subject.identification_confidence
                quadrant_confidence[quadrant] = 0

                unallocated_subject.identification_confidence = dict(
                    sorted(quadrant_confidence.items(), key=operator.itemgetter(1), reverse=True))
                unallocated_subject.quadrant = list(
                    unallocated_subject.identification_confidence.keys())[0]
                unallocated_subject.confidence = unallocated_subject.identification_confidence[
                    unallocated_subject.quadrant]

                if unallocated_subject.confidence == 0:
                    print("No confidence, discard this mf")
                    return allocated_subjects

                return self.allocate_subjects(allocated_subjects, frame)
        else:
            print("Not Person. Is body part")
            if unallocated_subject.confidence > 0:
                print("Join part to subject in quadrant")
                # unidentified_subject_valid_keypoints = unallocated_subject.get_valid_keypoints()
                return allocated_subjects
            else:
                print("Discard loose part or unwanted person in background")
                return allocated_subjects

        return allocated_subjects

    def assign_quadrant(self):
        # TODO: possibly need to complement this method
        # with densepose and openface data
        id_weighing = dict().fromkeys(
            SUBJECT_IDENTIFICATION_GRID[self.camera].keys(), 0)
        openpose_pose_features = self.pose['openpose']
        # openpose_face_features = self.face['openpose']

        for keypoint in openpose_pose_features.values():
            # print(keypoint)
            keypoint_x, keypoint_y, keypoint_confidence = keypoint[0], keypoint[1], keypoint[2]
            point = Point(keypoint_x, keypoint_y)
            for quadrant, polygon in SUBJECT_IDENTIFICATION_GRID[self.camera].items():
                # print(quadrant, polygon)
                if point.intersects(polygon):
                    id_weighing[quadrant] += keypoint_confidence

        # print((id_weighing, max(id_weighing, key=id_weighing.get)))

        identification_confidence = dict(
            sorted(id_weighing.items(), key=operator.itemgetter(1), reverse=True))
        self.identification_confidence = identification_confidence
        self.quadrant = list(identification_confidence.keys())[0]
        self.confidence = identification_confidence[self.quadrant]

        return identification_confidence

    def is_valid_keypoint(self, keypoint: list):
        return keypoint != [-1, -1, 0]

    def is_person(self, key='openpose'):
        subject_keypoints = self.pose[key]
        for keypoint in VALID_SUBJECT_POSE_KEYPOINTS:
            if not self.is_valid_keypoint(subject_keypoints[keypoint]):
                return False

        return True

    def get_valid_keypoints(self, key: str = 'openpose'):
        valid_keypoints = dict()
        for keypoint_index, keypoint_value in self.pose[key].items():
            if self.is_valid_keypoint(keypoint_value):
                valid_keypoints[keypoint_index] = keypoint_value

        return valid_keypoints

    def has_keypoints(self, keypoints: list, key: str = 'openpose'):
        has_keypoints = False
        pose = self.pose[key]

        for keypoint_index in keypoints.keys():
            if self.is_valid_keypoint(pose[keypoint_index]):
                has_keypoints = True
                break

        return has_keypoints

    def attach_keypoints(self, features: list, key: str = 'openpose'):
        merged_keypoints = {**self.pose[key], **features}
        self.pose[key] = merged_keypoints
        return True

    def parse_face_features(self, face_features):
        keypoints = parse_keypoints('FACE', face_features)
        return keypoints

    def parse_pose_features(self, pose_features):
        # TODO: Integrate densepose features too
        keypoints = parse_keypoints('POSE', pose_features)
        return keypoints

    def to_json(self):

        obj = {
            "id": self.quadrant,
            "pose": self.pose,
            "face": self.face
        }

        return obj

    def from_json(self):
        return None

    def __str__(self):
        return "Subject { id: %s, pose: %s }" % (self.quadrant, "(...)")
