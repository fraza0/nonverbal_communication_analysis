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


def is_relevant_pose_keypoint(entry):
    """Check if pose keypoint is valid

    Args:
        entry (enumerate): keypoints enumerate

    Returns:
        bool: True if keypoint is valid. False otherwise
    """
    if entry[0] in RELEVANT_POSE_KEYPOINTS:
        return True
    return False


def is_relevant_face_keypoint(entry):
    """Check if face keypoint is valid

    Args:
        entry (enumerate): keypoints enumerate

    Returns:
        bool: True if keypoint is valid. False otherwise
    """
    if entry[0] in RELEVANT_FACE_KEYPOINTS:
        return True
    return False


def parse_keypoints(_type: str, keypoints: enumerate):
    """Parsing keypoints.
    Keypoints are composed of 3 value list [x, y, c]
    x, y coordinates and confidence coefficient

    Args:
        _type (str): keypoint type (POSE or FACE)
        keypoints (enumerate): keypoints enumerate

    Returns:
        list: List of filtered keypoints
    """

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
    return keypoints_filtered


class Subject(object):
    """Subject class.
    Subject is a person after pose data parsing/processing
    From now on, every person in the experiment is called a Subject
    """

    def __init__(self, camera: str, face_features: list, pose_features: list, verbose: bool = False, display: bool = False):
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
        self.verbose = verbose
        self.display = display
        if verbose:
            matplotlib.use('QT5Agg')

    @property
    def quadrant(self):
        return self.__quadrant

    @quadrant.setter
    def quadrant(self, value):
        assert value < 5, "Invalid quadrant property value"
        self.__quadrant = value

    def allocate_subjects(self, allocated_subjects: dict, frame: int, vis: Visualizer = None):
        """Allocate subject to quadrant

        Args:
            allocated_subjects (dict): Already allocated subjects
            frame (int): Frame number
            vis (Visualizer, optional): Visualizer instance. Defaults to None. If None, visualization will not be available

        Returns:
            dict: Allocated Subjects in quadrants

        See also:
            80963_18032020_WEEKLY_REPORT - Report containing flowchart of subject allocation process
        """
        unallocated_subject = self
        quadrant = unallocated_subject.quadrant

        if self.display and vis is not None:
            vis.show(self.camera, frame, unallocated_subject)

        if quadrant not in allocated_subjects:
            if self.verbose:
                print("Previous", unallocated_subject.quadrant)
                print("Assign Subject to Quadrant")
            allocated_subjects[quadrant] = unallocated_subject
            return allocated_subjects
        elif unallocated_subject.is_person():
            if self.verbose:
                print("Is a person!")
                print("Allocated Sub confidence:",
                      allocated_subjects[quadrant].confidence, "Unallocated Sub confidence:", unallocated_subject.confidence)
            if unallocated_subject.confidence > allocated_subjects[quadrant].confidence:
                if self.verbose:
                    print(
                        "Need to replace unidentified subject with subject in quadrant")

                replace_subject = allocated_subjects[quadrant]
                allocated_subjects[unallocated_subject.quadrant] = unallocated_subject
                return replace_subject.allocate_subjects(allocated_subjects, frame)
            else:
                if self.verbose:
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
                    if self.verbose:
                        print("No confidence, discard this mf")
                    return allocated_subjects

                allocated_subjects[unallocated_subject.quadrant] = unallocated_subject

                return allocated_subjects
        else:
            if self.verbose:
                print("Not a person. Might be body part or misidentified subject")
                print(unallocated_subject.pose['openpose'])
            if unallocated_subject.confidence > 0:
                if self.verbose:
                    print("Join part to subject in quadrant")
                allocated_subjects[quadrant].attach_keypoints(
                    unallocated_subject.get_valid_keypoints())
                return allocated_subjects
            else:
                if self.verbose:
                    print("Discard loose part or unwanted person in background")
                if self.display:
                    vis.show(self.camera, frame, unallocated_subject)
                return allocated_subjects

        return allocated_subjects

    def assign_quadrant(self):
        """Assign subject to quadrant based on its keypoints position

        Returns:
            dict: Quadrant identification and associated confidence
        """
        # TODO: possibly need to complement this method with densepose and openface data
        id_weighing = dict().fromkeys(
            SUBJECT_IDENTIFICATION_GRID[self.camera].keys(), 0)
        openpose_pose_features = self.pose['openpose']
        # openpose_face_features = self.face['openpose']

        for keypoint in openpose_pose_features.values():
            keypoint_x, keypoint_y, keypoint_confidence = keypoint[0], keypoint[1], keypoint[2]
            point = Point(keypoint_x, keypoint_y)
            for quadrant, polygon in CAMERA_ROOM_GEOMETRY[self.camera].items():
                if point.intersects(polygon):
                    id_weighing[quadrant] += keypoint_confidence
        identification_confidence = dict(
            sorted(id_weighing.items(), key=operator.itemgetter(1), reverse=True))
        self.identification_confidence = identification_confidence
        self.quadrant = list(identification_confidence.keys())[0]
        self.confidence = identification_confidence[self.quadrant]

        return identification_confidence

    def is_valid_keypoint(self, keypoint: list):
        """Check if keypoint is valid

        Args:
            keypoint (list): Keypoint list

        Returns:
            bool: True if keypoint is valid. False otherwise
        """
        return keypoint != [-1, -1, 0]

    def is_person(self, key='openpose'):
        """Check if subject is a person. If it's not, might be a loose part.
        Person is considered a subject with upper trunk keypoints

        Args:
            key (str, optional): Defaults to 'openpose'.

        Returns:
            bool: True if subject is a person. False otherwise
        """
        subject_keypoints = self.pose[key]
        for keypoint in VALID_SUBJECT_POSE_KEYPOINTS:
            if not self.is_valid_keypoint(subject_keypoints[keypoint]):
                return False

        return True

    def get_valid_keypoints(self, key: str = 'openpose'):
        """Get valid keypoints

        Args:
            key (str, optional): Defaults to 'openpose'.

        Returns:
            dict: Valid index and keypoints's value
        """
        valid_keypoints = dict()
        for keypoint_index, keypoint_value in self.pose[key].items():
            if self.is_valid_keypoint(keypoint_value):
                valid_keypoints[keypoint_index] = keypoint_value

        return valid_keypoints

    def has_keypoints(self, keypoints: list, key: str = 'openpose'):
        """Check if subject has specific keypoints

        Args:
            keypoints (list): Keypoints list a subject should contain
            key (str, optional): Defaults to 'openpose'.

        Returns:
            bool: True if Subject contains keypoints. False otherwise
        """
        has_keypoints = False
        pose = self.pose[key]

        for keypoint_index in keypoints.keys():
            if self.is_valid_keypoint(pose[keypoint_index]):
                has_keypoints = True
                break

        return has_keypoints

    def attach_keypoints(self, features: list, key: str = 'openpose'):
        """Attach Subject keypoints.
        This might happen when subject instance is not a person, but instead is a loose part.

        Args:
            features (list): Subject pose features
            key (str, optional): Defaults to 'openpose'.

        Returns:
            bool: True for attachment confirmation
        """
        merged_keypoints = {**self.pose[key], **features}
        self.pose[key] = merged_keypoints
        return True

    def parse_face_features(self, face_features: list):
        """Parse Openpose face features

        Args:
            face_features (list): Openpose face features list

        Returns:
            dict: Parsed face keypoints
        """
        keypoints = parse_keypoints('FACE', face_features)
        return keypoints

    def parse_pose_features(self, pose_features):
        """Parse Openpose pose features

        Args:
            face_features (list): Openpose pose features list

        Returns:
            dict: Parsed pose keypoints
        """
        # TODO: Integrate densepose features too
        keypoints = parse_keypoints('POSE', pose_features)
        return keypoints

    def to_json(self):
        """Transform Subject object to JSON format

        Returns:
            str: JSON formatted Subject object
        """
        obj = {
            "id": self.quadrant,
            "pose": self.pose,
            "face": self.face
        }

        return obj

    def from_json(self):
        """Create Subject object from JSON string

        Returns:
            Subject: Subject object
        """

        return None

    def __str__(self):
        return "Subject { id: %s, pose: %s }" % (self.quadrant, "(...)")
