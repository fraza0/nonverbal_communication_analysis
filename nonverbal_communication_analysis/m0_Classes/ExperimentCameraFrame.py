import json
import cv2

import operator
import pandas as pd

from nonverbal_communication_analysis.environment import (
    CAMERA_ROOM_GEOMETRY, PEOPLE_FIELDS, RELEVANT_FACE_KEYPOINTS, NUM_EYE_LANDMARKS,
    SUBJECT_IDENTIFICATION_GRID, OPENPOSE_KEY, OPENFACE_KEY, DENSEPOSE_KEY)
from nonverbal_communication_analysis.m0_Classes.Subject import Subject
from nonverbal_communication_analysis.utils import log

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from nonverbal_communication_analysis.m6_Visualization.simple_visualization import SimpleVisualizer


class ExperimentCameraFrame(object):
    """ExperimentCameraFrame class

    Experiment frame. Analyze frame from a specific camera.
    Identify users based on their position in the experiment room.
    """

    def __init__(self, camera: str, frame: int, people_data: pd.DataFrame, library: str, verbose: bool = False, display: bool = False):
        self.frame_data_validity = False
        self.verbose = verbose
        self.display = display
        self.vis = SimpleVisualizer('3CLC9VWRSAMPLE')
        self.camera = camera
        self.frame = frame
        if library == OPENPOSE_KEY:
            self.subjects = self.parse_subjects_data(
                people_data, key=OPENPOSE_KEY)
        elif library == OPENFACE_KEY:
            self.subjects = self.parse_subjects_data(
                people_data, key=OPENFACE_KEY)
        elif library == DENSEPOSE_KEY:
            self.subjects = self.parse_subjects_data(
                people_data, key=DENSEPOSE_KEY)
        if display:
            matplotlib.use('QT5Agg')

    @property
    def subjects(self):
        return self.__subjects

    @subjects.setter
    def subjects(self, value):
        self.__subjects = value
        try:
            assert len(value) == 4
            self.__subjects = value
            self.frame_data_validity = True
        except AssertionError:
            total = {1, 2, 3, 4}
            found = set()
            for sub in value:
                found.add(sub.quadrant)
            log("WARN", "Invalid number of subjects. Found %s out of 4 required in frame %s of camera %s. Missing: %s" % (
                len(value), self.frame, self.camera, total.difference(found)))

    def parse_subjects_data(self, people_data: pd.Series, key: str):
        """Parse subjects openpose data

        Args:
            people_data (pd.Series): Openpose people's data

        Returns:
            list: List of identified experiment Subjects
        """
        allocated_subjects = dict()

        if self.verbose:
            print("Camera", self.camera, "Frame", self.frame)

        if key == OPENPOSE_KEY:
            for _, person in people_data[PEOPLE_FIELDS].iterrows():
                unconfirmed_identity_subject = Subject(
                    self.camera, openpose_pose_features=person['pose_keypoints_2d'],
                    openpose_face_features=person['face_keypoints_2d'],
                    verbose=self.verbose, display=self.display)
                unconfirmed_identity_subject.assign_quadrant(key=OPENPOSE_KEY)

                if self.verbose:
                    print(unconfirmed_identity_subject)
                allocated_subjects = unconfirmed_identity_subject.allocate_subjects(
                    allocated_subjects, self.frame, self.vis)

            for _, subject in allocated_subjects.copy().items():
                if not subject.is_person():
                    del allocated_subjects[subject.quadrant]
        elif key == OPENFACE_KEY:
            for _, person in people_data.iterrows():
                unconfirmed_identity_subject = Subject(
                    self.camera, openface_face_features=person,
                    verbose=self.verbose, display=self.display)

                unconfirmed_identity_subject.assign_quadrant(key=OPENFACE_KEY)
                if self.verbose:
                    print(unconfirmed_identity_subject)

            allocated_subjects = unconfirmed_identity_subject.allocate_subjects(
                allocated_subjects, self.frame, self.vis)
        elif key == DENSEPOSE_KEY:
            print("DP")

        if self.display and self.vis is not None:
            print(self.camera, self.frame, allocated_subjects)
            self.vis.show_subjects_frame(self.camera, self.frame,
                                         assigned_subjects=allocated_subjects, key=OPENFACE_KEY)

        return list(dict(sorted(allocated_subjects.items())).values())

    def to_json(self):
        """Transform ExperimentCameraFrame object to JSON format

        Returns:
            str: JSON formatted ExperimentCameraFrame object
        """

        obj = {
            "frame": self.frame,
            "is_raw_data_valid": self.frame_data_validity,
            "subjects": [subject.to_json() for subject in self.subjects]
        }

        return obj

    def from_json(self):
        """Create ExperimentCameraFrame object from JSON string

        Returns:
            Experiment: ExperimentCameraFrame object
        """
        return None

    def __str__(self):
        return "ExperimentFrame { frame: %s, subjects: %s }" % (self.frame, [str(subject) for subject in self.subjects])
