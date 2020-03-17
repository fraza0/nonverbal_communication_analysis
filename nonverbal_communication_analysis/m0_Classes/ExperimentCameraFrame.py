import json
import cv2

import operator
import pandas as pd

from nonverbal_communication_analysis.environment import (
    CAMERA_ROOM_GEOMETRY, PEOPLE_FIELDS, RELEVANT_FACE_KEYPOINTS,
    RELEVANT_POSE_KEYPOINTS, SUBJECT_IDENTIFICATION_GRID)
from nonverbal_communication_analysis.m0_Classes.Subject import Subject
from nonverbal_communication_analysis.utils import log

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from nonverbal_communication_analysis.m6_Visualization.simple_openpose_visualization import Visualizer
matplotlib.use('QT5Agg')


class ExperimentCameraFrame(object):

    def __init__(self, camera: str, frame: int, people_data: pd.DataFrame, vis: Visualizer = None, verbose: bool = False):
        self.is_valid = False
        self.verbose = verbose
        self.vis = vis
        self.camera = camera
        self.frame = frame
        self.subjects = self.parse_subjects_data(people_data)

    @property
    def subjects(self):
        return self.__subjects

    @subjects.setter
    def subjects(self, value):
        self.__subjects = value
        try:
            assert len(value) == 4
            self.__subjects = value
            self.is_valid = True
        except AssertionError:
            log("WARN", "Invalid number of subjects. Found %s out of 4 required in frame %s of camera %s.\n \
                This frame will be discarded" % (len(value), self.frame, self.camera))

    def parse_subjects_data(self, people_data: pd.Series):
        allocated_subjects = dict()

        if self.verbose:
            print("Camera", self.camera, "Frame", self.frame)

        # First try on subject ID assingment
        for _, person in people_data[PEOPLE_FIELDS].iterrows():
            unconfirmed_identity_subject = Subject(
                self.camera, person['face_keypoints_2d'], person['pose_keypoints_2d'], verbose=self.verbose)
            unconfirmed_identity_subject.assign_quadrant()

            allocated_subjects = unconfirmed_identity_subject.allocate_subjects(
                allocated_subjects, self.frame, self.vis)

        if self.verbose and vis is not None:
            self.vis.show(self.camera, self.frame,
                          assigned_subjects=allocated_subjects)

        return list(dict(sorted(allocated_subjects.items())).values())

    def to_json(self):

        obj = {
            "frame": self.frame,
            "subjects": [subject.to_json() for subject in self.subjects]
        }

        return obj

    def from_json(self):
        return None

    def __str__(self):
        return "ExperimentFrame { frame: %s, subjects: %s }" % (self.frame, [str(subject) for subject in self.subjects])
