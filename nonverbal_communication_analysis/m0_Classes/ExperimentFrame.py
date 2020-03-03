import pandas as pd

from nonverbal_communication_analysis.environment import PEOPLE_FIELDS
from nonverbal_communication_analysis.m0_Classes.Subject import Subject


class ExperimentCameraFrame(object):

    def __init__(self, camera: str, frame: int, people_data: pd.DataFrame):
        self.camera = camera
        self.frame = frame
        self.subjects = self.parse_subjects_features(people_data)

    def parse_subjects_features(self, people_data: pd.Series):
        subject_list = list()
        for _, person in people_data[PEOPLE_FIELDS].iterrows():
            subject_list.append(
                Subject(self.camera, person['face_keypoints_2d'], person['pose_keypoints_2d']))

        return subject_list

    def __str__(self):
        return "ExperimentFrame { frame: %s, subjects: %s }" % (self.frame, [str(subject) for subject in self.subjects])
