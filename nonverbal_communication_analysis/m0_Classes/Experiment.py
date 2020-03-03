from nonverbal_communication_analysis.environment import DATASET_SYNC, PEOPLE_FIELDS
import pandas as pd
from nonverbal_communication_analysis.m0_Classes.Subject import Subject


class Experiment(object):
    _n_subjects = 4
    _n_tasks = 2
    _n_cameras = 3

    def __init__(self, _id: str):
        self._id = _id
        self.type = self.match_id_type(_id)
        self.people = dict()  # {camera_id:x ,people:[]}

    def match_id_type(self, _id: str):
        df = pd.read_csv(DATASET_SYNC + 'groups_info.csv')
        return df[df['Group ID'] == _id]['Conflict Type'].tolist()[0]

    def __str__(self):
        return "Experiment { id: %s, type: %s, people: %s }" % (self._id, self.type,  self.people)


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
