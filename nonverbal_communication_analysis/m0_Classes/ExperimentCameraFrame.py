import json
import cv2

import pandas as pd

from nonverbal_communication_analysis.environment import (
    CAMERA_ROOM_GEOMETRY, PEOPLE_FIELDS, RELEVANT_FACE_KEYPOINTS,
    RELEVANT_POSE_KEYPOINTS, SUBJECT_IDENTIFICATION_GRID)
from nonverbal_communication_analysis.m0_Classes.Subject import Subject

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

matplotlib.use('QT5Agg')


class ExperimentCameraFrame(object):

    def __init__(self, camera: str, frame: int, people_data: pd.DataFrame):
        self.camera = camera
        self.frame = frame
        self.subjects = self.parse_subjects_data(people_data)

    def parse_subjects_data(self, people_data: pd.Series):
        subject_assignment = dict()

        # First try on subject ID assingment
        for _, person in people_data[PEOPLE_FIELDS].iterrows():
            unidentified_subject = Subject(
                self.camera, person['face_keypoints_2d'], person['pose_keypoints_2d'])
            subject_identification_list = unidentified_subject.identify_subject()
            _id = subject_identification_list[0]

            # Validate Assignments. No repeated IDs.
            # Check function documentation for separated keypoints attachment criteria
            if _id not in subject_assignment:
                subject_assignment[_id] = unidentified_subject
            else:
                unidentified_subject_valid_keypoints = unidentified_subject.get_valid_keypoints(
                    'openpose')
                if not subject_assignment[_id].has_keypoints(unidentified_subject_valid_keypoints):
                    subject_assignment[_id].attach_keypoints(
                        unidentified_subject_valid_keypoints)

                    print("NEW ME")
                else:
                    print("What should i do now? Possible intruder")

            if True:
                subject_pose = subject_assignment[_id].pose['openpose']
                pose_keypoints_df = pd.DataFrame(
                    subject_pose.values(), columns=['x', 'y', 'c'])
                _, ax = plt.subplots()
                image = cv2.imread(
                    '/home/fraza0/Desktop/MEI/TESE/nonverbal_communication_analysis/DATASET_DEP/SYNC/3CLC9VWR/last_frame_vid%s.png' % self.camera)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.set_xlim(-1, 1)
                ax.set_ylim(1, -1)
                ax.imshow(image, aspect='auto',
                          extent=(-1, 1, 1, -1), alpha=1, zorder=-1)
                ax.set_title('Subjects Keypoints Position')
                for quadrant, polygon in SUBJECT_IDENTIFICATION_GRID[self.camera].items():
                    pol_x, pol_y = polygon.exterior.xy
                    plt.plot(pol_x, pol_y)
                ax.scatter(
                    x=pose_keypoints_df['x'], y=pose_keypoints_df['y'], c=pose_keypoints_df['c'], cmap=cm.rainbow)
                plt.show()

        return list()

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
