import cv2
import matplotlib.pyplot as plt
import pandas as pd

from nonverbal_communication_analysis.environment import DATASET_SYNC, VALID_VIDEO_TYPES, SUBJECT_IDENTIFICATION_GRID
from nonverbal_communication_analysis.utils import fetch_files_from_directory, filter_files


class Visualizer(object):

    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.cap = None

    def get_frame(self, cap, frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def show(self, camera: str, frame: int, highlighted_subject, assigned_subjects: dict = None, key: str = 'openpose'):
        _, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(1, -1)
        ax.set_title('Subjects Keypoints Position')

        file_path = DATASET_SYNC+self.experiment_id+"/"
        video_file = filter_files(fetch_files_from_directory(
            [file_path]), VALID_VIDEO_TYPES, include=camera)
        cap = cv2.VideoCapture(file_path+video_file[0])

        display_frame = self.get_frame(cap, frame)
        ax.imshow(display_frame, aspect='auto',
                  extent=(-1, 1, 1, -1), alpha=1, zorder=-1)

        highlighted_subject_pose = highlighted_subject.pose[key]
        highlighted_subject_pose_keypoints_df = pd.DataFrame(
            highlighted_subject_pose.values(), columns=['x', 'y', 'c'])

        ax.scatter(x=highlighted_subject_pose_keypoints_df['x'], y=highlighted_subject_pose_keypoints_df['y'],
                   c='red')

        if assigned_subjects is not None:
            for _, assigned_subject in assigned_subjects.items():
                subject_pose = assigned_subject.pose[key]
                pose_keypoints_df = pd.DataFrame(
                    subject_pose.values(), columns=['x', 'y', 'c'])
                ax.scatter(x=pose_keypoints_df['x'],
                           y=pose_keypoints_df['y'], c='blue')

        for _, polygon in SUBJECT_IDENTIFICATION_GRID[camera].items():
            pol_x, pol_y = polygon.exterior.xy
            plt.plot(pol_x, pol_y)

        plt.show()
