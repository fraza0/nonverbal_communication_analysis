import argparse
import re
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from nonverbal_communication_analysis.environment import (
    DATASET_SYNC, SUBJECT_IDENTIFICATION_GRID, VALID_VIDEO_TYPES, VALID_OUTPUT_FILE_TYPES, OPENPOSE_OUTPUT_DIR, DATASET_SYNC)
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files, list_dirs)


class Visualizer(object):

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id

    def get_frame(self, cap, frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def show_subjects_frame(self, camera: str, frame: int, highlighted_subject=None, assigned_subjects: dict = None, key: str = 'openpose'):
        # TODO: REPLACE WITH show_frame()
        _, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(1, -1)
        ax.set_title('Subjects Keypoints Position')

        file_path = DATASET_SYNC / self.experiment_id / 'task_2'
        video_file = filter_files(fetch_files_from_directory(
            [file_path]), VALID_VIDEO_TYPES, include=camera)[0]
        path = file_path / video_file
        cap = cv2.VideoCapture(str(path))

        display_frame = self.get_frame(cap, frame)
        ax.imshow(display_frame, aspect='auto',
                  extent=(-1, 1, 1, -1), alpha=1, zorder=-1)

        if highlighted_subject is not None:
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

    def show_frame(self, video_directory: str, frame: int, openpose: bool = True, verbose: bool = False):
        group_id = video_directory.split(
            DATASET_SYNC.split('/')[-2])[1].replace('/', '')
        clean_dir = OPENPOSE_OUTPUT_DIR+group_id+"/"+group_id+"_clean/"
        clean_camera_dirs = list_dirs(clean_dir)
        caps = dict()
        color_map = {1: 'blue',
                     2: 'red',
                     3: 'green',
                     4: 'yellow'}

        assigned_subjects = dict()

        # load video files
        video_files = filter_files(fetch_files_from_directory(
            [video_directory]), valid_types=VALID_VIDEO_TYPES)

        for video_file in video_files:
            video_camera = 'pc'+video_file.split('pc')[1][0]
            caps[video_camera] = cv2.VideoCapture(video_directory+video_file)

        # load openpose files
        if openpose:
            openpose_camera_files = dict()
            for cam_dir in clean_camera_dirs:
                openpose_camera = cam_dir.replace('_clean', '')
                clean_path = clean_dir+cam_dir+'/'
                openpose_files = [clean_path+file for file in filter_files(
                    fetch_files_from_directory([clean_path]), valid_types=VALID_OUTPUT_FILE_TYPES)]
                openpose_files.sort()
                openpose_camera_files[openpose_camera] = openpose_files[frame]

                subjects_data = json.load(
                    open(openpose_camera_files[openpose_camera], 'r'))
                assigned_subjects[openpose_camera] = subjects_data['subjects']

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Subjects Keypoints Position')

        axis_idx = 0
        for cam, assigned_subjects in assigned_subjects.items():
            ax[axis_idx].set_xlim(-1, 1)
            ax[axis_idx].set_ylim(1, -1)
            ax[axis_idx].set_title(cam)

            display_frame = self.get_frame(caps[cam], frame)
            ax[axis_idx].imshow(display_frame, aspect='auto',
                                extent=(-1, 1, 1, -1), alpha=1, zorder=-1)

            for assigned_subject in assigned_subjects:
                subject_id = assigned_subject['id']
                subject_pose = assigned_subject['pose']['openpose']

                pose_keypoints_df = pd.DataFrame(
                    subject_pose.values(), columns=['x', 'y', 'c'])
                ax[axis_idx].scatter(x=pose_keypoints_df['x'],
                                     y=pose_keypoints_df['y'], c=color_map[subject_id])

            for _, polygon in SUBJECT_IDENTIFICATION_GRID[cam].items():
                pol_x, pol_y = polygon.exterior.xy
                ax[axis_idx].plot(pol_x, pol_y)

            axis_idx += 1

        plt.show()

    def play_vid(self, video_directory: str, verbose: bool = False):
        group_id = video_directory.split(
            DATASET_SYNC.split('/')[-2])[1].replace('/', '')
        clean_dir = OPENPOSE_OUTPUT_DIR+group_id+"/"+group_id+"_clean/"
        clean_camera_dirs = list_dirs(clean_dir)
        video_captures = dict()
        color_map = {1: 'blue',
                     2: 'red',
                     3: 'green',
                     4: 'yellow'}

        # load video files
        video_files = filter_files(fetch_files_from_directory(
            [video_directory]), valid_types=VALID_VIDEO_TYPES)

        for video_file in video_files:
            video_camera = 'pc'+video_file.split('pc')[1][0]
            video_captures[video_camera] = cv2.VideoCapture(video_directory+video_file)

        
        plt.show()


def main(video_directory: str, openpose: bool = True, frame: int = None, verbose: bool = False):

    visualizer = Visualizer('default_group_id')
    if frame is not None:
        visualizer.show_frame(video_directory, frame, openpose, verbose)
    else:
        visualizer.play_vid(video_directory, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract facial data using OpenFace')
    parser.add_argument('video_directory', type=str,
                        help='Group video directory')
    parser.add_argument('-f', '--frame', dest='frame', type=int,
                        help='Process Specific frame')
    parser.add_argument('-op', '--openpose', dest='openpose', default=True,
                        help='Overlay openpose data', action='store_true')
    # parser.add_argument('-dp', '--densepose', type=str,
    #                     help='Group video directory')
    # parser.add_argument('-of', '--openface', type=str,
    #                     help='Group video directory')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    video_directory = args['video_directory']
    frame = args['frame']
    openpose = args['openpose']
    verbose = args['verbose']

    main(video_directory, openpose, frame, verbose)
