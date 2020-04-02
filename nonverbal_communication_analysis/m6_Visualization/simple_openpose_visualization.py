import argparse
import re
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from nonverbal_communication_analysis.environment import (
    DATASET_SYNC, SUBJECT_IDENTIFICATION_GRID, VALID_VIDEO_TYPES, VALID_OUTPUT_FILE_TYPES, OPENPOSE_OUTPUT_DIR, DATASET_SYNC)
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files, list_dirs)
from nonverbal_communication_analysis.m0_Classes.Experiment import get_group_from_file_path


class Visualizer(object):

    COLOR_MAP = {1: 'yellow',
                 2: 'red',
                 3: 'green',
                 4: 'blue'}

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id

    def get_frame(self, cap, frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def show_subjects_frame(self, camera: str, frame: int, highlighted_subject=None, assigned_subjects: dict = None, key: str = 'openpose'):
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

    def openpose_overlay(self, group_id: str, ax, specific_frame: int = None, specific_task: int = None, verbose: bool = False):
        openpose_group_data = [x for x in (OPENPOSE_OUTPUT_DIR / group_id).iterdir()
                               if '_clean' in x.name][0]

        if (specific_frame and specific_task) is not None:
            openpose_task_directory = [x for x in openpose_group_data.iterdir()
                                       if x.is_dir() and str(specific_task) in str(x)][0]
            openpose_task_cameras_directories = [
                x for x in openpose_task_directory.iterdir()]
            openpose_task_cameras_directories.sort()

            assigned_subjects = dict()
            for camera_directory in openpose_task_cameras_directories:
                camera = camera_directory.name
                openpose_files = [x for x in camera_directory.iterdir(
                ) if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES]
                openpose_files.sort()

                subjects_data = json.load(
                    open(openpose_files[specific_frame], 'r'))
                assigned_subjects[camera] = subjects_data['subjects']

            for idx, (camera, assigned_subjects) in enumerate(assigned_subjects.items()):
                ax[idx].set_xlim(-1, 1)
                ax[idx].set_ylim(1, -1)

                for assigned_subject in assigned_subjects:
                    subject_id = assigned_subject['id']
                    subject_pose = assigned_subject['pose']['openpose']

                    pose_keypoints_df = pd.DataFrame(
                        subject_pose.values(), columns=['x', 'y', 'c'])
                    ax[idx].scatter(x=pose_keypoints_df['x'],
                                    y=pose_keypoints_df['y'],
                                    c=self.COLOR_MAP[subject_id], marker='.')

                for _, polygon in SUBJECT_IDENTIFICATION_GRID[camera].items():
                    pol_x, pol_y = polygon.exterior.xy
                    ax[idx].plot(pol_x, pol_y)

    def show_frame(self, group_directory: str, specific_frame: int = None, specific_task: int = None, openpose: bool = False, openface: bool = False, densepose: bool = False, verbose: bool = False):
        group_directory = Path(group_directory)
        group_id = get_group_from_file_path(group_directory)
        tasks_directories = [x for x in group_directory.iterdir()
                             if x.is_dir() and 'task' in str(x)]

        if specific_task is not None:
            tasks_directories = [x for x in tasks_directories
                                 if str(specific_task) in x.name]

        video_captures = dict()
        video_files_paths = [x for x in tasks_directories[0].iterdir()
                             if not x.is_dir()]
        video_files_paths.sort()

        for video_file in video_files_paths:
            camera_id = re.search(
                r'(?<=Video)(pc\d{1})(?=\d{14})', video_file.name).group(0)
            video_captures[camera_id] = cv2.VideoCapture(str(video_file))

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Subjects Keypoints Position')

        for idx, (camera, video_cap) in enumerate(video_captures.items()):
            display_frame = self.get_frame(video_cap, specific_frame)
            ax[idx].set_title("Camera %s" % camera)
            ax[idx].imshow(display_frame, aspect='auto',
                           extent=(-1, 1, 1, -1), alpha=1, zorder=-1)

        if openpose:
            self.openpose_overlay(group_id, ax=ax, specific_frame=specific_frame,
                                  specific_task=specific_task, verbose=verbose)
        if openface:
            pass
        if densepose:
            pass

        plt.show()

    def play_vid(self, group_directory: str, openpose: bool = False, openface: bool = False, densepose: bool = False, verbose: bool = False):
        group_directory = Path(group_directory)
        group_id = get_group_from_file_path(group_directory)
        tasks_directories = [x for x in group_directory.iterdir()
                             if x.is_dir() and 'task' in str(x)]

        if specific_task is not None:
            tasks_directories = [x for x in tasks_directories]

        video_captures = dict()
        for task_directory in tasks_directories:
            print("Playing %s" % task_directory.name)
            video_files_paths = [x for x in task_directory.iterdir()
                                 if not x.is_dir()]
            video_files_paths.sort()

            for video_file in video_files_paths:
                camera_id = re.search(
                    r'(?<=Video)(pc\d{1})(?=\d{14})', video_file.name).group(0)
                video_captures[camera_id] = cv2.VideoCapture(str(video_file))

            frame_dim = None
            while(all(video_capture.isOpened() for video_capture in video_captures.values())):
                cameras_frames = dict()
                for camera in video_captures:
                    video_capture = video_captures[camera]
                    ret, frame = video_capture.read()
                    if ret == True:
                        cv2.imshow('Task %s - Camera %s' %
                                   (task_directory.name, camera), frame)
                    else:
                        break

                if cv2.waitKey(1) & 0xFF == ord('q') or ret is not True:
                    break

            for camera, video_capture in video_captures.items():
                video_capture.release()
                cv2.destroyAllWindows()


def main(group_directory: str, specific_frame: int = None, specific_task: int = None, openpose: bool = False, openface: bool = False, densepose: bool = False, verbose: bool = False):

    visualizer = Visualizer('default_group_id')
    if (specific_frame and specific_task) is not None:
        visualizer.show_frame(group_directory, specific_task=specific_task,
                              specific_frame=specific_frame, openpose=openpose,
                              openface=openface, densepose=densepose, verbose=verbose)
    else:
        visualizer.play_vid(group_directory, openpose=openpose,
                            openface=openface, densepose=densepose, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize data')
    parser.add_argument('group_directory', type=str,
                        help='Group video directory')
    parser.add_argument('-t', '--task', dest="specific_task", type=int, choices=[1, 2],
                        help='Specify Task frame')
    parser.add_argument('-f', '--frame', dest='specific_frame', type=int,
                        help='Process Specific frame')
    parser.add_argument('-op', '--openpose',
                        help='Overlay openpose data', action='store_true')
    parser.add_argument('-dp', '--densepose',
                        help='Overlay densepose data', action='store_true')
    parser.add_argument('-of', '--openface',
                        help='Overlay openface data', action='store_true')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    group_directory = args['group_directory']
    specific_task = args['specific_task']
    specific_frame = args['specific_frame']
    openpose = args['openpose']
    openface = args['openface']
    densepose = args['densepose']
    verbose = args['verbose']

    main(group_directory, specific_frame=specific_frame, specific_task=specific_task,
         openpose=openpose, openface=openface, densepose=densepose, verbose=verbose)
