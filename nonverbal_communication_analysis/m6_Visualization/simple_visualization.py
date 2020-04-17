import argparse
import re
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from nonverbal_communication_analysis.environment import (OPENPOSE_KEY, OPENFACE_KEY, DATASET_SYNC,
                                                          SUBJECT_IDENTIFICATION_GRID, VALID_VIDEO_TYPES,
                                                          VALID_OUTPUT_FILE_TYPES, OPENPOSE_OUTPUT_DIR,
                                                          OPENFACE_OUTPUT_DIR, DATASET_SYNC,
                                                          QUADRANT_MIN, QUADRANT_MAX, VIDEO_RESOLUTION,
                                                          FEATURE_AGGREGATE_DIR, OPENPOSE_KEYPOINT_LINKS,
                                                          OPENPOSE_KEYPOINT_MAP)
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files, list_dirs)
from nonverbal_communication_analysis.m0_Classes.Experiment import get_group_from_file_path


class Visualizer(object):

    COLOR_MAP = {0: (0, 0, 0),
                 1: (255, 0, 0),
                 2: (0, 255, 255),
                 3: (0, 255, 0),
                 4: (0, 0, 255)}

    COLORMAP = {0: 'black',
                1: 'blue',
                2: 'yellow',
                3: 'green',
                4: 'red'}

    def __init__(self, group_id: str):
        self.group_id = group_id
        self.frame_feature_files = dict()

    def get_frame(self, cap, frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def show_subjects_frame(self, camera: str, frame: int, highlighted_subject=None, assigned_subjects: dict = None, key: str = OPENPOSE_KEY):
        _, ax = plt.subplots()
        ax.set_xlim(QUADRANT_MIN, QUADRANT_MAX)
        ax.set_ylim(QUADRANT_MAX, QUADRANT_MIN)
        ax.set_title('Subjects Keypoints Position')

        file_path = DATASET_SYNC / self.group_id / 'task_1'
        video_file = filter_files(fetch_files_from_directory(
            [file_path]), VALID_VIDEO_TYPES, include=camera)[0]
        path = file_path / video_file
        cap = cv2.VideoCapture(str(path))

        display_frame = self.get_frame(cap, frame)
        ax.imshow(display_frame, aspect='auto',
                  extent=(QUADRANT_MIN, QUADRANT_MAX,
                          QUADRANT_MAX, QUADRANT_MIN),
                  alpha=1, zorder=-1)

        if key == OPENPOSE_KEY:
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

        elif key == OPENFACE_KEY:
            if assigned_subjects is not None:
                for _, assigned_subject in assigned_subjects.items():
                    subject_pose = assigned_subject.face['openface']['face']
                    keypoints_df = pd.DataFrame(
                        subject_pose.values())
                    for _, keypoint in keypoints_df.iterrows():
                        ax.scatter(x=keypoint['x'],
                                   y=keypoint['y'], c='blue')
                        pass

        for _, polygon in SUBJECT_IDENTIFICATION_GRID[camera].items():
            pol_x, pol_y = polygon.exterior.xy
            plt.plot(pol_x, pol_y)

        plt.show()

    def openpose_overlay_frame(self, ax, specific_frame: int = None, specific_task: int = None, verbose: bool = False):
        openpose_group_data = [x for x in (OPENPOSE_OUTPUT_DIR / self.group_id).iterdir()
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
                ax[idx].set_xlim(QUADRANT_MIN, QUADRANT_MAX)
                ax[idx].set_ylim(QUADRANT_MAX, QUADRANT_MIN)

                for assigned_subject in assigned_subjects:
                    subject_id = assigned_subject['id']
                    subject_pose = assigned_subject['pose']['openpose']

                    pose_keypoints_df = pd.DataFrame(
                        subject_pose.values(), columns=['x', 'y', 'c'])
                    ax[idx].scatter(x=pose_keypoints_df['x'],
                                    y=pose_keypoints_df['y'],
                                    c=self.COLORMAP[subject_id], marker='o')

                for _, polygon in SUBJECT_IDENTIFICATION_GRID[camera].items():
                    pol_x, pol_y = polygon.exterior.xy
                    ax[idx].plot(pol_x, pol_y)

    def openface_overlay_frame(self, ax, specific_frame: int = None, specific_task: int = None, verbose: bool = False):
        """Overlay Openface data on a frame

        This method needs to have a camera_files dict instead of a list as openface does
        not output an empty file for unsuccessful facial keypoints detection in a frame,
        unlike openpose.

        Args:
            ax (plt.Axes): matplotlib plot axes
            specific_frame (int, optional): Specific frame. Defaults to None.
            specific_task (int, optional): Specific task. Defaults to None.
            verbose (bool, optional): Verbose. Defaults to False.
        """
        openface_group_data = [x for x in (OPENFACE_OUTPUT_DIR / self.group_id).iterdir()
                               if '_clean' in x.name][0]

        if (specific_frame and specific_task) is not None:
            openface_task_directory = [x for x in openface_group_data.iterdir()
                                       if x.is_dir() and str(specific_task) in str(x)][0]
            openface_task_cameras_directories = [
                x for x in openface_task_directory.iterdir()]
            openface_task_cameras_directories.sort()

            assigned_subjects = dict()
            for camera_directory in openface_task_cameras_directories:
                camera = camera_directory.name
                openface_files = {int(re.search(r'(?<=_)(\d{12})(?=_)',
                                                x.name).group(0)): x for x in camera_directory.iterdir(
                ) if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES}

                if specific_frame in openface_files:
                    subjects_data = json.load(
                        open(openface_files[specific_frame], 'r'))
                    assigned_subjects[camera] = subjects_data['subjects']

            for camera, assigned_subjects in assigned_subjects.items():
                idx = int(camera[2])-1
                ax[idx].set_xlim(QUADRANT_MIN, QUADRANT_MAX)
                ax[idx].set_ylim(QUADRANT_MAX, QUADRANT_MIN)

                for assigned_subject in assigned_subjects:
                    subject_id = assigned_subject['id']
                    subject_eyes = assigned_subject['face']['openface']['eye']
                    subject_face = assigned_subject['face']['openface']['face']

                    face_keypoints_df = pd.DataFrame(
                        subject_face.values())
                    eyes_keypoints_df = pd.DataFrame(subject_eyes.values())

                    ax[idx].scatter(x=face_keypoints_df['x'],
                                    y=face_keypoints_df['y'],
                                    c=self.COLOR_MAP[subject_id], marker='.')

                    ax[idx].scatter(x=eyes_keypoints_df['x'],
                                    y=eyes_keypoints_df['y'],
                                    c=self.COLOR_MAP[subject_id], marker='.')

    def show_frame(self, group_directory: str, specific_frame: int = None, specific_task: int = None, openpose: bool = False, openface: bool = False, densepose: bool = False, verbose: bool = False):
        group_directory = Path(group_directory)
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
                           extent=(QUADRANT_MIN, QUADRANT_MAX,
                                   QUADRANT_MAX, QUADRANT_MIN),
                           alpha=1, zorder=-1)

        if openpose:
            print("Overlay Openpose")
            self.openpose_overlay_frame(ax=ax, specific_frame=specific_frame,
                                        specific_task=specific_task, verbose=verbose)

        if openface:
            print("Overlay Openface")
            self.openface_overlay_frame(ax=ax, specific_frame=specific_frame,
                                        specific_task=specific_task, verbose=verbose)

        if densepose:
            print("Overlay Densepose")
            # self.densepose_overlay(ax=ax, specific_frame=specific_frame,
            #                        specific_task=specific_task, verbose=verbose)

        plt.show()

    #########
    # VIDEO #
    #########

    def rescale_frame(self, frame):
        width = VIDEO_RESOLUTION['pc3']['x']
        height = VIDEO_RESOLUTION['pc3']['y']
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    def openpose_overlay_video(self, openpose_data: dict, img_frame: np.ndarray, camera: str):
        subject_id = openpose_data['id'] if 'id' in openpose_data else 0
        del openpose_data['id']

        for key in openpose_data.keys():
            for keypoint_idx, keypoint_values in openpose_data[key].items():
                keypoint_x = round(
                    keypoint_values[0] * VIDEO_RESOLUTION[camera]['x'])
                keypoint_y = round(
                    keypoint_values[1] * VIDEO_RESOLUTION[camera]['y'])
                keypoint_c = round(keypoint_values[2] * 5)

                cv2.circle(img_frame, (keypoint_x, keypoint_y),
                           keypoint_c, self.COLOR_MAP[subject_id], -1)

                if key == 'pose':
                    keypoint_idx = int(keypoint_idx)
                    if keypoint_idx in OPENPOSE_KEYPOINT_LINKS:
                        for keypoint_link_idx in OPENPOSE_KEYPOINT_LINKS[keypoint_idx]:
                            pose_keypoints = openpose_data['pose']
                            if str(keypoint_link_idx) in pose_keypoints:
                                keypoint = pose_keypoints[str(
                                    keypoint_link_idx)]
                                keypoint_link_x = round(
                                    keypoint[0] * VIDEO_RESOLUTION[camera]['x'])
                                keypoint_link_y = round(
                                    keypoint[1] * VIDEO_RESOLUTION[camera]['y'])

                                if keypoint_x == 0 or keypoint_y == 0 or keypoint_link_x == 0 or keypoint_link_y == 0:
                                    break

                                cv2.line(img_frame, (keypoint_x, keypoint_y),
                                         (keypoint_link_x, keypoint_link_y), self.COLOR_MAP[subject_id], 2)

                                if camera == 'pc1' and int(keypoint_idx) == int(OPENPOSE_KEYPOINT_MAP['NECK']):
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    if subject_id <= 2:
                                        cv2.line(img_frame, (keypoint_x, keypoint_y),
                                                 (keypoint_x-5, keypoint_y+15), (0, 0, 255), 2)
                                        cv2.putText(
                                            img_frame, 'x', (keypoint_x, keypoint_y+15), font, 0.5, (50, 50, 255), 1, cv2.LINE_AA)
                                        cv2.line(img_frame, (keypoint_x, keypoint_y),
                                                 (keypoint_x+30, keypoint_y), (0, 255, 0), 2)
                                        cv2.putText(
                                            img_frame, 'y', (keypoint_x+30, keypoint_y), font, 0.5, (124, 247, 2), 1, cv2.LINE_AA)
                                        cv2.line(img_frame, (keypoint_x, keypoint_y),
                                                 (keypoint_x, keypoint_y-30), (255, 0, 0), 2)
                                        cv2.putText(
                                            img_frame, 'z', (keypoint_x, keypoint_y-30), font, 0.5, (247, 247, 2), 1, cv2.LINE_AA)
                                    else:
                                        cv2.line(img_frame, (keypoint_x-5, keypoint_y-20),
                                                 (keypoint_x, keypoint_y), (0, 0, 255), 2)
                                        cv2.putText(
                                            img_frame, 'x', (keypoint_x-15, keypoint_y-20), font, 0.5, (50, 50, 255), 1, cv2.LINE_AA)
                                        cv2.line(img_frame, (keypoint_x, keypoint_y),
                                                 (keypoint_x-30, keypoint_y), (0, 255, 0), 2)
                                        cv2.putText(
                                            img_frame, 'y', (keypoint_x-30, keypoint_y+10), font, 0.5, (124, 247, 2), 1, cv2.LINE_AA)
                                        cv2.line(img_frame, (keypoint_x, keypoint_y),
                                                 (keypoint_x, keypoint_y-30), (255, 0, 0), 2)
                                        cv2.putText(
                                            img_frame, 'z', (keypoint_x+2, keypoint_y-30), font, 0.5, (247, 247, 2), 1, cv2.LINE_AA)

    def feature_overlay_video(self, img_frame: np.ndarray, frame_idx: int, camera: str, openpose: bool = False, openface: bool = False, densepose: bool = False, verbose: bool = False):
        frame_data = json.load(
            open(self.frame_feature_files[camera][frame_idx], 'r'))
        frame_id = frame_data['frame']
        subjects_data = frame_data['subjects']

        if frame_id != frame_idx:
            print("ERROR, Frames numbers don't match")
            exit()

        for subject in subjects_data:
            if openpose:
                openpose_data = dict()
                openpose_data['id'] = subject['id'] if 'id' in subject else dict()
                openpose_data['pose'] = subject['pose']['openpose'] \
                    if 'pose' in subject else dict()
                openpose_data['face'] = subject['face']['openpose'] \
                    if 'face' in subject else dict()
                self.openpose_overlay_video(openpose_data, img_frame, camera)
            # if openface:
            #    self.openface_overlay_video()
            # if densepose:
            #     self.densepose_overlay_video()

    def play_video(self, group_directory: str, specific_task: int = None, openpose: bool = False, openface: bool = False, densepose: bool = False, verbose: bool = False):
        group_directory = Path(group_directory)
        tasks_directories = [x for x in group_directory.iterdir()
                             if x.is_dir() and 'task' in x.name]

        if specific_task is not None:
            tasks_directories = [x for x in tasks_directories
                                 if x.is_dir() and str(specific_task) in x.name]

        video_captures = dict()
        for task_directory in tasks_directories:
            print("Playing %s" % task_directory.name)
            video_files_paths = [x for x in task_directory.iterdir()
                                 if not x.is_dir() and x.suffix in VALID_VIDEO_TYPES]
            video_files_paths.sort()

            task_feature_data_path = group_directory / \
                FEATURE_AGGREGATE_DIR / task_directory.name

            # Load frames feature files
            camera_dirs_list = [x for x in task_feature_data_path.iterdir()
                                if x.is_dir() and 'pc' in x.name]

            for camera_dir in camera_dirs_list:
                if camera_dir.name not in camera_dirs_list:
                    self.frame_feature_files[camera_dir.name] = dict()

                self.frame_feature_files[camera_dir.name] = {int(re.search(r'(\d{12})', x.name).group(0)): x
                                                             for x in camera_dir.iterdir()
                                                             if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES}

            # Load videos
            for video_file in video_files_paths:
                camera_id = re.search(
                    r'(?<=Video)(pc\d{1})(?=\d{14})', video_file.name).group(0)
                video_captures[camera_id] = cv2.VideoCapture(str(video_file))

            frame_counter = 0
            while(all(video_capture.isOpened() for video_capture in video_captures.values())):
                for camera in video_captures:
                    video_capture = video_captures[camera]
                    ret, frame = video_capture.read()
                    if ret == True:
                        self.feature_overlay_video(frame, frame_counter, camera,
                                                   openpose=openpose,
                                                   openface=openface,
                                                   densepose=densepose)
                        frame = self.rescale_frame(frame)
                        cv2.imshow('Task %s - Camera %s' %
                                   (task_directory.name, camera), frame)

                        # cv2.imwrite("%s.jpg" % camera, frame)
                        # cv2.waitKey()
                        # return
                    else:
                        for camera, video_capture in video_captures.items():
                            video_capture.release()
                        break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 0x20:
                    while cv2.waitKey(-1) & 0xFF != 0x20:
                        pass

                frame_counter += 1

            cv2.destroyAllWindows()


def main(group_directory: str, specific_frame: int = None, specific_task: int = None, openpose: bool = False, openface: bool = False, densepose: bool = False, verbose: bool = False):
    group_id = get_group_from_file_path(group_directory)

    visualizer = Visualizer(group_id)
    if (specific_frame and specific_task) is not None:
        visualizer.show_frame(group_directory, specific_task=specific_task,
                              specific_frame=specific_frame, openpose=openpose,
                              openface=openface, densepose=densepose, verbose=verbose)
    else:
        visualizer.play_video(group_directory, specific_task=specific_task, openpose=openpose,
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
