import argparse
import json
import pandas as pd
import os
import re
from pathlib import Path
from nonverbal_communication_analysis.m0_Classes.Experiment import get_group_from_file_path
from nonverbal_communication_analysis.environment import OPENPOSE_OUTPUT_DIR, VALID_OUTPUT_FILE_TYPES, SUBJECT_AXES
from nonverbal_communication_analysis.utils import log


class OpenposeSubject(object):
    def __init__(self, _id):
        self.id = _id
        self.previous_pose = dict()
        self.current_pose = dict()
        self.expansiveness = dict()

    def __str__(self):
        return "OpenposeSubject: {id: %s}" % self.id

    def _update_pose(self, camera, current_pose, verbose: bool = False):
        if verbose:
            print("Subject", self.id,
                  "\nPrev.: ", self.previous_pose.keys())
        self.previous_pose = self.current_pose
        self.current_pose[camera] = current_pose
        if verbose:
            print("Current:", self.current_pose.keys())

    def metric_expansiveness(self, verbose: bool = False):
        if verbose:
            print("Expansiveness on ", self)
        # TODO: Change if necessary to use kypoint confidence to get
        # minimum keypoint value
        horizontal = {'min': None, 'max': None}
        vertical = {'min': None, 'max': None}

        print("=====", self, "=====")

        for camera, keypoints in self.current_pose.items():
            for _, keypoint in keypoints.items():
                if not horizontal['min']:
                    horizontal['min'] = keypoint[0]
                elif keypoint[0] < horizontal['min']:
                    horizontal['min'] = keypoint[0]

                if not horizontal['max']:
                    horizontal['max'] = keypoint[0]
                elif keypoint[0] > horizontal['max']:
                    horizontal['max'] = keypoint[0]

                if not vertical['min']:
                    vertical['min'] = keypoint[1]
                elif keypoint[1] < vertical['min']:
                    vertical['min'] = keypoint[1]

                if not vertical['max']:
                    vertical['max'] = keypoint[1]
                elif keypoint[1] > vertical['max']:
                    vertical['max'] = keypoint[1]

            if camera not in self.expansiveness:
                self.expansiveness[camera] = dict()

            self.expansiveness[camera]['x'] = [horizontal['min'],
                                               horizontal['max']]
            self.expansiveness[camera]['y'] = [vertical['min'],
                                               vertical['max']]

        print(self.expansiveness)


class OpenposeProcess(object):
    """ Openpose pose related metrics

        Subject-wise:
        * Horizontal/Vertical Expansion - Occupied Area
        * Number of Interactions with table center objects
        * Body Direction Vector
        * Hand / Head / Body energy (heatmap)

        Group-wise:
        * Intragroup Distance
        * Group Energy
    """

    def __init__(self, group_id: str, verbose: bool = False):
        self.group_id = group_id
        self.clean_group_dir = OPENPOSE_OUTPUT_DIR / \
            group_id / (group_id + '_clean')
        if not Path(self.clean_group_dir).is_dir():
            log('ERROR', 'This step requires the output of openpose data cleaning step')
        os.makedirs(self.clean_group_dir, exist_ok=True)

        self.current_frame = -1
        self.n_subjects = -1
        self.subjects = {subject_id: OpenposeSubject(
            subject_id) for subject_id in range(1, 5)}
        self.verbose = verbose

    @property
    def current_frame(self):
        try:
            return self.__current_frame
        except AttributeError:
            self.__current_frame = -1
            return self.__current_frame

    @current_frame.setter
    def current_frame(self, value):
        if value >= self.current_frame or value == -1:
            self.__current_frame = value
        else:
            log('ERROR', 'Analyzing previous frame. Frame processing should be ordered.')

    @property
    def n_subjects(self):
        try:
            return self.__n_subjects
        except AttributeError:
            self.__n_subjects = -1
            return self.__n_subjects

    @n_subjects.setter
    def n_subjects(self, value):
        if value == 4 or value == -1:
            self.__n_subjects = value
        else:
            log('ERROR', 'invalid number of subjects. Keep previous frame\'s subject pose')

    def has_required_cameras(self, subject):
        subject_cameras = set(subject.current_pose.keys())
        requirements = {'x': False,
                        'y': False,
                        'z': False}
        for subject_axis, required_cameras in SUBJECT_AXES[subject.id].items():
            if set(required_cameras).intersection(subject_cameras):
                requirements[subject_axis] = True

        return all([state for state in requirements.values()])

    def frame_data_transform(self, frame_data):
        # transform subjects list to dict
        frame_idx = frame_data['frame']
        frame_subjects = {subject['id']: subject['pose']['openpose']
                          for subject in frame_data['subjects']}

        data = {'frame': frame_idx,
                'subjects': frame_subjects}

        return data

    def camera_frame_parse_subjects(self, camera, frame_data):
        frame_subjects = frame_data['subjects']
        frame_idx = frame_data['frame']
        self.current_frame = frame_idx

        for subject_id, subject in self.subjects.items():
            if subject_id in frame_subjects:
                subject._update_pose(camera, frame_subjects[subject_id])
            elif not subject.previous_pose:
                if self.verbose:
                    log('INFO', 'No previous pose. Attention to when using 2nd ' +
                        'choice camera to calculate metrics ' +
                        '(frame: %s, camera: %s, subject: %s)' %
                        (frame_idx, camera, subject))

    def process_subject(self, subject):
        subject.metric_expansiveness()

    def handle_frames(self, camera_frame_files: dict, output_directory: str, prettify: bool = False, verbose: bool = False, display: bool = False):
        for frame_idx in sorted(camera_frame_files):
            frame_camera_dict = camera_frame_files[frame_idx]
            for camera, frame_file in frame_camera_dict.items():
                output_frame_directory = output_directory / camera
                output_frame_file = output_frame_directory / \
                    ("processed_%s_%.12d_clean.json" % (camera, frame_idx))
                if not output_frame_directory.is_dir():
                    log('ERROR', 'Directory does not exist')
                data = json.load(open(frame_file))
                data = self.frame_data_transform(data)
                self.camera_frame_parse_subjects(camera, data)

            for _, subject in self.subjects.items():
                if not self.has_required_cameras(subject):
                    log('ERROR', 'Subject does not have data from required cameras. ' +
                        'Not enough information to process frame')
                self.process_subject(subject)

            if frame_idx == 0:
                exit()

    def process(self, tasks_directories: dict, specific_frame: int = None, prettify: bool = False, verbose: bool = False, display: bool = False):
        clean_task_directory = self.clean_group_dir
        clean_tasks_directories = list()
        for task in tasks_directories:
            clean_tasks_directories += ([x for x in clean_task_directory.iterdir()
                                         if x.is_dir() and task.name in x.name])

        for task in clean_tasks_directories:
            clean_camera_directories = [x for x in task.iterdir()
                                        if x.is_dir()]
            camera_files = dict()
            for camera_id in clean_camera_directories:
                for frame_file in camera_id.iterdir():
                    frame_idx = int(re.search(r'(?<=_)(\d{12})(?=_)',
                                              frame_file.name).group(0))
                    if frame_idx not in camera_files:
                        camera_files[frame_idx] = dict()
                    if camera_id not in camera_files[frame_idx]:
                        camera_files[frame_idx][camera_id.name] = dict()
                    camera_files[frame_idx][camera_id.name] = frame_file

            if specific_frame is not None:
                specific_camera_files = dict()
                for camera_id in clean_camera_directories:
                    specific_camera_files[specific_frame] = camera_files[specific_frame]
                camera_files = specific_camera_files

            output_directory = self.clean_group_dir / task.name
            self.handle_frames(camera_files, output_directory, prettify=prettify,
                               verbose=verbose, display=display)
