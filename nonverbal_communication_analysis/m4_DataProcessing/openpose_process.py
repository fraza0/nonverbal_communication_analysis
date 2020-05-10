import argparse
import json
import pandas as pd
import os
import re
import numpy as np
from statistics import mean
from math import sqrt
from pathlib import Path
from nonverbal_communication_analysis.m0_Classes.Experiment import get_group_from_file_path, Experiment
from nonverbal_communication_analysis.m0_Classes.Subject import Subject
from nonverbal_communication_analysis.environment import OPENPOSE_OUTPUT_DIR, VALID_OUTPUT_FILE_TYPES, SUBJECT_AXES, OPENPOSE_KEYPOINT_MAP, OPENPOSE_KEY, CAMERAS_3D_AXES
from nonverbal_communication_analysis.utils import log
from sympy import Polygon


def distance_between_points(p1: tuple, p2: tuple):
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x2-x1)**2 + (y2-y1)**2)


class OpenposeSubject(Subject):
    # See Also: Between Shoulders Distance Analysis section of Openpose Analysis notebook
    _distance_threshold = 0.5

    def __init__(self, _id, prettify: bool = False, verbose: bool = False):
        self.id = _id
        self.previous_pose = dict()
        self.current_pose = dict()
        self.current_face = dict()
        self.expansiveness = dict()
        self.body_direction = dict()
        self.center_interaction = 0.0
        self.verbose = verbose
        self.prettify = prettify
        self.subject_shoulder_distances = dict()

    def __str__(self):
        return "OpenposeSubject: {id: %s}" % self.id

    def _update_pose(self, camera, current_pose, verbose: bool = False):
        verbose = False
        if verbose:
            print("Subject", self.id,
                  "\nPrev.: ", self.previous_pose.keys())

        self.previous_pose = self.current_pose

        shoulder_distance = distance_between_points(current_pose[str(OPENPOSE_KEYPOINT_MAP['R_SHOULDER'])][:2],
                                                    current_pose[str(OPENPOSE_KEYPOINT_MAP['L_SHOULDER'])][:2])

        if camera not in self.subject_shoulder_distances:
            self.subject_shoulder_distances[camera] = list()
            self.subject_shoulder_distances[camera].append(shoulder_distance)

        mean_shoulder_distance = mean(self.subject_shoulder_distances[camera])
        mean_min = mean_shoulder_distance-mean_shoulder_distance*self._distance_threshold
        mean_max = mean_shoulder_distance+mean_shoulder_distance*self._distance_threshold

        if mean_min <= shoulder_distance <= mean_max:
            # this is only updated when valid current pose so it doesn't push the mean value towards a more solid value
            self.subject_shoulder_distances[camera].append(shoulder_distance)
            self.current_pose[camera] = current_pose

        if verbose:
            print("Current:", self.current_pose.keys())

    def metric_expansiveness(self, verbose: bool = False):
        """Calculate subject expansiveness in 2D image.
        Get both maximum and minimum keypoint values from both X and Y coordinates.

        Coordinates X and Y are retrieved and available in every camera view (as it is 2D).
        If this was a 3D scenario (as it was intented to be by the use of different perspective
        cameras) X and Y would be X and Z. The Y coordinate would represent the deviation of the 
        subject's hands related to the body.

        Args:
            verbose (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        if verbose:
            print("Expansiveness on ", self)
        # TODO: Change if necessary to use keypoint confidence to get
        # minimum keypoint value
        horizontal = {'min': None, 'max': None}
        vertical = {'min': None, 'max': None}

        expansiveness = dict()

        # print("=====", self, "=====")
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

            if camera not in expansiveness:
                expansiveness[camera] = dict()

            expansiveness[camera]['x'] = [horizontal['min'],
                                          horizontal['max']]
            expansiveness[camera]['y'] = [vertical['min'],
                                          vertical['max']]
        return expansiveness

    def metric_body_direction(self, verbose: bool = False):
        """
        IMPOSSIBLE TO CREATE 3D Points necessary for the
        calculation of the direction vector

        See Also:
            http://chenlab.ece.cornell.edu/people/adarsh/publications/BackProjection.pdf
        """
        # print(self)
        # prefered_camera = True

        # subject_id = self.id
        # subject_pose = self.current_pose

        # axes = SUBJECT_AXES[self.id]

        # neck_keypoint = list()

        # if subject_id == 1 or subject_id == 2:
        #     for axis in axes.keys():
        #         cameras = axes[axis]
        #         prefered_camera = True
        #         for camera in cameras:
        #             print(axis, camera, prefered_camera)
        #             if camera in subject_pose:
        #                 neck_keypoint = subject_pose[camera][str(OPENPOSE_KEYPOINT_MAP['NECK'])]
        #                 print(axis, camera, neck_keypoint)
        #                 break
        #             else:
        #                 prefered_camera = False
        #                 print("=======",camera, "not in pose!!!")
        #             # if camera in subject_pose:
        #         # print(self.current_pose[])

        # elif subject_id == 3 or subject_id == 4:
        #     print("3 or 4")

        # neck_x = np.array(keypoints[str(OPENPOSE_KEYPOINT_MAP['NECK'])][:2])
        # l_shoulder = np.array(keypoints[str(OPENPOSE_KEYPOINT_MAP['L_SHOULDER'])][:2])
        # r_shoulder = np.array(keypoints[str(OPENPOSE_KEYPOINT_MAP['R_SHOULDER'])][:2])
        # neck_lshoulder_vector = neck-l_shoulder
        # neck_rshoulder_vector = neck-r_shoulder
        # print(self.id, camera, np.cross(neck_rshoulder_vector, neck_lshoulder_vector))

    def metric_center_interaction(self, group_data, subject_expansiveness):
        sideview_camera = 'pc1'
        camera_relative_coordinates = CAMERAS_3D_AXES[sideview_camera]
        hands_body_deviation_coordinate = list(camera_relative_coordinates.keys())[
            list(camera_relative_coordinates.values()).index('y')]
        table_center = group_data[sideview_camera]['center']
        hands_body_deviation = subject_expansiveness[sideview_camera][hands_body_deviation_coordinate]
        center_proximity = distance_between_points(
            table_center, hands_body_deviation)

        return center_proximity

    def to_json(self):
        """Transform Subject object to JSON format.
        If attribute is empty it is not printed

        Returns:
            str: JSON formatted Subject object
        """

        obj = {
            "id": self.id,
            "pose": {
                OPENPOSE_KEY.lower(): self.current_pose,
                "metrics": {
                    "expansiveness": self.expansiveness,
                    "center_interaction": self.center_interaction,
                    # "body_direction": self.body_direction, # Impossible to measure without 3D data
                },
            },
            "face": {
                OPENPOSE_KEY.lower(): self.current_face
            }
        }

        return obj


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

    def __init__(self, group_id: str, prettify: bool = False, verbose: bool = False):
        self.group_id = group_id
        self.clean_group_dir = OPENPOSE_OUTPUT_DIR / \
            group_id / (group_id + '_clean')
        if not Path(self.clean_group_dir).is_dir():
            log('ERROR', 'This step requires the output of openpose data cleaning step')
        os.makedirs(self.clean_group_dir, exist_ok=True)

        self.output_group_dir = OPENPOSE_OUTPUT_DIR / \
            group_id / (group_id + '_processed')
        os.makedirs(self.output_group_dir, exist_ok=True)

        self.experiment = Experiment(group_id)
        json.dump(self.experiment.to_json(),
                  open(self.output_group_dir / (self.group_id + '.json'), 'w'))

        self.current_frame = -1
        self.n_subjects = -1
        self.subjects = {subject_id: OpenposeSubject(
            subject_id, verbose) for subject_id in range(1, 5)}
        self.intragroup_distance = dict()
        self.prettify = prettify
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
        if value >= self.current_frame or value <= 0:
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

    def to_json(self):
        group_metrics = {
            "intragroup_distance": self.intragroup_distance,
        }
        return group_metrics

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
        frame_subjects_pose, frame_subjects_face = dict(), dict()
        for subject in frame_data['subjects']:
            subject_id = subject['id']
            frame_subjects_pose[subject_id] = subject['pose']['openpose']
            frame_subjects_face[subject_id] = subject['face']['openpose']

        data = {'frame': frame_idx,
                'subjects': {'pose': frame_subjects_pose,
                             'face': frame_subjects_face}}
        return data

    def camera_frame_parse_subjects(self, camera, frame_data):
        frame_subjects = frame_data['subjects']
        frame_subject_pose = frame_subjects['pose']
        frame_subject_face = frame_subjects['face']
        frame_idx = frame_data['frame']
        self.current_frame = frame_idx

        for subject_id, subject in self.subjects.items():
            if subject_id in frame_subject_pose:
                subject_data = frame_subject_pose[subject_id]
                subject._update_pose(
                    camera, subject_data, self.verbose)
            elif camera not in subject.previous_pose:
                # subject has no pose nor previous pose. Must skip frame.
                if self.verbose:
                    log('INFO', 'Subject has no previous pose. Attention when resorting to 2nd ' +
                        'choice camera to calculate metrics ' +
                        '(frame: %s, camera: %s, subject: %s)' %
                        (frame_idx, camera, subject))
                return False
            else:
                if self.verbose:
                    log('INFO', 'Subject (%s) has no pose in this frame (%s - %s), but previous pose on this camera can be used' %
                        (subject_id, frame_idx, camera))
                subject._update_pose(
                    camera, subject.previous_pose[camera], self.verbose)

            if subject_id in frame_subject_face:
                subject_data = frame_subject_face[subject_id]
                subject.current_face[camera] = subject_data
        return True

    def process_subject_individual_metrics(self, subject, group_data):
        subject.expansiveness = subject.metric_expansiveness()
        # subject.body_direction = subject.metric_body_direction() # SEE COMMENTS ON METHOD ABOVE
        subject.center_interaction = subject.metric_center_interaction(
            group_data, subject.expansiveness)

    def metric_intragroup_distance(self, subjects):
        subjects_distance = dict()
        for subject_id, subject in subjects.items():
            subject_pose = subject.current_pose
            for camera, keypoints in subject_pose.items():
                if camera not in subjects_distance:
                    subjects_distance[camera] = dict()
                neck_keypoint = tuple(
                    keypoints[str(OPENPOSE_KEYPOINT_MAP['NECK'])])
                subjects_distance[camera][subject_id] = neck_keypoint[:2]

        intragroup_distance = dict()
        for camera, subjects in subjects_distance.items():
            points = list(subjects.values())
            points.append(points[0])
            polygon = Polygon(*points)
            polygon_area = float(abs(polygon.area))
            centroid = polygon.centroid
            polygon_center = [float(centroid.x), float(centroid.y)]

            intragroup_distance[camera] = {'polygon': points,
                                           'area': polygon_area,
                                           'center': polygon_center}
        return intragroup_distance

    def save_output(self, output_directory, frame_validity):

        frame = self.current_frame
        output_frame_file = output_directory / \
            ("frame_%.12d_processed.json" % (frame))
        os.makedirs(output_directory, exist_ok=True)
        if not output_directory.is_dir():
            log('ERROR', 'Directory does not exist')

        obj = {
            "frame": self.current_frame,
            "is_processed_data_valid": frame_validity,
            "group": self.to_json(),
            "subjects": [subject.to_json() for subject_id, subject in self.subjects.items()],
        }

        if self.prettify:
            json.dump(obj, open(output_frame_file, 'w'), indent=2)
        else:
            json.dump(obj, open(output_frame_file, 'w'))

    def handle_frames(self, camera_frame_files: dict, output_directory: str, display: bool = False):
        for frame_idx in sorted(camera_frame_files):
            # print('=== FRAME %s ===' % frame_idx)
            self.current_frame = frame_idx
            frame_camera_dict = camera_frame_files[frame_idx]
            is_valid_frame = True
            for camera, frame_file in frame_camera_dict.items():
                data = json.load(open(frame_file))
                data = self.frame_data_transform(data)
                is_valid_frame = self.camera_frame_parse_subjects(camera, data)
                if not is_valid_frame:
                    if self.verbose:
                        log('INFO', 'Not enough poses detected. Skipping frame')
                    break

            if is_valid_frame:
                group_data = self.metric_intragroup_distance(
                    self.subjects)
                self.intragroup_distance = group_data
                for _, subject in self.subjects.items():
                    if not self.has_required_cameras(subject):
                        log('ERROR', 'Subject (%s) does not have data from required cameras. ' % subject.id +
                            'Not enough information to process frame (%s)' % frame_idx)
                    self.process_subject_individual_metrics(
                        subject, group_data)

            # writting every frame. Indent if invalid frames should not be saved
            self.save_output(output_directory, is_valid_frame)

            if frame_idx == 3:
                exit()

    def process(self, tasks_directories: dict, specific_frame: int = None, display: bool = False):
        clean_task_directory = self.clean_group_dir
        clean_tasks_directories = list()
        for task in tasks_directories:
            clean_tasks_directories += ([x for x in clean_task_directory.iterdir()
                                         if x.is_dir() and task.name in x.name])

        for task in clean_tasks_directories:
            clean_camera_directories = [x for x in task.iterdir()
                                        if x.is_dir()]
            clean_camera_directories.sort()
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

            output_directory = self.output_group_dir / task.name
            self.handle_frames(camera_files, output_directory, display=display)
