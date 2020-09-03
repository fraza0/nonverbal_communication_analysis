import argparse
import json
import os
import re
from itertools import permutations, combinations
from math import sqrt
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from sympy import Point, Polygon
from shapely import geometry as shapely

from nonverbal_communication_analysis.environment import (
    CAMERAS, DENSEPOSE_KEY, DENSEPOSE_KEYPOINT_MAP,
    DENSEPOSE_OUTPUT_DIR, SCALE_FACTOR, SCALE_SUBJECTS, SUBJECT_AXES,
    VALID_OUTPUT_FILE_TYPES, SIDEVIEW_CAMERA)
from nonverbal_communication_analysis.m0_Classes.Experiment import (
    Experiment, get_group_from_file_path)
from nonverbal_communication_analysis.m0_Classes.Subject import Subject
from nonverbal_communication_analysis.utils import (
    log, polygon_vertices_from_2_points)


def distance_between_points(p1: tuple, p2: tuple):
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x2-x1)**2 + (y2-y1)**2)


class DenseposeSubject(Subject):
    _distance_threshold = 0.5

    def __init__(self, _id, prettify: bool = False, verbose: bool = False):
        self.id = _id
        self.previous_pose = dict()
        self.current_pose = dict()
        self.expansiveness = dict()
        self.overlap = dict()
        self.body_direction = dict()
        self.energy = dict()
        self.center_interaction = dict()
        self.verbose = verbose
        self.prettify = prettify
        self.subject_shoulder_distances = dict()

    def __str__(self):
        return "DenseposeSubject: {id: %s}" % self.id

    def _update_pose(self, camera, current_pose, verbose: bool = False):
        verbose = False
        if verbose:
            print("Subject", self.id,
                  "\nPrev.: ", self.previous_pose.keys())

        if camera not in self.current_pose:
            self.current_pose[camera] = dict()

        self.previous_pose[camera] = self.current_pose[camera].copy()

        shoulder_distance = distance_between_points(current_pose[str(DENSEPOSE_KEYPOINT_MAP['right_shoulder'])][:2],
                                                    current_pose[str(DENSEPOSE_KEYPOINT_MAP['left_shoulder'])][:2])

        if camera not in self.subject_shoulder_distances:
            self.subject_shoulder_distances[camera] = list()
            self.subject_shoulder_distances[camera].append(
                shoulder_distance)

        mean_shoulder_distance = mean(
            self.subject_shoulder_distances[camera])
        mean_min = mean_shoulder_distance-mean_shoulder_distance*self._distance_threshold
        mean_max = mean_shoulder_distance+mean_shoulder_distance*self._distance_threshold

        if mean_min <= shoulder_distance <= mean_max:
            # this is only updated when valid current pose so it doesn't push
            # the mean value towards a more solid value
            self.subject_shoulder_distances[camera].append(
                shoulder_distance)
            self.current_pose[camera] = current_pose
            # if self.id == 1:
            #     print('Prev.', self.previous_pose, '\nCurr.', self.current_pose,
            #           '\n\n')

        if verbose:
            print("Current:", self.current_pose.keys())

    def metric_expansiveness(self):
        """Calculate subject expansiveness in 2D image.
        Get both maximum and minimum keypoint values from both X and Y coordinates.

        Coordinates X and Y are retrieved and available in every camera view (as it is 2D).
        If this was a 3D scenario (as it was intented to be by the use of different perspective
        cameras) X and Y would be X and Z. The Y coordinate would represent the deviation of the 
        subject's hands related to the body.

        Args:

        Returns:
            [type]: [description]

        TODO:
            Change output to polygon format instead of {'x': [x_min, x_max], 'y': [y_min, y_max]}
        """
        if self.verbose:
            print("Expansiveness on ", self)
        # TODO: Change if necessary to use keypoint confidence to get
        # minimum keypoint value

        expansiveness = dict()

        # print("=====", self, "=====")
        for camera, keypoints in self.current_pose.items():

            horizontal = {'min': None, 'max': None}
            vertical = {'min': None, 'max': None}

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

            edge_x = float(horizontal['max'] - horizontal['min'])
            edge_y = float(vertical['max'] - vertical['min'])

            scale_factor = 1
            if self.id in SCALE_SUBJECTS[camera]:
                scale_factor = SCALE_FACTOR[camera]

            polygon_area = edge_x * scale_factor * edge_y * scale_factor
            expansiveness[camera]['area'] = np.round(polygon_area, 6)

        return expansiveness

    def metric_keypoint_energy(self, verbose: bool = False):
        previous_pose = self.previous_pose
        current_pose = self.current_pose

        # keypoint_variability contains the variation of each individual keypoint.
        # If this is needed, just return it.
        keypoint_variability = dict()
        keypoints_energy = {'pc1': 0,
                            'pc2': 0,
                            'pc3': 0}

        if previous_pose == current_pose:
            return keypoints_energy

        previous_pose_cameras = set(previous_pose.keys())
        current_pose_cameras = set(current_pose.keys())
        intersection_cameras = previous_pose_cameras & current_pose_cameras

        for camera in intersection_cameras:
            previous_pose_camera = previous_pose[camera]
            current_pose_camera = current_pose[camera]
            previous_pose_camera_keypoints = set(previous_pose_camera.keys())
            current_pose_camera_keypoints = set(current_pose_camera.keys())
            intersection_keypoints = previous_pose_camera_keypoints & current_pose_camera_keypoints

            if camera not in keypoint_variability:
                keypoint_variability[camera] = dict()

            for keypoint_idx in intersection_keypoints:
                previous_keypoint = previous_pose_camera[keypoint_idx][:2]
                current_keypoint = current_pose_camera[keypoint_idx][:2]

                if keypoint_idx not in keypoint_variability[camera]:
                    keypoint_variability[camera][keypoint_idx] = 0

                if not previous_keypoint == current_keypoint:
                    variation = distance_between_points(
                        previous_keypoint, current_keypoint)

                    keypoint_variability[camera][keypoint_idx] += variation
                    keypoints_energy[camera] += variation

        return keypoints_energy

    def metric_center_interaction(self, group_data, subject_expansiveness):
        table_center = group_data[SIDEVIEW_CAMERA]['center']

        hands_body_deviation = subject_expansiveness[SIDEVIEW_CAMERA]['x']
        subject_half_point_y = float(
            np.sum(subject_expansiveness[SIDEVIEW_CAMERA]['y']) / 2)

        table_center_x = table_center[0]

        if self.id <= 2:
            hands_body_deviation_x = max(hands_body_deviation)
        else:
            hands_body_deviation_x = min(hands_body_deviation)

        subject_point = [hands_body_deviation_x, subject_half_point_y]

        point_in_center_line = [table_center_x, subject_half_point_y]

        center_proximity = distance_between_points(subject_point,
                                                   point_in_center_line)

        scale_factor = 1
        if self.id in SCALE_SUBJECTS[SIDEVIEW_CAMERA]:
            scale_factor = SCALE_FACTOR[SIDEVIEW_CAMERA]
        center_proximity = center_proximity * scale_factor

        interaction = {
            'value': np.round(center_proximity, 6),
            'subject_point': subject_point,
            'center': point_in_center_line
        }

        return interaction

    def to_json(self):
        """Transform Subject object to JSON format.
        If attribute is empty it is not printed

        Returns:
            str: JSON formatted Subject object
        """

        obj = {
            "id": self.id,
            "pose": {
                DENSEPOSE_KEY.lower(): self.current_pose,
                "metrics": {
                    "expansiveness": self.expansiveness,
                    "center_interaction": self.center_interaction,
                    "overlap": self.overlap,
                    "keypoint_energy": self.energy,
                    # "body_direction": self.body_direction, # Impossible to measure without 3D data
                },
            },
        }

        return obj


class DenseposeProcess(object):
    """ Densepose pose related metrics

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
        self.clean_group_dir = DENSEPOSE_OUTPUT_DIR / \
            group_id / (group_id + '_clean')
        if not Path(self.clean_group_dir).is_dir():
            log('ERROR', 'This step requires the output of densepose data cleaning step')
        os.makedirs(self.clean_group_dir, exist_ok=True)

        self.output_group_dir = DENSEPOSE_OUTPUT_DIR / \
            group_id / (group_id + '_processed')
        os.makedirs(self.output_group_dir, exist_ok=True)

        self.experiment = Experiment(group_id)
        json.dump(self.experiment.to_json(),
                  open(self.output_group_dir / (self.group_id + '.json'), 'w'))

        self.current_frame = -1
        self.n_subjects = -1
        self.subjects = {subject_id: DenseposeSubject(
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
        frame_subjects_pose = dict()
        for subject in frame_data['subjects']:
            subject_id = subject['id']
            frame_subjects_pose[subject_id] = subject['pose']['densepose']

        data = {'frame': frame_idx,
                'subjects': {'pose': frame_subjects_pose}}
        return data

    def camera_frame_parse_subjects(self, camera, frame_data):
        frame_subjects = frame_data['subjects']
        frame_subject_pose = frame_subjects['pose']
        frame_idx = frame_data['frame']
        self.current_frame = frame_idx
        for subject_id, subject in self.subjects.items():
            if subject_id in frame_subject_pose:
                subject_data = frame_subject_pose[subject_id]
                subject._update_pose(
                    camera, subject_data, self.verbose)
            elif camera not in subject.previous_pose:
                # Subject has no pose nor previous pose.
                # Must skip camera frame as it is impossible to reuse keypoints
                # from other camera due to lack of 3D alignment. Future Consideration.
                if self.verbose:
                    log('INFO', 'Subject has no previous pose.' +
                        '(frame: %s, camera: %s, subject: %s)' %
                        (frame_idx, camera, subject))
                return False
            else:
                if self.verbose:
                    log('INFO', 'Subject (%s) has no pose in this frame (%s - %s), but previous pose on this camera can be used' %
                        (subject_id, frame_idx, camera))

                if subject.previous_pose[camera]:
                    subject._update_pose(
                        camera, subject.previous_pose[camera], self.verbose)
                else:
                    return False
                
        return True

    def process_subject_individual_metrics(self, subject, group_data):
        subject.expansiveness = subject.metric_expansiveness()
        subject.energy = subject.metric_keypoint_energy()
        # subject.body_direction = subject.metric_body_direction() # SEE COMMENTS ON METHOD ABOVE
        subject.center_interaction = subject.metric_center_interaction(group_data,
                                                                       subject.expansiveness)

    def metric_overlap(self, subjects: dict):
        """Calculate overlap between subjects.
        Only considering overlap on side-by-side subjects.
        Subjects in front of each other can have an almost coincident posture/expansiveness.

        Args:
            subjects: (dict): subjects dict
        """

        # Reset overlap values
        for subject_id, subject in self.subjects.items():
            subject.overlap = dict()

        for camera in CAMERAS:
            camera_expensiveness = dict()
            for subject_id, subject in subjects.items():
                if camera in subject.expansiveness:
                    camera_expensiveness[subject_id] = subject.expansiveness[camera]
                    vertices = polygon_vertices_from_2_points(subject.expansiveness[camera]['x'],
                                                              subject.expansiveness[camera]['y'])
                    camera_expensiveness[subject_id] = vertices

            overlap_permutations = list(combinations(
                camera_expensiveness.keys(), 2)).copy()

            for perm in overlap_permutations:
                s1, s2 = perm[0], perm[1]
                subject1, subject2 = self.subjects[s1], self.subjects[s2]
                vertices1, vertices2 = camera_expensiveness[s1], camera_expensiveness[s2]
                polygon1, polygon2 = shapely.Polygon(
                    vertices1), shapely.Polygon(vertices2)
                intersection = polygon1.intersection(polygon2)
                if intersection and intersection.geom_type == 'Polygon':
                    polygon_area = float(abs(intersection.area))

                    for subject in [subject1, subject2]:
                        if camera in subject.overlap:
                            subject.overlap[camera]['area'] += polygon_area
                        else:
                            overlap_dict = {'polygon': intersection.exterior.coords[:],
                                            'area': polygon_area}
                            subject.overlap[camera] = overlap_dict

    def metric_intragroup_distance(self, subjects: dict):
        subjects_distance = dict()
        neck_keypoint = None
        for subject_id, subject in subjects.items():
            subject_pose = subject.current_pose
            for camera, keypoints in subject_pose.items():
                if camera not in subjects_distance:
                    subjects_distance[camera] = dict()
                
                neck_keypoint = tuple(
                    keypoints[str(DENSEPOSE_KEYPOINT_MAP['neck'])])
                subjects_distance[camera][subject_id] = neck_keypoint[:2]

        intragroup_distance = dict()
        for camera, subjects in subjects_distance.items():
            points = list(subjects.values())
            points.append(points[0])
            polygon = Polygon(*points)
            polygon_area = None
            try:
                polygon_area = float(abs(polygon.area))
            except AttributeError:
                print(self.current_frame, camera, polygon, polygon_area)
            centroid = (sum([point[0] for point in points]) / len(points),
                        sum([point[1] for point in points]) / len(points))
            polygon_center = [float(centroid[0]), float(centroid[1])]
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
            "is_enhanced_data_valid": frame_validity,
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
            is_valid_frame = None
            for camera, frame_file in frame_camera_dict.items():
                data = json.load(open(frame_file))
                data = self.frame_data_transform(data)
                is_valid_frame = self.camera_frame_parse_subjects(camera, data)
                if not is_valid_frame:
                    if self.verbose:
                        log('INFO', 'Not enough poses detected. Skipping camera frame')
                    continue

            if is_valid_frame:
                group_data = self.metric_intragroup_distance(self.subjects)
                self.intragroup_distance = group_data
                for _, subject in self.subjects.items():
                    if not self.has_required_cameras(subject):
                        log('WARN', 'Subject (%s) does not have data from required cameras. ' % subject.id +
                            'Not enough information to process frame (%s)' % frame_idx)
                    else:
                        self.process_subject_individual_metrics(subject,
                                                                group_data)

            self.metric_overlap(self.subjects)
            # writting every frame
            self.save_output(output_directory, is_valid_frame)

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
