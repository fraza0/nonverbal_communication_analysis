import argparse
import json
import os
import re
from math import degrees, pi
from os.path import isfile, join, splitext
from pathlib import Path

import numpy as np
import pandas as pd

from nonverbal_communication_analysis.environment import (
    CAMERAS_3D_AXES, EMOTIONS_ENCONDING, FRAME_THRESHOLD,
    HEAD_MOV_VARIANCE_THRESHOLD, NUM_EYE_LANDMARKS, NUM_FACE_LANDMARKS,
    NUM_NON_RIGID, OPENFACE_KEY, OPENFACE_OUTPUT_DIR, SUBJECT_AXES,
    VALID_OUTPUT_FILE_TYPES)
from nonverbal_communication_analysis.m0_Classes.Experiment import (
    Experiment, get_group_from_file_path)
from nonverbal_communication_analysis.m0_Classes.Subject import Subject
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files, log,
                                                    strided_split)


'''
https://www.cs.cmu.edu/~face/facs.htm
'''


class OpenfaceSubject(Subject):
    # See Also: Between Shoulders Distance Analysis section of Openpose Analysis notebook
    _distance_threshold = 0.5

    def __init__(self, _id):
        self.id = _id

        self.data_buffer = {
            'head': list(),
            'eye': list()
        }

        self.face = None
        self.emotion = None
        self.eye_gaze = None
        self.head_rotation = None

    def to_json(self):

        obj = {
            "id": self.id,
            "face": {
                "openface": {
                    "raw": self.face,
                    "enhanced": None
                },
                "metrics": {
                    "emotion": self.emotion,
                    "head_movement": self.head_rotation,
                    "gaze_movement": self.eye_gaze
                }
            }
        }

        return obj

    def __str__(self):
        return "OpenposeSubject: {id: %s}" % self.id


class OpenfaceProcess(object):

    def __init__(self, group_id: str, prettify: bool = False, verbose: bool = False):
        self.group_id = group_id
        self.clean_group_dir = OPENFACE_OUTPUT_DIR / \
            group_id / (group_id + '_clean')
        if not Path(self.clean_group_dir).is_dir():
            log('ERROR', 'This step requires the output of openface data cleaning step')
        os.makedirs(self.clean_group_dir, exist_ok=True)

        self.output_group_dir = OPENFACE_OUTPUT_DIR / \
            group_id / (group_id + '_processed')
        os.makedirs(self.output_group_dir, exist_ok=True)

        self.experiment = Experiment(group_id)
        json.dump(self.experiment.to_json(),
                  open(self.output_group_dir / (self.group_id + '.json'), 'w'))

        self.subjects = dict()
        self.is_valid_frame = None

        self.prettify = prettify
        self.verbose = verbose

    columns_basic = [
        # 'frame',       # Frame number
        'face_id',     # Face id - No guarantee this is consistent across frames in case of FaceLandmarkVidMulti
        'timestamp',   # Timer of video being processed in seconds
        'success',     # Is the track successful - Is there a face in the frame or do we think we tracked it well
        'confidence'   # How confident is the tracker in current landmark detection estimation
    ]

    columns_gaze = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z',  # Left eye
                    'gaze_1_x', 'gaze_1_y', 'gaze_1_z',  # Right eye
                    'gaze_angle_x', 'gaze_angle_y']      # Gaze angle in radians

    columns_2d_eye_lmks = [('eye_lmk_x_%s' % lmk_idx) for lmk_idx in range(NUM_EYE_LANDMARKS)] + \
        [('eye_lmk_y_%s' % lmk_idx) for lmk_idx in range(
            NUM_EYE_LANDMARKS)]  # 2D Eye Landmarks (x,y) coordinates

    columns_3d_eye_lmks = [('eye_lmk_X_%s' % lmk_idx) for lmk_idx in range(
        NUM_EYE_LANDMARKS)] + \
        [('eye_lmk_Y_%s' % lmk_idx) for lmk_idx in range(
            NUM_EYE_LANDMARKS)] + \
        [('eye_lmk_Z_%s' % lmk_idx) for lmk_idx in range(
            NUM_EYE_LANDMARKS)]  # 3D Eye Landmarks (X,Y,Z) coordinates

    # Relative Location to Camera
    columns_head_loc = ['pose_Tx', 'pose_Ty', 'pose_Tz']
    columns_head_rot = ['pose_Rx', 'pose_Ry',
                        'pose_Rz']  # Rotation pitch, yaw, roll

    columns_facial_lmks = [('X_%s' % lmk_idx) for lmk_idx in range(NUM_FACE_LANDMARKS)] + \
        [('Y_%s' % lmk_idx) for lmk_idx in range(NUM_FACE_LANDMARKS)] + \
        [('Z_%s' % lmk_idx) for lmk_idx in range(
            NUM_FACE_LANDMARKS)]  # 3D Face Landmarks (X,Y,Z) coordinate

    # Rigid face shape (location, scale, rotation)
    columns_rigid_shape = ['p_scale',                   # Face scale
                           'p_rx', 'p_ry', 'p_rz',      #
                           'p_tx', 'p_ty']              #

    columns_aus_intensity = ['AU01_r',  # Inner Brow Raiser
                             'AU02_r',  # Outer Brow Raiser
                             'AU04_r',  # Brow Lowerer
                             'AU05_r',  # Upper Lid Raiser
                             'AU06_r',  # Cheek Raiser
                             'AU07_r',  # Lid Tightener
                             'AU09_r',  # Nose Wrinkler
                             'AU10_r',  # Upper Lip Raiser
                             'AU12_r',  # Lip Corner Puller
                             'AU14_r',  # Dimpler
                             'AU15_r',  # Lip Corner Depressor
                             'AU17_r',  # Chin Raiser
                             'AU20_r',  # Lip stretcher
                             'AU23_r',  # Lip Tightener
                             'AU25_r',  # Lips part
                             'AU26_r',  # Jaw Drop
                             'AU45_r']  # Blink

    columns_features = ['emotion',
                        'head_movement_pitch', 'head_movement_yaw', 'head_movement_roll',
                        'eye_movement_x', 'eye_movement_y']

    columns_relevant = columns_basic + \
        columns_facial_lmks + \
        columns_aus_intensity + \
        columns_head_rot + \
        columns_gaze

    columns_output = columns_relevant + columns_features

    def calculate_au_vector_coef(self, au_intensity_list: list, decrease_factor: float):
        weight = 1
        au_coef = 0
        for val in au_intensity_list:
            au_coef += val * (weight)
            weight -= decrease_factor
        return au_coef

    def identify_emotion(self, action_units_data: dict):
        # TODO: update docs
        """Emotion Identification based on relation matrix using \
            Discriminative Power method as Velusamy et al. used \
            in "A METHOD TO INFER EMOTIONS FROM FACIAL ACTION UNITS"

        Args:
            aus_vector (dict): Vectors of Action Units intensities

        See Also:
            "A METHOD TO INFER EMOTIONS FROM FACIAL ACTION UNITS":
            https://ieeexplore.ieee.org/document/5946910/

        To Do:
            predict_emotion(aus_vector: pd.Series) using a ML classifier

        Returns:
            str: identified emotion
        """

        aus_vector = dict()
        for au_col in self.columns_aus_intensity:
            if au_col in action_units_data:
                aus_vector[au_col] = action_units_data[au_col]

        emotion_vector_coefs = dict()
        for emotion_name, emotion_aus_vector in EMOTIONS_ENCONDING.items():
            decrease_factor = 1/len(emotion_aus_vector)
            emotion_aus = list()
            for au in emotion_aus_vector:
                emotion_aus.append(aus_vector[au+"_r"])

            emotion_vector_coefs[emotion_name] = self.calculate_au_vector_coef(
                emotion_aus, decrease_factor)

        emotion_vector_coefs['NEUTRAL'] = 1
        emotion_pred = max(emotion_vector_coefs, key=emotion_vector_coefs.get)

        return emotion_pred

    def is_sequence_increasing(self, seq: list):
        return all(earlier <= later for earlier, later in zip(seq, seq[1:]))

    def is_sequence_decreasing(self, seq: list):
        return all(earlier >= later for earlier, later in zip(seq, seq[1:]))

    def identify_head_movement(self, data_buffer: str):

        orientation_labels = {
            'x': ['UP', 'DOWN'],
            'y': ['LEFT', 'RIGHT'],
            'z': ['CW', 'CCW']
        }

        x_vector = list()
        y_vector = list()
        z_vector = list()
        for entry in data_buffer:
            x_vector.append(degrees(entry[0]))
            y_vector.append(degrees(entry[1]))
            z_vector.append(degrees(entry[2]))

        vectors = {'x': x_vector, 'y': y_vector, 'z': z_vector}
        orientation = {'x': None, 'y': None, 'z': None}

        for axis, vector in vectors.items():
            if np.var(vector) >= HEAD_MOV_VARIANCE_THRESHOLD:
                if self.is_sequence_increasing(vector):
                    orientation[axis] = orientation_labels[axis][0]
                elif self.is_sequence_decreasing(vector):
                    orientation[axis] = orientation_labels[axis][1]
                else:
                    orientation[axis] = 'CENTER'
            else:
                orientation[axis] = 'CENTER'

        return orientation

    def identify_eye_gaze_movement(self, data_buffer: str):
        """If a person is looking left-right this will results in the change of gaze_angle_x (from positive to negative) \
        If a person is looking up-down this will result in change of gaze_angle_y (from negative to positive) \
        If a person is looking straight ahead both of the angles will be close to 0 (within measurement error)

        See Also:
            Check 'Gaze related' section in OpenFace Documentation
            https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format

        Args:
            col (str): [description]
            vector (str): [description]

        Returns:
            str: [description]
        """
        orientation_labels = {
            'x': ['RIGHT', 'CENTER', 'LEFT'],
            'y': ['UP', 'CENTER', 'DOWN'],
        }

        x_vector = list()
        y_vector = list()
        for entry in data_buffer:
            x_vector.append(degrees(entry[0]))
            y_vector.append(degrees(entry[1]))

        vectors = {'x': x_vector, 'y': y_vector}
        orientation = {'x': None, 'y': None}

        for axis, vector in vectors.items():
            vector_mean = np.mean(vector)
            vector_std = np.std(vector)

            if (vector_mean-vector_std <= 0 <= vector_mean+vector_std):
                orientation[axis] = orientation_labels[axis][1]
            elif vector_mean > 0:
                orientation[axis] = orientation_labels[axis][2]
            elif vector_mean < 0:
                orientation[axis] = orientation_labels[axis][0]
            else:
                orientation[axis] = 'MEASUREMENT ERROR'

        return orientation

    def save_output(self, output_path, frame_subjects):

        obj = {
            "frame": self.current_frame,
            "is_processed_data_valid": self.is_valid_frame,
            "subjects": [subject.to_json() for subject in frame_subjects.values()]
        }

        if self.prettify:
            json.dump(obj, open(output_path, 'w'), indent=2)
        else:
            json.dump(obj, open(output_path, 'w'))

        return obj

    def handle_frames(self, camera_frame_files: dict, output_directory: str, display: bool = False):

        for frame_idx in sorted(camera_frame_files):
            # print('=== FRAME %s ===' % frame_idx)
            self.current_frame = frame_idx
            frame_camera_dict = camera_frame_files[frame_idx]
            for camera, frame_file in frame_camera_dict.items():
                output_path_dir = output_directory / camera
                output_path = output_path_dir / \
                    ("%s_%.12d_processed.json" % (camera, frame_idx))
                os.makedirs(output_path_dir, exist_ok=True)
                data = json.load(open(frame_file))
                # TODO: Replace by condition if needed
                self.is_valid_frame = data['is_raw_data_valid']

                frame_subjects = dict()

                for subject in data['subjects']:
                    subject_id = subject['id']
                    # print("== SUBJECT %s ==" % subject_id)
                    if subject_id in self.subjects:
                        openface_subject = self.subjects[subject_id]
                    else:
                        openface_subject = OpenfaceSubject(subject['id'])
                        self.subjects[subject_id] = openface_subject

                    # Parse data
                    subject_openface_data = subject['face']['openface']

                    # EMOTION IDENTIFICATION
                    openface_subject.face = subject_openface_data
                    openface_subject.emotion = self.identify_emotion(
                        subject_openface_data['AUs'])

                    # HEAD MOVEMENT DIRECTION
                    head_rotation_data = subject_openface_data['head']
                    if len(openface_subject.data_buffer['head']) >= FRAME_THRESHOLD:
                        openface_subject.data_buffer['head'].pop(0)

                    openface_subject.data_buffer['head'].append(
                        head_rotation_data)
                    if len(openface_subject.data_buffer['head']) == FRAME_THRESHOLD:
                        openface_subject.head_rotation = self.identify_head_movement(
                            openface_subject.data_buffer['head'])

                    # EYE MOVEMENT DIRECTION
                    eye_gaze_data = subject_openface_data['gaze']
                    if len(openface_subject.data_buffer['eye']) >= FRAME_THRESHOLD:
                        openface_subject.data_buffer['eye'].pop(0)

                    openface_subject.data_buffer['eye'].append(eye_gaze_data)
                    if len(openface_subject.data_buffer['eye']) == FRAME_THRESHOLD:
                        openface_subject.eye_gaze = self.identify_eye_gaze_movement(
                            openface_subject.data_buffer['eye'])

                    # Update subject - Not needed?
                    frame_subjects[subject_id] = openface_subject
            write = self.save_output(output_path, frame_subjects)
            if not write:
                log('ERROR', 'Could not save frame %s to %s' %
                    (frame_idx, output_path))

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


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description='Extract facial data using OpenFace')
#     parser.add_argument('csv_file', type=str, nargs='*', help='CSV file path')
#     parser.add_argument('-dir', '--directory',
#                         action="store_true", help='CSV files path')
#     parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
#                         action='store_true')

#     args = vars(parser.parse_args())

#     csv_files = args['csv_file']
#     directory = args['directory']
#     verbose = args['verbose']

#     print(csv_files)
#     if directory:
#         csv_files = fetch_files_from_directory(csv_files)
#     exit()

#     csv_files = filter_files(csv_files, VALID_OUTPUT_FILE_TYPES)

#     for csv_file in csv_files:
#         df = pd.read_csv(csv_file)
#         df.rename(columns=lambda x: x.strip(), inplace=True)
#         df.set_index('frame', inplace=True)

#         # Identify emotions
#         DF_OUTPUT = pd.DataFrame(df[columns_relevant], columns=columns_output)
#         DF_OUTPUT['emotion'] = DF_OUTPUT.apply(identify_emotion, axis=1)

#         # Transform head rotation from rads to degrees
#         DF_OUTPUT[columns_head_rot] = DF_OUTPUT[columns_head_rot].apply(
#             radians_to_degrees_df, axis=1)

#         # Identify head movement
#         head_movement_features = columns_features[1:4]
#         DF_OUTPUT[head_movement_features] = identify_head_movement(
#             DF_OUTPUT[columns_head_rot], head_movement_features)

#         # Transform gaze angles from rads to degrees
#         gaze_angles = columns_gaze[6:8]
#         DF_OUTPUT[gaze_angles] = DF_OUTPUT[gaze_angles].apply(
#             radians_to_degrees_df, axis=1)

#         # Identify eye gaze movement
#         eye_movement_features = columns_features[4:6]
#         DF_OUTPUT[eye_movement_features] = identify_eye_gaze_movement(
#             DF_OUTPUT[gaze_angles], eye_movement_features)

#         DF_OUTPUT.dropna(axis=0, inplace=True)
#         # DF_OUTPUT.apply(df_format_to_json)

#         print(DF_OUTPUT.head())
