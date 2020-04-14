import argparse
import json
from math import pi
from os.path import isfile, join, splitext

import numpy as np
import pandas as pd

from environment import (NUM_EYE_LANDMARKS, NUM_FACE_LANDMARKS, NUM_NON_RIGID,
                         OPENFACE_OUTPUT_DIR, VALID_OUTPUT_FILE_TYPES, EMOTIONS_ENCONDING,
                         FRAME_THRESHOLD, HEAD_MOV_VARIANCE_THRESHOLD)
from utils import fetch_files_from_directory, filter_files, strided_split

'''
https://www.cs.cmu.edu/~face/facs.htm
'''

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


def radians_to_degrees(rads):
    """Convert Radians to Degrees

    Arguments:
        rads {float} -- Radians

    Returns:
        float -- Degrees
    """
    return (rads * 180) / pi


def radians_to_degrees_df(rads_list: pd.Series):

    for a, val_rad in rads_list.items():
        rads_list[a] = radians_to_degrees(val_rad)

    return rads_list


def calculate_au_vector_coef(au_intensity_list: list, decrease_factor: float):
    weight = 1
    au_coef = 0
    for val in au_intensity_list:
        au_coef += val * (weight)
        weight -= decrease_factor
    return au_coef


def identify_emotion(aus_vector: pd.Series):
    """Emotion Identification based on relation matrix using \
        Discriminative Power method as Velusamy et al. used \
        in "A METHOD TO INFER EMOTIONS FROM FACIAL ACTION UNITS"

    Args:
        aus_vector (pd.Series): Vectors of Action Units intensities

    See Also:
        "A METHOD TO INFER EMOTIONS FROM FACIAL ACTION UNITS":
        https://ieeexplore.ieee.org/document/5946910/

    Returns:
        str: identified emotion
    """

    aus_vector = aus_vector[columns_aus_intensity]

    emotion_vector_coefs = dict()
    for emotion_name, emotion_aus_vector in EMOTIONS_ENCONDING.items():
        decrease_factor = 1/len(emotion_aus_vector)
        emotion_aus = list()
        for au in emotion_aus_vector:
            emotion_aus.append(au+"_r")

        emotion_vector_coefs[emotion_name] = calculate_au_vector_coef(
            aus_vector.filter(emotion_aus), decrease_factor)

    emotion_vector_coefs['NEUTRAL'] = 1
    emotion_pred = max(emotion_vector_coefs, key=emotion_vector_coefs.get)

    return emotion_pred

# TODO: predict_emotion(aus_vector: pd.Series) using a ML classifier


def is_sequence_increasing(seq: list):
    return all(earlier <= later for earlier, later in zip(seq, seq[1:]))


def is_sequence_decreasing(seq: list):
    return all(earlier >= later for earlier, later in zip(seq, seq[1:]))


def identify_head_movement_orientation(col: str, vector: str):
    col = col.lower()
    orientation = {
        'x': ['UP', 'DOWN'],
        'y': ['LEFT', 'RIGHT'],
        'z': ['CW', 'CCW']
    }

    for axis, label in orientation.items():
        if axis in col:
            break

    vector_variance = np.var(vector)
    valid_variance = vector_variance >= HEAD_MOV_VARIANCE_THRESHOLD

    if is_sequence_increasing(vector) and valid_variance:
        return label[0]
    elif is_sequence_decreasing(vector) and valid_variance:
        return label[1]
    else:
        return 'CENTER'


def identify_head_movement(head_pose_df: pd.DataFrame, columns_head_rotation_out: list):
    head_pose_out = pd.DataFrame(columns=columns_head_rotation_out)

    for df_split in strided_split(head_pose_df, FRAME_THRESHOLD):
        movement_split = list()
        for col in df_split.columns:
            movement = identify_head_movement_orientation(
                col, df_split[col])
            movement_split.append(movement)

        movement_split_df = pd.DataFrame([movement_split for _ in range(
            len(df_split))], columns=columns_head_rotation_out)
        head_pose_out = head_pose_out.append(
            movement_split_df, ignore_index=True)

    return head_pose_out


def identify_gaze_movement_orientation(col: str, vector: str):
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
    col = col.lower()
    orientation = {
        'x': ['RIGHT', 'CENTER', 'LEFT'],
        'y': ['UP', 'CENTER', 'DOWN'],
    }

    for axis, label in orientation.items():
        if axis in col:
            vector_mean = np.mean(vector)
            vector_std = np.std(vector)

            if (vector_mean-vector_std <= 0 <= vector_mean+vector_std):
                return label[1]
            elif vector_mean > 0:
                return label[2]
            elif vector_mean < 0:
                return label[0]

    return "MEASUREMENT ERROR"


def identify_eye_gaze_movement(eye_gaze_df: pd.DataFrame, columns_eye_movement_out: list):
    """If a person is looking left-right this will results in the change of gaze_angle_x (from positive to negative) \
    If a person is looking up-down this will result in change of gaze_angle_y (from negative to positive) \
    If a person is looking straight ahead both of the angles will be close to 0 (within measurement error)

    See Also:
        Check 'Gaze related' section in OpenFace Documentation
        https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format

    Args:
        eye_gaze_df (pd.DataFrame): Gaze angles dataframe
        columns_eye_movement_out (list): Output columns' names

    Returns:
        pd.DataFrame: Returns output columns dataframe
    """
    eye_movement_out = pd.DataFrame(columns=columns_eye_movement_out)

    for df_split in strided_split(eye_gaze_df, FRAME_THRESHOLD):
        movement_split = list()
        for col in df_split.columns:
            movement = identify_gaze_movement_orientation(
                col, df_split[col])
            movement_split.append(movement)

        movement_split_df = pd.DataFrame([movement_split for _ in range(
            len(df_split))], columns=columns_eye_movement_out)
        eye_movement_out = eye_movement_out.append(
            movement_split_df, ignore_index=True)

    return eye_movement_out


def df_format_to_json(line):
    # print(list(line))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract facial data using OpenFace')
    parser.add_argument('csv_file', type=str, nargs='*', help='CSV file path')
    parser.add_argument('-dir', '--directory',
                        action="store_true", help='CSV files path')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    csv_files = args['csv_file']
    directory = args['directory']
    verbose = args['verbose']

    print(csv_files)
    if directory:
        csv_files = fetch_files_from_directory(csv_files)
    exit()

    csv_files = filter_files(csv_files, VALID_OUTPUT_FILE_TYPES)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        df.set_index('frame', inplace=True)

        # Assign people IDs

        # exit()

        # Identify emotions
        DF_OUTPUT = pd.DataFrame(df[columns_relevant], columns=columns_output)
        DF_OUTPUT['emotion'] = DF_OUTPUT.apply(identify_emotion, axis=1)

        # Transform head rotation from rads to degrees
        DF_OUTPUT[columns_head_rot] = DF_OUTPUT[columns_head_rot].apply(
            radians_to_degrees_df, axis=1)

        # Identify head movement
        head_movement_features = columns_features[1:4]
        DF_OUTPUT[head_movement_features] = identify_head_movement(
            DF_OUTPUT[columns_head_rot], head_movement_features)

        # Transform gaze angles from rads to degrees
        gaze_angles = columns_gaze[6:8]
        DF_OUTPUT[gaze_angles] = DF_OUTPUT[gaze_angles].apply(
            radians_to_degrees_df, axis=1)

        # Identify eye gaze movement
        eye_movement_features = columns_features[4:6]
        DF_OUTPUT[eye_movement_features] = identify_eye_gaze_movement(
            DF_OUTPUT[gaze_angles], eye_movement_features)

        DF_OUTPUT.dropna(axis=0, inplace=True)
        # DF_OUTPUT.apply(df_format_to_json)

        print(DF_OUTPUT.head())
