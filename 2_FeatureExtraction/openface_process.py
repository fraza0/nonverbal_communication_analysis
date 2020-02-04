import argparse
import json
from math import pi
from os.path import isfile, join, splitext

import numpy as np
import pandas as pd

from environment import (NUM_EYE_LANDMARKS, NUM_FACE_LANDMARKS, NUM_NON_RIGID,
                         OPENFACE_OUTPUT_DIR, VALID_FILE_TYPES, EMOTIONS_ENCONDING)
from utils import fetch_files_from_directory, filter_files

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

columns_head = ['pose_Tx', 'pose_Ty', 'pose_Tz',  # Relative Location to Camera
                'pose_Rx', 'pose_Ry', 'pose_Rz']  # Rotation pitch, yaw, roll

columns_facial_lmks = [('X_%s' % lmk_idx) for lmk_idx in range(NUM_FACE_LANDMARKS)] + \
    [('Y_%s' % lmk_idx) for lmk_idx in range(NUM_FACE_LANDMARKS)] + \
    [('Z_%s' % lmk_idx) for lmk_idx in range(
        NUM_FACE_LANDMARKS)]  # 3D Face Landmarks (X,Y,Z) coordinate

# Rigid face shape (location, scale, rotation)
columns_rigid_shape = ['p_scale',                   # Face scale
                       'p_rx', 'p_ry', 'p_rz',      #
                       'p_tx', 'p_ty']              #

columns_non_rigid_shape = [('p_%s' % lmk_idx)
                           for lmk_idx in range(NUM_NON_RIGID)]

columns_aus_presence = ['AU01_c',  # Inner Brow Raiser
                        'AU02_c',  # Outer Brow Raiser
                        'AU04_c',  # Brow Lowerer
                        'AU05_c',  # Upper Lid Raiser
                        'AU06_c',  # Cheek Raiser
                        'AU07_c',  # Lid Tightener
                        'AU09_c',  # Nose Wrinkler
                        'AU10_c',  # Upper Lip Raiser
                        'AU12_c',  # Lip Corner Puller
                        'AU14_c',  # Dimpler
                        'AU15_c',  # Lip Corner Depressor
                        'AU17_c',  # Chin Raiser
                        'AU20_c',  # Lip stretcher
                        'AU23_c',  # Lip Tightener
                        'AU25_c',  # Lips part
                        'AU26_c',  # Jaw Drop
                        'AU28_c',  # Lip Suck
                        'AU45_c']  # Blink

columns_aus_intensity = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
                         'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
                         'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

columns_features = ['emotion',
                    'head_movement_pitch', 'head_movement_yaw', 'head_movement_roll',
                    'eye_movement_x', 'eye_movement_y']

columns_relevant = columns_basic + \
    columns_facial_lmks + \
    columns_aus_intensity

columns_output = columns_relevant + columns_features


def radians_to_degrees(rads):
    """Conver Radians to Degrees

    Arguments:
        rads {float} -- Radians

    Returns:
        float -- Degrees
    """
    return (rads * 180) / pi


def calculate_au_vector_coef(au_intensity_list: list, decrease_factor: float):
    weight = 1
    au_coef = 0
    for val in au_intensity_list:
        au_coef += val * (weight)
        weight -= decrease_factor
    return au_coef


def predict_emotion(aus_vector: pd.Series):
    aus_vector = aus_vector[columns_aus_intensity]
    # print(calculate_au_vector_coef(aus_vector, decrease_factor))

    emotion_vector_coefs = dict()
    for emotion_name, emotion_aus_vector in EMOTIONS_ENCONDING.items():
        decrease_factor = 1/len(emotion_aus_vector)
        emotion_aus = list()
        for au in emotion_aus_vector:
            emotion_aus.append(au+"_r")

        emotion_vector_coefs[emotion_name] = calculate_au_vector_coef(
            aus_vector.filter(emotion_aus), decrease_factor)

    emotion_pred = max(emotion_vector_coefs, key=emotion_vector_coefs.get)

    if verbose:
        print("PREDICTED EMOTION: %s" % emotion_pred)

    return emotion_pred


# calculate pitch movement
# calculate yaw movement
# calculate roll movement
# calculate eye movement (x,y)


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

    if directory:
        csv_files = fetch_files_from_directory(csv_files)

    csv_files = filter_files(csv_files, VALID_FILE_TYPES)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        df.set_index('frame', inplace=True)

        df_basic = pd.DataFrame(df[columns_basic], columns=columns_basic)
        df_gaze = pd.DataFrame(df[columns_gaze], columns=columns_gaze)
        df_2d_eye_lmks = pd.DataFrame(
            df[columns_2d_eye_lmks], columns=columns_2d_eye_lmks)
        df_3d_eye_lmks = pd.DataFrame(
            df[columns_3d_eye_lmks], columns=columns_3d_eye_lmks)
        df_head = pd.DataFrame(df[columns_head], columns=columns_head)
        df_face = pd.DataFrame(df[columns_facial_lmks],
                               columns=columns_facial_lmks)
        df_rigid = pd.DataFrame(df[columns_rigid_shape],
                                columns=columns_rigid_shape)
        df_non_rigid = pd.DataFrame(
            df[columns_non_rigid_shape], columns=columns_non_rigid_shape)
        df_aus_presence = pd.DataFrame(
            df[columns_aus_presence], columns=columns_aus_presence)
        df_aus_intensity = pd.DataFrame(
            df[columns_aus_intensity], columns=columns_aus_intensity)

        DF_OUTPUT = pd.DataFrame(
            df[columns_relevant], columns=columns_output)
        DF_OUTPUT['emotion'] = DF_OUTPUT.apply(predict_emotion, axis=1)

        print(DF_OUTPUT)
