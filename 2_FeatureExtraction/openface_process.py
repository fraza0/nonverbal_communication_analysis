import numpy as np
import pandas as pd
import json

from environment import (OPENFACE_OUTPUT_DIR,
                         NUM_EYE_LANDMARKS,
                         NUM_FACE_LANDMARKS,
                         NUM_NON_RIGID)

'''
https://www.cs.cmu.edu/~face/facs.htm
'''

df_basic = [
    'frame',       # Frame number
    'face_id',     # Face id - No guarantee this is consistent across frames in case of FaceLandmarkVidMulti
    'timestamp',   # Timer of video being processed in seconds
    'success',     # Is the track successful - Is there a face in the frame or do we think we tracked it well
    'confidence'   # How confident is the tracker in current landmark detection estimation
]


df_gaze = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z',  # Left eye
           'gaze_1_x', 'gaze_1_y', 'gaze_1_z',  # Right eye
           'gaze_angle_x', 'gaze_angle_y']      # Gaze angle in radians

df_2d_eye_lmks = [[('eye_lmk_x_%s' % lmk_idx) for lmk_idx in range(NUM_EYE_LANDMARKS)],  # 2D Eye Landmarks x coordinate
                  [('eye_lmk_y_%s' % lmk_idx) for lmk_idx in range(NUM_EYE_LANDMARKS)]]  # 2D Eye Landmarks y coordinate

df_3d_eye_lmks = [[('eye_lmk_X_%s' % lmk_idx) for lmk_idx in range(NUM_EYE_LANDMARKS)],  # 3D Eye Landmarks X coordinate
                  [('eye_lmk_Y_%s' % lmk_idx) for lmk_idx in range(
                      NUM_EYE_LANDMARKS)],  # 3D Eye Landmarks Y coordinate
                  [('eye_lmk_Z_%s' % lmk_idx) for lmk_idx in range(NUM_EYE_LANDMARKS)]]  # 3D Eye Landmarks Z coordinate

df_head = ['pose_Tx', 'pose_Ty', 'pose_Tz',  # Relative Location to Camera
           'pose_Rx', 'pose_Ry', 'pose_Rz']  # Rotation pitch, yaw, roll


df_facial_lmks = [[('X_%s' % lmk_idx) for lmk_idx in range(NUM_FACE_LANDMARKS)],  # 3D Face Landmarks X coordinate
                  # 3D Face Landmarks Y coordinate
                  [('Y_%s' % lmk_idx)
                   for lmk_idx in range(NUM_FACE_LANDMARKS)],
                  [('Z_%s' % lmk_idx) for lmk_idx in range(NUM_FACE_LANDMARKS)]]  # 3D Face Landmarks Z coordinate

# Rigid face shape (location, scale, rotation)
columns_rigid_shape = ['p_scale',                   # Face scale
                       'p_rx', 'p_ry', 'p_rz',      #
                       'p_tx', 'p_ty']              #

columns_non_rigid_shape = [[('p_%s' % lmk_idx)
                            for lmk_idx in range(NUM_NON_RIGID)]]

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


def parse_input(csv):
    pass