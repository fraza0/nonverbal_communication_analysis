import argparse
import csv
import json
import re
import os
from pathlib import Path

import pandas as pd
import yaml

from nonverbal_communication_analysis.environment import (
    VALID_OUTPUT_FILE_TYPES, OPENFACE_OUTPUT_DIR, NUM_EYE_LANDMARKS, NUM_FACE_LANDMARKS, OPENFACE_KEY, VIDEO_RESOLUTION)
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files, log)

from nonverbal_communication_analysis.m0_Classes.Experiment import Experiment
from nonverbal_communication_analysis.m0_Classes.ExperimentCameraFrame import ExperimentCameraFrame


class OpenfaceClean(object):

    columns_basic = ['frame',       # Frame number
                     'face_id',     # Face id - No guarantee this is consistent across frames in case of FaceLandmarkVidMulti
                     'timestamp',   # Timer of video being processed in seconds
                     'success',     # Is the track successful - Is there a face in the frame or do we think we tracked it well
                     'confidence']  # How confident is the tracker in current landmark detection estimation

    columns_gaze = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z',  # Left eye
                    'gaze_1_x', 'gaze_1_y', 'gaze_1_z',  # Right eye
                    'gaze_angle_x', 'gaze_angle_y']      # Gaze angle in radians

    columns_2d_eye_lmks_x = [('eye_lmk_x_%s' % lmk_idx)
                             for lmk_idx in range(NUM_EYE_LANDMARKS)]

    columns_2d_eye_lmks_y = [('eye_lmk_y_%s' % lmk_idx)
                             for lmk_idx in range(NUM_EYE_LANDMARKS)]

    columns_2d_eye_lmks = columns_2d_eye_lmks_x + \
        columns_2d_eye_lmks_y  # 2D Eye Landmarks (x,y) coordinates

    # columns_3d_eye_lmks = [('eye_lmk_X_%s' % lmk_idx) for lmk_idx in range(
    #     NUM_EYE_LANDMARKS)] + \
    #     [('eye_lmk_Y_%s' % lmk_idx) for lmk_idx in range(
    #         NUM_EYE_LANDMARKS)] + \
    #     [('eye_lmk_Z_%s' % lmk_idx) for lmk_idx in range(
    #         NUM_EYE_LANDMARKS)]  # 3D Eye Landmarks (X,Y,Z) coordinates

    # Relative Location to Camera
    columns_head_loc = ['pose_Tx', 'pose_Ty', 'pose_Tz']
    columns_head_rot = ['pose_Rx', 'pose_Ry',
                        'pose_Rz']  # Rotation pitch, yaw, roll

    columns_2d_facial_lmks_x = [('x_%s' % lmk_idx)
                                for lmk_idx in range(NUM_FACE_LANDMARKS)]

    columns_2d_facial_lmks_y = [('y_%s' % lmk_idx)
                                for lmk_idx in range(NUM_FACE_LANDMARKS)]

    columns_2d_facial_lmks = columns_2d_facial_lmks_x + \
        columns_2d_facial_lmks_y  # 2D Face Landmarks (x,y) coordinates

    # columns_3d_facial_lmks = [('X_%s' % lmk_idx) for lmk_idx in range(NUM_FACE_LANDMARKS)] + \
    #     [('Y_%s' % lmk_idx) for lmk_idx in range(NUM_FACE_LANDMARKS)] + \
    #     [('Z_%s' % lmk_idx) for lmk_idx in range(NUM_FACE_LANDMARKS)
    #      ]  # 3D Face Landmarks (X,Y,Z) coordinate

    # Rigid face shape (location, scale, rotation)
    columns_rigid_shape = ['p_scale',                   # Face scale
                           'p_rx', 'p_ry', 'p_rz',      #
                           'p_tx', 'p_ty']              #

    columns_aus_classification = ['AU01_c',  # Inner Brow Raiser
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
                                  'AU28_c',  # Lip suck
                                  'AU45_c']  # Blink

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

    def __init__(self, group_id):
        self.experiment = Experiment(group_id)
        self.group_id = group_id
        self.base_output_dir = OPENFACE_OUTPUT_DIR / \
            group_id / (group_id + '_clean')
        os.makedirs(self.base_output_dir, exist_ok=True)

    def scaling_min_max(self, df_entry: pd.DataFrame, axis: str, camera: str, a: int = -1, b: int = 1):
        resolution_min, resolution_max = 0, VIDEO_RESOLUTION[camera][axis]
        normalized_entry = a + (df_entry - resolution_min) * (b-a) \
            / (resolution_max-resolution_min)

        return normalized_entry

    def process_frames(self, task_frame_df: pd.DataFrame, output_directory: str, prettify: bool = False, verbose: bool = False, display: bool = False):
        frames = list(task_frame_df['frame'].unique())
        for frame in frames:
            df = task_frame_df.loc[task_frame_df.frame == frame]
            df = df.loc[df.success == 1]

            frame_cameras_count = df.groupby('camera').size()
            for frame_camera in frame_cameras_count.iteritems():
                _cam = frame_camera[0]
                _cam_count = frame_camera[1]
                if verbose:
                    if _cam_count < self.experiment._n_subjects:
                        log("WARN", "Camera %s only found %s subjects faces" %
                            (_cam, _cam_count))

                output_frame_directory = output_directory / _cam
                output_frame_file = output_frame_directory / \
                    ("%s_%.12d_clean.json" % (_cam, frame))
                os.makedirs(output_frame_directory, exist_ok=True)

                experiment_frame = ExperimentCameraFrame(
                    _cam, int(frame), df[['confidence'] + self.columns_2d_facial_lmks + self.columns_2d_eye_lmks], OPENFACE_KEY, verbose=verbose, display=display)

                if prettify:
                    json.dump(experiment_frame.to_json(), open(
                        output_frame_file, 'w'), indent=2)
                else:
                    json.dump(experiment_frame.to_json(), open(
                        output_frame_file, 'w'))

    def clean(self, tasks_directories: dict, specific_frame: int = None, prettify: bool = False, verbose: bool = False, display: bool = False):
        for task in tasks_directories:
            camera_files = dict()
            task_directory = OPENFACE_OUTPUT_DIR / self.group_id / task.name
            openface_camera_directories = [x for x in task_directory.iterdir()
                                           if x.is_dir()]

            camera_files = pd.DataFrame()
            for openface_camera_dir in openface_camera_directories:
                openface_output_files = [x for x in openface_camera_dir.iterdir()
                                         if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES]
                for openface_output_file in openface_output_files:
                    camera_id = re.search(r'(?<=Video)(pc\d{1})(?=\d{14})',
                                          openface_output_file.name).group(0)
                    openface_file_df = pd.read_csv(openface_output_file)

                    tmp_df = openface_file_df
                    if specific_frame is not None:
                        tmp_df = openface_file_df.loc[openface_file_df.frame ==
                                                      specific_frame]
                    tmp_df = tmp_df.assign(camera=camera_id)

                    # Scaling
                    to_normalize_cols_x = self.columns_2d_facial_lmks_x + self.columns_2d_eye_lmks_x
                    to_normalize_cols_y = self.columns_2d_facial_lmks_y + self.columns_2d_eye_lmks_y
                    to_normalize_df_x = tmp_df[to_normalize_cols_x]
                    to_normalize_df_y = tmp_df[to_normalize_cols_y]

                    to_normalize_df_x = self.scaling_min_max(
                        to_normalize_df_x, axis='x', camera=camera_id)
                    to_normalize_df_y = self.scaling_min_max(
                        to_normalize_df_y, axis='y', camera=camera_id)

                    tmp_df.drop(self.columns_2d_facial_lmks,
                                axis=1, inplace=True)
                    tmp_df.drop(self.columns_2d_eye_lmks, axis=1, inplace=True)
                    tmp_df = tmp_df.join(to_normalize_df_x)
                    tmp_df = tmp_df.join(to_normalize_df_y)

                    camera_files = camera_files.append(
                        tmp_df, ignore_index=True)

            output_directory = self.base_output_dir / task.name
            self.process_frames(camera_files, output_directory, prettify=prettify,
                                verbose=verbose, display=display)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description='Extract facial data using OpenFace')
#     parser.add_argument('openpose_group_data_dir', type=str,
#                         help='Openpose output group data directory')
#     parser.add_argument('-o', '--output-file', dest="output_file", type=str,
#                         help='Output file path and filename')
#     parser.add_argument('-p', '--prettify', dest="prettify", action='store_true',
#                         help='Output pretty printed JSON')
#     parser.add_argument('-f', '--frame', dest="frame", type=int,
#                         help='Process Specific frame')
#     parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
#                         action='store_true')
#     parser.add_argument('-d', '--display', help='Whether or not image output should be displayed',
#                         action='store_true')

#     args = vars(parser.parse_args())

#     input_directory = args['openpose_group_data_dir']
#     prettify = args['prettify']
#     frame = args['frame']
#     verbose = args['verbose']
#     display = args['display']

#     main(input_directory, prettify, frame, verbose, display)
