import argparse
import csv
import json
import re

import pandas as pd
import yaml

from nonverbal_communication_analysis.environment import (
    CONFIDENCE_THRESHOLD, DATASET_SYNC, PEOPLE_FIELDS, VALID_OUTPUT_FILE_TYPES, CONFIG_FILE, OPENPOSE_OUTPUT_DIR)
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files, log)

from nonverbal_communication_analysis.m0_Classes.Experiment import Experiment
from nonverbal_communication_analysis.m0_Classes.ExperimentCameraFrame import ExperimentCameraFrame


def main(input_directories: list, prettify: bool, specific_frame: int = None, verbose: bool = False, display: bool = False):
    group_id = re.match('OPENPOSE/(.*)/task', input_directories[0]).group(1)
    output_dir = OPENPOSE_OUTPUT_DIR+group_id+"/"+group_id+"_clean/"
    experiment = Experiment(group_id)

    # output_file = output_dir + "/" + group_id+"_clean.json"
    # json.dump(json.loads(experiment.to_json()), open(
    #     output_file, 'w'), separators=(',', ':'))

    # camera_files = dict()

    # # load files
    # for cam_dir in input_directories:
    #     camera = cam_dir.replace('/', '').split('output_')[1]
    #     files = [cam_dir+file for file in filter_files(
    #         fetch_files_from_directory([cam_dir]), valid_types=VALID_OUTPUT_FILE_TYPES)]
    #     files.sort()

    #     if specific_frame is not None:
    #         camera_files[camera] = [files[specific_frame]]
    #     else:
    #         camera_files[camera] = files

    # process_frame = None

    # if specific_frame is not None:
    #     num_frames = 1
    #     # process_frame = specific_frame
    #     for cam, files in camera_files.items():
    #         camera_files[cam] = camera_files[cam]
    # else:
    #     num_frames = min(len(files) for files in camera_files.values())
    #     # process_frame = 0
    #     for cam, files in camera_files.items():
    #         camera_files[cam] = camera_files[cam][:num_frames]

    # last_checkpoint = 0
    # for frame_file_idx in range(num_frames):
    #     for camera in camera_files:
    #         output_frame_file = output_dir + \
    #             "%s_clean/%s_%s_clean.json" % (camera,
    #                                            camera, str(frame_file_idx).zfill(12))
    #         frame_file = camera_files[camera][frame_file_idx]

    #         if verbose:
    #             progress = round(frame_file_idx / num_frames * 100)
    #             if progress % 10 == 0:
    #                 if progress != last_checkpoint:
    #                     last_checkpoint = progress
    #                     print("Progress: %s %% (%s / %s)" %
    #                           (progress, frame_file_idx, num_frames))

    #         with open(frame_file) as json_data:
    #             data = json.load(json_data)
    #             file_people_df = pd.json_normalize(data['people'])
    #             frame = ExperimentCameraFrame(
    #                 camera, frame_file_idx, file_people_df, experiment._vis, verbose=verbose)
    #         json_data.close()

    #         if prettify:
    #             json.dump(frame.to_json(), open(
    #                 output_frame_file, 'w'), indent=2)
    #         else:
    #             json.dump(frame.to_json(), open(
    #                 output_frame_file, 'w'))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract facial data using OpenFace')
    parser.add_argument('openpose_data_dir', nargs=3, type=str,
                        help='Openpose output data directory')
    parser.add_argument('-o', '--output-file', dest="output_file", type=str,
                        help='Output file path and filename')
    parser.add_argument('-p', '--prettify', dest="prettify", action='store_true',
                        help='Output pretty printed JSON')
    parser.add_argument('-f', '--frame', dest="frame", type=int,
                        help='Process Specific frame')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')
    parser.add_argument('-d', '--display', help='Whether or not image output should be displayed',
                        action='store_true')

    args = vars(parser.parse_args())

    input_directory = args['openpose_data_dir']
    prettify = args['prettify']
    frame = args['frame']
    verbose = args['verbose']
    display = args['display']

    # main(input_directory, prettify, frame, verbose, display)