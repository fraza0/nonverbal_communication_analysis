import argparse
import csv
import json
import re
import os
from pathlib import Path

import pandas as pd
import yaml

from nonverbal_communication_analysis.environment import (
    CONFIDENCE_THRESHOLD, DATASET_SYNC, PEOPLE_FIELDS, VALID_OUTPUT_FILE_TYPES, CONFIG_FILE, OPENPOSE_OUTPUT_DIR)
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files, log)

from nonverbal_communication_analysis.m0_Classes.Experiment import Experiment
from nonverbal_communication_analysis.m0_Classes.ExperimentCameraFrame import ExperimentCameraFrame


class OpenposeClean(object):

    def __init__(self, group_id):
        self.experiment = Experiment(group_id)
        self.group_id = group_id
        self.base_output_dir = OPENPOSE_OUTPUT_DIR / \
            group_id / (group_id + '_clean')
        os.makedirs(self.base_output_dir, exist_ok=True)

    def process_frames(self, camera_frame_files: dict, output_directory: str, prettify: bool = False, verbose: bool = False, display: bool = False):
        for camera in camera_frame_files:
            for frame_file in camera_frame_files[camera]:
                frame_idx = re.search(r'(?<=_)(\d{12})(?=_)',
                                      frame_file.name).group(0)
                output_frame_directory = output_directory / camera
                output_frame_file = output_frame_directory / \
                    ("%s_%s_clean.json" % (camera, frame_idx))
                os.makedirs(output_frame_directory, exist_ok=True)

                with open(frame_file) as json_data:
                    data = json.load(json_data)
                    file_people_df = pd.json_normalize(data['people'])
                    experiment_frame = ExperimentCameraFrame(
                        camera, int(frame_idx), file_people_df, self.experiment._vis, verbose=verbose, display=display)
                json_data.close()

                if prettify:
                    json.dump(experiment_frame.to_json(), open(
                        output_frame_file, 'w'), indent=2)
                else:
                    json.dump(experiment_frame.to_json(), open(
                        output_frame_file, 'w'))

    def clean(self, tasks_directories: dict, specific_frame: int = None, prettify: bool = False, verbose: bool = False, display: bool = False):
        for task in tasks_directories:
            camera_files = dict()
            task_directory = OPENPOSE_OUTPUT_DIR / self.group_id / task.name
            openpose_camera_directories = [
                x for x in task_directory.iterdir() if x.is_dir()]

            # load camera files
            for camera_id in openpose_camera_directories:
                openpose_camera_raw_files = [x for x in camera_id.iterdir()
                                             if x.suffix in VALID_OUTPUT_FILE_TYPES]
                openpose_camera_raw_files.sort()
                camera_files[camera_id.name] = openpose_camera_raw_files

                if specific_frame is not None:
                    camera_files[camera_id.name] = [
                        openpose_camera_raw_files[specific_frame]]
                else:
                    camera_files[camera_id.name] = openpose_camera_raw_files

            num_frames = 1
            if specific_frame is None:
                num_frames = min(len(files) for files in camera_files.values())
                for camera in camera_files:
                    camera_files[camera] = camera_files[camera][:num_frames]

            output_directory = self.base_output_dir / task.name
            self.process_frames(camera_files, output_directory, prettify=prettify,
                                verbose=verbose, display=display)


def main(input_directories: list, prettify: bool, specific_frame: int = None, verbose: bool = False, display: bool = False):
    group_id = re.compile('OPENPOSE/(.*)/task').split(input_directories[0])[-2]
    output_dir = OPENPOSE_OUTPUT_DIR / group_id / (group_id+"_clean")
    experiment = Experiment(group_id)

    output_file = output_dir / (group_id+"_clean.json")
    json.dump(json.loads(experiment.to_json()), open(
        output_file, 'w'), separators=(',', ':'))

    camera_files = dict()

    # load files
    for cam_dir in input_directories:
        camera = cam_dir.split('/')[-2]
        files = [cam_dir+file for file in filter_files(
            fetch_files_from_directory([cam_dir]), valid_types=VALID_OUTPUT_FILE_TYPES)]
        files.sort()

        if specific_frame is not None:
            camera_files[camera] = [files[specific_frame]]
        else:
            camera_files[camera] = files

    # process_frame = None

    if specific_frame is not None:
        num_frames = 1
        # process_frame = specific_frame
        for cam, files in camera_files.items():
            camera_files[cam] = camera_files[cam]
    else:
        num_frames = min(len(files) for files in camera_files.values())
        # process_frame = 0
        for cam, files in camera_files.items():
            camera_files[cam] = camera_files[cam][:num_frames]

    for frame_file_idx in range(num_frames):
        for camera in camera_files:
            output_frame_file = output_dir / ("%s_clean/%s_%s_clean.json" % (camera,
                                                                             camera, str(frame_file_idx).zfill(12)))
            os.makedirs(output_dir / ("%s_clean/" % camera), exist_ok=True)
            frame_file = camera_files[camera][frame_file_idx]

            with open(frame_file) as json_data:
                data = json.load(json_data)
                file_people_df = pd.json_normalize(data['people'])
                frame = ExperimentCameraFrame(
                    camera, frame_file_idx, file_people_df, experiment._vis, verbose=verbose, display=display)
            json_data.close()

            if prettify:
                json.dump(frame.to_json(), open(
                    output_frame_file, 'w'), indent=2)
            else:
                json.dump(frame.to_json(), open(
                    output_frame_file, 'w'))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract facial data using OpenFace')
    parser.add_argument('openpose_group_data_dir', type=str,
                        help='Openpose output group data directory')
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

    input_directory = args['openpose_group_data_dir']
    prettify = args['prettify']
    frame = args['frame']
    verbose = args['verbose']
    display = args['display']

    main(input_directory, prettify, frame, verbose, display)
