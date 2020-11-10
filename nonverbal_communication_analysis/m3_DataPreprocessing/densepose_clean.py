import argparse
import csv
import json
import os
import re
from pathlib import Path

import pandas as pd

from nonverbal_communication_analysis.environment import (DENSEPOSE_KEY,
                                                          DENSEPOSE_OUTPUT_DIR,
                                                          VALID_OUTPUT_FILE_TYPES)
from nonverbal_communication_analysis.m0_Classes.Experiment import Experiment
from nonverbal_communication_analysis.m0_Classes.ExperimentCameraFrame import \
    ExperimentCameraFrame
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files, log)


class DenseposeClean(object):

    def __init__(self, group_id):
        self.experiment = Experiment(group_id)
        self.group_id = group_id
        self.base_output_dir = DENSEPOSE_OUTPUT_DIR / \
            group_id / (group_id + '_clean')
        os.makedirs(self.base_output_dir, exist_ok=True)
        json.dump(self.experiment.to_json(),
                  open(self.base_output_dir / (self.group_id + '.json'), 'w'))

    def process_frames(self, camera_frame_files: dict, output_directory: str, prettify: bool = False, verbose: bool = False, display: bool = False):
        """Process each frame. Filter Skeleton parts detected and parse Subjects

        Args:
            camera_frame_files (dict): Camera frame files
            output_directory (str): Output directory path
            prettify (bool, optional): Pretty JSON print. Defaults to False.
            verbose (bool, optional): Verbose. Defaults to False.
            display (bool, optional): Display visualization. Defaults to False.
        """

        for camera in camera_frame_files:
            for frame_file in camera_frame_files[camera]:
                frame_idx = re.search(r'(?<=_)(\d{12})(?=.)',
                                      frame_file.name).group(0)
                output_frame_directory = output_directory / camera
                output_frame_file = output_frame_directory / \
                    ("%s_%s_clean.json" % (camera, frame_idx))
                os.makedirs(output_frame_directory, exist_ok=True)

                with open(frame_file) as json_data:
                    data = json.load(json_data)
                    metadata = data['meta']
                    del data['meta']
                    file_people_df = pd.json_normalize(data.values())
                    experiment_frame = ExperimentCameraFrame(
                        camera, int(frame_idx), file_people_df, DENSEPOSE_KEY, verbose=verbose, display=display, metadata=metadata)
                json_data.close()

                if prettify:
                    json.dump(experiment_frame.to_json(), open(
                        output_frame_file, 'w'), indent=2)
                else:
                    json.dump(experiment_frame.to_json(), open(
                        output_frame_file, 'w'))

    def clean(self, tasks_directories: dict, specific_frame: int = None, prettify: bool = False, verbose: bool = False, display: bool = False):
        """Densepose feature data cleansing and filtering

        Args:
            tasks_directories (dict): Experiment Group Tasks directory
            specific_frame (int, optional): Specify frame. Defaults to None.
            prettify (bool, optional): Pretty JSON print. Defaults to False.
            verbose (bool, optional): Verbose. Defaults to False.
            display (bool, optional): Enable visualization. Defaults to False.
        """

        for task in tasks_directories:
            camera_files = dict()
            task_directory = DENSEPOSE_OUTPUT_DIR / self.group_id / task.name
            densepose_camera_directories = [
                x for x in task_directory.iterdir() if x.is_dir()]

            # load camera files
            for camera_id in densepose_camera_directories:
                densepose_camera_raw_files = [x for x in camera_id.iterdir()
                                             if x.suffix in VALID_OUTPUT_FILE_TYPES]
                densepose_camera_raw_files.sort()
                camera_files[camera_id.name] = densepose_camera_raw_files

                if specific_frame is not None:
                    camera_files[camera_id.name] = [
                        densepose_camera_raw_files[specific_frame]]
                else:
                    camera_files[camera_id.name] = densepose_camera_raw_files

            num_frames = 1
            if specific_frame is None:
                num_frames = min(len(files) for files in camera_files.values())
                for camera in camera_files:
                    camera_files[camera] = camera_files[camera][:num_frames]

            output_directory = self.base_output_dir / task.name
            self.process_frames(camera_files, output_directory, prettify=prettify,
                                verbose=verbose, display=display)
