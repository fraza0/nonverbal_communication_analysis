import argparse
import csv
import json
import re

import pandas as pd
import yaml
from pandas.io.json import json_normalize

from nonverbal_communication_analysis.environment import (
    CONFIDENCE_THRESHOLD, DATASET_SYNC, PEOPLE_FIELDS, VALID_OUTPUT_FILE_TYPES, CONFIG_FILE)
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files, log)

from nonverbal_communication_analysis.m0_Classes.Experiment import Experiment
from nonverbal_communication_analysis.m0_Classes.ExperimentCameraFrame import ExperimentCameraFrame


def main(input_directories: list, output_file: str, verbose: bool):

    group_id = re.match('Openpose/(.*)/output', input_directories[0]).group(1)
    experiment = Experiment(group_id)

    for directory in input_directories:
        if verbose:
            print("Directory:", directory)

        input_files = [directory+file for file in filter_files(
            fetch_files_from_directory([directory]), valid_types=VALID_OUTPUT_FILE_TYPES)]
        input_files.sort()
        total_files = len(input_files)

        camera = directory.replace('/', '').split('output_')[1]
        experiment.people[camera] = list()

        # frame_counter is equal to frame number written on file name as we sort the files.
        frame_counter = 0
        last_checkpoint = 0
        for file in input_files[:]:
            if verbose:
                progress = round(frame_counter / total_files * 100)
                if progress % 10 == 0:
                    if progress != last_checkpoint:
                        last_checkpoint = progress
                        print("Progress: %s %% (%s / %s)" %
                              (progress, frame_counter, total_files))
            frame_counter += 1
            with open(file) as json_data:
                data = json.load(json_data)
                file_people_df = json_normalize(data['people'])
                frame = ExperimentCameraFrame(
                    camera, frame_counter, file_people_df)
                experiment.people[camera].append(frame)

    if output_file is None:
        log("WARN", "No output_file passed. Not writting to file. Printing output to terminal instead.")
        print(experiment.to_json())
        return

    with open(output_file, 'w') as output:
        output.write(experiment.to_json())
    output.close()

    return


if __name__ == "__main__":
    # TODO: adjust program to receive GROUP directory with 3 cameras' outputs
    parser = argparse.ArgumentParser(
        description='Extract facial data using OpenFace')
    parser.add_argument('openpose_data_dir', nargs=3, type=str,
                        help='Openpose output data directory')
    parser.add_argument('-o', '--output_file', type=str,
                        help='Output file path and filename')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    input_directory = args['openpose_data_dir']
    output_file = args['output_file']
    verbose = args['verbose']

    main(input_directory, output_file, verbose)
