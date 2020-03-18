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


def main(input_directories: list, single_step: bool, prettify: bool, verbose: bool = False):
    group_id = re.match('Openpose/(.*)/output', input_directories[0]).group(1)
    output_dir = OPENPOSE_OUTPUT_DIR+group_id+"/"+group_id+"_clean"
    experiment = Experiment(group_id)

    to_discard = list()

    for directory in input_directories:
        if verbose:
            print("Directory:", directory)

        input_files = [directory+file for file in filter_files(
            fetch_files_from_directory([directory]), valid_types=VALID_OUTPUT_FILE_TYPES)]
        input_files.sort()
        total_files = len(input_files)

        camera = directory.replace('/', '').split('output_')[1]
        frames_list = list()

        if single_step:
            output_file = output_dir + "/" + group_id + "_clean.json"
        else:
            output_file = output_dir + "/" + \
                re.match('Openpose/(.*)/output_(.*)/',
                         directory).group(2) + "_clean.json"

        # frame_counter is equal to frame number written on file name as we sort the files.
        frame_counter = 0
        last_checkpoint = 0

        specific_frame = None
        if specific_frame is not None:
            input_files = [input_files[specific_frame]]
            frame_counter = specific_frame

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
                file_people_df = pd.json_normalize(data['people'])
                frame = ExperimentCameraFrame(
                    camera, frame_counter, file_people_df, experiment._vis, verbose=verbose)
                frames_list.append(frame)
            json_data.close()

            if single_step:
                experiment.people[camera] = frames_list

        if not single_step:
            json.dump({camera: [frame.to_json() for frame in frames_list]}, open(
                output_file, 'w'), separators=(',', ':'))

    output_file = output_dir + "/" + group_id+"_clean.json"
    with open(output_file, 'w') as output:
        if prettify:
            json.dump(json.loads(experiment.to_json()), output, indent=2)
        else:
            output.write(experiment.to_json())
    output.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract facial data using OpenFace')
    parser.add_argument('openpose_data_dir', nargs=3, type=str,
                        help='Openpose output data directory')
    parser.add_argument('-o', '--output-file', dest="output_file", type=str,
                        help='Output file path and filename')
    parser.add_argument('-ss', '--single-step', dest="single_step", action='store_true',
                        help='Number of processing steps. Single or Multiple')
    parser.add_argument('-p', '--prettify', dest="prettify", action='store_true',
                        help='Output pretty printed JSON')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    input_directory = args['openpose_data_dir']
    single_step = args['single_step']
    prettify = args['prettify']
    verbose = args['verbose']

    main(input_directory, single_step, prettify, verbose)
