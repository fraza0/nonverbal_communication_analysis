import argparse
import csv
import json
import re
import os

import pandas as pd
import yaml

from nonverbal_communication_analysis.environment import OPENPOSE_OUTPUT_DIR, OPENFACE_OUTPUT_DIR, DENSEPOSE_OUTPUT_DIR, VALID_OUTPUT_FILE_TYPES
from nonverbal_communication_analysis.utils import filter_files, fetch_files_from_directory
from nonverbal_communication_analysis.m3_DataPreprocessing.openpose_clean import OpenposeClean
from nonverbal_communication_analysis.m0_Classes.Experiment import get_group_from_file_path
from pathlib import Path


def main(group_directory: str, task: int = None, specific_frame: int = None, openpose: bool = True, openface: bool = False, densepose: bool = False, prettify: bool = False, display: bool = False, verbose: bool = False,):
    group_directory = Path(group_directory)
    group_id = get_group_from_file_path(group_directory)
    tasks_directories = [x for x in group_directory.iterdir()
                         if x.is_dir() and 'task' in str(x)]

    if task is not None:
        tasks_directories = [x for x in tasks_directories
                             if str(task) in x.name]

    if openpose:
        if verbose:
            print("Cleaning Openpose data")
        opc = OpenposeClean(group_id)
        opc.clean(tasks_directories, specific_frame=specific_frame,
                  prettify=prettify, verbose=verbose, display=display)

    if openface:
        if verbose:
            print("Cleaning Openface data")

    if densepose:
        if verbose:
            print("Cleaning Densepose data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Data cleaning step')
    parser.add_argument('group_data', type=str,
                        help='Group data directory')
    parser.add_argument('-op', '--openpose', dest='openpose',
                        help='Process Openpose data', action='store_true')
    parser.add_argument('-dp', '--densepose', dest='densepose',
                        help='Process Densepose data', action='store_true')
    parser.add_argument('-of', '--openface', dest='openface',
                        help='Process OpenFace directory', action='store_true')
    parser.add_argument('-o', '--output-file', dest="output_file", type=str,
                        help='Output file path and filename')
    parser.add_argument('-p', '--prettify', dest="prettify", action='store_true',
                        help='Output pretty printed JSON')
    parser.add_argument('-f', '--frame', dest="specific_frame", type=int,
                        help='Process Specific frame')
    parser.add_argument('-t', '--task', dest="task", type=int, choices=[1, 2],
                        help='Soecify Task frame')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')
    parser.add_argument('-d', '--display', help='Whether or not image output should be displayed',
                        action='store_true')

    args = vars(parser.parse_args())

    group_directory = args['group_data']
    prettify = args['prettify']
    specific_frame = args['specific_frame']
    openpose = args['openpose']
    openface = args['openface']
    densepose = args['densepose']
    task = args['task']
    verbose = args['verbose']
    display = args['display']

    main(group_directory=group_directory, task=task, specific_frame=specific_frame,
         openpose=openpose, openface=openface, densepose=densepose,
         prettify=prettify, display=display, verbose=verbose)
