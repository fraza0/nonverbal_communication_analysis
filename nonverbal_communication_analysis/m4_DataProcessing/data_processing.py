import argparse
import csv
import json
import os
import re
from pathlib import Path

import pandas as pd

from nonverbal_communication_analysis.m0_Classes.Experiment import \
    get_group_from_file_path
from nonverbal_communication_analysis.m4_DataProcessing.video_process import \
    VideoProcess
from nonverbal_communication_analysis.m4_DataProcessing.openpose_process import \
    OpenposeProcess
from nonverbal_communication_analysis.m4_DataProcessing.openface_process import \
    OpenfaceProcess
from nonverbal_communication_analysis.m4_DataProcessing.densepose_process import \
    DenseposeProcess


def main(group_directory: str, task: int = None, specific_frame: int = None, video: bool = False, openpose: bool = False, openface: bool = False, densepose: bool = False, prettify: bool = False, display: bool = False, verbose: bool = False,):
    group_directory = Path(group_directory)
    group_id = get_group_from_file_path(group_directory)
    tasks_directories = [x for x in group_directory.iterdir()
                         if x.is_dir() and 'task' in str(x)]

    if task is not None:
        tasks_directories = [x for x in group_directory.iterdir()
                             if x.is_dir() and str(task) in str(x.name)]

    if video:
        if verbose:
            print("Processing Video data")
        vp = VideoProcess(group_id, prettify=prettify, verbose=verbose)
        vp.process(tasks_directories,
                   specific_frame=specific_frame, display=display)

    if openpose:
        if verbose:
            print("Processing Openpose data")
        opp = OpenposeProcess(group_id, prettify=prettify, verbose=verbose)
        opp.process(tasks_directories,
                    specific_frame=specific_frame, display=display)

    if openface:
        if verbose:
            print("Processing Openface data")
        ofp = OpenfaceProcess(group_id, prettify=prettify, verbose=verbose)
        ofp.process(tasks_directories,
                    specific_frame=specific_frame, display=display)

    if densepose:
        if verbose:
            print("Processing Densepose data")
        dpp = DenseposeProcess(group_id, prettify=prettify, verbose=verbose)
        dpp.process(tasks_directories,
                    specific_frame=specific_frame, display=display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Data processing step')
    parser.add_argument('group_data', type=str,
                        help='Group data directory')
    parser.add_argument('-vid', '--video', dest='video',
                        help='Process Video data', action='store_true')
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
                        help='Specify Task frame')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')
    parser.add_argument('-d', '--display', help='Whether or not image output should be displayed',
                        action='store_true')

    args = vars(parser.parse_args())

    group_directory = args['group_data']
    prettify = args['prettify']
    specific_frame = args['specific_frame']
    video = args['video']
    openpose = args['openpose']
    openface = args['openface']
    densepose = args['densepose']
    task = args['task']
    verbose = args['verbose']
    display = args['display']

    main(group_directory=group_directory, task=task, specific_frame=specific_frame,
         video=video, openpose=openpose, openface=openface, densepose=densepose,
         prettify=prettify, display=display, verbose=verbose)
