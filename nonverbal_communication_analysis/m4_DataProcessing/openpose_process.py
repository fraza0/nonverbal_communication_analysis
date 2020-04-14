import argparse
import json
import pandas as pd
from nonverbal_communication_analysis.m0_Classes.Experiment import get_group_from_file_path

class OpenPoseProcess:
    def __init__(self):
        pass


def main(group_directory: str, specific_frame: int = None, specific_task: int = None, openpose: bool = False, openface: bool = False, densepose: bool = False, verbose: bool = False):
    group_id = get_group_from_file_path(group_directory)

    # visualizer = Visualizer(group_id)
    # if (specific_frame and specific_task) is not None:
    #     visualizer.show_frame(group_directory, specific_task=specific_task,
    #                           specific_frame=specific_frame, openpose=openpose,
    #                           openface=openface, densepose=densepose, verbose=verbose)
    # else:
    #     visualizer.play_video(group_directory, specific_task=specific_task, openpose=openpose,
    #                           openface=openface, densepose=densepose, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize data')
    parser.add_argument('group_directory', type=str,
                        help='Group video directory')
    parser.add_argument('-t', '--task', dest="specific_task", type=int, choices=[1, 2],
                        help='Specify Task frame')
    parser.add_argument('-f', '--frame', dest='specific_frame', type=int,
                        help='Process Specific frame')
    parser.add_argument('-op', '--openpose',
                        help='Overlay openpose data', action='store_true')
    parser.add_argument('-dp', '--densepose',
                        help='Overlay densepose data', action='store_true')
    parser.add_argument('-of', '--openface',
                        help='Overlay openface data', action='store_true')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    group_directory = args['group_directory']
    specific_task = args['specific_task']
    specific_frame = args['specific_frame']
    openpose = args['openpose']
    openface = args['openface']
    densepose = args['densepose']
    verbose = args['verbose']

    main(group_directory, specific_frame=specific_frame, specific_task=specific_task,
         openpose=openpose, openface=openface, densepose=densepose, verbose=verbose)
