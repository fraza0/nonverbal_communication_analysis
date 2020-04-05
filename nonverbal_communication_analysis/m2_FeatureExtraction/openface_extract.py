from nonverbal_communication_analysis.m0_Classes.Experiment import get_group_from_file_path
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files,
                                                    print_assertion_error)
import argparse
import re
import subprocess
from datetime import datetime
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

from nonverbal_communication_analysis.environment import (DATASET_SYNC, VALID_VIDEO_TYPES,
                                                          OPENFACE_OUTPUT_DIR,
                                                          OPENFACE_FACE_LANDMARK_IMG,
                                                          OPENFACE_OUTPUT_FLAGS,
                                                          OPENFACE_FACE_LANDMARK_VID_MULTI)


"""
OpenFace Output Commands:
-au_static    : static models only rely on a single image to make an estimate of AU
                  presence or intensity, while dynamic ones calibrate to a person by performing person
                  normalization in the video, they also attempt to correct for over and under prediction of AUs.
                  By default OpenFace uses static models on images and dynamic models on image sequences and videos.
                  DOC Ref: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units#static-vs-dynamic
-2Dfp         : output 2D landmarks in pixels
-3Dfp         : output 3D landmarks in milimeters
-pdmparams    : output rigid and non-rigid shape parameters:
                  * Rigid shape parameters describe the placement of the face in the image
                      (scaling, rotation, and translation);
                  * Non-rigid shape parameters on the other hand describe the deformation
                      of the face due to identity or expression (wider or taller faces, smiles, blinks etc.)
-pose         : output head pose (location and rotation)
-aus          : output the Facial Action Units
-gaze         : output gaze and related features (2D and 3D locations of eye landmarks)
-hogalign     : output extracted HOG feaure file
-simalign     : output similarity aligned images of the tracked faces
-nobadaligned : if outputting similarity aligned images, do not output from frames where detection failed
                  or is unreliable (thus saving some disk space)
-tracked      : output video with detected landmarks
"""

# def min_max_scaling():
#     normalized_df=(df-df.min())/(df.max()-df.min())


def format_output_string(file_path: str, group_id: str = None, directory: bool = False):

    output_string = group_id+'/' if group_id is not None else ''

    if directory:
        output_string += str(Path(file_path).parent).split('/')[-1]
    else:
        if "Videopc" not in file_path:
            output_string = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
            print("INFO: Media files do not follow naming of experiment videos. Writting Output to: %s " % (
                OPENFACE_OUTPUT_DIR / output_string))
        else:
            file_timestamp = re.compile(
                "(?<=Videopc.{1})(.*)(?=.avi)").split(file_path.split("/")[-1])[1][:-4]
            output_string = "%s_%s_%s_%s" % (
                file_timestamp[:2], file_timestamp[2:4], file_timestamp[4:8], file_timestamp[8:10])
            print("INFO: Output directory: %s" %
                  (OPENFACE_OUTPUT_DIR / output_string))

    return output_string


def openface_img(img_files: list, write: bool, verbose: bool = False):
    '''
    Process single image input. Only supports single face processing.
    '''
    assert isinstance(img_files, list), print_assertion_error(
        "img_files", "list")
    cmd_list = [OPENFACE_FACE_LANDMARK_IMG]
    for file_path in img_files:
        cmd_list += ['-f', file_path]

    cmd_list += OPENFACE_OUTPUT_FLAGS
    cmd_list += ['-out_dir', OPENFACE_OUTPUT_DIR]
    if write:
        cmd_list += ['-tracked']
    if verbose:
        print(cmd_list)
    subprocess.call(cmd_list)


def openface_vid(video_file: str, output_path: str, write: bool = False, verbose: bool = False):
    '''
    Process video input with multiple faces.

    In this case, OPENFACE_FEATURE_EXTRACTION will be used instead of OPENFACE_FACE_LANDMARK_VID
    as they both serve the same purpose of single face tracking, but only OPENFACE_FEATURE_EXTRACTION
    processed the images regarding the frame sequence, while OPENFACE_FACE_LANDMARK_VID processes
    each frame as it was a random image. (Unsure what is the difference, but such inference is
    described in documentation - https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments#example-uses)
    '''

    cmd_list = [OPENFACE_FACE_LANDMARK_VID_MULTI]
    cmd_list += ['-f', video_file]

    cmd_list += OPENFACE_OUTPUT_FLAGS
    cmd_list += ['-out_dir', output_path]
    if write:
        cmd_list += ['-tracked']
    if verbose:
        print(cmd_list)
    subprocess.call(cmd_list)


def main(group_directory: str, write: bool = False, verbose: bool = False):
    group_directory = Path(group_directory)
    group_id = get_group_from_file_path(group_directory)
    tasks_directories = [x for x in group_directory.iterdir()
                         if x.is_dir() and 'task' in str(x)]

    for task in tasks_directories:
        video_files = [x for x in task.iterdir()
                       if x.suffix in VALID_VIDEO_TYPES]
        for video in video_files:
            camera_id = re.search(r'(?<=Video)(pc\d{1})(?=\d{14})',
                                  video.name).group(0)

            output_path = OPENFACE_OUTPUT_DIR / group_id / task.name / camera_id
            openface_vid(video, output_path, write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract facial data using OpenFace')
    parser.add_argument('group_directory', type=str,
                        help='Group data Directory')
    parser.add_argument('-w', '--write', help="Write video with facial features",
                        action='store_true')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    group_directory = args['group_directory']
    write = args['write']
    verbose = args['verbose']

    main(group_directory, write=write, verbose=verbose)
