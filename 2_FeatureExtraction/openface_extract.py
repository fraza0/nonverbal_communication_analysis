import argparse
import re
import subprocess
from datetime import datetime
from os import listdir
from os.path import isfile, join, splitext

from environment import (OPENFACE_FACE_LANDMARK_IMG,
                         OPENFACE_FACE_LANDMARK_VID_MULTI,
                         OPENFACE_FEATURE_EXTRACTION, OPENFACE_OUTPUT_COMMANDS,
                         OPENFACE_OUTPUT_DIR, VALID_IMAGE_TYPES,
                         VALID_VIDEO_TYPES)
from utils import print_assertion_error, fetch_files_from_directory, filter_files

OPENFACE_OUTPUT = OPENFACE_OUTPUT_DIR + "/"


def format_output_string(file_path):

    output_string = ""

    if "Videopc" not in file_path:
        output_string = OPENFACE_OUTPUT + datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
        print("INFO: Media files do not follow naming of experiment videos. Writting Output to: %s " % output_string)
    else:
        file_timestamp = re.compile(
            "(?<=Videopc.{1})(.*)(?=.avi)").split(file_path.split("/")[-1])[1][:-4]
        output_string = "%s_%s_%s_%s" % (
            file_timestamp[:2], file_timestamp[2:4], file_timestamp[4:8], file_timestamp[8:10])
        print("INFO: Output directory: %s" % (OPENFACE_OUTPUT + output_string))

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

    cmd_list += OPENFACE_OUTPUT_COMMANDS
    cmd_list += ['-out_dir', OPENFACE_OUTPUT]
    if write:
        cmd_list += ['-tracked']
    if verbose:
        print(cmd_list)
    subprocess.call(cmd_list)


def openface_vid(vid_files: list, multi: bool, write: bool, verbose: bool = False):
    '''
    Process video input with single of multiple faces.

    In this case, OPENFACE_FEATURE_EXTRACTION will be used instead of OPENFACE_FACE_LANDMARK_VID
    as they both serve the same purpose of single face tracking, but only OPENFACE_FEATURE_EXTRACTION
    processed the images regarding the frame sequence, while OPENFACE_FACE_LANDMARK_VID processes
    each frame as it was a random image. (Unsure what is the difference, but such inference is
    described in documentation - https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments#example-uses)
    '''
    assert isinstance(vid_files, list), print_assertion_error(
        "vid_files", "list")
    assert isinstance(multi, bool), print_assertion_error(
        "multi", "bool")

    if not multi:
        cmd_list = [OPENFACE_FEATURE_EXTRACTION]
    else:
        cmd_list = [OPENFACE_FACE_LANDMARK_VID_MULTI]

    for file_path in vid_files:
        cmd_list += ['-f', file_path]

    cmd_list += OPENFACE_OUTPUT_COMMANDS
    cmd_list += ['-out_dir', OPENFACE_OUTPUT]
    if write:
        cmd_list += ['-tracked']
    if verbose:
        print(cmd_list)
    subprocess.call(cmd_list)


def openface_cam(device: int, write: bool, verbose: bool = False):
    '''
    Process data directly from device (webcam = 0) input
    '''
    assert isinstance(device, int), print_assertion_error(
        "device", "int")

    cmd_list = [OPENFACE_FEATURE_EXTRACTION]
    cmd_list += OPENFACE_OUTPUT_COMMANDS
    cmd_list += ['-device', '0']
    cmd_list += ['-out_dir', OPENFACE_OUTPUT]
    if verbose:
        print(cmd_list)
    if write:
        cmd_list += ['-tracked']
    subprocess.call(cmd_list)


if __name__ == "__main__":

    MEDIA_TYPE_IMAGE = 'img'
    MEDIA_TYPE_VIDEO = 'vid'
    MEDIA_TYPE_CAM = 'cam'

    parser = argparse.ArgumentParser(
        description='Extract facial data using OpenFace')
    parser.add_argument('media_type', type=str, choices=[
                        MEDIA_TYPE_IMAGE, MEDIA_TYPE_VIDEO, MEDIA_TYPE_CAM], help='Media type')
    parser.add_argument('media_files', type=str, nargs='*', help='Media files')
    parser.add_argument('-dir', '--directory', action="store_true",
                        help='Media files path')
    parser.add_argument('--multi', action="store_true",
                        help='Multiple individuals')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')
    parser.add_argument('-w', '--write', help="Write video with facial features",
                        action='store_true')

    args = vars(parser.parse_args())

    media_type = args['media_type']
    valid_types = (VALID_IMAGE_TYPES if media_type ==
                   MEDIA_TYPE_IMAGE else VALID_VIDEO_TYPES)
    media_files = (args['media_files'] if 'media_files' in args else None)
    directory = args['directory']
    multi = args['multi']
    verbose = args['verbose']
    write = args['write']

    if directory:
        media_files = fetch_files_from_directory(media_files)

    if media_type == MEDIA_TYPE_IMAGE:
        media_files = filter_files(media_files, VALID_IMAGE_TYPES)
    elif media_type == MEDIA_TYPE_VIDEO:
        media_files = filter_files(media_files, VALID_VIDEO_TYPES)
        
    if not media_files and media_type != MEDIA_TYPE_CAM:
        print("Error: No media files passed or no valid media files in directory")
        exit()

    OPENFACE_OUTPUT += format_output_string(media_files[0])

    if media_type == MEDIA_TYPE_IMAGE:
        media_files = filter_files(media_files, VALID_VIDEO_TYPES)
        openface_img(media_files, write)
    elif media_type == MEDIA_TYPE_VIDEO:
        openface_vid(media_files, multi, write)
    elif media_type == MEDIA_TYPE_CAM:
        openface_cam(0, write)
