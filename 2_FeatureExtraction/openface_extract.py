from utils import print_assertion_error
from os.path import isfile, join, splitext
from os import listdir
import subprocess
import argparse

from environment import (OPENFACE_OUTPUT_DIR, OPENFACE_FACE_LANDMARK_IMG,
                         OPENFACE_FEATURE_EXTRACTION,
                         OPENFACE_FACE_LANDMARK_VID_MULTI,
                         OPENFACE_OUTPUT_COMMANDS,
                         OPENFACE_OUTPUT,
                         VALID_IMAGE_TYPES, VALID_VIDEO_TYPES)


def openface_img(img_files: list, verbose: bool = False):
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
    print(cmd_list)
    subprocess.call(cmd_list)


def openface_vid(vid_files: list, multi: bool = False, verbose: bool = False):
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
    print(cmd_list)
    subprocess.call(cmd_list)


def openface_cam(device: list = 0, verbose: bool = False):
    '''
    Process data directly from device (webcam = 0) input
    '''
    assert isinstance(device, int), print_assertion_error(
        "device", "int")

    cmd_list = [OPENFACE_FEATURE_EXTRACTION]
    cmd_list += OPENFACE_OUTPUT_COMMANDS
    cmd_list += ['-device', '0']
    cmd_list += ['-out_dir', OPENFACE_OUTPUT]
    print(cmd_list)
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

    args = vars(parser.parse_args())

    media_type = args['media_type']
    valid_types = (VALID_IMAGE_TYPES if media_type ==
                   MEDIA_TYPE_IMAGE else VALID_VIDEO_TYPES)
    media_files = (args['media_files'] if 'media_files' in args else None)
    directory = args['directory']
    multi = args['multi']
    verbose = args['verbose']

    if directory:
        media_files_directory = media_files
        for _dir in media_files_directory:
            media_files = [f for f in listdir(_dir) if isfile(join(_dir, f))]

    if not media_files and media_type != MEDIA_TYPE_CAM:
        print("Error: No media files passed")
        exit()

    for file in media_files:
        _, file_extension = splitext(file)
        if file_extension not in valid_types:
            print("Not supported or invalid file type (%s). File must be {%s}" % (
                file, valid_types))
            exit()

    if media_type == MEDIA_TYPE_IMAGE:
        openface_img(media_files)
    elif media_type == MEDIA_TYPE_VIDEO:
        openface_vid(media_files, multi=multi)
    elif media_type == MEDIA_TYPE_CAM:
        openface_cam(device=0)
