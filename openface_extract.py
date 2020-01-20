import os
import subprocess
import argparse

import environment as env
from utils import print_assertion_error

MEDIA_TYPE_IMAGE = 'img'
MEDIA_TYPE_VIDEO = 'vid'
MEDIA_TYPE_FEATURES = ''
MEDIA_TYPE_CAM = 'cam'

# class OpenFaceExtractor:

parser = argparse.ArgumentParser(
    description='Process Facial Data using OpenFace')
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


def openface_img(img_files: list, verbose: bool = False):
    '''
    Process single image input. Only supports single face processing.
    '''
    assert isinstance(img_files, list), print_assertion_error(
        "img_files", "list")
    cmd_list = [env.OPENFACE_FACE_LANDMARK_IMG]
    for file_path in img_files:
        cmd_list += ['-f', file_path]

    cmd_list += ['-out_dir', env.OPENFACE_OUTPUT_DIR]
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
        cmd_list = [env.OPENFACE_FEATURE_EXTRACTION]
    else:
        cmd_list = [env.OPENFACE_FACE_LANDMARK_VID_MULTI]

    for file_path in vid_files:
        cmd_list += ['-f', file_path]

    cmd_list += env.OPENFACE_OUTPUT_COMMANDS
    cmd_list += ['-out_dir', env.OPENFACE_OUTPUT_DIR]
    print(cmd_list)
    subprocess.call(cmd_list)


def openface_cam(device: list = 0, verbose: bool = False):
    '''
    Process data directly from device (webcam = 0) input
    '''
    assert isinstance(device, int), print_assertion_error(
        "device", "int")

    cmd_list = [env.OPENFACE_FEATURE_EXTRACTION]
    cmd_list += ['-device', '0']
    cmd_list += ['-out_dir', env.OPENFACE_OUTPUT_DIR]
    print(cmd_list)
    subprocess.call(cmd_list)

# print(args)


media_type = args['media_type']
valid_types = (env.VALID_IMAGE_TYPES if media_type ==
               MEDIA_TYPE_IMAGE else env.VALID_VIDEO_TYPES)
media_files = (args['media_files'] if 'media_files' in args else None)
directory = args['directory']
multi = args['multi']
verbose = args['verbose']

if not media_files and media_type != MEDIA_TYPE_CAM:
    print("Error: No media files passed")
    exit()


# TODO: if directory, retrieve all files, pass through filetype validation

for file in media_files:
    _, file_extension = os.path.splitext(file)
    if file_extension not in valid_types:
        print("Not supported or invalid file type (%s). Check input files" % file)
        exit()

if media_type == MEDIA_TYPE_IMAGE:
    openface_img(media_files)
elif media_type == MEDIA_TYPE_VIDEO:
    openface_vid(media_files, multi=multi)
elif media_type == MEDIA_TYPE_CAM:
    openface_cam(device=0)
