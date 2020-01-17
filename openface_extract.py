import os
import subprocess
import argparse

import environment as env
from utils import print_assertion_error

MEDIA_TYPE_IMAGE = 'img'
MEDIA_TYPE_VIDEO = 'vid'

# class OpenFaceExtractor:

parser = argparse.ArgumentParser(
    description='Process Facial Data using OpenFace')
parser.add_argument('media_type', type=str, choices=[
                    'img', 'vid'], help='Media type')
parser.add_argument('media_files', type=str, nargs='*', help='Media files')
parser.add_argument('-dir', '--directory', action="store_true",
                    help='Media files path')
parser.add_argument('--multi', action="store_true",
                    help='Multiple individuals')
parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                    action='store_true')

args = vars(parser.parse_args())


def openface_landmark_img(img_files: list, verbose: bool = False):
    assert isinstance(img_files, list), print_assertion_error(
        "img_files", "list")
    cmd_list = [env.OPENFACE_FACE_LANDMARK_IMG]
    for file_path in img_files:
        cmd_list += ['-f', file_path]

    print(cmd_list)
    subprocess.call(cmd_list)


def openface_landmark_vid(vid_files: list, multi: bool = False, verbose: bool = False):
    assert isinstance(vid_files, list), print_assertion_error(
        "vid_files", "list")
    assert isinstance(multi, bool), print_assertion_error(
        "multi", "bool")

    if not multi:
        cmd_list = [env.OPENFACE_FACE_LANDMARK_VID]
    else:
        cmd_list = [env.OPENFACE_FACE_LANDMARK_VID_MULTI]

    for file_path in vid_files:
        cmd_list += ['-f', file_path]

    print(cmd_list)
    # subprocess.call(cmd_list)


# print(args)

media_type = args['media_type']
valid_types = (env.VALID_IMAGE_TYPES if media_type ==
               MEDIA_TYPE_IMAGE else env.VALID_VIDEO_TYPES)
media_files = (args['media_files'] if 'media_files' in args else None)
directory = args['directory']
multi = args['multi']
verbose = args['verbose']

if not media_files:
    print("Error: No media files passed")
    exit()


# TODO: if directory, retrieve all files, pass through filetype validation

for file in media_files:
    _, file_extension = os.path.splitext(file)
    if file_extension not in valid_types:
        print("Not supported or invalid file type (%s). Check input files" % file)
        exit()

if media_type == MEDIA_TYPE_IMAGE:
    openface_landmark_img(media_files)
elif media_type == MEDIA_TYPE_VIDEO:
    openface_landmark_vid(media_files, multi=multi)
