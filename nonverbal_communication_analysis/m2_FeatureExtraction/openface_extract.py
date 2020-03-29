import argparse
import re
import subprocess
from datetime import datetime
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

from nonverbal_communication_analysis.environment import (
    OPENFACE_FACE_LANDMARK_IMG, OPENFACE_FACE_LANDMARK_VID_MULTI,
    OPENFACE_FEATURE_EXTRACTION, OPENFACE_OUTPUT_DIR, OPENFACE_OUTPUT_FLAGS,
    VALID_IMAGE_TYPES, VALID_VIDEO_TYPES)
from nonverbal_communication_analysis.utils import (fetch_files_from_directory,
                                                    filter_files,
                                                    print_assertion_error)


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


def format_output_string(file_path: str, group_id:str = None, directory: bool = False):

    output_string = group_id+'/' if group_id is not None else ''

    if directory:
        output_string += str(Path(file_path).parent).split('/')[-1]
    else:
        if "Videopc" not in file_path:
            output_string = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
            print("INFO: Media files do not follow naming of experiment videos. Writting Output to: %s " % (
                OPENFACE_OUTPUT_DIR + output_string))
        else:
            file_timestamp = re.compile(
                "(?<=Videopc.{1})(.*)(?=.avi)").split(file_path.split("/")[-1])[1][:-4]
            output_string = "%s_%s_%s_%s" % (
                file_timestamp[:2], file_timestamp[2:4], file_timestamp[4:8], file_timestamp[8:10])
            print("INFO: Output directory: %s" %
                  (OPENFACE_OUTPUT_DIR + output_string))

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

    cmd_list += OPENFACE_OUTPUT_FLAGS
    cmd_list += ['-out_dir', OPENFACE_OUTPUT_DIR]
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
    cmd_list += OPENFACE_OUTPUT_FLAGS
    cmd_list += ['-device', '0']
    cmd_list += ['-out_dir', OPENFACE_OUTPUT_DIR]
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

    group_id = None
    if directory:
        group_id = re.compile('SYNC/(.*)/(.*)/').split(media_files[0])[-3]
        media_files = [media_files[0] +
                       file for file in fetch_files_from_directory(media_files)]

    if media_type == MEDIA_TYPE_IMAGE:
        media_files = filter_files(media_files, VALID_IMAGE_TYPES)
    elif media_type == MEDIA_TYPE_VIDEO:
        media_files = filter_files(media_files, VALID_VIDEO_TYPES)

    if not media_files and media_type != MEDIA_TYPE_CAM:
        print("Error: No media files passed or no valid media files in directory")
        exit()

    OPENFACE_OUTPUT_DIR += format_output_string(media_files[0], group_id, directory)

    if media_type == MEDIA_TYPE_IMAGE:
        media_files = filter_files(media_files, VALID_VIDEO_TYPES)
        openface_img(media_files, write)
    elif media_type == MEDIA_TYPE_VIDEO:
        openface_vid(media_files, multi, write)
    elif media_type == MEDIA_TYPE_CAM:
        openface_cam(0, write)
