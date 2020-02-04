import os
from datetime import datetime

TESE_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

####################
# OPENFACE_EXTRACT #
####################

# VARIABLES
VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
VALID_VIDEO_TYPES = ['.avi', '.wmv', '.mp4']

# PATHS
# OpenFace
# FaceLandmarkImg executable is for individual image analysis (can either contain one or more faces)
# FaceLandmarkVidMulti is intended for sequence analysis that contain multiple faces
# FeatureExtraction executable is used for sequence analysis that contain a single face
OPENFACE_HOME = TESE_HOME + "/packages/openface"
OPENFACE_BUILD = OPENFACE_HOME + "/build/bin/"
OPENFACE_FACE_LANDMARK_IMG = OPENFACE_BUILD + "FaceLandmarkImg"
OPENFACE_FACE_LANDMARK_VID = OPENFACE_BUILD + "FaceLandmarkVid"
OPENFACE_FACE_LANDMARK_VID_MULTI = OPENFACE_BUILD + "FaceLandmarkVidMulti"
OPENFACE_FEATURE_EXTRACTION = OPENFACE_BUILD + "FeatureExtraction"
OPENFACE_OUTPUT_DIR = TESE_HOME + "/output/openface_output"

NUM_EYE_LANDMARKS = 56
NUM_FACE_LANDMARKS = 68
NUM_NON_RIGID = 34

# OpenFace Output Commands:
# -au_static    : static models only rely on a single image to make an estimate of AU
#                   presence or intensity, while dynamic ones calibrate to a person by performing person
#                   normalization in the video, they also attempt to correct for over and under prediction of AUs.
#                   By default OpenFace uses static models on images and dynamic models on image sequences and videos.
#                   DOC Ref: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units#static-vs-dynamic
# -2Dfp         : output 2D landmarks in pixels
# -3Dfp         : output 3D landmarks in milimeters
# -pdmparams    : output rigid and non-rigid shape parameters:
#                   * Rigid shape parameters describe the placement of the face in the image
#                       (scaling, rotation, and translation);
#                   * Non-rigid shape parameters on the other hand describe the deformation
#                       of the face due to identity or expression (wider or taller faces, smiles, blinks etc.)
# -pose         : output head pose (location and rotation)
# -aus          : output the Facial Action Units
# -gaze         : output gaze and related features (2D and 3D locations of eye landmarks)
# -hogalign     : output extracted HOG feaure file
# -simalign     : output similarity aligned images of the tracked faces
# -nobadaligned : if outputting similarity aligned images, do not output from frames where detection failed
#                   or is unreliable (thus saving some disk space)
# -tracked      : output video with detected landmarks


OPENFACE_OUTPUT_COMMANDS = ['-3Dfp', '-pdmparams',
                            '-pose', '-aus', '-gaze']


####################
# OPENFACE_PROCESS #
####################

# VARIABLES
VALID_FILE_TYPES = ['.csv']


# OpenPose
OPENPOSE_HOME = TESE_HOME + "/packages/openpose"
OPENPOSE_OUT = "output/openpose_output"

# OpenCV
OPENCV_HOME = TESE_HOME + "/packages/opencv"
OPENCV_OUT = "output/opencv_output"
