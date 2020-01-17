import os

# VARIABLES
VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
VALID_VIDEO_TYPES = ['.avi', '.wmv']

# PATHS
TESE_HOME = os.path.dirname(os.path.abspath(__file__))

## OpenFace
OPENFACE_HOME = TESE_HOME + "/packages/openface"
OPENFACE_BUILD = OPENFACE_HOME + "/build/bin/"
OPENFACE_FACE_LANDMARK_IMG = OPENFACE_BUILD + "FaceLandmarkImg"
OPENFACE_FACE_LANDMARK_VID = OPENFACE_BUILD + "FaceLandmarkVid"
OPENFACE_FACE_LANDMARK_VID_MULTI = OPENFACE_BUILD + "FaceLandmarkVidMulti"
OPENFACE_FEATURE_EXTRACTION = OPENFACE_BUILD + "FeatureExtraction"
OPENFACE_OUT = TESE_HOME + "/packages/openface_output"

## OpenPose
OPENPOSE_HOME = TESE_HOME + "/packages/openpose"
OPENPOSE_OUT = TESE_HOME + "/packages/openpose_output"
