import os
from datetime import datetime

TESE_HOME = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))

#########################
# VIDEO SYNCHRONIZATION #
#########################

# VARIABLES
VALID_VIDEO_TYPES = ['.avi', '.wmv', '.mp4']
VALID_TIMESTAMP_FILES = ['.txt']
TIMESTAMP_THRESHOLD = 100 #ms
FOURCC = 'X264'
FRAME_SKIP = 100

CAM_ROI = {
    '1': {'xmin': 130, 'xmax': 520, 'ymin': 150, 'ymax': 400},
    '2': {'xmin': 80, 'xmax': 510, 'ymin': 110, 'ymax': 420},
    '3': {'xmin': 70, 'xmax': 570, 'ymin': 110, 'ymax': 460}
}

# Actually not needed as I could just calculate the center using the ROI info,
# but I prefered to specify it, because of some cam specific offsets on the axis
PERSON_IDENTIFICATION_GRID = {
    '1': {'horizontal': {'x0': 130, 'x1': 520, 'y': 230}, 'vertical': {'x': 315, 'y0': 150, 'y1': 400}},
    '2': {'horizontal': {'x0': 80, 'x1': 510, 'y': 265}, 'vertical': {'x': 295, 'y0': 110, 'y1': 420}},
    '3': {'horizontal': {'x0': 70, 'x1': 570, 'y': 255}, 'vertical': {'x': 320, 'y0': 110, 'y1': 460}}
}


# PATHS
DATASET_DIR = TESE_HOME + "/DATASET_DEP/"
DATASET_PC1 = DATASET_DIR + "Videos_LAB_PC1/"
DATASET_PC2 = DATASET_DIR + "Videos_LAB_PC2/"
DATASET_PC3 = DATASET_DIR + "Videos_LAB_PC3/"

DATASET_SYNC = DATASET_DIR + "SYNC/"

####################
# OPENFACE_EXTRACT #
####################

# VARIABLES
VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
VALID_VIDEO_TYPES = ['.avi']

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
OPENFACE_OUTPUT_DIR = TESE_HOME + "/output/openface_output/extract/"

NUM_EYE_LANDMARKS = 56
NUM_FACE_LANDMARKS = 68
NUM_NON_RIGID = 34

OPENFACE_OUTPUT_FLAGS = ['-3Dfp', '-pdmparams',
                         '-pose', '-aus', '-gaze']


####################
# OPENFACE_PROCESS #
####################

# VARIABLES
VALID_OUTPUT_FILE_TYPES = ['.csv', '.json']

'''
Emotions enconding. AUs are ordered by relevance to emotion identification according to 
'''
EMOTIONS_ENCONDING = {
    'ANGER': ['AU23', 'AU07', 'AU17', 'AU04'],
    'FEAR': ['AU20', 'AU04', 'AU01', 'AU05'],
    'SADNESS': ['AU15', 'AU01', 'AU04', 'AU17'],
    'HAPPINESS': ['AU12', 'AU06', 'AU26', 'AU10'],
    'SURPRISE': ['AU02', 'AU01', 'AU05', 'AU26'],
    'DISGUST': ['AU09', 'AU07', 'AU04', 'AU17']
}

FRAME_THRESHOLD = 5
HEAD_MOV_VARIANCE_THRESHOLD = .3
