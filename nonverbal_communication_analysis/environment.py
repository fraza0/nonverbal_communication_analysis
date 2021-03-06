import os
from datetime import datetime
from pathlib import Path
from nonverbal_communication_analysis.utils import person_identification_grid_rescaling

###########
# GENERAL #
###########

TESE_HOME = Path(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
CONFIG_FILE = TESE_HOME / 'config.yaml'

CAMERAS = ['pc1', 'pc2', 'pc3']
SIDEVIEW_CAMERA = 'pc1'

# Table dimensions to calculate scale factor.
# edge0: closer to the camera point
# edge1: furthest
# measurement in milimeters (mm)
TABLE_DIMENSIONS = {
    'pc1': {
        'edge0': 627,
        'edge1': 371
    },
    'pc2': {
        'edge0': 13900,
        'edge1': 10660
    },
    'pc3': {
        'edge0': 17678,
        'edge1': 12710
    }
}

SCALE_FACTOR = {
    'pc1': TABLE_DIMENSIONS['pc1']['edge0'] / TABLE_DIMENSIONS['pc1']['edge1'],
    'pc2': TABLE_DIMENSIONS['pc2']['edge0'] / TABLE_DIMENSIONS['pc2']['edge1'],
    'pc3': TABLE_DIMENSIONS['pc3']['edge0'] / TABLE_DIMENSIONS['pc3']['edge1']
}

SCALE_SUBJECTS = {
    'pc1': [2, 4],
    'pc2': [3, 4],
    'pc3': [1, 2]
}


#################
# DATASET PATHS #
#################

DATASET_DIR = TESE_HOME / "DATASET_DEP/"
DATASET_SYNC = DATASET_DIR / "SYNC/"
GROUPS_INFO_FILE = DATASET_DIR / 'groups_info.csv'

VIDEO_OUTPUT_DIR = DATASET_DIR / 'VIDEO/'
OPENPOSE_OUTPUT_DIR = DATASET_DIR / "OPENPOSE/"
OPENFACE_OUTPUT_DIR = DATASET_DIR / "OPENFACE/"
DENSEPOSE_OUTPUT_DIR = DATASET_DIR / "DENSEPOSE/"


#########################
# VIDEO SYNCHRONIZATION #
#########################

# VARIABLES
VALID_VIDEO_TYPES = ['.avi']
VALID_TIMESTAMP_FILES = ['.txt']
TIMESTAMP_THRESHOLD = 100  # ms
FOURCC = 'X264'  # CODEC
FRAME_SKIP = 100

CAM_ROI = {
    'pc1': {'xmin': 130, 'xmax': 520, 'ymin': 150, 'ymax': 400},
    'pc2': {'xmin': 80, 'xmax': 510, 'ymin': 110, 'ymax': 420},
    'pc3': {'xmin': 120, 'xmax': 620, 'ymin': 110, 'ymax': 460}
}

VIDEO_RESOLUTION = {
    'pc1': {'x': CAM_ROI['pc1']['xmax'] - CAM_ROI['pc1']['xmin'], 'y': CAM_ROI['pc1']['ymax'] - CAM_ROI['pc1']['ymin']},
    'pc2': {'x': CAM_ROI['pc2']['xmax'] - CAM_ROI['pc2']['xmin'], 'y': CAM_ROI['pc2']['ymax'] - CAM_ROI['pc2']['ymin']},
    'pc3': {'x': CAM_ROI['pc3']['xmax'] - CAM_ROI['pc3']['xmin'], 'y': CAM_ROI['pc3']['ymax'] - CAM_ROI['pc3']['ymin']}
}

# Actually not needed as I could just calculate the center using the ROI info,
# but I prefered to specify it, because of some cam specific offsets on the axis
PERSON_IDENTIFICATION_GRID = {
    'pc1': {'horizontal': {'x0': 130, 'x1': 520, 'y': 260}, 'vertical': {'x': 315, 'y0': 150, 'y1': 400}},
    'pc2': {'horizontal': {'x0': 80, 'x1': 510, 'y': 265}, 'vertical': {'x': 295, 'y0': 110, 'y1': 420}},
    'pc3': {'horizontal': {'x0': 120, 'x1': 620, 'y': 255}, 'vertical': {'x': 370, 'y0': 110, 'y1': 460}}
}

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
OPENFACE_HOME = TESE_HOME / "packages/openface"
OPENFACE_BUILD = OPENFACE_HOME / "build/bin/"
OPENFACE_FACE_LANDMARK_IMG = OPENFACE_BUILD / "FaceLandmarkImg"
OPENFACE_FACE_LANDMARK_VID = OPENFACE_BUILD / "FaceLandmarkVid"
OPENFACE_FACE_LANDMARK_VID_MULTI = OPENFACE_BUILD / "FaceLandmarkVidMulti"
OPENFACE_FEATURE_EXTRACTION = OPENFACE_BUILD / "FeatureExtraction"


NUM_EYE_LANDMARKS = 56
NUM_FACE_LANDMARKS = 68
NUM_NON_RIGID = 34

OPENFACE_OUTPUT_FLAGS = ['-pose', '-2Dfp', '-aus', '-gaze']

####################
# OPENFACE_PROCESS #
####################

# VARIABLES
VALID_OUTPUT_FILE_TYPES = ['.csv', '.json']
VALID_OUTPUT_IMG_TYPES = ['.png']

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

###################
# DATA FILTERING #
##################

QUADRANT_MIN = 0
QUADRANT_MAX = 1

OPENPOSE_KEY = 'OPENPOSE'

# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
OPENPOSE_KEYPOINT_MAP = {
    'NOSE': 0,
    'NECK': 1,
    'R_SHOULDER': 2,
    'R_ELBOW': 3,
    'R_WRIST': 4,
    'L_SHOULDER': 5,
    'L_ELBOW': 6,
    'L_WRIST': 7,
    'M_HIP': 8,
    'R_HIP': 9,
    'R_KNEE': 10,
    'R_ANKLE': 11,
    'L_HIP': 12,
    'L_KNEE': 13,
    'L_ANKLE': 14,
    'R_EYE': 15,
    'L_EYE': 16,
    'R_EAR': 17,
    'L_EAR': 18,
    'L_BIGTOE': 19,
    'L_SMALLTOE': 20,
    'L_HEEL': 21,
    'R_BIGTOE': 22,
    'R_SMALLTOE': 23,
    'R_HEEL': 24,
    'BACKGROUND': 25,
}

# Densepose keypoints adapted to BODY_25 (openpose keypoint mapping)
DENSEPOSE_KEYPOINT_MAP = {
    'nose': 0,
    'neck': 1,
    'right_shoulder': 2,
    'right_elbow': 3,
    'right_wrist': 4,
    'left_shoulder': 5,
    'left_elbow': 6,
    'left_wrist': 7,
    'mid_hip': 8,
    'right_hip': 9,
    'right_knee': 10,
    'right_ankle': 11,
    'left_hip': 12,
    'left_knee': 13,
    'left_ankle': 14,
    'right_eye': 15,
    'left_eye': 16,
    'right_ear': 17,
    'left_ear': 18,
}

KEYPOINT_CONFIDENCE_THRESHOLD = 0.10
SUBJECT_CONFIDENCE_THRESHOLD = 0.10
PEOPLE_FIELDS = ['person_id', 'pose_keypoints_2d', 'face_keypoints_2d']
RELEVANT_POSE_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 18]
RELEVANT_FACE_KEYPOINTS = range(0, 70)

VALID_SUBJECT_POSE_KEYPOINTS = [OPENPOSE_KEYPOINT_MAP['NECK'],
                                OPENPOSE_KEYPOINT_MAP['R_SHOULDER'], OPENPOSE_KEYPOINT_MAP['L_SHOULDER']]

SUBJECT_IDENTIFICATION_GRID = person_identification_grid_rescaling(
    PERSON_IDENTIFICATION_GRID, CAM_ROI, QUADRANT_MIN, QUADRANT_MAX)

CAMERA_ROOM_GEOMETRY = {
    'pc1': {
        1: SUBJECT_IDENTIFICATION_GRID['pc1'][3],
        2: SUBJECT_IDENTIFICATION_GRID['pc1'][1],
        3: SUBJECT_IDENTIFICATION_GRID['pc1'][4],
        4: SUBJECT_IDENTIFICATION_GRID['pc1'][2]
    },
    'pc2': {
        1: SUBJECT_IDENTIFICATION_GRID['pc2'][4],
        2: SUBJECT_IDENTIFICATION_GRID['pc2'][3],
        3: SUBJECT_IDENTIFICATION_GRID['pc2'][2],
        4: SUBJECT_IDENTIFICATION_GRID['pc2'][1]
    },
    'pc3': {
        1: SUBJECT_IDENTIFICATION_GRID['pc3'][1],
        2: SUBJECT_IDENTIFICATION_GRID['pc3'][2],
        3: SUBJECT_IDENTIFICATION_GRID['pc3'][3],
        4: SUBJECT_IDENTIFICATION_GRID['pc3'][4]
    }
}

OPENFACE_KEY = 'OPENFACE'
DENSEPOSE_KEY = 'DENSEPOSE'
OPENCV_KEY = 'OPENCV'
VIDEO_KEY = 'VIDEO'

FEATURE_AGGREGATE_DIR = 'FEATURE_DATA'

OPENPOSE_KEYPOINT_LINKS = {
    0: [1],
    1: [2, 5, 8],
    2: [3],
    3: [4],
    5: [6],
    6: [7],
    8: [9, 12]
}

# Camera list by user, ordered by choice preference
# based on view perspective. If 2nd preference is used,
# calculations, flip coordinates signals
SUBJECT_AXES = {
    1: {'x': ['pc2', 'pc3'],
        'y': ['pc1'],
        'z': ['pc2', 'pc3'],
        },
    2: {'x': ['pc2', 'pc3'],
        'y': ['pc1'],
        'z': ['pc2', 'pc3'],
        },
    3: {'x': ['pc3', 'pc2'],
        'y': ['pc1'],
        'z': ['pc3', 'pc2'],
        },
    4: {'x': ['pc3', 'pc2'],
        'y': ['pc1'],
        'z': ['pc3', 'pc2'],
        },
}

# CAMERAS_3D_AXES = {
#     'pc1': {
#         'x': 'y',
#         'y': 'z'
#     },
#     'pc2': {
#         'x': 'x',
#         'y': 'z'
#     },
#     'pc3': {
#         'x': 'x',
#         'y': 'z'
#     }
# }

SCALE_FACTORS = {
    'pc1': {
        'closer_edge': 627,
        'farther_edge': 371,
        'scale_factor': 627/371,
    },
    'pc2': {
        'closer_edge': 13900,
        'farther_edge': 10660,
        'scale_factor': 13900/10660,
    },
    'pc3': {
        'closer_edge': 17680,
        'farther_edge': 12710,
        'scale_factor': 17680/12710,
    },
}

PLOT_INTRAGROUP_DISTANCE = 'intragroup_distance'
PLOT_GROUP_ENERGY = 'energy'
PLOT_KEYPOINT_ENERGY = 'keypoint_energy'
PLOT_SUBJECT_OVERLAP = 'overlap'
PLOT_CENTER_INTERACTION = 'center_interaction'

PLOTS_LIB = {
    PLOT_INTRAGROUP_DISTANCE: OPENPOSE_KEY,
    PLOT_GROUP_ENERGY: OPENCV_KEY, 
    PLOT_KEYPOINT_ENERGY: OPENPOSE_KEY,
    PLOT_SUBJECT_OVERLAP: OPENPOSE_KEY,
    PLOT_CENTER_INTERACTION: OPENPOSE_KEY,
}

# Visualization
COLOR_MAP = {0: (0, 0, 0, 255), # Invalid/debug
             1: (255, 0, 0, 255), # Subject 1
             2: (0, 255, 255, 255), # 2
             3: (0, 255, 0, 255), # 3
             4: (0, 0, 255, 255), # 4
             'overlap': (252, 239, 93, 255), # Overlap
             'intragroup_distance': (163, 32, 219, 255), # Intragroup distance
             'center_interaction': (255, 179, 0, 255)} # Center interaction

PLOT_CANVAS_COLOR_ENCODING = {
        'pc1': 'tab:red',
        'pc1_splinefit': 'r-',
        'pc1_polyfit': 'r--',
        'pc1_rolling_meanfit': 'r:',

        'pc2': 'tab:green',
        'pc2_splinefit': 'g-',
        'pc2_polyfit': 'g--',
        'pc2_rolling_meanfit': 'g:',

        'pc3': 'tab:blue',
        'pc3_splinefit': 'b-',
        'pc3_polyfit': 'b--',
        'pc3_rolling_meanfit': 'b:',

        'energy': 'tab:olive',
        'energy_splinefit': 'y-',
        'energy_polyfit': 'y--',
        'energy_rolling_meanfit': 'y:',

        '1': 'tab:red',
        '1_splinefit': 'r-',
        '1_polyfit': 'r--',
        '1_rolling_meanfit': 'r:',

        '2': 'tab:cyan',
        '2_splinefit': 'c-',
        '2_polyfit': 'c--',
        '2_rolling_meanfit': 'c:',

        '3': 'tab:green',
        '3_splinefit': 'g-',
        '3_polyfit': 'g--',
        '3_rolling_meanfit': 'g:',

        '4': 'tab:blue',
        '4_splinefit': 'b-',
        '4_polyfit': 'b--',
        '4_rolling_meanfit': 'b:',
    }

ROLLING_WINDOW_SIZE = 1800 # 1800 = 30fps * 60seg <=> 60seg = 1min
LINESTYLES = ['-', '--', ':']
CMP_PLOT_COLORS = ['tab:blue', 'tab:orange', 'tab:green']

TASK_2_MARK = 9000
