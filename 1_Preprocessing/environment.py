import os

TESE_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# VARIABLES
VALID_VIDEO_TYPES = ['.avi', '.wmv', '.mp4']
VALID_TIMESTAMP_FILES = ['.txt']
TIMESTAMP_THRESHOLD = 100
FOURCC = 'MJPG'  # 'X264' # Error -- https://github.com/skvark/opencv-python/issues/81, https://github.com/skvark/opencv-python/issues/100
FRAME_SKIP = 100

CAM_ROI = {
    '1': {'xmin': 130, 'xmax': 520, 'ymin': 150, 'ymax': 400},
    '2': {'xmin': 80, 'xmax': 510, 'ymin': 110, 'ymax': 420},
    '3': {'xmin': 80, 'xmax': 570, 'ymin': 110, 'ymax': 460}
}


# PATHS
DATASET_DIR = TESE_HOME + "/DATASET_DEP"
DATASET_PC1 = DATASET_DIR + "/" + "Videos_LAB_PC1"
DATASET_PC2 = DATASET_DIR + "/" + "Videos_LAB_PC2"
DATASET_PC3 = DATASET_DIR + "/" + "Videos_LAB_PC3"

DATASET_SYNC = DATASET_DIR + "/" + "SYNC"
