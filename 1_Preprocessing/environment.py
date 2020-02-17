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
DATASET_DIR = TESE_HOME + "/DATASET_DEP"
DATASET_PC1 = DATASET_DIR + "/" + "Videos_LAB_PC1"
DATASET_PC2 = DATASET_DIR + "/" + "Videos_LAB_PC2"
DATASET_PC3 = DATASET_DIR + "/" + "Videos_LAB_PC3"

DATASET_SYNC = DATASET_DIR + "/" + "SYNC"
