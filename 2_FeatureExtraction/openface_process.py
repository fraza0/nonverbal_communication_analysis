import numpy as np
import pandas as pd
import json

from environment import (OPENFACE_OUTPUT_DIR)

columns_basic = [
    'frame',       # Number of the frame
    'face_id',     # Face id - No guarantee this is consistent across frames in case of FaceLandmarkVidMulti
    'timestamp',   # Timer of video being processed in seconds
    'success',     # Is the track successful - Is there a face in the frame or do we think we tracked it well
    'confidence'   # How confident is the tracker in current landmark detection estimation
]

columns_gaze = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z',  # Left eye
                'gaze_1_x', 'gaze_1_y', 'gaze_1_z',  # Right eye
                'gaze_angle_x', 'gaze_angle_y']      # Gaze angle in radians

# columns_
