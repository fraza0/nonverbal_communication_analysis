import os
os.chdir('/home/fraza0/Desktop/MEI/TESE/nonverbal_communication_analysis')
print("Working Directory:", os.getcwd())

import re
import json

import pandas as pd
import ipywidgets as widgets
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
from nonverbal_communication_analysis.environment import (DATASET_SYNC,
                                                          OPENPOSE_OUTPUT_DIR, OPENFACE_OUTPUT_DIR,
                                                          DENSEPOSE_OUTPUT_DIR,
                                                          VALID_OUTPUT_FILE_TYPES)

GROUPS = ['3CLC9VWR', '4ZAI4OPO', '0436INH4']
df_structure = pd.DataFrame(columns=['group', 'task', 'camera', 'frame', 'n_subjects', 'n_valids'])

class ExtractionPlotGenerator:
    
