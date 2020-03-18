from os import listdir
from os.path import isfile, join, splitext
from math import ceil
import numpy as np
import logging
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point


def log(_type: str, msg: str):
    _type = _type.upper()
    if _type == 'INFO':
        logging.info(msg)
    elif _type == 'WARN' or _type == 'WARNING':
        logging.warn(msg)
    elif _type == 'ERROR':
        logging.error(msg)


def print_assertion_error(_obj, _type):
    """
    Return Assertion Error Message
    """

    return "Parameter %s type must be %s" % (_obj, _type)


def fetch_files_from_directory(dir_path: list):
    """Fetch files from given directory

    Arguments:
        dir_path {list} -- path to directory

    Returns:
        list -- list of file paths
    """
    files = list()
    for _dir in dir_path:
        files = [f for f in listdir(_dir) if isfile(join(_dir, f))]
    return files


def filter_files(files: list, valid_types: list, verbose: bool = False, include: str = None):
    """Filter files based on file type

    Args:
        files (list): files list
        valid_types (list): valid file types
        verbose (bool, optional): verbose. Defaults to False.
        include (str, optional): filename must include given terms

    Returns:
        [type]: valid files
    """
    valid_files = list()

    for _file in files:
        file_name, file_extension = splitext(_file)
        if file_extension in valid_types:
            if include is not None:
                if include in file_name:
                    valid_files.append(_file)
            else:
                valid_files.append(_file)
        else:
            if verbose:
                print("WARN: Not supported or invalid file type (%s). File must be {%s}" % (
                    _file, valid_types))
    return valid_files


def index_marks(nrows, chunk_size):
    return range(chunk_size, ceil(nrows / chunk_size) * chunk_size, chunk_size)


def strided_split(df, chunk_size):
    indices = index_marks(df.shape[0], chunk_size)
    return np.split(df, indices)


def set_quadrants(center, _min=-1, _max=1):
    x = center[0]
    y = center[1]
    q1 = Polygon([(center), (_min, y), (_min, _min), (x, _min)])
    q2 = Polygon([(center), (x, _min), (_max, _min), (_max, y)])
    q3 = Polygon([(center), (x, _max), (_min, _max), (_min, y)])
    q4 = Polygon([(center), (_max, y), (_max, _max), (x, _max)])

    quadrants = {
        1: q1,
        2: q2,
        3: q3,
        4: q4
    }
    return quadrants


def person_identification_grid_rescaling(identification_grid, roi_grid, a: int = -1, b: int = 1):
    subject_identification_grid = dict()
    for camera, grid in identification_grid.items():
        x_min, x_max = roi_grid[camera]['xmin'], roi_grid[camera]['xmax']
        y_min, y_max = roi_grid[camera]['ymin'], roi_grid[camera]['ymax']
        x_max_min = x_max - x_min
        y_max_min = y_max - y_min
        x_grid = grid['vertical']['x']
        y_grid = grid['horizontal']['y']

        x_rescaled = a + (((x_grid - x_min)*(b-a)) / x_max_min)
        y_rescaled = a + (((y_grid - y_min)*(b-a)) / y_max_min)

        center = (x_rescaled, y_rescaled)
        subject_identification_grid[camera] = set_quadrants(center)

    return subject_identification_grid
