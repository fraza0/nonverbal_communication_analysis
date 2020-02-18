from os import listdir
from os.path import isfile, join, splitext
from math import ceil
import numpy as np


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
        if file_extension in valid_types and include in file_name:
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