from os import listdir
from os.path import isfile, join, splitext
from math import ceil
import numpy as np
import logging

# print("UTILS HERE")

# # create logger
# logger = logging.getLogger('simple_example')
# logger.setLevel(logging.DEBUG)

# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

# # create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # add formatter to ch
# ch.setFormatter(formatter)

# # add ch to logger
# logger.addHandler(ch)

# # 'application' code
# logger.debug('debug message')
# logger.info('info message')
# logger.warn('warn message')
# logger.error('error message')
# logger.critical('critical message')

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
