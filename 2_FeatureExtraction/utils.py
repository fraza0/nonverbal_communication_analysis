from os import listdir
from os.path import isfile, join, splitext


def print_assertion_error(_obj, _type):
    """
    Return Assertion Error Message
    """

    return "Parameter %s type must be %s" % (_obj, _type)


def fetch_files_from_directory(dir_path: str):
    """Fetch files from given directory

    Arguments:
        dir_path {str} -- path to directory

    Returns:
        list -- list of file paths
    """
    for _file in dir_path:
        files = [f for f in listdir(_file) if isfile(join(_file, f))]
    return files


def filter_files(files: list, valid_types: list, verbose: bool = False):
    valid_files = list()

    for _file in files:
        _, file_extension = splitext(_file)
        if file_extension in valid_types:
            valid_files.append(_file)
        else:
            if verbose:
                print("WARN: Not supported or invalid file type (%s). File must be {%s}" % (
                    _file, valid_types))
    return valid_files
