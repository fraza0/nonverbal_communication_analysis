import argparse
import pandas as pd
from os.path import splitext
from nonverbal_communication_analysis.utils import fetch_files_from_directory, filter_files, log
from nonverbal_communication_analysis.environment import VALID_OUTPUT_FILE_TYPES
from pandas_profiling import ProfileReport

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Openface Output Statistics')
    parser.add_argument('input_files', type=str, nargs='*', help='Input files')
    parser.add_argument('-dir', '--directory', action="store_true",
                        help='Media files path')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')
    parser.add_argument('-w', '--write', help="Write video with facial features",
                        action='store_true')

    args = vars(parser.parse_args())

    input_files = (args['input_files'] if 'input_files' in args else None)
    directory = args['directory']
    verbose = args['verbose']
    write = args['write']

    if directory:
        input_files = fetch_files_from_directory(input_files)

    input_files = filter_files(input_files, VALID_OUTPUT_FILE_TYPES)

    if not input_files:
        log('ERROR', 'No media files passed or no valid media files in directory')
        exit()

    for file in input_files:
        df = pd.read_csv(file)
        df_description = df.describe()
        if verbose:
            print(df_description)
        if write:
            filename, _ = splitext(file)
            df_description.to_csv(filename+"_description.csv")
            profile = ProfileReport(df, minimal=True)
            profile.to_file(output_file=filename+"_description.html")