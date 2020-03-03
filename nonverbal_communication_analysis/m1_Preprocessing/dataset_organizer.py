import argparse
import pandas as pd
import re
from nonverbal_communication_analysis.utils import fetch_files_from_directory, filter_files
from nonverbal_communication_analysis.environment import VALID_VIDEO_TYPES, DATASET_DIR


def timestamp_from_filename(name: str):
    full_timestamp = re.compile('Videopc.(.*?).avi').split(name)
    return full_timestamp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Match Videos from different cameras of the experiment based on timestamp values')
    parser.add_argument('labels_file', type=str,
                        help='Groups labels file path')
    parser.add_argument('pc1', type=str, help='PC1 Camera files')
    parser.add_argument('pc2', type=str, help='PC1 Camera files')
    parser.add_argument('pc3', type=str, help='PC1 Camera files')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')
    args = vars(parser.parse_args())

    labels_file = args['labels_file']
    pc_dir = [args['pc1'], args['pc2'], args['pc3']]
    verbose = args['verbose']

    labels = pd.read_csv(labels_file)

    for index, row in labels.iterrows():
        path = DATASET_DIR + row['Group ID'] + "_" + row['Conflict Type']
        date_split = row['Date'].split()
        date_list = ''.join(date_split[0].split('/'))
        time_list = ''.join(date_split[1].split(':'))
        full_group_timestamp = date_list + time_list
        group_timestamp = full_group_timestamp[:8]
        files = list()
        print(full_group_timestamp)

        for pc in pc_dir:
            files = filter_files(fetch_files_from_directory(
                [pc]), valid_types=VALID_VIDEO_TYPES, include=group_timestamp)
            print(files)
            # candidates = min(files, key=lambda x:abs(int(group_timestamp)-int(timestamp_from_filename(x))))

        if index == 2:
            exit()