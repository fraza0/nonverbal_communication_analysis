import argparse
import json

from utils import fetch_files_from_directory, filter_files, strided_split
from environment import OPENPOSE_OUTPUT_FILE_TYPE, OPENPOSE_OUT


class OpenposeOutputFile:
    def __init__(self, filename: str):
        self.filename = filename
        self.version = None
        self.people = None
        self.frame = int(filename.split('_')[-2])
        self.fetch_people_info()

    def fetch_people_info(self):
        file = open(self.filename, 'r')
        file_json = json.load(file)
        self.version = file_json['version']
        self.people = file_json['people']
        file.close()


class OpenPoseOutputAggregator:
    def __init__(self, op_output_directory: str):
        self.dir = op_output_directory
        self.version = None
        self.json_files = list()
        self.output_file = None
        self.fetch_files()
        self.json_output = dict()

    def fetch_files(self):
        _dir = self.dir
        if _dir[-1] != '/':
            _dir += '/'
            self.dir = _dir

        self.output_file = _dir[:-1].split('/')[-1]+"_agg.json"
        self.json_files = filter_files(fetch_files_from_directory(
            [_dir]), OPENPOSE_OUTPUT_FILE_TYPE, include='keypoints')
        self.json_files.sort()

    def aggregate(self):
        json_output = dict()
        output_file = open(self.dir+self.output_file, 'w+')
        
        for file in self.json_files:
            file = self.dir+file
            openpose_output_file = OpenposeOutputFile(file)
            json_output[openpose_output_file.frame] = {"people": openpose_output_file.people}

        self.json_output['frame'] = json_output
        output_file.write(json.dumps(self.json_output))
        output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='OpenPose output files aggregator')
    parser.add_argument('files_directory', type=str,
                        help='Openpose output directory')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    files_directory = args['files_directory']
    verbose = args['verbose']

    openpose_agg = OpenPoseOutputAggregator(files_directory)
    openpose_agg.aggregate()
