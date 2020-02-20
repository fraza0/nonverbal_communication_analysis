import argparse
import json
import pandas as pd
from pandas.io.json import json_normalize
from openpose_output_aggregator import OpenPoseOutputAggregator


class OpenPoseProcess:
    def __init__(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='OpenPose output files aggregator')
    parser.add_argument('json_file', type=str,
                        help='Openpose input file')
    parser.add_argument('-dir', '--directory', action="store_true",
                        help="Openpose output directory")
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    input_file = args['json_file']
    directory = args['directory']
    verbose = args['verbose']

    if directory:
        openpose_agg = OpenPoseOutputAggregator(input_file)
        openpose_agg.aggregate()
        input_file = openpose_agg.dir + openpose_agg.output_file

    with open(input_file) as f:
        d = json.load(f)

    # input_file = open(input_file, 'r')
    json_file = json_normalize(data=d['frame'],
                               record_path='people',
                               meta=['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d', "pose_keypoints_3d", "face_keypoints_3d", "hand_left_keypoints_3d", "hand_right_keypoints_3d"])
    print(json_file.head(3))
    # df = pd.read_json(d)
    # print(df.head())
