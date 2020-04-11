import argparse
import json
import re
from os import makedirs
from pathlib import Path

from nonverbal_communication_analysis.environment import OPENPOSE_OUTPUT_DIR, OPENFACE_OUTPUT_DIR, DENSEPOSE_OUTPUT_DIR, VALID_OUTPUT_FILE_TYPES, OPENPOSE_KEY, OPENFACE_KEY, DENSEPOSE_KEY
from nonverbal_communication_analysis.m0_Classes.Experiment import Experiment, get_group_from_file_path
from nonverbal_communication_analysis.utils import log


class SubjectDataAggregator:
    def __init__(self, group_directory: str, openpose: bool = False, openface: bool = False, densepose: bool = False, prettify: bool = False, verbose: bool = False):
        self.group_directory = Path(group_directory)
        self.group_id = get_group_from_file_path(group_directory)
        self.prettify = prettify
        self.verbose = verbose
        self.openpose_data = dict()
        self.openface_data = dict()
        self.densepose_data = dict()
        experiment_data = dict()

        if openpose:
            openpose_group_directory = OPENPOSE_OUTPUT_DIR / self.group_id
            experiment_data[OPENPOSE_KEY] = self.get_experiment_data(
                openpose_group_directory)
            self.openpose_data = self.get_data(
                openpose_group_directory)

        if openface:
            openface_group_directory = OPENFACE_OUTPUT_DIR / self.group_id
            experiment_data[OPENFACE_KEY] = self.get_experiment_data(
                openface_group_directory)
            self.openface_data = self.get_data(
                openface_group_directory)

        if densepose:
            densepose_group_directory = DENSEPOSE_OUTPUT_DIR / self.group_id
            experiment_data[DENSEPOSE_KEY] = self.get_experiment_data(
                densepose_group_directory)
            self.densepose_data = self.get_densepose_data(
                densepose_group_directory)

        experiment_data_output = self.group_directory / \
            (self.group_id + '.json')

        experiment_data = experiment_data[OPENPOSE_KEY]
        if prettify:
            json.dump(experiment_data, open(
                experiment_data_output, 'w'), indent=2)
        else:
            json.dump(experiment_data, open(experiment_data_output, 'w'))

    def get_experiment_data(self, group_directory):
        clean_dir = [x for x in group_directory.iterdir()
                     if x.is_dir() and 'clean' in x.name][0]

        experiment_file = [x for x in clean_dir.iterdir()
                           if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES][0]

        experiment_file = json.load(open(experiment_file, 'r'))
        experiment_data = experiment_file['experiment']

        experiment = Experiment(experiment_data['id'])
        return experiment.to_json()

    def get_data(self, group_directory):
        clean_dir = [x for x in group_directory.iterdir()
                     if x.is_dir() and 'clean' in x.name][0]

        task_dirs = [x for x in clean_dir.iterdir()
                     if x.is_dir() and 'task' in x.name]

        files = dict()
        for task in task_dirs:
            task_camera_directory = [x for x in task.iterdir()
                                     if x.is_dir() and 'pc' in x.name]

            camera_files = dict()
            for camera_directory in task_camera_directory:
                camera_id = camera_directory.name
                camera_frame_files = [x for x in camera_directory.iterdir()
                                      if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES]
                for frame_file in camera_frame_files:
                    frame = int(re.search(r'(?<=_)(\d{12})(?=_)',
                                          frame_file.name).group(0))
                    if camera_directory.name not in camera_files:
                        camera_files[camera_id] = dict()

                    camera_files[camera_id][frame] = frame_file
            files[task.name] = camera_files
        # print(files['task_1']['pc1'].keys())
        return files

    def get_densepose_data(self, densepose_group_directory):
        print("Densepose Directory: ", densepose_group_directory)
        return dict()

    def merge_subject_dicts(self, subject_list1: list, subject_list2: dict):
        pose_key = 'pose'
        face_key = 'face'
        subjects_list = dict()
        for subject1 in subject_list1:
            for subject2 in subject_list2:
                if subject1['id'] == subject2['id']:
                    if face_key in subject1.keys() and face_key in subject2.keys():
                        subject1[face_key].update(subject2[face_key])
                    if pose_key in subject1.keys() and pose_key in subject2.keys():
                        subject1[pose_key].update(subject2[pose_key])
                subjects_list[subject1['id']] = subject1
                break
        return list(subjects_list.values())

    def aggregate(self, prettify: bool = False):
        # Follow openpose frame list as openpose always outputs a file per frame
        openpose_data = self.openpose_data
        openface_data = self.openface_data
        densepose_data = self.densepose_data

        for task in openpose_data:
            for camera in openpose_data[task]:
                for frame, file in openpose_data[task][camera].items():
                    frame_file = dict()
                    frame_file['frame'] = frame
                    frame_file['subjects'] = dict()

                    openpose_frame_file = json.load(open(file, 'r'))
                    openpose_subjects = openpose_frame_file['subjects']
                    frame_file['subjects'] = openpose_subjects

                    if task in openface_data and camera in openface_data[task]:
                        if frame in openface_data[task][camera]:
                            openface_frame_file = json.load(
                                open(openface_data[task][camera][frame], 'r'))
                            openface_subjects = openface_frame_file['subjects']
                            frame_file['subjects'] = self.merge_subject_dicts(
                                frame_file['subjects'], openface_subjects)

                    # TODO: CHECK DENSEPOSE IMPLEMENTATION AFTER DENSEPOSE EXTRACTION
                    if task in densepose_data and camera in densepose_data[task]:
                        if frame in densepose_data[task][camera]:
                            densepose_frame_file = json.load(
                                open(densepose_data[task][camera][frame], 'r'))
                            densepose_subjects = densepose_frame_file['subjects']
                            frame_file['subjects'] = self.merge_subject_dicts(
                                frame_file['subjects'], densepose_subjects['subjects'])

                    output_frame_directory = self.group_directory / \
                        'FEATURE_DATA' / task / camera
                    makedirs(output_frame_directory, exist_ok=True)
                    output_frame_file = output_frame_directory / \
                        ("%.12d" % frame + '.json')

                    if prettify:
                        json.dump(frame_file, open(
                            output_frame_file, 'w'), indent=2)
                    else:
                        json.dump(frame_file, open(output_frame_file, 'w'))


def main(group_directory: str, specific_frame: int = None, specific_task: int = None, openpose: bool = False, openface: bool = False, densepose: bool = False, prettify: bool = False, verbose: bool = False):
    group_directory = Path(group_directory)
    aggregator = SubjectDataAggregator(group_directory, openpose=openpose,
                                       openface=openface, densepose=densepose,
                                       verbose=verbose)
    aggregator.aggregate(prettify=prettify)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize data')
    parser.add_argument('group_directory', type=str,
                        help='Group video directory')
    parser.add_argument('-t', '--task', dest="specific_task", type=int, choices=[1, 2],
                        help='Specify Task frame')
    parser.add_argument('-f', '--frame', dest='specific_frame', type=int,
                        help='Process Specific frame')
    parser.add_argument('-op', '--openpose',
                        help='Overlay openpose data', action='store_true')
    parser.add_argument('-dp', '--densepose',
                        help='Overlay densepose data', action='store_true')
    parser.add_argument('-of', '--openface',
                        help='Overlay openface data', action='store_true')
    parser.add_argument('-p', '--prettify', dest="prettify", action='store_true',
                        help='Output pretty printed JSON')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')

    args = vars(parser.parse_args())

    group_directory = args['group_directory']
    specific_task = args['specific_task']
    specific_frame = args['specific_frame']
    openpose = args['openpose']
    openface = args['openface']
    densepose = args['densepose']
    prettify = args['prettify']
    verbose = args['verbose']

    main(group_directory, specific_frame=specific_frame, specific_task=specific_task,
         openpose=openpose, openface=openface, densepose=densepose, prettify=prettify, verbose=verbose)
