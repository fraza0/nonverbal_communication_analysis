import argparse
import json
import re
from os import makedirs
from pathlib import Path

from nonverbal_communication_analysis.environment import OPENPOSE_OUTPUT_DIR, OPENFACE_OUTPUT_DIR, DENSEPOSE_OUTPUT_DIR, VALID_OUTPUT_FILE_TYPES, OPENPOSE_KEY, OPENFACE_KEY, DENSEPOSE_KEY, FEATURE_AGGREGATE_DIR
from nonverbal_communication_analysis.m0_Classes.Experiment import Experiment, get_group_from_file_path
from nonverbal_communication_analysis.utils import log


class AggregateSubject(object):
    def __init__(self, subject_id):
        self.id = subject_id
        self.clean_data = dict()
        self.clean_pose_data = dict()
        self.clean_face_data = dict()
        self.processed_pose_data = dict()
        self.processed_face_data = dict()

    def to_json(self):
        obj = {
            "id": self.id,
            "raw": {
                "pose": self.clean_pose_data,
                "face": self.clean_pose_data,
            },
            "custom": {
                "pose": self.processed_pose_data,
                "face": self.processed_face_data
            }
        }

        return obj

    def __str__(self):
        return "AggregateSubject: {id: %s; clean_data: %s; processed_data: %s}" % (self.id, self.clean_data, self.processed_data)


class AggregateFrame(object):

    def __init__(self, frame_idx):
        self.frame = frame_idx
        self.is_raw_data_valid = None
        self.is_processed_data_valid = None
        self.group = dict()
        self.subjects = dict()

    def to_json(self):
        obj = {
            "frame": self.frame,
            "is_raw_data_valid": self.is_raw_data_valid,
            "is_custom_data_valid": self.is_processed_data_valid,
            "group": self.group,
            "subjects": [subject.to_json() for _, subject in self.subjects.items()]
        }

        return obj

    def __str__(self):
        return "AggregateFrame {index: %s; is_raw_data_valid: %s; is_processed_data_valid: %s; subjects: %s}" % (self.frame, self.is_raw_data_valid, self.is_processed_data_valid, self.subjects)


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
            self.openpose_data['cleaned'] = self.get_clean_data(
                openpose_group_directory)
            self.openpose_data['processed'] = self.get_processed_data(
                openpose_group_directory)

        # if openface:
        #     openface_group_directory = OPENFACE_OUTPUT_DIR / self.group_id
        #     experiment_data[OPENFACE_KEY] = self.get_experiment_data(
        #         openface_group_directory)
        #     self.openface_data['cleaned'] = self.get_clean_data(
        #         openface_group_directory)

        # if densepose:
        #     densepose_group_directory = DENSEPOSE_OUTPUT_DIR / self.group_id
        #     experiment_data[DENSEPOSE_KEY] = self.get_experiment_data(
        #         densepose_group_directory)
        #     self.densepose_data = self.get_densepose_data(
        #         densepose_group_directory)

        experiment_data_output = self.group_directory / \
            (self.group_id + '.json')

        experiment_data = experiment_data[OPENPOSE_KEY]
        if prettify:
            json.dump(experiment_data, open(
                experiment_data_output, 'w'), indent=2)
        else:
            json.dump(experiment_data, open(experiment_data_output, 'w'))

    def get_experiment_data(self, group_directory):
        # TODO: Change if processed group info changes. So far nothing is added to the experiment
        # JSON file in the processing step
        """Get experiment information

        Args:
            group_directory (str): Dataset Group directory path

        Returns:
            dict: Group basic information: ID and TYPE
        """
        clean_dir = [x for x in group_directory.iterdir()
                     if x.is_dir() and 'clean' in x.name][0]

        experiment_file = [x for x in clean_dir.iterdir()
                           if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES][0]

        experiment_file = json.load(open(experiment_file, 'r'))
        experiment_data = experiment_file['experiment']

        experiment = Experiment(experiment_data['id'])
        return experiment.to_json()

    def get_clean_data(self, group_directory):
        """Get framework clean data

        Args:
            group_directory ([type]): Framework output group directory path

        Returns:
            dict: Framework output files paths
        """
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

        return files

    def get_processed_data(self, group_directory):
        """Get processed data. This includes calculated metrics

        Args:
            group_directory ([type]): Group's processed data directory path

        Returns:
            dict: Processed data output files paths
        """
        processed_dir = [x for x in group_directory.iterdir()
                         if x.is_dir() and 'processed' in x.name][0]

        task_dirs = [x for x in processed_dir.iterdir()
                     if x.is_dir() and 'task' in x.name]

        files = dict()
        for task in task_dirs:
            task_frame_files = [x for x in task.iterdir()
                                if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES]
            for frame_file in task_frame_files:
                frame = int(re.search(r'(?<=_)(\d{12})(?=_)',
                                      frame_file.name).group(0))
                if task.name not in files:
                    files[task.name] = dict()
                files[task.name][frame] = frame_file

        return files

    def read_frame_data(self, agg_frame: AggregateFrame, frame_data: dict, key: str, framework: str = None, camera: str = None):
        frame_data_type = None

        agg_frame_subjetcs = agg_frame.subjects

        if 'is_raw_data_valid' in frame_data:
            agg_frame.is_raw_data_valid = frame_data['is_raw_data_valid']
            frame_data_type = 'raw'

        if 'is_processed_data_valid' in frame_data:
            agg_frame.is_processed_data_valid = frame_data['is_processed_data_valid']
            frame_data_type = 'processed'

        if 'group' in frame_data:
            agg_frame.group = frame_data['group']

        if 'subjects' in frame_data:
            frame_subjects = frame_data['subjects']
            for subject in frame_subjects:
                subject_id = subject['id']

                agg_subject = agg_frame_subjetcs[subject_id] if subject_id in agg_frame_subjetcs \
                    else AggregateSubject(subject_id)

                if camera is not None:
                    if framework not in agg_subject.clean_pose_data:
                        agg_subject.clean_pose_data[framework] = dict()

                    if camera not in agg_subject.clean_pose_data[framework]:
                        agg_subject.clean_pose_data[framework][camera] = dict()

                    if framework not in agg_subject.clean_face_data:
                        agg_subject.clean_face_data[framework] = dict()

                    if camera not in agg_subject.clean_face_data:
                        agg_subject.clean_face_data[framework][camera] = dict()

                if frame_data_type == 'raw':
                    if 'pose' in subject:
                        agg_subject.clean_pose_data[framework][camera].update(
                            subject['pose'][framework])
                    if 'face' in subject:
                        agg_subject.clean_face_data[framework][camera].update(
                            subject['face'][framework])
                elif frame_data_type == 'processed':
                    if 'pose' in subject:
                        agg_subject.processed_pose_data.update(
                            subject['pose'])
                    if 'face' in subject:
                        agg_subject.processed_face_data.update(
                            subject['face'])

                agg_frame.subjects[subject_id] = agg_subject

        return agg_frame

    # def merge_subject_dicts(self, subject_list1: list, subject_list2: list):
    #     """Merge subjects data from different frameworks

    #     Args:
    #         subject_list1 (list): Subject data
    #         subject_list2 (list): Subject data

    #     Returns:
    #         list: Merged subjects attributes
    #     """
    #     pose_key = 'pose'
    #     face_key = 'face'
    #     subjects_list = dict()
    #     for subject1 in subject_list1:
    #         for subject2 in subject_list2:
    #             if subject1['id'] == subject2['id']:
    #                 if face_key in subject1.keys() and face_key in subject2.keys():
    #                     subject1[face_key].update(subject2[face_key])
    #                 if pose_key in subject1.keys() and pose_key in subject2.keys():
    #                     subject1[pose_key].update(subject2[pose_key])
    #             subjects_list[subject1['id']] = subject1
    #             break
    #     return list(subjects_list.values())

    def aggregate(self, prettify: bool = False):
        """Aggregate framework clean output data
        Follow openpose frame list as openpose always outputs a file per frame

        Args:
            prettify (bool, optional): Pretty JSON print. Defaults to False.
        """
        openpose_data = self.openpose_data
        openface_data = self.openface_data

        # TODO: CHECK DENSEPOSE IMPLEMENTATION AFTER
        # IMPLEMENTING DENSEPOSE EXTRACTION
        # densepose_data = self.densepose_data

        cleaned_openpose = openpose_data['cleaned']
        processed_openpose = openpose_data['processed']

        for task in cleaned_openpose:
            output_frame_directory = self.group_directory / \
                FEATURE_AGGREGATE_DIR / task
            makedirs(output_frame_directory, exist_ok=True)

            processed_openpose_files = processed_openpose[task]
            for frame in processed_openpose_files:
                output_frame_file = output_frame_directory / \
                    ("%.12d" % frame + '.json')
                frame_file = AggregateFrame(frame)
                if frame != 3:
                    continue

                if self.verbose:
                    print("Processed Openpose")
                openpose_processed_frame_data = json.load(
                    open(processed_openpose_files[frame], 'r'))
                frame_file = self.read_frame_data(
                    frame_file, openpose_processed_frame_data, 'pose')

                for camera in cleaned_openpose[task]:
                    cleaned_openpose_files = cleaned_openpose[task][camera]
                    # Openpose
                    # Open and read each frame file into a joint structure
                    if self.verbose:
                        print("Clean OpenPose")
                    openpose_clean_frame_data = json.load(
                        open(cleaned_openpose_files[frame], 'r'))
                    frame_file = self.read_frame_data(
                        frame_file, openpose_clean_frame_data, 'pose', framework='openpose', camera=camera)

                    if self.verbose:
                        print(frame, frame_file)

                # if task in openface_data and camera in openface_data[task]:
                #     if frame in openface_data[task][camera]:
                #         openface_frame_file = json.load(
                #             open(openface_data[task][camera][frame], 'r'))
                #         openface_subjects = openface_frame_file['subjects']
                #         frame_file['subjects'] = self.merge_subject_dicts(
                #             frame_file['subjects'], openface_subjects)

                if prettify:
                    json.dump(frame_file.to_json(), open(
                        output_frame_file, 'w'), indent=2)
                else:
                    json.dump(frame_file.to_json(),
                              open(output_frame_file, 'w'))


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
