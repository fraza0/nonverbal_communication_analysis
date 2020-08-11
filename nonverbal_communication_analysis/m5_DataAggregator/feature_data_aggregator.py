import argparse
import csv
import json
import re
import shutil
from os import makedirs
from pathlib import Path

from nonverbal_communication_analysis.environment import (
    DENSEPOSE_KEY, DENSEPOSE_OUTPUT_DIR, FEATURE_AGGREGATE_DIR, OPENFACE_KEY,
    OPENFACE_OUTPUT_DIR, OPENPOSE_KEY, OPENPOSE_OUTPUT_DIR,
    VALID_OUTPUT_FILE_TYPES, VALID_OUTPUT_IMG_TYPES, VIDEO_KEY, OPENCV_KEY,
    VIDEO_OUTPUT_DIR, PLOT_INTRAGROUP_DISTANCE, PLOT_GROUP_ENERGY, PLOT_SUBJECT_OVERLAP,
    PLOT_KEYPOINT_ENERGY, PLOT_CENTER_INTERACTION)
from nonverbal_communication_analysis.m0_Classes.Experiment import (
    Experiment, get_group_from_file_path)
from nonverbal_communication_analysis.utils import log


class AggregateSubject(object):
    def __init__(self, subject_id):
        self.id = subject_id
        self.clean_pose_data = dict()  # openpose, densepose
        self.clean_face_data = dict()  # openpose, openface
        self.processed_pose_data = dict()  # openpose, densepose
        self.processed_face_data = dict()  # openpose, openface
        self.metrics = dict()

    def to_json(self):
        obj = {
            "id": self.id,
            "raw": {
                "pose": self.clean_pose_data,
                "face": self.clean_face_data,
            },
            "enhanced": {
                "pose": self.processed_pose_data,
                "face": self.processed_face_data
            },
            "metrics": self.metrics
        }

        return obj

    def __str__(self):

        obj = {
            "id": self.id,
            "raw": {
                "pose": self.clean_pose_data.keys(),
                "face": self.clean_face_data.keys(),
            },
            "enhanced": {
                "pose": self.processed_pose_data.keys(),
                "face": self.processed_face_data.keys()
            },
            "metrics": self.metrics
        }

        return "AggregateSubject: %s" % (obj)


class AggregateFrame(object):
    """
        frame,
        is_raw_data_valid,
        is_enhanced_data_valid,
        group: {
            framework: {intragroup_distance},
        }
        subjects: {
            id,
            raw: {
                pose: {
                    openpose: {cams},
                    densepose: {cams}
                },
                face: {
                    openpose: {cams},
                    openface: {cams}
                }
            }
            enhanced: {
                pose: {
                    openpose: {cams},
                    densepose: {cams}
                },
                face: {
                    openpose: {cams},
                    openface: {cams}
                }
            }
            metrics: {
                framework: {cams}
            }
        }
    """

    def __init__(self, frame_idx):
        self.frame_idx = frame_idx
        self.is_raw_data_valid = None
        self.is_processed_data_valid = None
        self.group = dict()
        self.subjects = dict()

    def to_json(self):
        obj = {
            "frame": self.frame_idx,
            "is_raw_data_valid": self.is_raw_data_valid,
            "is_enhanced_data_valid": self.is_processed_data_valid,
            "group": self.group,
            "subjects": [subject.to_json() for _, subject in self.subjects.items()]
        }

        return obj

    def __str__(self):

        obj = {
            "index": self.frame_idx,
            "is_raw_data_valid": self.is_raw_data_valid,
            "is_enhanced_data_valid": self.is_processed_data_valid,
            "group": self.group,
            "subjects": [str(subject) for _, subject in self.subjects.items()]
        }

        return "AggregateFrame %s" % obj


class SubjectDataAggregator:

    def __init__(self, group_directory: str, openpose: bool = False, openface: bool = False, densepose: bool = False, video: bool = False, specific_task: int = None, specific_frame: int = None, prettify: bool = False, verbose: bool = False):
        self.group_directory = Path(group_directory)
        self.group_id = get_group_from_file_path(group_directory)
        self.prettify = prettify
        self.verbose = verbose

        self.specific_task = specific_task
        self.specific_frame = specific_frame

        self.framework_being_processed = None

        self.openpose_data = dict()
        self.openface_data = dict()
        self.densepose_data = dict()
        self.video_data = dict()

        experiment_data = dict()

        self.reset_files = True

        if openpose:
            openpose_group_directory = OPENPOSE_OUTPUT_DIR / self.group_id
            experiment_data[OPENPOSE_KEY] = self.get_experiment_data(
                openpose_group_directory)
            self.openpose_data['cleaned'] = self.get_clean_data(
                openpose_group_directory)
            self.openpose_data['processed'] = self.get_processed_data(
                openpose_group_directory)

            experiment_data = experiment_data[OPENPOSE_KEY]

        if openface:
            openface_group_directory = OPENFACE_OUTPUT_DIR / self.group_id
            experiment_data[OPENFACE_KEY] = self.get_experiment_data(
                openface_group_directory)
            self.openface_data['cleaned'] = self.get_clean_data(
                openface_group_directory)
            self.openface_data['processed'] = self.get_processed_data(
                openface_group_directory)

        # if densepose:
        #     densepose_group_directory = DENSEPOSE_OUTPUT_DIR / self.group_id
        #     experiment_data[DENSEPOSE_KEY] = self.get_experiment_data(
        #         densepose_group_directory)
        #     self.densepose_data = self.get_densepose_data(
        #         densepose_group_directory)

        if video:
            video_data_group_directory = VIDEO_OUTPUT_DIR / self.group_id
            experiment_data[VIDEO_KEY] = self.get_experiment_data(
                video_data_group_directory, dir_type='processed')
            self.video_data['processed'] = self.get_processed_data(
                video_data_group_directory)

            processed_path = video_data_group_directory / \
                (self.group_id + '_processed')
            tasks = [x for x in processed_path.iterdir() if x.is_dir()
                     and 'task' in x.name]
            for task in tasks:
                self.video_data['heatmap'] = {task.name: [x for x in task.iterdir() if not x.is_dir()
                                                          and x.suffix in VALID_OUTPUT_IMG_TYPES]}

        experiment_data_output = self.group_directory / \
            (self.group_id + '.json')

        if prettify:
            json.dump(experiment_data, open(
                experiment_data_output, 'w'), indent=2)
        else:
            json.dump(experiment_data, open(experiment_data_output, 'w'))

    def get_experiment_data(self, group_directory, dir_type: str = 'clean'):
        # TODO: Change if processed group info changes. So far nothing is added to the experiment
        # JSON file in the processing step
        """Get experiment information

        Args:
            group_directory (str): Dataset Group directory path

        Returns:
            dict: Group basic information: ID and TYPE
        """
        clean_dir = [x for x in group_directory.iterdir()
                     if x.is_dir() and dir_type in x.name][0]

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
            task_camera_dirs = [x for x in task.iterdir()
                                if x.is_dir() and 'pc' in x.name]

            task_frame_files = list()
            if task_camera_dirs:
                task_frame_files = dict()
                for camera_dir in task_camera_dirs:
                    task_frame_files[camera_dir.name] = [x for x in camera_dir.iterdir()
                                                         if not x.is_dir() and x.suffix in VALID_OUTPUT_FILE_TYPES]

                for camera, frame_files in task_frame_files.items():
                    for frame_file in frame_files:
                        frame = int(re.search(r'(?<=_)(\d{12})(?=_)',
                                              frame_file.name).group(0))
                        if task.name not in files:
                            files[task.name] = dict()

                        if frame not in files[task.name]:
                            files[task.name][frame] = dict()

                        files[task.name][frame][camera] = frame_file

            else:
                task_frame_files = [x for x in task.iterdir()
                                    if not x.is_dir()
                                    and x.suffix in VALID_OUTPUT_FILE_TYPES]

                for frame_file in task_frame_files:
                    frame = int(re.search(r'(?<=_)(\d{12})(?=_)',
                                          frame_file.name).group(0))
                    if task.name not in files:
                        files[task.name] = dict()
                    files[task.name][frame] = frame_file

        return files

    def parse_data(self, agg_subject_data, frame_data, camera: str = None):
        subject_framework_metrics = None
        for framework, data in frame_data.items():

            if framework == 'metrics':
                subject_framework_metrics = {
                    self.framework_being_processed.lower(): data
                }
                continue

            if framework not in agg_subject_data:
                agg_subject_data[framework] = dict()

            if camera:
                if camera not in agg_subject_data[framework]:
                    agg_subject_data[framework][camera] = dict()
                agg_subject_data[framework][camera].update(
                    data)
            else:
                agg_subject_data[framework].update(
                    data)

        return agg_subject_data, subject_framework_metrics

    def read_frame_data(self, agg_frame: AggregateFrame, frame_data: dict, camera: str = None, frame_data_type: str = None):
        agg_frame_subjects = agg_frame.subjects

        if 'is_raw_data_valid' in frame_data:
            if agg_frame.is_raw_data_valid is not False:
                agg_frame.is_raw_data_valid = frame_data['is_raw_data_valid']
            frame_data_type = 'raw'

        if agg_frame.is_processed_data_valid is None and 'is_enhanced_data_valid' in frame_data:
            agg_frame.is_processed_data_valid = frame_data['is_enhanced_data_valid']
            frame_data_type = 'processed'

        if 'group' in frame_data:
            agg_frame.group.update({
                self.framework_being_processed.lower(): frame_data['group']
            })

        if 'subjects' in frame_data:
            frame_subjects = frame_data['subjects']
            for subject in frame_subjects:

                # ID
                if 'id' not in subject:
                    log('ERROR', 'Invalid Subject. Cannot parse subject ID.')
                subject_id = subject['id']
                agg_subject = agg_frame_subjects[subject_id] if subject_id in agg_frame_subjects \
                    else AggregateSubject(subject_id)

                # FACE
                if 'face' in subject:
                    subject_face_data = subject['face']
                    if frame_data_type == 'raw':
                        agg_subject.clean_face_data, _ = self.parse_data(
                            agg_subject.clean_face_data, subject_face_data, camera=camera)

                    if frame_data_type == 'processed':
                        processed_face_data, processed_face_metrics = self.parse_data(
                            agg_subject.processed_face_data, subject_face_data, camera=camera)

                        agg_subject.processed_face_data = processed_face_data
                        if processed_face_metrics:
                            agg_subject.metrics.update(processed_face_metrics)

                # POSE
                if 'pose' in subject:
                    subject_pose_data = subject['pose']

                    if frame_data_type == 'raw':
                        agg_subject.clean_pose_data, _ = self.parse_data(
                            agg_subject.clean_pose_data, subject_pose_data, camera)

                    if frame_data_type == 'processed':
                        processed_pose_data, processed_pose_metrics = self.parse_data(
                            agg_subject.processed_pose_data, subject_pose_data)

                        agg_subject.processed_pose_data = processed_pose_data
                        if processed_pose_metrics:
                            agg_subject.metrics.update(processed_pose_metrics)

                agg_frame.subjects[subject_id] = agg_subject

        return agg_frame

    def aggregate(self, prettify: bool = False):
        """Aggregate framework clean output data
        Follow openpose frame list as openpose always outputs a file per frame

        Args:
            prettify (bool, optional): Pretty JSON print. Defaults to False.
        """
        openpose_data = self.openpose_data
        openface_data = self.openface_data
        video_data = self.video_data

        # TODO: CHECK DENSEPOSE IMPLEMENTATION AFTER
        # IMPLEMENTING DENSEPOSE EXTRACTION
        # densepose_data = self.densepose_data

        if not openpose_data:
            log('ERROR', 'Nothing to Aggregate. Use -op -of and -dp to include openpose, openface and densepose data. The use of Openpose data is mandatory.')

        if openface_data:
            cleaned_openface = openface_data['cleaned']
            processed_openface = openface_data['processed']

        cleaned_openpose = openpose_data['cleaned']
        processed_openpose = openpose_data['processed']
        tasks = cleaned_openpose.keys()

        if self.specific_task is not None:
            tasks = [task for task in tasks if str(self.specific_task) in task]

        for task in tasks:
            self.reset_files = True
            output_frame_directory = self.group_directory / \
                FEATURE_AGGREGATE_DIR / task
            makedirs(output_frame_directory, exist_ok=True)

            processed_openpose_files = processed_openpose[task]

            if self.specific_frame is not None:
                processed_openpose_files = {
                    self.specific_frame: processed_openpose_files[self.specific_frame]}

            for frame_idx in processed_openpose_files:
                output_frame_file = output_frame_directory / \
                    ("%.12d" % frame_idx + '.json')
                aggregate_frame = AggregateFrame(frame_idx)

                # OPENPOSE
                if self.verbose:
                    print("Cleaned OpenPose")
                self.framework_being_processed = OPENPOSE_KEY
                for camera in cleaned_openpose[task]:
                    cleaned_openpose_files = cleaned_openpose[task][camera]
                    openpose_clean_frame_data = json.load(
                        open(cleaned_openpose_files[frame_idx], 'r'))
                    aggregate_frame = self.read_frame_data(aggregate_frame,
                                                           openpose_clean_frame_data,
                                                           camera=camera,
                                                           frame_data_type='raw')

                if self.verbose:
                    print("Processed Openpose")
                openpose_processed_frame_data = json.load(
                    open(processed_openpose_files[frame_idx], 'r'))

                aggregate_frame = self.read_frame_data(aggregate_frame,
                                                       openpose_processed_frame_data,
                                                       frame_data_type='processed')

                # OPENFACE
                if openface_data:
                    if self.verbose:
                        print("Cleaned OpenFace")
                    self.framework_being_processed = OPENFACE_KEY
                    cleaned_task_openface = cleaned_openface[task]
                    for camera in cleaned_task_openface:
                        cleaned_openface_files = cleaned_task_openface[camera]
                        if frame_idx in cleaned_openface_files:
                            openface_clean_frame_data = json.load(
                                open(cleaned_openface_files[frame_idx], 'r'))
                            aggregate_frame = self.read_frame_data(aggregate_frame,
                                                                   openface_clean_frame_data,
                                                                   camera=camera,
                                                                   frame_data_type='raw')

                    if self.verbose:
                        print("Processed Openface")

                    processed_task_openface = processed_openface[task]
                    if frame_idx in processed_task_openface:
                        processed_task_frame = processed_task_openface[frame_idx]
                        for camera, frame_file in processed_task_frame.items():
                            openface_processed_frame_data = json.load(
                                open(frame_file, 'r'))
                            aggregate_frame = self.read_frame_data(aggregate_frame,
                                                                   openface_processed_frame_data,
                                                                   camera=camera,
                                                                   frame_data_type='processed')

                # DENSEPOSE

                # VIDEO
                if video_data:
                    self.framework_being_processed = OPENCV_KEY
                    processed_video_data = video_data['processed']
                    if task in processed_video_data:
                        processed_video_data_task = processed_video_data[task]
                        if frame_idx in processed_video_data_task:
                            processed_video_data_frame = processed_video_data_task[frame_idx]
                            for camera, frame_file in processed_video_data_frame.items():
                                video_data_processed_frame_data = json.load(
                                    open(frame_file, 'r'))
                                aggregate_frame = self.read_frame_data(aggregate_frame,
                                                                       video_data_processed_frame_data,
                                                                       camera=camera,
                                                                       frame_data_type='processed')

                self.plot_generator(aggregate_frame, output_frame_directory)

                if prettify:
                    json.dump(aggregate_frame.to_json(), open(
                        output_frame_file, 'w'), indent=2)
                else:
                    json.dump(aggregate_frame.to_json(),
                              open(output_frame_file, 'w'))

            if video_data:
                video_data_heatmaps = video_data['heatmap']
                if task in video_data_heatmaps:
                    video_data_heatmaps_task = video_data_heatmaps[task]
                    for file_name in video_data_heatmaps_task:
                        shutil.copy(file_name, output_frame_directory)

    def plot_generator(self, aggregate_frame, output_directory):
        output_directory = output_directory / 'PLOTS'
        makedirs(output_directory, exist_ok=True)
        frame_idx = aggregate_frame.frame_idx

        # Group metrics
        # Intragroup Distance
        for lib in aggregate_frame.group:
            lib_group_data = aggregate_frame.group[lib]
            if PLOT_INTRAGROUP_DISTANCE in lib_group_data:
                if self.reset_files:
                    file_ig = open(output_directory /
                                   (lib+'_'+PLOT_INTRAGROUP_DISTANCE+'.csv'), 'w')
                    file_ig.flush()
                    self.file_ig = csv.writer(file_ig)
                    self.file_ig.writerow(
                        ['frame', 'camera', PLOT_INTRAGROUP_DISTANCE])

                for camera, value in lib_group_data[PLOT_INTRAGROUP_DISTANCE].items():
                    intragroup_entry = [frame_idx, camera, value['area']]
                    self.file_ig.writerow(intragroup_entry)

            # Group Energy
            if PLOT_GROUP_ENERGY in lib_group_data:
                if self.reset_files:
                    file_energy = open(
                        output_directory/(lib+'_'+PLOT_GROUP_ENERGY+'.csv'), 'w')
                    file_energy.flush()
                    self.file_energy = csv.writer(file_energy)
                    self.file_energy.writerow(['frame', PLOT_GROUP_ENERGY])
                energy_value = lib_group_data[PLOT_GROUP_ENERGY]
                energy_entry = [frame_idx, energy_value]
                self.file_energy.writerow(energy_entry)

        # Subject metrics
        for subject_id, subject in aggregate_frame.subjects.items():
            for lib in subject.metrics:
                lib_subjects_data = subject.metrics[lib]

                if self.reset_files:
                    file_overlap = open(
                        output_directory/(lib+'_'+PLOT_SUBJECT_OVERLAP+'.csv'), 'w')
                    file_overlap.flush()
                    self.file_overlap = csv.writer(file_overlap)
                    self.file_overlap.writerow(['frame', 'camera',
                                                'subject', PLOT_SUBJECT_OVERLAP])
                    file_center_interaction = open(
                        output_directory/(lib+'_'+PLOT_CENTER_INTERACTION+'.csv'), 'w')
                    file_center_interaction.flush()
                    self.file_center_interaction = csv.writer(
                        file_center_interaction)
                    self.file_center_interaction.writerow(['frame', 'subject',
                                                           PLOT_CENTER_INTERACTION])

                    file_keypoint_energy = open(
                        output_directory/(lib+'_'+PLOT_KEYPOINT_ENERGY+'.csv'), 'w')
                    file_keypoint_energy.flush()
                    self.file_keypoint_energy = csv.writer(
                        file_keypoint_energy)
                    self.file_keypoint_energy.writerow(['frame', 'camera',
                                                        'subject', PLOT_KEYPOINT_ENERGY])

                # overlap
                if PLOT_SUBJECT_OVERLAP in lib_subjects_data:
                    for camera, value in lib_subjects_data['overlap'].items():
                        overlap_entry = [frame_idx,  camera,
                                         subject_id, value['area']]
                        self.file_overlap.writerow(overlap_entry)

                # Center interaction
                if PLOT_CENTER_INTERACTION in lib_subjects_data:
                    center_interaction_value = lib_subjects_data['center_interaction']['value']
                    center_interaction_entry = [frame_idx, subject_id,
                                                center_interaction_value]
                    self.file_center_interaction.writerow(
                        center_interaction_entry)

                if PLOT_KEYPOINT_ENERGY in lib_subjects_data:
                    for camera, value in lib_subjects_data['keypoint_energy'].items():
                        keypoint_energy_entry = [frame_idx,  camera,
                                                 subject_id, value]
                        self.file_keypoint_energy.writerow(
                            keypoint_energy_entry)

        self.reset_files = False

        return None


def main(group_directory: str, specific_frame: int = None, specific_task: int = None, openpose: bool = False, openface: bool = False, densepose: bool = False, video: bool = False, prettify: bool = False, verbose: bool = False):
    group_directory = Path(group_directory)
    aggregator = SubjectDataAggregator(group_directory, openpose=openpose,
                                       openface=openface, densepose=densepose,
                                       video=video, specific_task=specific_task,
                                       specific_frame=specific_frame, verbose=verbose)
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
    parser.add_argument('-vid', '--video',
                        help='Overlay video energy data', action='store_true')
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
    video = args['video']
    prettify = args['prettify']
    verbose = args['verbose']

    main(group_directory, specific_frame=specific_frame, specific_task=specific_task,
         openpose=openpose, openface=openface, densepose=densepose, video=video, prettify=prettify, verbose=verbose)
