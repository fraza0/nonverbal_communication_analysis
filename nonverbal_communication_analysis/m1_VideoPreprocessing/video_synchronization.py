from os.path import isfile, splitext
from datetime import datetime
import argparse
import cv2
import os
import errno
from pathlib import Path
import numpy as np
from itertools import combinations

from nonverbal_communication_analysis.environment import (VALID_VIDEO_TYPES, VALID_TIMESTAMP_FILES,
                                                          TIMESTAMP_THRESHOLD, DATASET_SYNC, FOURCC,
                                                          FRAME_SKIP, CAM_ROI, PERSON_IDENTIFICATION_GRID)

from nonverbal_communication_analysis.utils import log, fetch_files_from_directory, filter_files
from nonverbal_communication_analysis.m0_Classes.Experiment import Experiment

'''
Video synchronization and cut for DEP experiment Dataset.

This script allows video synchronization from 3 different data sources (easily adapted to more).
Needs videos timestamp files for the synchronization process.

3 Windows are opened as soon as the videos are in sync.
After that, use 'a' and 'd' keys to navigate through the video.
Use 1-8 keys to set markers. Only 4 clips are allowed.
Use 's' key to save.
Use 'q' key to quit without saving.


Example command:
$ python 1_Preprocessing/video_synchronization.py -f DATASET_DEP/Videos_LAB_PC1/Videopc118102019021136.avi
                                                -f DATASET_DEP/Videos_LAB_PC2/Videopc218102019021117.avi
                                                -f DATASET_DEP/Videos_LAB_PC3/Videopc318102019021104.avi
OR
$ python 1_Preprocessing/video_synchronization.py -d DATASET_DEP/CC/3CLC9VWR/
'''


class CameraVideo:
    """
    Camera Video Object: Unit to Synchronize

    title: video/camera identification. Visual purposes only
    video_path: video file path
    timestamp_path: timestamp file path
    output_path: output path
    """

    def __init__(self, title: str, video_path: str, roi: dict, grid: dict, timestamp_path: str, output_paths: str, filename: str):
        self.title = title
        self.cap = cv2.VideoCapture(video_path)
        self.timestamps = open(timestamp_path, 'r')
        self.current_frame_idx = 0
        self.current_timestamp = 0
        self.init_synced_frame = 0
        self.markers = dict()
        self.ret = None
        self.frame = None
        self.grid = grid
        self.roi = roi
        frame_width = int(roi['xmax']-roi['xmin'])
        frame_height = int(roi['ymax']-roi['ymin'])
        size = (frame_width, frame_height)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.writer_task1 = cv2.VideoWriter(output_paths[0]+filename, cv2.VideoWriter_fourcc(
            *FOURCC), fps, size)
        self.writer_task2 = cv2.VideoWriter(output_paths[1]+filename, cv2.VideoWriter_fourcc(
            *FOURCC), fps, size)
        self.task_separator = 0

    def __str__(self):
        print("%s: %s" % (self.title, self.current_timestamp))


def timestamp_align(cap_list: list):
    """ Determine which video(s) need to be aligned

    cap_list: list of cv2 captures

    Arguments:
        cap_list {list} -- [description]

    Returns:
        Tuple:
        * (True, None) - If they are aligned. Ignore first iteration when frame and timestamp is read.
        * (True, True) - If they are aligned.
        * (to_align: list, align_by: CameraVideo) - List of videos that need to be aligned, CameraVideo object of reference video
    """
    align_by = max(cap_list, key=lambda x: x.current_timestamp)
    combs = combinations([vid.current_timestamp for vid in cap_list], 2)
    is_synced = max([abs(ts1-ts2)for ts1, ts2 in combs]) <= TIMESTAMP_THRESHOLD

    if is_synced == True:
        if align_by.current_timestamp == 0:
            return True, None
        return True, True

    align_ts = align_by.current_timestamp

    to_align = list()

    for vid in cap_list:
        if align_ts - vid.current_timestamp > TIMESTAMP_THRESHOLD:
            to_align.append(vid)

    return to_align, align_by


def cut_from_until(vid, _from: int, _until: int):
    """ Write video from frame f_init until f_end

    Arguments:
        vid {[type]} -- [description]
        _from {int} -- [description]
        _until {int} -- [description]
    """

    vid.cap.set(cv2.CAP_PROP_POS_FRAMES, _from)

    roi = vid.roi
    for frame_idx in range(_from, _until):
        vid.ret, vid.frame = vid.cap.read()
        vid.frame = vid.frame[roi['ymin']:roi['ymax'], roi['xmin']:roi['xmax']]
        if frame_idx < vid.task_separator:
            vid.writer_task1.write(vid.frame)
        else:
            vid.writer_task2.write(vid.frame)


def main(video_files: list, verbose: bool = False):
    group_id = list(filter(None, video_files[0].split('/')))[-1]
    experiment = Experiment(group_id)

    video_files = [directory + filename for filename in filter_files(fetch_files_from_directory(
        [directory]), valid_types=VALID_VIDEO_TYPES)]

    video_files.sort()
    timestamp_files = [splitext(f)[0][::-1].replace('Video'[::-1],
                                                    'Timestamp'[::-1], 1)[::-1] + ".txt" for f in video_files]

    if len(video_files) != 3 and len(timestamp_files) != 3:
        log('ERROR', 'Specify only 3 video files (and corresponding timestamps - Optional: Default is searching for same file name)')
        exit()

    out_dir_base = '%s/%s' % (DATASET_SYNC, str(
        Path(video_files[0]).parent).split('/')[-1])
    out_dirs = list()

    for n in range(experiment._n_tasks):
        out_dirs.append('%s/%s_%s/' % (out_dir_base, 'task', n+1))

    if verbose:
        print('Saving to: ', out_dirs)

    try:
        for _dir in out_dirs:
            os.makedirs(_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    cap_list = list()
    for i in range(len(video_files)):
        _id = str(i+1)
        vid = CameraVideo("VID"+_id, video_files[i], CAM_ROI['pc'+_id], PERSON_IDENTIFICATION_GRID['pc'+_id],
                          timestamp_files[i], out_dirs, splitext(video_files[i])[0].split('/')[-1]+"_sync.avi")
        cap_list.append(vid)

    if not all(vid.cap.isOpened() for vid in cap_list):
        log('ERROR', 'Error opening video stream or file')
        exit()

    marker_validator = {ord(str(i)): False for i in range(1, 9)}

    precision_step = 10

    while(all(vid.cap.isOpened() for vid in cap_list)):

        alignment, align_by = timestamp_align(cap_list)
        to_align_list = cap_list if alignment is True else alignment

        for vid in to_align_list:
            # Read frame
            vid.ret, vid.frame = vid.cap.read()
            vid.current_frame_idx += 1
            # print(vid.current_frame_idx)
            # Update current_timestamp
            file_ts = vid.timestamps.readline()
            vid.current_timestamp = int(file_ts) if file_ts is not '' else -1

        key = cv2.waitKey(25) & 0xff

        # TODO: While paused, allow to set markers
        if key == 0x20:                             # Pause Video
            while cv2.waitKey(-1) & 0xFF != 0x20:   # Resume Video
                pass

        if key == ord('d'):                         # Skip
            for vid in cap_list:
                vid.current_frame_idx += FRAME_SKIP
                vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)

        if key == ord('a'):                         # Jump Back
            for vid in cap_list:
                vid.current_frame_idx -= FRAME_SKIP
                if vid.current_frame_idx < vid.init_synced_frame:
                    vid.current_frame_idx = vid.init_synced_frame

                vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)

        if key == ord('f'):                         # Jump Back
            for vid in cap_list:
                vid.current_frame_idx -= FRAME_SKIP*10
                if vid.current_frame_idx < vid.init_synced_frame:
                    vid.current_frame_idx = vid.init_synced_frame

                vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)

        # if key == ord('s'): # Set sync point
        #     print("SET SYNC POINT")

        if key == ord('u'):
            vid = cap_list[0]
            vid.current_frame_idx -= precision_step
            vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)

        if key == ord('i'):
            vid = cap_list[0]
            vid.current_frame_idx += precision_step
            vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)

        if key == ord('j'):
            vid = cap_list[1]
            vid.current_frame_idx -= precision_step
            vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)

        if key == ord('k'):
            vid = cap_list[1]
            vid.current_frame_idx += precision_step
            vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)

        if key == ord('n'):
            vid = cap_list[2]
            vid.current_frame_idx -= precision_step
            vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)

        if key == ord('m'):
            vid = cap_list[2]
            vid.current_frame_idx += precision_step
            vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)

        if key >= ord('1') and key <= ord('8'):
            print("Marker %s set" % chr(key))
            marker_validator[key] = True
            for vid in cap_list:
                vid.markers[key] = vid.current_frame_idx

        if key == ord('t'):
            print("Task Separator set")
            for vid in cap_list:
                vid.task_separator = vid.current_frame_idx

        if all([vid.ret for vid in cap_list]):
            if alignment == True and align_by is not None:  # Videos are in sync
                for vid in cap_list:
                    if vid.init_synced_frame == 0:
                        vid.init_synced_frame = vid.current_frame_idx

                    roi = vid.roi
                    grid = vid.grid
                    grid_horitonzal_axis = grid['horizontal']
                    grid_vertical_axis = grid['vertical']
                    cv2.rectangle(
                        vid.frame, (roi['xmin'], roi['ymin']), (roi['xmax'], roi['ymax']), (0, 255, 0), 2)
                    cv2.line(
                        vid.frame, (grid_horitonzal_axis['x0'], grid_horitonzal_axis['y']), (grid_horitonzal_axis['x1'], grid_horitonzal_axis['y']), (0, 0, 255), 1)
                    cv2.line(
                        vid.frame, (grid_vertical_axis['x'], grid_vertical_axis['y0']), (grid_vertical_axis['x'], grid_vertical_axis['y1']), (0, 0, 255), 1)
                    cv2.imshow(vid.title, vid.frame)

            if key == ord('g'):                 # Save
                if verbose:
                    print("Start writting phase")

                for vid in cap_list:
                    valid_markers = [
                        marker for marker in marker_validator.items() if marker[1] == True]
                    if len(valid_markers) % 2 != 0:
                        log('ERROR', 'Odd number of markers. Number of markers should be an even number.')
                        break

                    first_experience_markers = list(
                        vid.markers.keys())[:2]
                    second_experience_markers = list(
                        vid.markers.keys())[2:4]
                    third_experience_markers = list(
                        vid.markers.keys())[4:6]
                    fourth_experience_markers = list(
                        vid.markers.keys())[6:8]

                    for i in range(0, (len(vid.markers.keys())+1)-1, 2):
                        task_markers = list(vid.markers.keys())[i:i+2]

                        if verbose:
                            print("Vid %s Saving Task" % (vid.title))
                        cut_from_until(
                            vid, vid.markers[task_markers[0]], vid.markers[task_markers[1]])

                break

            if key == ord('q'):
                break

        else:
            break

    for vid in cap_list:
        vid.cap.release()
        vid.writer_task1.release()
        vid.writer_task2.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Syncronize Videos')
    parser.add_argument('-d', '--directory', type=str, required=True,
                        dest='group_files', help='Group Video files path')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')
    args = vars(parser.parse_args())

    directory = args['group_files']
    verbose = args['verbose']
    video_files = [directory]

    main(video_files, verbose)
