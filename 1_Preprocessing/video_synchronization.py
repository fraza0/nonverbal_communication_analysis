from os.path import isfile, splitext
from datetime import datetime
import argparse
import cv2
import os
import errno
import cv2
import os
import errno

from environment import (VALID_VIDEO_TYPES, VALID_TIMESTAMP_FILES,
                         TIMESTAMP_THRESHOLD, DATASET_SYNC, FOURCC,
                         FRAME_SKIP, CAM_ROI)

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
                                                -t DATASET_DEP/Videos_LAB_PC1/Timestamppc118102019021136.txt
                                                -f DATASET_DEP/Videos_LAB_PC2/Videopc218102019021117.avi
                                                -t DATASET_DEP/Videos_LAB_PC2/Timestamppc218102019021117.txt
                                                -f DATASET_DEP/Videos_LAB_PC3/Videopc318102019021104.avi
                                                -t DATASET_DEP/Videos_LAB_PC3/Timestamppc318102019021104.txt
'''


class CameraVideo:
    """
    Camera Video Object: Unit to Synchronize

    title: video/camera identification. Visual purposes only
    video_path: video file path
    timestamp_path: timestamp file path
    output_path: output path
    """

    current_timestamp = 0

    def __init__(self, title: str, video_path: str, roi: dict, timestamp_path: str, output_path: str):
        self.title = title
        self.cap = cv2.VideoCapture(video_path)
        self.timestamps = open(timestamp_path, 'r')
        self.current_frame_idx = 0
        self.current_timestamp = 0
        self.markers = dict()
        self.ret = None
        self.frame = None
        self.roi = roi
        frame_width = int(roi['xmax']-roi['xmin'])
        frame_height = int(roi['ymax']-roi['ymin'])
        size = (frame_width, frame_height)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
            *FOURCC), fps, size)

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
    is_synced = all(align_by.current_timestamp - vid.current_timestamp <=
                    TIMESTAMP_THRESHOLD for vid in cap_list)
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
    for _ in range(_from, _until):
        vid.ret, vid.frame = vid.cap.read()
        vid.frame = vid.frame[roi['ymin']:roi['ymax'], roi['xmin']:roi['xmax']]
        vid.writer.write(vid.frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Syncronize Videos')
    parser.add_argument('-f', '--file', type=str, nargs=1, dest='video_files',
                        action='append', help='Video file path')
    parser.add_argument('-t' '--timestamp', type=str, nargs=1,
                        dest='timestamp_files', action='append', help='Media files')
    parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                        action='store_true')
    args = vars(parser.parse_args())

    if not args['video_files']:
        print("Error: No camera video files passed")
        exit()

    video_files = [vf[0] for vf in args['video_files']]
    timestamp_files = [tf[0] for tf in args['timestamp_files']] if args['timestamp_files'] is not None else [
        splitext(f)[0][::-1].replace('Video'[::-1], 'Timestamp'[::-1], 1)[::-1] + ".txt" for f in video_files]
    verbose = args['verbose']

    for file in video_files:
        _, file_extension = splitext(file)
        if file_extension not in VALID_VIDEO_TYPES:
            print("Not supported or invalid video file type (%s). File must be {%s}" % (
                file, VALID_VIDEO_TYPES))
            exit()

    for file in timestamp_files:
        _, file_extension = splitext(file)
        if file_extension not in VALID_TIMESTAMP_FILES:
            print("Not supported or invalid timestamp file type (%s). Check input files" %
                  VALID_TIMESTAMP_FILES)
            exit()

    if len(video_files) != 3 and len(timestamp_files) != 3:
        print("Specify only 3 video files (and corresponding timestamps - Optional: Default is searching for same file name)")
        exit()

    cap_list = list()
    out_dir = DATASET_SYNC+"/%s/" % video_files[0][-18:-8]

    if verbose:
        print("Saving to: ", out_dir)

    try:
        os.makedirs(out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    for i in range(len(video_files)):
        _id = str(i+1)
        vid = CameraVideo("VID"+_id, video_files[i], CAM_ROI[_id], timestamp_files[i], out_dir +
                          "sync_vid"+_id+"_"+splitext(video_files[i])[0].split('/')[-1]+".avi")
        cap_list.append(vid)

    if not all(vid.cap.isOpened() for vid in cap_list):
        print("Error opening video stream or file")
        exit()

    frame_count = 0
    marker_validator = {ord(str(i)): False for i in range(1, 9)}

    INIT_TIME = datetime.now()

    while(all(vid.cap.isOpened() for vid in cap_list)):

        alignment, align_by = timestamp_align(cap_list)
        to_align_list = cap_list if alignment is True else alignment

        for vid in to_align_list:
            # Read frame
            vid.ret, vid.frame = vid.cap.read()
            vid.current_frame_idx += 1
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
                # vid.timestamps.seek(vid.timestamps.tell())

        if key == ord('a'):                         # Jump Back
            for vid in cap_list:
                vid.current_frame_idx -= FRAME_SKIP
                vid.cap.set(cv2.CAP_PROP_POS_FRAMES, vid.current_frame_idx)
                # vid.timestamps.seek(vid.timestamps.tell())

        if key >= ord('1') and key <= ord('8'):
            print("Marker %s set" % chr(key))
            marker_validator[key] = True
            for vid in cap_list:
                vid.markers[key] = vid.current_frame_idx

        if all([vid.ret for vid in cap_list]):
            for vid in cap_list:
                if alignment == True and align_by is not None:  # Videos are in sync
                    roi = vid.roi
                    cv2.rectangle(
                        vid.frame, (roi['xmin'], roi['ymin']), (roi['xmax'], roi['ymax']), (0, 255, 0), 2)
                    cv2.imshow(vid.title, vid.frame)

            if key == ord('s'):                 # Save
                if verbose:
                    print("Start writting phase")

                for vid in cap_list:

                    valid_markers = [
                        marker for marker in marker_validator.items() if marker[1] == True]
                    if len(valid_markers) % 2 != 0:
                        print(
                            "Odd number of markers. Number of markers should be an even number.")
                        exit()

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
        vid.writer.release()

    cv2.destroyAllWindows()
