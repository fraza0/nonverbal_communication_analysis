from os.path import isfile, splitext
import argparse
import cv2

from environment import VALID_VIDEO_TYPES, VALID_TIMESTAMP_FILES, TIMESTAMP_THRESHOLD

# python Preprocessing/video_synchronization.py -f DATASET_DEP/Videos_LAB_PC1/Videopc118102019021136.avi -t DATASET_DEP/Videos_LAB_PC1/Timestamppc118102019021136.txt -f DATASET_DEP/Videos_LAB_PC2/Videopc218102019021117.avi -t DATASET_DEP/Videos_LAB_PC2/Timestamppc218102019021117.txt -f DATASET_DEP/Videos_LAB_PC3/Videopc318102019021104.avi -t DATASET_DEP/Videos_LAB_PC3/Timestamppc318102019021104.txt


class CameraVideo:
    '''
    '''
    current_timestamp = 0

    def __init__(self, title: str, video_path: str, timestamp_path: str):
        self.title = title
        self.cap = cv2.VideoCapture(video_path)
        self.timestamps = open(timestamp_path, 'r')
        self.current_timestamp = 0
        self.ret = None
        self.frame = None

    def __str__(self):
        print("%s: %s" % (self.title, self.current_timestamp))


def timestamp_align(cap_list: list):
    '''
    Determine which video(s) that need to be aligned

    Returns: True if they are aligned
             List of videos that need to be aligned
    '''

    align_by = max(cap_list, key=lambda x: x.current_timestamp)
    is_synced = all(align_by.current_timestamp - vid.current_timestamp <=
                    TIMESTAMP_THRESHOLD for vid in cap_list)
    if is_synced:
        return True

    # align_by = max(cap_list, key=lambda x: x.current_timestamp)
    align_ts = align_by.current_timestamp

    to_align = list()

    for vid in cap_list:
        if align_ts - vid.current_timestamp > TIMESTAMP_THRESHOLD:
            to_align.append(vid)

    # print([vid.current_timestamp for vid in cap_list])
    # print([vid.title for vid in to_align])
    return to_align


parser = argparse.ArgumentParser(
    description='Syncronize Videos')
parser.add_argument('-f', '--file', type=str, nargs=1, dest='video_files',
                    action='append', help='Video file path')
parser.add_argument('-t' '--timestamp', type=str, nargs=1,
                    dest='timestamp_files', action='append', help='Media files')
parser.add_argument('-v', '--verbose', help='Whether or not responses should be printed',
                    action='store_true')
# TODO: Add read all files directly from directory. Timestamp files must have same file_name as video file name
# parser.add_argument('-dir', '--directory', action="store_true", help='Media files path')
args = vars(parser.parse_args())

video_files = [vf[0] for vf in args['video_files']]
timestamp_files = [tf[0] for tf in args['timestamp_files']] if args['timestamp_files'] is not None else [
    splitext(f[0])[::-1].replace('Video'[::-1], 'Timestamp'[::-1])[::-1] + ".txt" for f in video_files]
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
    print("Only 3 video files and corresponding timestamps available")
    exit()

vid1 = CameraVideo("PC1", video_files[0], timestamp_files[0])
vid2 = CameraVideo("PC2", video_files[1], timestamp_files[1])
vid3 = CameraVideo("PC3", video_files[2], timestamp_files[2])

cap_list = [vid1, vid2, vid3]

if (vid1.cap.isOpened() == False or vid2.cap.isOpened() == False or vid3.cap.isOpened() == False):
    print("Error opening video stream or file")
    exit()

cap = (vid1.cap.isOpened() and vid1.cap.isOpened() and vid1.cap.isOpened())
# CAP_PROP_FRAME_COUNT

fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
writer_list = [cv2.VideoWriter('', fourcc, vid.cap.get(cv2.CAP_PROP_FPS), (vid.cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), vid.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) for vid in cap_list]


while(cap):

    alignment = timestamp_align(cap_list)
    to_align_list = cap_list if alignment is True else alignment

    for vid in to_align_list:
        # Read frame
        vid.ret, vid.frame = vid.cap.read()
        # Update current_timestamp
        vid.current_timestamp = int(vid.timestamps.readline())

    key = cv2.waitKey(25) & 0xff

    if key == ord('p'):                             # Pause Video
        while cv2.waitKey(-1) & 0xFF != ord('p'):   # Resume Video
            pass

    if all([vid.ret for vid in cap_list]):

        for vid in cap_list:
            cv2.imshow(vid.title, vid.frame)

        if key == ord('q'):
            break

    else:
        break

for vid in cap_list:
    vid.cap.release()

cv2.destroyAllWindows()
