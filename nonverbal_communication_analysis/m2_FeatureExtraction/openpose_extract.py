import argparse
import os
import re
import subprocess
import time
from pathlib import Path

OPENPOSE_BIN = Path('openpose.bin')


def main(base_directory, output_directory, verbose):
    dir_path = os.getcwd()  # os.path.dirname(os.path.realpath(__file__))
    base_directory = Path(dir_path+'/'+base_directory)
    output_directory = Path(dir_path+'/'+output_directory)

    groups_dirs = [g for g in base_directory.iterdir() if g.is_dir()]

    for group_dir in groups_dirs:
        group = group_dir.name
        tasks = [t for t in group_dir.iterdir() if t.is_dir()]
        for task_dir in tasks:
            task = task_dir.name
            video_files = [v for v in task_dir.iterdir() if v.suffix == '.avi']
            for video_file in video_files:
                camera = re.search(r'(?<=Video)(pc\d{1})(?=\d{14})',
                                   video_file.name).group(0)
                t0 = time.time()
                cmd_args = [OPENPOSE_BIN, '--video', video_file, '--write-json', output_directory /
                            group/task/camera, '--face', '--keypoint_scale 3', '--display 0', '--render_pose 0']
                print(cmd_args)
                # subprocess.call(cmd_args)
                time.sleep(2)
                t1 = time.time()
                t = t1-t0
                print("%s %s %s: %s" % (group, task, camera, t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Openpose Data Extraction')
    parser.add_argument('base_path', type=str, help='Groups data directory')
    parser.add_argument('output_path', type=str, help='Output directory')
    parser.add_argument('-v', '--verbose',
                        action='store_true', help='Verbosity')

    args = vars(parser.parse_args())

    base_directory = args['base_path']
    output_directory = args['output_path']
    verbose = args['verbose']

    main(base_directory, output_directory, verbose)
