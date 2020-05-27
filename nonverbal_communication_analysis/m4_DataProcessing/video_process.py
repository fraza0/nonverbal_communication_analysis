import json
import os
import re
import threading
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from nonverbal_communication_analysis.environment import (DATASET_SYNC,
                                                          VALID_VIDEO_TYPES,
                                                          VIDEO_OUTPUT_DIR)
from nonverbal_communication_analysis.m0_Classes.Experiment import Experiment
from nonverbal_communication_analysis.utils import log


class Worker(threading.Thread):
    def __init__(self, worker_id, camera, video_cap, output_directory, verbose: bool = False, prettify: bool = False, display: bool = False):
        threading.Thread.__init__(self)
        self.worker_id = worker_id
        self.camera = camera
        self.video_cap = video_cap
        self.output_directory = output_directory

        self.verbose = verbose
        self.prettify = prettify
        self.display = display

        self.previous_frame = None
        self.current_frame = None
        self.frame_idx = 0

        self.cumulative_energy_frame = None

    def frame_energy(self, current_frame):
        frame_energy = np.zeros(current_frame.shape[:2], np.int)

        if self.previous_frame is None:
            return frame_energy

        frame_diff = cv2.absdiff(self.previous_frame, current_frame)
        gs_frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, frame_energy = cv2.threshold(
            gs_frame_diff, 127, 1, cv2.THRESH_BINARY)
        return frame_energy

    def save_frame(self, frame_data):
        output_directory = self.output_directory
        output_frame_file = output_directory / \
            ("frame_%.12d_processed.json" % (self.frame_idx))

        metrics_dict = dict()
        metrics_dict["energy"] = int(np.sum(frame_data))

        obj = dict()
        obj["frame"] = self.frame_idx
        obj["group"] = metrics_dict

        if self.prettify:
            json.dump(obj, open(output_frame_file, 'w'), indent=2)
        else:
            json.dump(obj, open(output_frame_file, 'w'))

    def run(self):
        cap = self.video_cap
        cumulative_energy_frame = self.cumulative_energy_frame

        while(cap.isOpened()):
            ret, current_frame = cap.read()
            if ret == True:
                if self.display:
                    cv2.imshow('Camera %s' % self.camera, current_frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

                if cumulative_energy_frame is None:
                    cumulative_energy_frame = np.zeros(
                        current_frame.shape[:2], np.int)

                frame_energy = self.frame_energy(
                    current_frame)
                cumulative_energy_frame += frame_energy

                self.save_frame(frame_energy)

                self.previous_frame = current_frame
                self.frame_idx += 1
            else:
                break

        cap.release()
        self.cumulative_energy_frame = cumulative_energy_frame
        # print("Worker %s finished with: %s" %
        #       (self.worker_id, np.unique(cumulative_energy_frame)))


class VideoProcess(object):

    def __init__(self, group_id: str, prettify: bool = False, verbose: bool = False):
        self.group_id = group_id
        self.prettify = prettify
        self.verbose = verbose

        self.experiment = Experiment(group_id)
        self.output_group_dir = VIDEO_OUTPUT_DIR / \
            group_id / (group_id + '_processed')
        os.makedirs(self.output_group_dir, exist_ok=True)
        json.dump(self.experiment.to_json(),
                  open(self.output_group_dir / (self.group_id + '.json'), 'w'))

    def save_output(self, metrics, task, camera):

        energy_heatmap_path = self.output_group_dir / task / \
            ('htmp_energy_pc%s.png' % camera)
        energy_heatmap = metrics['energy_htmp']

        if self.verbose:
            plt.imshow(energy_heatmap)
            plt.show()

        cv2.imwrite(str(energy_heatmap_path), energy_heatmap)

    def process(self, tasks_directories: dict, specific_frame: int = None, display: bool = False):

        if specific_frame is not None:
            log('WARN', 'Impossible to calculate energy in a single specific frame. Skipping video processing step')
        else:
            for task in tasks_directories:
                output_directory = self.output_group_dir / task.name
                os.makedirs(output_directory, exist_ok=True)
                if not output_directory.is_dir():
                    log('ERROR', 'Directory does not exist')

                task_video_files = [
                    x for x in task.iterdir() if x.suffix in VALID_VIDEO_TYPES]
                video_caps = {int(re.search(r'(?<=Videopc)(\d{1})(?=\d{12})', x.name).group(
                    0)): cv2.VideoCapture(str(x)) for x in task_video_files}

                thread_list = list()
                for cap_cam, video_cap in video_caps.items():
                    thread = Worker(cap_cam, cap_cam, video_cap, output_directory,
                                    verbose=self.verbose, prettify=self.prettify, display=display)
                    thread.start()
                    thread_list.append(thread)

                for thread in thread_list:
                    thread.join()

                metrics = dict()
                for thread in thread_list:
                    metrics['energy_htmp'] = thread.cumulative_energy_frame
                    self.save_output(metrics, task.name, thread.camera)
