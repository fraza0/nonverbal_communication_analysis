import json
import random
import threading
import time
from queue import Queue

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from nonverbal_communication_analysis.m6_Visualization.feature_analyzer_gui import \
    Ui_FeatureAnalyzer

matplotlib.use('Qt5Agg')

DATA_QUEUE = Queue()


class ReaderThread(threading.Thread):
    def __init__(self, thread_id: int, group_files_data: dict):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.group_files_data = group_files_data
        self.finish_signal = QtCore.pyqtSignal()

    def run(self):
        for frame_idx, file_path in sorted(self.group_files_data.items()):
            file = json.load(open(file_path, 'r'))
            DATA_QUEUE.put((frame_idx, file))

        return True


class ProcessorThread(threading.Thread):
    def __init__(self, thread_id: int, plots_threads: dict):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.plots_threads = plots_threads

        self.group_metrics = pd.DataFrame(columns=['frame',
                                                   'camera',
                                                   'intragroup_distance',
                                                   'energy'])

        self.subject_metrics = pd.DataFrame(columns=['frame',
                                                     'camera',
                                                     'subject',
                                                     'expansiveness',
                                                     'center_interaction'])

    def run(self):
        while True:
            try:
                file_info = DATA_QUEUE.get(block=True, timeout=2)
                frame_idx = file_info[0]
                frame_data = file_info[1]
                self.process_file(frame_idx, frame_data)
            except:
                print("No more data in queue.")
                break

        intragroup_dist_thread = self.plots_threads['intragroup_distance']
        intragroup_distance_data = self.group_metrics[['frame',
                                                       'camera',
                                                       'intragroup_distance']]
        intragroup_dist_thread.data = intragroup_distance_data
        intragroup_dist_thread.start()

        group_energy_thread = self.plots_threads['group_energy']
        group_energy_data = self.group_metrics[['frame',
                                                'energy']]
        group_energy_thread.data = group_energy_data
        group_energy_thread.start()

        expansiveness_thread = self.plots_threads['expansiveness']
        expansiveness_data = self.subject_metrics[['frame',
                                                   'camera',
                                                   'subject',
                                                   'expansiveness']]
        expansiveness_thread.data = expansiveness_data
        expansiveness_thread.start()

        center_interaction_thread = self.plots_threads['center_interaction']
        center_interaction_data = self.subject_metrics[['frame',
                                                        'subject',
                                                        'center_interaction']]
        center_interaction_thread.data = center_interaction_data
        center_interaction_thread.start()

    def process_file(self, frame_idx, frame_data: dict):
        group_data = frame_data['group']
        subjects_data = frame_data['subjects']

        energy_data = group_data['energy']
        for camera, camera_intragroup_distance in group_data['intragroup_distance'].items():
            group_metrics_frame = dict()
            group_metrics_frame['frame'] = [frame_idx]
            group_metrics_frame['camera'] = [camera]
            group_metrics_frame['intragroup_distance'] = [
                camera_intragroup_distance['area']]
            group_metrics_frame['energy'] = [energy_data]

            tmp_df = pd.DataFrame(group_metrics_frame)
            self.group_metrics = self.group_metrics.append(tmp_df,
                                                           ignore_index=True)

        for subject in subjects_data:
            subject_id = subject['id']
            subject_metrics = subject['metrics']
            subject_metrics_frame = dict()
            subject_metrics_frame['frame'] = [frame_idx]
            subject_metrics_frame['subject'] = subject_id
            for camera, expansiveness_data in subject_metrics['expansiveness'].items():
                subject_metrics_frame['camera'] = [camera]
                subject_metrics_frame['expansiveness'] = [
                    expansiveness_data['area']]
                subject_metrics_frame['center_interaction'] = [
                    subject_metrics['center_interaction']]

                tmp_df = pd.DataFrame(subject_metrics_frame)
                self.subject_metrics = self.subject_metrics.append(tmp_df,
                                                                   ignore_index=True)



class PlotThread(QtCore.QThread):
    def __init__(self, thread_id, parent):
        QtCore.QThread.__init__(self)
        self.thread_id = thread_id
        print("Thread started", thread_id)
        self.parent = parent
        self.data = None

        if not isinstance(self.parent, list):
            self.canvas = PlotCanvas(parent=self.parent)
        else:
            self.canvas = [PlotCanvas(parent=self.parent[0]),
                           PlotCanvas(parent=self.parent[1]),
                           PlotCanvas(parent=self.parent[2])]

    def __del__(self):
        self.wait()

    def run(self):
        print("Print plot", self.thread_id)

        columns = self.data.columns

        if 'subject' in columns:
            if 'camera' in columns:
                for idx, camera in enumerate(self.data['camera'].unique()):
                    data = self.data[self.data['camera'] == camera]
                    data = data.drop('camera', axis=1).pivot(index='frame',
                                                             columns='subject')
                    self.canvas[idx].plot_subject(data)
            else:
                data = self.data.drop_duplicates(keep='first')
                data = data.pivot(index='frame', columns='subject')
                self.canvas.plot_subject(data)

        elif 'camera' in columns:
            data = self.data.pivot(index='frame', columns='camera')
            self.canvas.plot_camera(data)
        else:
            data = self.data.set_index('frame')
            self.canvas.plot_single(data)

        return


class PlotCanvas(QtWidgets.QWidget):
    # TODO: set xlim and/or ylim if needed
    # TODO: Save plots as images (Add save button)

    _color_encoding = {
        'pc1': 'tab:red',
        'pc2': 'tab:green',
        'pc3': 'tab:blue',
        'energy': 'tab:olive',
        '1': 'red',
        '2': 'cyan',
        '3': 'lime',
        '4': 'blue'
    }

    def __init__(self, parent, width=5, height=4, dpi=100):
        QtWidgets.QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())
        self.canvas.axes = self.canvas.figure.add_subplot()
        parent.layout().addWidget(self.canvas)

    def plot_single(self, data):
        for camera_index in data.columns:
            camera = camera_index
            self.canvas.axes.plot(data[camera_index],
                                  color=self._color_encoding[camera],
                                  label=camera)
        self.canvas.axes.legend(loc='upper right')
        self.canvas.draw()
        return

    def plot_camera(self, data):
        self.canvas.axes.cla()
        for camera_index in data.columns:
            camera = camera_index[1]
            self.canvas.axes.plot(data[camera_index],
                                  color=self._color_encoding[camera],
                                  label=camera)
        self.canvas.axes.legend(loc='upper right')
        self.canvas.draw()
        return

    def plot_subject(self, data):
        for subject_index in data.columns:
            subject = subject_index[1]
            self.canvas.axes.plot(data[subject_index],
                                  color=self._color_encoding[str(subject)],
                                  label=subject)
        self.canvas.axes.legend(loc='upper right')
        self.canvas.draw()
        return


class FeatureAnalyzer(object):
    def __init__(self):
        self.widget = QtWidgets.QWidget()
        self.ui = Ui_FeatureAnalyzer()
        self.ui.setupUi(self.widget)

        self.is_active = True  # Check if necessary. Probably not.
        self.has_initial_data = False
        self.group_id = None
        self.group_task = None
        self.group_files_data = None

    def set_data(self, group_info_data: dict):
        self.has_initial_data = True
        self.group_id = group_info_data['group_id']
        self.group_task = group_info_data['task']
        self.group_files_data = group_info_data['group_data']
        self.data_length = len(self.group_files_data)

    def open(self):
        self.widget.show()

        if self.is_active and self.has_initial_data:

            reader_thread = ReaderThread(0, self.group_files_data)
            reader_thread.start()

            intragroup_dist_thread = PlotThread(1, self.ui.cvs_intragroup_dist)
            group_energy_thread = PlotThread(2, self.ui.cvs_group_energy)
            expansiveness_thread = PlotThread(3, [self.ui.cvs_expansiveness_1,
                                                  self.ui.cvs_expansiveness_2,
                                                  self.ui.cvs_expansiveness_3])
            env_interaction_thread = PlotThread(4, self.ui.cvs_env_interaction)

            plots_threads = {
                'intragroup_distance': intragroup_dist_thread,
                'group_energy': group_energy_thread,
                'expansiveness': expansiveness_thread,
                'center_interaction': env_interaction_thread,
            }

            processor_thread = ProcessorThread(1, plots_threads)
            processor_thread.start()

            # self.intragroup_dist_thread.start()
            # self.group_energy_thread.start()
            # self.expansiveness_thread.start()
            # self.env_interaction_thread.start()
