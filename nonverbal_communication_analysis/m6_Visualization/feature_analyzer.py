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

from nonverbal_communication_analysis.environment import PLOT_INTRAGROUP_DISTANCE, PLOT_GROUP_ENERGY, PLOT_SUBJECT_EXPANSIVENESS, PLOT_CENTER_INTERACTION, DATASET_SYNC

matplotlib.use('Qt5Agg')


class PlotThread(QtCore.QThread):
    def __init__(self, thread_id, group_path, plot_type, parent):
        QtCore.QThread.__init__(self)
        self.thread_id = thread_id
        print("Thread started", thread_id)
        self.parent = parent
        self.plot_path = group_path / 'PLOTS'
        self.plot_type = plot_type

        if not isinstance(self.parent, list):
            self.canvas = PlotCanvas(parent=self.parent)
        else:
            self.canvas = [PlotCanvas(parent=self.parent[0]),
                           PlotCanvas(parent=self.parent[1]),
                           PlotCanvas(parent=self.parent[2])]

        plot_types = [PLOT_INTRAGROUP_DISTANCE, PLOT_GROUP_ENERGY,
                      PLOT_SUBJECT_EXPANSIVENESS, PLOT_CENTER_INTERACTION]
        if plot_type not in plot_types:
            print("Invalid plot type")
            exit()

    def __del__(self):
        self.wait()

    def run(self):
        print("Print plot", self.thread_id)

        data = pd.read_csv(str(self.plot_path / (self.plot_type+'.csv')))
        data = data.sort_values(by=['frame'])

        if self.plot_type in [PLOT_INTRAGROUP_DISTANCE, PLOT_GROUP_ENERGY, PLOT_CENTER_INTERACTION]:
            self.canvas.draw_plot(data, self.plot_type)

        elif isinstance(self.canvas, list):
            for idx, canvas in enumerate(self.canvas):
                camera = 'pc'+str(idx+1)
                camera_data = data[data['camera'] == camera]
                canvas.draw_plot(camera_data, self.plot_type)

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

    def draw_plot(self, data, data_column):
        data = data.sort_values(by=['frame'])
        if 'subject' in data:
            data = data.sort_values(by=['frame', 'subject'])
            for subject_index in data['subject'].unique():
                subject_data = data[data['subject'] == subject_index]
                self.canvas.axes.plot(subject_data['frame'], subject_data[data_column],
                                      color=self._color_encoding[str(
                                          subject_index)],
                                      label=subject_index)
        elif 'camera' in data:
            for camera in data['camera'].unique():
                camera_data = data[data['camera'] == camera]
                self.canvas.axes.plot(camera_data['frame'], camera_data[data_column],
                                      color=self._color_encoding[camera],
                                      label=camera)
        else:
            self.canvas.axes.plot(data['frame'], data[data_column],
                                  color=self._color_encoding[data_column],
                                  label=data_column)
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
        self.group_plot_path = None

    def set_data(self, group_info_data: dict):
        self.has_initial_data = True
        self.group_id = group_info_data['group_id']
        self.group_task = group_info_data['task']
        self.group_plot_path = group_info_data['path']

    def open(self):
        self.widget.show()

        if self.is_active and self.has_initial_data:

            intragroup_dist_thread = PlotThread(1, self.group_plot_path,
                                                PLOT_INTRAGROUP_DISTANCE,
                                                self.ui.cvs_intragroup_dist)
            group_energy_thread = PlotThread(2, self.group_plot_path,
                                             PLOT_GROUP_ENERGY,
                                             self.ui.cvs_group_energy)
            expansiveness_thread = PlotThread(3, self.group_plot_path,
                                              PLOT_SUBJECT_EXPANSIVENESS,
                                              [self.ui.cvs_expansiveness_1,
                                               self.ui.cvs_expansiveness_2,
                                               self.ui.cvs_expansiveness_3])
            env_interaction_thread = PlotThread(4, self.group_plot_path,
                                                PLOT_CENTER_INTERACTION,
                                                self.ui.cvs_env_interaction)

            intragroup_dist_thread.start()
            group_energy_thread.start()
            expansiveness_thread.start()
            env_interaction_thread.start()
