import json
import random
import threading
import time
import warnings
from queue import Queue
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as I
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy.polynomial import polynomial as P
from PyQt5 import QtCore, QtGui, QtWidgets

from nonverbal_communication_analysis.environment import (
    DATASET_SYNC, PLOT_CENTER_INTERACTION, PLOT_GROUP_ENERGY,
    PLOT_INTRAGROUP_DISTANCE, PLOT_SUBJECT_OVERLAP, ROLLING_WINDOW_SIZE,
    PLOT_CANVAS_COLOR_ENCODING, PLOTS_LIB, PLOT_KEYPOINT_ENERGY)
from nonverbal_communication_analysis.m6_Visualization.feature_analyzer_gui import \
    Ui_FeatureAnalyzer

matplotlib.use('Qt5Agg')
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)

# plt.ion()


class PlotThread(QtCore.QThread):
    def __init__(self, thread_id, group_path, plot_type, parent_info, group):
        QtCore.QThread.__init__(self)
        self.thread_id = thread_id
        self.parent_info = parent_info
        self.plot_path = group_path / 'PLOTS'
        self.plot_type = plot_type

        if not isinstance(self.parent_info, list):
            self.canvas = PlotCanvas(
                parent=self.parent_info[0], name=self.parent_info[1], group=group)
        else:
            self.canvas = [PlotCanvas(parent=self.parent_info[0][0], name=self.parent_info[0][1], group=group),
                           PlotCanvas(
                               parent=self.parent_info[1][0], name=self.parent_info[1][1], group=group),
                           PlotCanvas(parent=self.parent_info[2][0], name=self.parent_info[2][1], group=group)]

        plot_types = [PLOT_INTRAGROUP_DISTANCE, PLOT_GROUP_ENERGY,
                      PLOT_SUBJECT_OVERLAP, PLOT_CENTER_INTERACTION, PLOT_KEYPOINT_ENERGY]

        if plot_type not in plot_types:
            print("Invalid plot type")
            exit()

    def __del__(self):
        self.wait()

    def save_plots(self):
        if isinstance(self.canvas, list):
            for canvas in self.canvas:
                canvas.save_plots(self.plot_path)
        else:
            self.canvas.save_plots(self.plot_path)

    def update_plots(self, line_type):
        self.draw_plots(line_type=line_type, update=True)

    def draw_plots(self, line_type=None, update: bool = True):
        data = self.data
        line_type = line_type if line_type is not None else ['spline']
        if isinstance(self.canvas, list):
            for idx, canvas in enumerate(self.canvas):
                camera = 'pc'+str(idx+1)
                camera_data = data[data['camera'] == camera]
                canvas.draw_plot(camera_data, self.plot_type,
                                 line_type=line_type, update=update)
        else:
            self.canvas.draw_plot(data, self.plot_type,
                                  line_type=line_type, update=update)

    def run(self):
        # print("Print plot", self.thread_id)
        plot_type = self.plot_type
        lib = PLOTS_LIB[plot_type].lower()
        ext = '.csv'
        file_name = lib + '_' + plot_type + ext

        data = pd.read_csv(str(self.plot_path / file_name))
        data = data.sort_values(by=['frame'])
        self.data = data

        self.draw_plots(update=False)


class PlotCanvas(QtWidgets.QWidget):
    _color_encoding = PLOT_CANVAS_COLOR_ENCODING

    def __init__(self, parent, name, group, width=5, height=4, dpi=100):
        QtWidgets.QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())
        self.canvas.axes = self.canvas.figure.add_subplot()
        parent.layout().addWidget(self.canvas)
        self.group = group

        self.name = name

    def save_plots(self, plot_path):
        self.canvas.figure.set_size_inches(12, 7, forward=False)
        self.canvas.figure.savefig(
            plot_path / ('plot_'+self.name+'.png'), dpi=100)
        return True

    def smoothing_factor(self, number_datapoints):
        return number_datapoints+sqrt(2*number_datapoints)-1

    def draw_plot(self, data: pd.DataFrame, data_column: str, line_type: list = ['spline'], update: bool = False, save: bool = True):
        """[summary]

        Args:
            data (pd.DataFrame): DataFrame
            data_column (str): Data column name
            line_type (list, optional): Types may be 'raw', 'spline' or 'poly'. Defaults to ['spline'].
            save (bool, optional): Save plot figures. Defaults to True.
        """
        self.canvas.axes.clear()
        data = data.sort_values(by=['frame'])
        poly_degree = 50
        self.canvas.axes.set_title(self.group + ' ' + self.name)

        if 'subject' in data:
            for subject_index in data['subject'].unique():
                subject_data = data[data['subject'] == subject_index]
                subject_data = subject_data.sort_values(
                    by=['frame', 'subject'])

                subject_index = str(subject_index)
                x = subject_data['frame']
                y = subject_data[data_column]

                if 'raw' in line_type:
                    self.canvas.axes.scatter(subject_data['frame'], subject_data[data_column],
                                             color=self._color_encoding[subject_index],
                                             label=subject_index,
                                             marker='.')

                if 'spline' in line_type:
                    s_value = self.smoothing_factor(len(x))
                    bspl = I.splrep(x, y, s=s_value)
                    bspl_y = I.splev(x, bspl)
                    self.canvas.axes.plot(x, bspl_y,
                                          self._color_encoding[subject_index +
                                                               '_splinefit'],
                                          label=subject_index+'_spline')

                if 'poly' in line_type:
                    z = np.polyfit(x, y, poly_degree)
                    f = np.poly1d(z)
                    self.canvas.axes.plot(x, f(x),
                                          self._color_encoding[subject_index+'_polyfit'],
                                          label=subject_index+'_poly')

                if 'rolling' in line_type:
                    data_size = len(x)
                    roling_window_size = ROLLING_WINDOW_SIZE \
                        if data_size > ROLLING_WINDOW_SIZE*3 else round(data_size/5)
                    self.canvas.axes.plot(x, y.rolling(window=roling_window_size).mean(),
                                          self._color_encoding[subject_index +
                                                               '_rolling_meanfit'],
                                          label=subject_index+'_rolling_mean')

        elif 'camera' in data:
            for camera in sorted(data['camera'].unique()):
                camera_data = data[data['camera'] == camera]

                x = camera_data['frame']  # .loc[::5]
                y = camera_data[data_column]  # .loc[::5]

                if 'raw' in line_type:
                    self.canvas.axes.scatter(camera_data['frame'], camera_data[data_column],
                                             color=self._color_encoding[camera],
                                             label=camera,
                                             marker='.')

                if 'spline' in line_type:
                    s_value = self.smoothing_factor(len(x))
                    bspl = I.splrep(x, y, s=s_value)
                    bspl_y = I.splev(x, bspl)
                    self.canvas.axes.plot(x, bspl_y,
                                          self._color_encoding[camera +
                                                               '_splinefit'],
                                          label=camera+'_spline')

                if 'poly' in line_type:

                    z = np.polyfit(x, y, poly_degree)
                    p = np.poly1d(z)
                    self.canvas.axes.plot(x, p(x),
                                          self._color_encoding[camera +
                                                               '_polyfit'],
                                          label=camera+'_poly')

                if 'rolling' in line_type:
                    data_size = len(x)
                    roling_window_size = ROLLING_WINDOW_SIZE \
                        if data_size > ROLLING_WINDOW_SIZE*3 else round(data_size/5)
                    self.canvas.axes.plot(x, y.rolling(window=roling_window_size).mean(),
                                          self._color_encoding[camera +
                                                               '_rolling_meanfit'],
                                          label=camera+'_rolling_mean')

        else:
            x = data['frame']  # .loc[::5]
            y = data[data_column]  # .loc[::5]

            if 'raw' in line_type:
                self.canvas.axes.scatter(data['frame'], data[data_column],
                                         color=self._color_encoding[data_column],
                                         label=data_column,
                                         marker='.')

            if 'spline' in line_type:
                s_value = self.smoothing_factor(len(x))
                bspl = I.splrep(x, y, s=s_value)
                bspl_y = I.splev(x, bspl)
                self.canvas.axes.plot(x, bspl_y,
                                      self._color_encoding[data_column +
                                                           '_splinefit'],
                                      label=data_column+'_spline')

            if 'poly' in line_type:
                z = np.polyfit(x, y, poly_degree)
                p = np.poly1d(z)
                self.canvas.axes.plot(x, p(x),
                                      self._color_encoding[data_column+'_polyfit'],
                                      label=data_column+'_poly')

            if 'rolling' in line_type:
                self.canvas.axes.plot(x, y.rolling(window=ROLLING_WINDOW_SIZE).mean(),
                                      self._color_encoding[data_column +
                                                           '_rolling_meanfit'],
                                      label=data_column+'_rolling_mean')

        self.canvas.axes.legend(loc='upper right')
        self.canvas.draw()
        return


class FeatureAnalyzer(object):
    def __init__(self):
        self.widget = QtWidgets.QWidget()
        self.ui = Ui_FeatureAnalyzer()
        self.ui.setupUi(self.widget)

        self.ui.btn_save.setIcon(self.widget.style().standardIcon(
            QtWidgets.QStyle.SP_DialogSaveButton))

        self.ui.btn_save.clicked.connect(self.save_plots)
        self.ui.chb_raw.stateChanged.connect(self.update_plots)
        self.ui.chb_spline.stateChanged.connect(self.update_plots)
        self.ui.chb_poly.stateChanged.connect(self.update_plots)
        self.ui.chb_rolling.stateChanged.connect(self.update_plots)

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

    def save_plots(self):
        self.intragroup_dist_thread.save_plots()
        self.group_energy_thread.save_plots()
        self.subject_energy_thread.save_plots()
        self.overlap_thread.save_plots()
        self.env_interaction_thread.save_plots()

    def update_plots(self):
        state = []
        if self.ui.chb_raw.isChecked():
            state.append('raw')
        if self.ui.chb_poly.isChecked():
            state.append('poly')
        if self.ui.chb_rolling.isChecked():
            state.append('rolling')
        if self.ui.chb_spline.isChecked():
            state.append('spline')

        self.intragroup_dist_thread.update_plots(state)
        self.group_energy_thread.update_plots(state)
        self.subject_energy_thread.update_plots(state)
        self.overlap_thread.update_plots(state)
        self.env_interaction_thread.update_plots(state)

    def open(self, group_feature_path):
        self.widget.show()

        if self.is_active and self.has_initial_data:

            self.intragroup_dist_thread = PlotThread(1, self.group_plot_path,
                                                     PLOT_INTRAGROUP_DISTANCE,
                                                     (self.ui.cvs_intragroup_dist, PLOT_INTRAGROUP_DISTANCE), self.group_id+'_'+self.group_task)
            self.group_energy_thread = PlotThread(2, self.group_plot_path,
                                                  PLOT_GROUP_ENERGY,
                                                  (self.ui.cvs_group_energy, PLOT_GROUP_ENERGY), self.group_id+'_'+self.group_task)
            self.subject_energy_thread = PlotThread(3, self.group_plot_path,
                                                    PLOT_KEYPOINT_ENERGY,
                                                    ([(self.ui.cvs_kp_energy_1,
                                                       PLOT_KEYPOINT_ENERGY+'_pc1'),
                                                      (self.ui.cvs_kp_energy_2,
                                                       PLOT_KEYPOINT_ENERGY+'_pc2'),
                                                      (self.ui.cvs_kp_energy_3,
                                                       PLOT_KEYPOINT_ENERGY+'_pc3')]), self.group_id+'_'+self.group_task)
            self.overlap_thread = PlotThread(4, self.group_plot_path,
                                             PLOT_SUBJECT_OVERLAP,
                                             [(self.ui.cvs_overlap_1,
                                               PLOT_SUBJECT_OVERLAP+'_pc1'),
                                              (self.ui.cvs_overlap_2,
                                               PLOT_SUBJECT_OVERLAP+'_pc2'),
                                              (self.ui.cvs_overlap_3,
                                               PLOT_SUBJECT_OVERLAP+'_pc3')], self.group_id+'_'+self.group_task)
            self.env_interaction_thread = PlotThread(5, self.group_plot_path,
                                                     PLOT_CENTER_INTERACTION,
                                                     (self.ui.cvs_env_interaction, PLOT_CENTER_INTERACTION), self.group_id+'_'+self.group_task)

            self.intragroup_dist_thread.start()
            self.group_energy_thread.start()
            self.subject_energy_thread.start()
            self.overlap_thread.start()
            self.env_interaction_thread.start()
