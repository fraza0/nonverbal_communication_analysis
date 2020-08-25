import json
import warnings
from math import sqrt

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.interpolate as I
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from nonverbal_communication_analysis.environment import (
    DATASET_SYNC, FEATURE_AGGREGATE_DIR, GROUPS_INFO_FILE, LINESTYLES,
    PLOT_CANVAS_COLOR_ENCODING, PLOTS_LIB, ROLLING_WINDOW_SIZE,
    VALID_OUTPUT_FILE_TYPES, PLOT_CENTER_INTERACTION)
from nonverbal_communication_analysis.m6_Visualization.feature_comparator_gui import \
    Ui_FeatureComparator

matplotlib.use('Qt5Agg')
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)


class PlotCanvas(QtWidgets.QWidget):
    _color_encoding = PLOT_CANVAS_COLOR_ENCODING

    def __init__(self, parent, name, width=5, height=4, dpi=100):
        QtWidgets.QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())
        self.canvas.axes = self.canvas.figure.add_subplot()
        parent.layout().removeItem(parent.layout().itemAt(0))
        parent.layout().addWidget(self.canvas)

        self.name = name

    def save_plots(self, plot_path):
        self.canvas.figure.set_size_inches(18.5, 10.5, forward=False)
        self.canvas.figure.savefig(
            plot_path / (self.name+'.png'), dpi=100)
        return True

    def smoothing_factor(self, number_datapoints):
        _max = number_datapoints+sqrt(2*number_datapoints)
        _min = number_datapoints-sqrt(2*number_datapoints)
        return (_max + _min) / 2

    def draw_plot(self, data: pd.DataFrame, gid_group: tuple, metric: str, linetype: str = 'spline'):
        data = data.sort_values(by=['frame'])
        group_id, group = gid_group

        data_size = len(data)
        poly_degree = 50
        _roling_window_size = ROLLING_WINDOW_SIZE \
            if data_size > ROLLING_WINDOW_SIZE*3 else round(data_size/5)

        x = data['frame'].astype('int64')
        y = data[metric]

        if 'subject' in data:
            subjects = sorted(data['subject'].unique())
            for subject_index in subjects:
                subject_data = data[data['subject'] == subject_index]
                subject_data = subject_data.sort_values(
                    by=['frame', 'subject'])

                subject_index = str(subject_index)
                x = subject_data['frame'].astype('int64')
                y = subject_data[metric]

                label = 'S'+subject_index+'_'+group

                if 'raw' in linetype:
                    self.canvas.axes.scatter(x, y,
                                             color=self._color_encoding[subject_index],
                                             marker='.',
                                             label=label)
                elif 'spline' in linetype:
                    s_value = self.smoothing_factor(len(x))
                    bspl = I.splrep(x, y, s=s_value)
                    bspl_y = I.splev(x, bspl)
                    self.canvas.axes.plot(x, bspl_y,
                                          color=self._color_encoding[subject_index],
                                          linestyle=LINESTYLES[group_id],
                                          label=label)
                elif 'poly' in linetype:
                    z = np.polyfit(x, y, poly_degree)
                    f = np.poly1d(z)
                    self.canvas.axes.plot(x, f(x),
                                          color=self._color_encoding[subject_index],
                                          linestyle=LINESTYLES[group_id],
                                          label=label)
                elif 'rolling' in linetype:
                    self.canvas.axes.plot(x, y.rolling(window=_roling_window_size).mean(),
                                          color=self._color_encoding[subject_index],
                                          linestyle=LINESTYLES[group_id],
                                          label=label)
        else:
            if 'raw' in linetype:
                self.canvas.axes.scatter(x, y,
                                         label=group,
                                         marker='.')
            elif 'spline' in linetype:
                s_value = self.smoothing_factor(len(x))
                bspl = I.splrep(x, y, s=s_value)
                bspl_y = I.splev(x, bspl)
                self.canvas.axes.plot(x, bspl_y,
                                      label=group)
            elif 'poly' in linetype:
                z = np.polyfit(x, y, poly_degree)
                p = np.poly1d(z)
                self.canvas.axes.plot(x, p(x),
                                      label=group)
            elif 'rolling' in linetype:
                self.canvas.axes.plot(x, y.rolling(window=_roling_window_size).mean(),
                                      label=group)

        self.canvas.axes.set_title(metric)
        self.canvas.axes.legend(loc='upper right')
        self.canvas.draw()
        return


class FeatureComparator(object):
    def __init__(self):
        self.widget = QtWidgets.QWidget()
        self.ui = Ui_FeatureComparator()
        self.ui.setupUi(self.widget)

        self.linestyles = LINESTYLES

        self.group_dirs = {x.name: x for x in DATASET_SYNC.iterdir()
                           if x.is_dir()}

        self.groups_info = pd.read_csv(GROUPS_INFO_FILE)

        # Initial state
        group_names = list(self.group_dirs.keys())
        group_names.sort()
        self.ui.cb_group1.addItems(group_names)
        self.ui.cb_group2.addItems(group_names)
        self.ui.cb_group3.addItems(group_names)

        self.ui.btn_save.setIcon(self.widget.style().standardIcon(
            QtWidgets.QStyle.SP_DialogSaveButton))

        self.subjects_cb = [self.ui.chb_subject1, self.ui.chb_subject2,
                            self.ui.chb_subject3, self.ui.chb_subject4]

        self.linetype_rad = {'raw': self.ui.rad_raw,
                             'poly': self.ui.rad_poly,
                             'rolling': self.ui.rad_moving_avg,
                             'spline': self.ui.rad_spline}

        self.compare_groups_state = [(self.ui.cb_group1, self.ui.cb_task1),
                                     (self.ui.cb_group2, self.ui.cb_task2),
                                     (self.ui.cb_group3, self.ui.cb_task3)]

        # Actions
        self.ui.btn_close.clicked.connect(self.close)
        self.ui.btn_compare.clicked.connect(self.compare)
        self.ui.cb_group1.currentIndexChanged.connect(
            lambda: self.on_change_cb_group(self.ui.cb_group1, self.ui.cb_task1, self.ui.tb_type1))
        self.ui.cb_group2.currentIndexChanged.connect(
            lambda: self.on_change_cb_group(self.ui.cb_group2, self.ui.cb_task2, self.ui.tb_type2))
        self.ui.cb_group3.currentIndexChanged.connect(
            lambda: self.on_change_cb_group(self.ui.cb_group3, self.ui.cb_task3, self.ui.tb_type3))

    def open(self):
        self.widget.show()

    def close(self):
        self.widget.close()

    def save(self):
        print("Save")

    def load_data(self, groups):
        subjects = list()

        for idx, subject_cb in enumerate(self.subjects_cb):
            if subject_cb.isChecked():
                subjects.append(idx+1)

        camera = self.ui.cb_camera.currentText().lower()
        metric = self.ui.cb_metric.currentText().lower()
        linetype = [k for k, v in self.linetype_rad.items()
                    if v.isChecked()]

        plot_name = 'compare_' + \
            '_'.join(['_'.join(group_tuple) for group_tuple in groups])

        self.canvas = PlotCanvas(parent=self.ui.cvs_plot, name=plot_name)

        groups = sorted(groups)
        for idx, (group, task) in enumerate(groups):
            group_task_data_path = DATASET_SYNC / group / \
                FEATURE_AGGREGATE_DIR / task / 'PLOTS'
            metric_name = metric.replace(' ', '_')
            metric_lib = PLOTS_LIB[metric_name].lower()
            metric_file = [x for x in group_task_data_path.iterdir()
                           if x.suffix in VALID_OUTPUT_FILE_TYPES
                           and metric_lib+'_'+metric_name in x.name]

            if not metric_file:
                print("Metric file not found", metric_name)
                break

            metric_file = metric_file[0]
            data = pd.read_csv(metric_file)

            if 'camera' not in data.columns:
                self.ui.cb_camera.setCurrentIndex(0)
                self.ui.cb_camera.setEnabled(False)
                camera = self.ui.cb_camera.currentText().lower()
            else:
                self.ui.cb_camera.setEnabled(True)
                if self.ui.cb_camera.currentIndex() == 0:
                    self.ui.cb_camera.setCurrentIndex(1)

                camera = self.ui.cb_camera.currentText().lower()
                data = data[data['camera'] == camera]

            if 'subject' not in data.columns:
                self.ui.chb_subject1.setChecked(True)
                self.ui.chb_subject2.setChecked(True)
                self.ui.chb_subject3.setChecked(True)
                self.ui.chb_subject4.setChecked(True)
            else:
                subjects_data = pd.DataFrame(columns=data.columns)
                for subject in subjects:
                    subjects_data = subjects_data.append(
                        data[data['subject'] == subject], ignore_index=True)
                data = subjects_data

            self.canvas.draw_plot(data, (idx, group), metric_name,
                                  linetype=linetype)

    def compare(self):
        self.groups_to_compare = set()

        for (cb_group, cb_task) in self.compare_groups_state:
            if cb_group.currentIndex() != 0:
                self.groups_to_compare.add((cb_group.currentText(),
                                            cb_task.currentText()))

        self.load_data(self.groups_to_compare)

    def on_change_cb_group(self, cb_group, cb_task, tb_type):
        if cb_group.currentIndex() == 0:
            cb_task.setCurrentIndex(0)
            cb_task.clear()
            cb_task.addItem('Select Task')
            tb_type.setText('')
        else:
            group_name = cb_group.currentText()
            group_tasks = [str(x.name) for x in self.group_dirs[group_name].iterdir()
                           if x.is_dir() and 'task' in x.name]
            cb_task.clear()
            cb_task.addItems(group_tasks)

            conflict_type = self.groups_info[self.groups_info['Group ID']
                                             == group_name]['Conflict Type'].values[0]
            tb_type.setText(conflict_type)
