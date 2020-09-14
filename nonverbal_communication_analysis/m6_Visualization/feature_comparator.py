import json
import warnings
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
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
    VALID_OUTPUT_FILE_TYPES, PLOT_CENTER_INTERACTION, TASK_2_MARK)
from nonverbal_communication_analysis.m6_Visualization.feature_comparator_gui import \
    Ui_FeatureComparator

matplotlib.use('Qt5Agg')
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)


class PlotCanvas(QtWidgets.QWidget):
    _color_encoding = PLOT_CANVAS_COLOR_ENCODING

    def __init__(self, parent, name, camera, width=5, height=4, dpi=100):
        QtWidgets.QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())
        self.canvas.axes = self.canvas.figure.add_subplot()
        parent.layout().removeItem(parent.layout().itemAt(0))
        parent.layout().addWidget(self.canvas)

        self.name = name
        self.camera = camera

    def save_plots(self, metric, camera, linetype):
        # self.canvas.figure.set_size_inches(12.5, 9.5, forward=True)
        name_parts = self.name.split('_')[1:]
        for n_group in range(0, len(name_parts)//3):
            group_name = name_parts[n_group*3]
            group_task = name_parts[n_group*3+1] + \
                '_' + name_parts[n_group*3+2]

            save_dir = DATASET_SYNC/group_name/FEATURE_AGGREGATE_DIR/group_task/'PLOTS'
            save_name = metric + '_' + self.name
            if camera is not None:
                save_name += '_' + camera + '_'
            save_name += linetype + '.png'
            self.canvas.figure.savefig(save_dir / save_name, dpi=100)
        return True

    def smoothing_factor(self, number_datapoints):
        _sqrt_value = sqrt(2*number_datapoints)
        _max = number_datapoints+_sqrt_value
        _min = number_datapoints-_sqrt_value
        return (_max + _min) / 2

    def draw_plot(self, data: pd.DataFrame, gid_group: tuple, metric: str, linetype: str = 'spline'):
        data = data.sort_values(by=['frame'])
        group_id, group, (task_name, num_tasks), conflict_type = gid_group

        data_size = len(data)
        poly_degree = 50
        _roling_window_size = ROLLING_WINDOW_SIZE \
            if data_size > ROLLING_WINDOW_SIZE*3 else round(data_size/5)
        
        if 'subject' in data:
            subjects = sorted(data['subject'].unique())
            for subject_index in subjects:
                subject_data = data[data['subject'] == subject_index]
                subject_data = subject_data.sort_values(
                    by=['frame', 'subject'])

                subject_index = str(subject_index)
                x = subject_data['frame'].astype('int64')
                y = subject_data[metric]
                
                x_data = pd.DataFrame(columns=['raw', 'norm'])
                
                x = subject_data['frame'].astype('int64')
                x_data['raw'] = x
                
                x_min, x_max = x.min(), x.max()
                x_norm = (x-x_min) / \
                    (x_max-x_min)
                x_data['norm'] = x_norm

                x = x_data['norm']
                y = subject_data[metric].astype('float64')
                y_pos = y.max()

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
                    
            if task_name == 'task_2' and group_id == 0:
                vertical_line_drawn = True
                five_min_mark = x_data[x_data['raw'] == TASK_2_MARK]
                five_min_mark = float(five_min_mark['norm'].unique()[0])
                self.canvas.axes.axvline(x=five_min_mark, color='k', linestyle='--', alpha=0.3)
                
        else:
            x_data = pd.DataFrame(columns=['raw', 'norm'])
            
            x = data['frame'].astype('int64')
            x_data['raw'] = x
            
            x_min, x_max = x.min(), x.max()
            x_norm = (x-x_min) / \
                (x_max-x_min)
            x_data['norm'] = x_norm

            x = x_data['norm']
            y = data[metric].astype('float64')
            y_pos = y.max()
            
            if task_name == 'task_2' and group_id == 0:
                vertical_line_drawn = True
                five_min_mark = x_data[x_data['raw'] == TASK_2_MARK]
                five_min_mark = float(five_min_mark['norm'].unique()[0])
                self.canvas.axes.axvline(x=five_min_mark, color='k', linestyle='--', alpha=0.3)
            
            label = group + ' (%s)' % conflict_type
            if num_tasks > 1:
                label = group + '_' + task_name + ' (%s)' % conflict_type

            if 'raw' in linetype:
                self.canvas.axes.scatter(x, y,
                                         label=label,
                                         marker='.')
            elif 'spline' in linetype:
                s_value = self.smoothing_factor(len(x))
                bspl = I.splrep(x, y, s=s_value)
                bspl_y = I.splev(x, bspl)

                self.canvas.axes.plot(x, bspl_y,
                                      label=label)
            elif 'poly' in linetype:
                z = np.polyfit(x, y, poly_degree)
                p = np.poly1d(z)
                self.canvas.axes.plot(x, p(x),
                                      label=label)
            elif 'rolling' in linetype:
                self.canvas.axes.plot(x, y.rolling(window=_roling_window_size).mean(),
                                      label=label)

        self.canvas.axes.set_title(metric+'_'+self.camera)
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

        self.canvas = None

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

        self.ui.btn_save.clicked.connect(self.save)

    def open(self):
        self.widget.show()

    def close(self):
        self.widget.close()

    def save(self):
        if self.canvas:
            self.canvas.save_plots(self.metric, self.camera, self.linetype)

    def load_data(self, groups):
        subjects = list()

        for idx, subject_cb in enumerate(self.subjects_cb):
            if subject_cb.isChecked():
                subjects.append(idx+1)

        camera = self.ui.cb_camera.currentText().lower()
        self.camera = camera
        metric = self.ui.cb_metric.currentText().lower()
        self.metric = metric
        linetype = [k for k, v in self.linetype_rad.items()
                    if v.isChecked()]
        self.linetype = linetype[0]

        plot_name = 'comparison_' + \
            '_'.join(['_'.join(group_tuple) for group_tuple in groups])

        self.canvas = PlotCanvas(
            parent=self.ui.cvs_plot, name=plot_name, camera=camera)

        fixed_columns = ['frame', 'group_idx', 'group', 'task']

        data = None

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
            group_metric_data = pd.read_csv(metric_file)

            group_metric_data['group_idx'] = idx
            group_metric_data['group'] = group
            group_metric_data['task'] = task

            if data is None:
                data = pd.DataFrame(columns=group_metric_data.columns)

            if 'camera' not in group_metric_data.columns:
                self.ui.cb_camera.setCurrentIndex(1)
                self.ui.cb_camera.setEnabled(False)
                camera = self.ui.cb_camera.currentText().lower()
            else:
                self.ui.cb_camera.setEnabled(True)
                if self.ui.cb_camera.currentIndex() == 0:
                    self.ui.cb_camera.setCurrentIndex(1)

                camera = self.ui.cb_camera.currentText().lower()
                group_metric_data = group_metric_data[group_metric_data['camera'] == camera]
                fixed_columns.append('camera')

            if 'subject' not in group_metric_data.columns:
                self.ui.chb_subject1.setChecked(True)
                self.ui.chb_subject2.setChecked(True)
                self.ui.chb_subject3.setChecked(True)
                self.ui.chb_subject4.setChecked(True)
            else:
                subjects_data = pd.DataFrame(columns=data.columns)
                for subject in subjects:
                    subjects_data = subjects_data.append(
                        group_metric_data[group_metric_data['subject'] == subject], ignore_index=True)
                group_metric_data = subjects_data
                fixed_columns.append('subject')

            data = data.append(group_metric_data, ignore_index=True)

        to_normalize_column = set(
            data.columns).symmetric_difference(fixed_columns)
        to_normalize_data = data[to_normalize_column]
        normalized_min = to_normalize_data.min()
        normalized_data = (to_normalize_data-normalized_min) / \
            (to_normalize_data.max()-normalized_min)

        data[list(to_normalize_column)] = normalized_data
        
        group_idx = -1

        for group in data['group'].unique():
            group_conflict_type = self.groups_info[self.groups_info['Group ID']
                                             == group]['Conflict Type'].values[0]
            tasks = data['task'].unique()
            for task in tasks:
                task_info = (task, len(tasks))
                group_data = data[data['group'] == group]
                group_idx += 1 #group_data['group_idx'].unique()[0]
                group_data = group_data[data['task'] == task]
                group_data = group_data.drop(
                    columns=['group', 'group_idx', 'task'])
                
                self.canvas.draw_plot(group_data, (group_idx, group, task_info, group_conflict_type),
                                      metric_name, linetype=linetype)

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
