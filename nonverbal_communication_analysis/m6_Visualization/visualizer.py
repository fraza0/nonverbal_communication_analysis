import sys

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from nonverbal_communication_analysis.environment import DATASET_SYNC
from nonverbal_communication_analysis.m6_Visualization.visualizer_gui import Ui_Visualizer


class Visualizer:

    def __init__(self):
        self.group_dirs = {
            x.name: x for x in DATASET_SYNC.iterdir() if x.is_dir()}

    # GROUP FRAME METHODS
    def group_tasks(self, group_id):
        return [x.name for x in self.group_dirs[group_id].iterdir()
                if x.is_dir() and 'task' in x.name]

    def combo_on_select(self):
        cb_group = self.ui.cb_groupId
        cb_task = self.ui.cb_task
        group_cb_idx = int(cb_group.currentIndex())
        # task_cb_idx = int(cb_task.currentIndex())

        if group_cb_idx != 0:
            cb_task.setEnabled(True)
            cb_task.clear()
            cb_task.addItems(self.group_tasks(cb_group.currentText()))

        else:
            cb_task.clear()
            cb_task.setEnabled(False)

    def group_select_confirm(self, widgets):
        for widget in widgets:
            widget.setEnabled(False)

        # PLAYER FRAMES METHODS

        # OVERLAYS FRAMES METHODS

        # CONTROLS FRAMES METHODS

    def main(self):
        app = QtWidgets.QApplication(sys.argv)
        Visualizer = QtWidgets.QMainWindow()
        self.ui = Ui_Visualizer()
        self.ui.setupUi(Visualizer)

        # Group Frame
        cb_group_id = self.ui.cb_groupId
        cb_task = self.ui.cb_task
        btn_confirm = self.ui.btn_confirm
        group_select_frame = [cb_group_id, cb_task, btn_confirm]

        cb_group_id.addItems(
            ['Select Group ID']+list(self.group_dirs.keys()))
        cb_group_id.currentIndexChanged.connect(self.combo_on_select)
        btn_confirm.clicked.connect(
            lambda: self.group_select_confirm(group_select_frame))

        Visualizer.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.main()
