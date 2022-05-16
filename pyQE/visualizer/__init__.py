# -*- coding: utf-8 -*-
#=========================================================
# Beginning of __init__.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains the functionality to 
# render the visualizer with dispersion widget
#
# This software is part of a package distributed under the 
# GPLv3 license, see ..LICENSE.txt
#=========================================================
import json
import qdarkstyle
from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import os
from .visualizer import ObjectWidget, HELP_WINDOW, ObjectParam, OBJECT, CustomDialog
from .atomic_motion import VisualizerWidget
from .dispersion_widget import DispersionWindow
from contextlib import contextmanager

scriptDir = os.path.dirname(os.path.realpath(__file__))

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self, input_json_data):
        QtWidgets.QMainWindow.__init__(self)
        self.input_json_data = input_json_data
        WIDTH = 1800
        HEIGHT = 800
        self.resize(WIDTH, HEIGHT)
        self.setWindowTitle('Phonon viewer')
        self.setWindowIcon(QtGui.QIcon('icon.png'))        
        self.props_widget = ObjectWidget(self)
        self.props_widget.signal_objet_changed.connect(self.update_view)

        self.splitter_v = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter_v.addWidget(self.props_widget)
        self.splitter_v.addWidget(QtWidgets.QLabel(HELP_WINDOW))
        self.save_gif_widget = QtWidgets.QPushButton("Save animation")
        self.save_gif_widget.setCheckable(True)
        self.save_gif_widget.toggle()
        self.save_gif_widget.clicked.connect(self.btnstate)
        self.splitter_v.addWidget(self.save_gif_widget)
        VisWidget = VisualizerWidget(input_json_data=self.input_json_data)
        self.canvas = VisWidget.canvas
        # self.canvas = VisualizerCanvas(input_json_data=self.input_json_data)
        # self.canvas.create_native()
        # self.canvas.native.setParent(self)
        self.canvas.measure_fps(0.1, self.show_fps)

        # Central Widget
        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(self.splitter_v)
        splitter1.addWidget(VisWidget)
        # splitter1.addWidget(self.canvas.native)
        DispWidget = DispersionWindow(input_json_data=self.input_json_data)
        DispWidget.signal.connect(self.update_motion)

        splitter1.addWidget(DispWidget)

        self.setCentralWidget(splitter1)

        # FPS message in statusbar:
        self.status = self.statusBar()
        self.status_label = QtWidgets.QLabel('...')
        self.status.addWidget(self.status_label)

        self.update_view(self.props_widget.param)
        # self.update_motion(self.props_widget.momentum_param)
        self.show()


    def list_objectChanged(self):
        row = self.list_object.currentIndex().row()
        name = self.list_object.currentIndex().data()
        if row != -1:
            self.props_widget.deleteLater()
            self.props_widget = ObjectWidget(self, param=ObjectParam(name,
                                                                     OBJECT[name]))
            self.splitter_v.addWidget(self.props_widget)
            self.props_widget.signal_objet_changed.connect(self.update_view)
            # self.props_widget.signal_q_changed.connect(self.update_motion)
            self.update_view(self.props_widget.param)
            # self.update_motion(self.props_widget.momentum_param)

    def show_fps(self, fps):
        msg = "FPS - %0.2f  " % (float(fps))
        # NOTE: We can't use showMessage in PyQt5 because it causes
        #       a draw event loop (show_fps for every drawing event,
        #       showMessage causes a drawing event, and so on).
        self.status_label.setText(msg)

    def btnstate(self):
        dlg = CustomDialog(self)
        dlg.setNameFilter("GIF (*.gif)")
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            self.canvas.save_animation(filenames[0])

    # def keyPressEvent(selfz

    def update_view(self, param):
        self.canvas.visible = param.props['visible']
        self.canvas.draw_bonds_bool = param.props['show_bonds']
        self.canvas.set_view( param.props['amplitude'], param.props['speed'])

    def update_motion(self, momentum_param):
        self.canvas.set_motion(momentum_param.props['momentum'], momentum_param.props['branch'])

@contextmanager
def gui_environment():
    """ 
    Prepare the environment in which GUI will run by setting 
    the PyQtGraph QT library to PyQt5 while GUI is running. Revert back when done.
    """
    old_qt_lib = os.environ.get(
        "PYQTGRAPH_QT_LIB", "PyQt5"
    )  # environment variable might not exist
    os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
    yield
    os.environ["PYQTGRAPH_QT_LIB"] = old_qt_lib

# Start Qt event loop unless running in interactive mode.
def run_visualizer(input_json_data):
    f = open(input_json_data,'r')
    data = json.load(f)
    with gui_environment():        
        appQt = QtWidgets.QApplication(sys.argv)
        appQt.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        win = MainWindow(input_json_data=data)
        appQt.exec_()
#=========================================================
# End of __init__.py
#=========================================================