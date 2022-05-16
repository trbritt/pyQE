# -*- coding: utf-8 -*-
#=========================================================
# Beginning of dispersion_widget.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains everything needed for the 
# plotting of phonon bands and allows interaction with the
# atomic motion visualization. Right clicks on the band structure
# will update the atomic motion visualizer and display the
# motion of the supercell at that phonon mode and momentum
#
# This software is part of a package distributed under the 
# GPLv3 license, see ..LICENSE.txt
#=========================================================
import numpy as np
from PyQt5 import QtWidgets, QtCore
pyqtsignal = QtCore.pyqtSignal if hasattr(QtCore, 'pyqtSignal') else QtCore.Signal

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
class MomentumParam(object):
    """
    momentum OBJECT class
    """
    def __init__(self, idq, nu):
        self.props={}
        self.props['momentum'] = idq
        self.props['branch'] = nu

class Snapper:
    """Snaps to data points"""

    def __init__(self, data, callback):
        self.data = data
        self.callback = callback

    def snap(self, x, y):
        dataidx = np.argmin(np.linalg.norm(self.data[:,:2] - np.array([x, y]), axis=1))
        datapos = self.data[dataidx,:]
        self.callback(datapos[0], datapos[1])
        return int(self.data[dataidx,2]), int(self.data[dataidx,3]) #just first two comp of datapos for toolbar

class Highlighter:
    def __init__(self, ax):
        self.ax = ax
        self.marker = None
        self.markerpos = None

    def draw(self, x, y):
        """draws a marker at plot position (x,y)"""
        if (x, y) != self.markerpos:
            if self.marker:
                self.marker.remove()
                del self.marker
            self.marker = self.ax.scatter(x, y, s=50, color='r')
            self.markerpos = (x, y)
            self.ax.figure.canvas.draw()

class SnappingNavigationToolbar(NavigationToolbar):
    """Navigation toolbar with data snapping"""

    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        self.snapper = None

    def set_snapper(self, snapper):
        self.snapper = snapper

    def mouse_move(self, event):
        if self.snapper and event.xdata and event.ydata:
            event.xdata, event.ydata = self.snapper.snap(event.xdata, event.ydata)
        super().mouse_move(event)


class MplWidget(FigureCanvasQTAgg):
    signal_q_changed = pyqtsignal(MomentumParam, name="momentumChanged")

    def __init__(self, original_data, parent=None):
        fig = Figure()
        super(MplWidget, self).__init__(fig)
        self.setParent(parent)
        self.original_data = original_data
        self.momentum_param = MomentumParam(2348, 0)
        # Set default colors array
        self.defaultColors = np.array(
            [
                [0, 0.4470, 0.7410],
                [0.8500, 0.3250, 0.0980],
                [0.9290, 0.6940, 0.1250],
                [0.4660, 0.6740, 0.1880],
                [0.6350, 0.0780, 0.1840],
                [0.4940, 0.1840, 0.5560],
                [0.3010, 0.7450, 0.9330],
            ]
        )

        # Create a figure with axes
        self.ax = self.figure.add_subplot(111)
        self.highlighter = Highlighter(self.ax)
        self.snapper = Snapper(self.original_data, self.highlighter.draw)
        self.ax.set_xlabel('Wavevector [au]')
        self.ax.set_ylabel('Wavenumber [cm'+ r"$^{-1}$" + ']')
        # Form the plot and shading
        self.bottomLeftX = 0
        self.bottomLeftY = 0
        self.topRightX = 0
        self.topRightY = 0
        self.x = np.array(
            [
                self.bottomLeftX,
                self.bottomLeftX,
                self.topRightX,
                self.topRightX,
                self.bottomLeftX,
            ]
        )
        self.y = np.array(
            [
                self.bottomLeftY,
                self.topRightY,
                self.topRightY,
                self.bottomLeftY,
                self.bottomLeftY,
            ]
        )

        (self.myPlot,) = self.ax.plot(self.x, self.y, color=self.defaultColors[0, :])
        self.aspan = self.ax.axvspan(
            self.bottomLeftX, self.topRightX, color=self.defaultColors[0, :], alpha=0
        )


        # Set moving flag false (determines if mouse is being clicked and dragged inside plot). Set graph snap
        self.moving = False
        self.plotSnap = 5

        # Set up connectivity
        self.cid1 = self.mpl_connect("button_press_event", self.onclick)
        self.cid2 = self.mpl_connect("button_release_event", self.onrelease)
        self.cid3 = self.mpl_connect("motion_notify_event", self.onmotion)

    def setSnapBase(self, base):
        return lambda value: base * float(value) / base

    def onclick(self, event):
        if event.inaxes is None:
            return

        if self.plotSnap <= 0:
            self.bottomLeftX = event.xdata
            self.bottomLeftY = event.ydata
        else:
            self.calculateSnapCoordinates = self.setSnapBase(self.plotSnap)
            self.bottomLeftX = self.calculateSnapCoordinates(event.xdata)
            self.bottomLeftY = self.calculateSnapCoordinates(event.ydata)
        if event.button == 1:
            try:
                self.aspan.remove()
            except:
                pass
            self.moving = True

        elif event.button == 3:
            dataidq, branch = self.snapper.snap(self.bottomLeftX, self.bottomLeftY)
            self.momentum_param.props['momentum'] = dataidq
            self.momentum_param.props['branch'] = branch
            self.signal_q_changed.emit(self.momentum_param)
        else:
            return


    def onrelease(self, event):
        if event.inaxes is None:
            return
        if self.plotSnap <= 0:
            self.topRightX = event.xdata
            self.topRightY = event.ydata
        else:
            try:
                calculateSnapCoordinates = self.setSnapBase(self.plotSnap)
                self.topRightX = calculateSnapCoordinates(event.xdata)
                self.topRightY = calculateSnapCoordinates(event.ydata)
            except:
                pass
        if event.button == 1:

            self.x = np.array(
                [
                    self.bottomLeftX,
                    self.bottomLeftX,
                    self.topRightX,
                    self.topRightX,
                    self.bottomLeftX,
                ]
            )
            self.y = np.array(
                [
                    self.bottomLeftY,
                    self.topRightY,
                    self.topRightY,
                    self.bottomLeftY,
                    self.bottomLeftY,
                ]
            )

            ylimDiff = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
            self.aspan = self.ax.axvspan(
                self.bottomLeftX,
                self.topRightX,
                (self.bottomLeftY - self.ax.get_ylim()[0]) / ylimDiff,
                (self.topRightY - self.ax.get_ylim()[0]) / ylimDiff,
                color=self.defaultColors[0, :],
                alpha=0.25,
            )
            LEFT = self.bottomLeftX
            RIGHT = self.topRightX
            BOTTOM = self.bottomLeftY
            TOP = self.topRightY
            if not ((LEFT==RIGHT) or (TOP==BOTTOM)):
                self.ax.set_xlim([LEFT, RIGHT])
                self.ax.set_ylim([BOTTOM, TOP])

            self.moving = False
            self.draw()

    def onmotion(self, event):
        if event.inaxes is None:
            return
        if self.plotSnap <= 0:
            self.topRightX = event.xdata
            self.topRightY = event.ydata
        else:
            self.calculateSnapCoordinates = self.setSnapBase(self.plotSnap)
            self.topRightX = self.calculateSnapCoordinates(event.xdata)
            self.topRightY = self.calculateSnapCoordinates(event.ydata)
        if not self.moving:
            return
        if event.button == 1:
            
            self.x = np.array(
                [
                    self.bottomLeftX,
                    self.bottomLeftX,
                    self.topRightX,
                    self.topRightX,
                    self.bottomLeftX,
                ]
            )
            self.y = np.array(
                [
                    self.bottomLeftY,
                    self.topRightY,
                    self.topRightY,
                    self.bottomLeftY,
                    self.bottomLeftY,
                ]
            )
            self.myPlot.set_xdata(self.x)
            self.myPlot.set_ydata(self.y)
            self.draw()

        else:
            return


class DispersionWindow(QtWidgets.QMainWindow):

    def __init__(self, input_json_data):
        super().__init__()
        WIDTH = 800
        HEIGHT = 800
        self.input_json_data = input_json_data
        
        self.resize(WIDTH, HEIGHT)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        NATOMS = self.input_json_data['natoms']
        NQPTS = len(self.input_json_data['distances'])
        QPOINTS = np.array(self.input_json_data['distances'])
        FREQS   = np.array(self.input_json_data['eigenvalues'])
        self.data = np.zeros((3*NATOMS*NQPTS, 4))
        acc = 0
        for i in range(NQPTS):
            for j in range(3*NATOMS):
                self.data[acc,:] = np.array((QPOINTS[i], FREQS[i, j], i, j))
                acc += 1
        self.canvas = MplWidget(original_data = self.data)
        self.signal = self.canvas.signal_q_changed
        layout.addWidget(self.canvas)

        self.reset_widget = QtWidgets.QPushButton("Reset")
        self.reset_widget.setCheckable(True)
        self.reset_widget.toggle()
        self.reset_widget.clicked.connect(self.btnstate)

        layout.addWidget(self.reset_widget)

        for j in range(3*NATOMS):
            self.canvas.ax.plot(QPOINTS, FREQS[:,j], 'b-')
        self.canvas.original_xlim = self.canvas.ax.get_xlim()
        self.canvas.original_ylim = self.canvas.ax.get_ylim()


    def btnstate(self):
        self.canvas.ax.set_xlim(self.canvas.original_xlim)
        self.canvas.ax.set_ylim(self.canvas.original_ylim)
        self.canvas.draw()
#=========================================================
# End of dispersion_widget.py
#=========================================================