# -*- coding: utf-8 -*-
#=========================================================
# Beginning of visualizer.py
# @author: Tristan Britt
# @email: tristan.britt@mail.mcgill.ca
# @description: This file contains util functions for the 
# visualizer including the widget to save dialog, etc.
#
# This software is part of a package distributed under the 
# GPLv3 license, see ..LICENSE.txt
#=========================================================
try:
    from sip import setapi
    setapi("QVariant", 2)
    setapi("QString", 2)
except ImportError:
    pass


# To switch between PyQt5 and PySide2 bindings just change the from import
from PyQt5 import QtCore, QtWidgets

# Provide automatic signal function selection for PyQt5/PySide2
pyqtsignal = QtCore.pyqtSignal if hasattr(QtCore, 'pyqtSignal') else QtCore.Signal

class CustomDialog(QtWidgets.QFileDialog):

   def __init__(self, *args, **kwargs):
      super(CustomDialog, self).__init__(*args, **kwargs)
      
      self.setWindowTitle("Where are we saving the animation?")

      QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

      self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
      self.buttonBox.accepted.connect(self.accept)
      self.buttonBox.rejected.connect(self.reject)

      self.layout = QtWidgets.QVBoxLayout()
      self.layout.addWidget(self.buttonBox)
    #   self.setLayout(self.layout)

OBJECT = {'spheres': [
                     ('speed', 1, 3, 'double', 2),
                     ('amplitude', 0.001, 0.10, 'double', 0.06),
                     ],
         }

# -----------------------------------------------------------------------------
class ObjectParam(object):
    """
    OBJECT parameter test
    """

    def __init__(self, name, list_param):
        self.name = name
        self.list_param = list_param
        self.props = {}
        self.props['visible'] = True
        self.props['show_bonds'] = False
        for nameV, minV, maxV, typeV, iniV in list_param:
            self.props[nameV] = iniV



# -----------------------------------------------------------------------------
class ObjectWidget(QtWidgets.QWidget):
    """
    Widget for editing OBJECT parameters
    """
    signal_objet_changed = pyqtsignal(ObjectParam, name='objectChanged')
    # signal_q_changed = pyqtsignal(MomentumParam, name="momentumChanged")

    def __init__(self, parent=None, param=None, momentum_param=None):
        super(ObjectWidget, self).__init__(parent)

        if param is None:
            self.param = ObjectParam('spheres', OBJECT['spheres'])
        else:
            self.param = param

        # if momentum_param is None:
        #     self.momentum_param = MomentumParam(2348, 0)
        # else:
        #     self.momentum_param = momentum_param

        self.gb_c = QtWidgets.QGroupBox(u"Hide/Show %s" % self.param.name)
        self.gb_c.setCheckable(True)
        self.gb_c.setChecked(self.param.props['visible'])
        self.gb_c.toggled.connect(self.update_param)

        self.gb_c2 = QtWidgets.QGroupBox(u"Hide/Show Bonds")
        self.gb_c2.setCheckable(True)
        self.gb_c2.setChecked(self.param.props['show_bonds'])
        self.gb_c2.toggled.connect(self.update_param)

        lL = []
        self.sp = []
        gb_c_lay = QtWidgets.QGridLayout()
        
        for nameV, minV, maxV, typeV, iniV in self.param.list_param:
            lL.append(QtWidgets.QLabel(nameV, self.gb_c))
            if typeV == 'double':
                self.sp.append(QtWidgets.QDoubleSpinBox(self.gb_c))
                self.sp[-1].setDecimals(2)
                self.sp[-1].setSingleStep(0.1)
                self.sp[-1].setLocale(QtCore.QLocale(QtCore.QLocale.English))
            elif typeV == 'int':
                self.sp.append(QtWidgets.QSpinBox(self.gb_c))
            self.sp[-1].setMinimum(minV)
            self.sp[-1].setMaximum(maxV)
            self.sp[-1].setValue(iniV)

        # Layout
        for pos in range(len(lL)):
            gb_c_lay.addWidget(lL[pos], pos, 0)
            gb_c_lay.addWidget(self.sp[pos], pos, 1)
            # Signal
            self.sp[pos].valueChanged.connect(self.update_param)

        self.gb_c.setLayout(gb_c_lay)
        # self.gb_c2.setLayout(gb_c_lay)
        vbox = QtWidgets.QVBoxLayout()
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.gb_c)
        vbox.addWidget(self.gb_c2)
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)
        self.setLayout(vbox)

    def update_param(self, option):
        """
        update param and emit a signal
        """
        self.param.props['visible'] = self.gb_c.isChecked()
        self.param.props['show_bonds'] = self.gb_c2.isChecked()
        keys = map(lambda x: x[0], self.param.list_param)
        for pos, nameV in enumerate(keys):
            self.param.props[nameV] = self.sp[pos].value()
        # emit signal
        self.signal_objet_changed.emit(self.param)
        # self.signal_q_changed.emit(self.momentum_param)
# -----------------------------------------------------------------------------
HELP_WINDOW="""

     Commands for atomic motion
     ------------------------------------

     X - show projection from x axis

     Y - show projection from y axis

     Z - show projection from z axis
     
     W - advance in y

     S - retreat in y

     D - advance in x

     A - retreat in x

     Left/Right - rotate azimuth

     Up/Down - rotate polar

     N - toggle normal projection

     P - toggle perspective projection

     B - toggle bonds

     Commands for dispersion viewer
     ------------------------------------

     Left click + drag - zoom to ROI

     Right click - adjust atomic motion
                   to nearest phonon mode
                   and momentum

"""
#=========================================================
# End of visualizer.py
#=========================================================