# This Python file uses the following encoding: utf-8
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QMessageBox
from PyQt6 import QtWebEngineWidgets
from PyQt6 import uic
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import numpy as np
from functions import *
import pyqtgraph as pg
import random
import math
import pandas as pd
from user_defined_func import *


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('form.ui', self)

        self.setWindowTitle('PCube')

        # Loading data in function combo box
        self.function_box.addItems(['', 'Low pass filter', 'High pass filter', 'Differentiation', 'Integration', 'Windowed Phasor', 'User defined function'])

        # Plotting of signals from users
        self.plot_button.clicked.connect(self.plotter)

        # Selecting the function to apply
        self.select_button.clicked.connect(self.selected)

        # Selecting default test case
        self.use_test.clicked.connect(self.useTest)

        # Night mode
        self.plotwidget.setBackground('w')
        self.background.clicked.connect(self.setBackG)

        # Show grid
        self.grid_check.clicked.connect(self.changeGrid)

        # Clear plot
        self.clear_button.clicked.connect(self.clearPlot)

        # Select between file input and function input
        self.fileinput.clicked.connect(self.changeFormat)

        # Default values of the buttons
        self.time_signal.setEnabled(True)
        self.signal.setEnabled(True)
        self.function_box.setEnabled(True)
        self.plot_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.file_1.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.file_signal_1.setEnabled(False)
        self.file_signal_2.setEnabled(False)
        self.threshold.setEnabled(False)
        self.samplingrate.setEnabled(False)
        self.forcycles.setEnabled(False)
        self.hyperparams.setEnabled(False)
        self.clear_file.setEnabled(False)

        # Browse File
        self.browse_button.clicked.connect(self.getfile)

        # Clear file input
        self.clear_file.clicked.connect(self.clearFile)

        # Help guide
        self.guide = guide1()
        self.help_button.clicked.connect(self.guide.show)

        # About
        self.about_button.clicked.connect(self.About)

# ---------------------------------------------------------------------------------------------------------------

    def plotter(self):
        pen = pg.mkPen(color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), width = 3)

        if not(self.fileinput.isChecked()):
            if not(self.keep_plot.isChecked()):
                self.plotwidget.clear()

            if self.use_test.isChecked():
                t = np.arange(0, 2, 1e-5)
                x = np.zeros(len(t))
                for i in range(len(t)):
                    if t[i] < 0.5:
                        x[i] = 10*np.sin(2*np.pi*50*t[i])
                    elif 0.5 <= t[i] < 1:
                        x[i] = 10*np.sin(2*np.pi*50*t[i])+2*np.sin(2*np.pi*500*t[i])
                    else:
                        x[i] = 8
            else:
                t = eval(self.time_signal.text())
                x = eval(self.signal.text())

            if self.function_box.currentText() == '':
                self.plotwidget.plot(t, x, pen=pen)
            elif self.function_box.currentText() == 'Low pass filter':
                self.threshold.setEnabled(True)
                y = mylowpass(t, x, float(self.threshold.text()))
                self.plotwidget.plot(t, y, pen=pen)
            elif self.function_box.currentText() == 'High pass filter':
                self.threshold.setEnabled(True)
                y = myhighpass(t, x, float(self.threshold.text()))
                self.plotwidget.plot(t, y, pen=pen)
            elif self.function_box.currentText() == 'Differentiation':
                y = derivative(t, x)
                self.plotwidget.plot(t, y, pen=pen)
            elif self.function_box.currentText() == 'Integration':
                y = integration(t, x)
                self.plotwidget.plot(t, y, pen=pen)
            elif self.function_box.currentText() == 'Windowed Phasor':
                y, t_new = window_phasor(x, t, int(self.samplingrate.text()), float(self.forcycles.text()))
                self.plotwidget.plot(t_new, y, pen=pen)
            elif self.function_box.currentText() == 'User defined function':
                hyper = eval(self.hyperparams.text())
                y, tnew = user_func(t, x, hyper)
                self.plotwidget.plot(tnew, y, pen=pen)
        else:

            if not (self.keep_plot.isChecked()):
                self.plotwidget.clear()

            df = pd.read_csv(self.file_1.text())

            t = df[self.file_signal_1.currentText()]
            x = df[self.file_signal_2.currentText()]

            if self.function_box.currentText() == '':
                self.plotwidget.plot(df[self.file_signal_1.currentText()], df[self.file_signal_2.currentText()], pen=pen)
            elif self.function_box.currentText() == 'Low pass filter':
                self.threshold.setEnabled(True)
                y = mylowpass(t, x, float(self.threshold.text()))
                self.plotwidget.plot(t, y, pen=pen)
            elif self.function_box.currentText() == 'High pass filter':
                self.threshold.setEnabled(True)
                y = myhighpass(t, x, float(self.threshold.text()))
                self.plotwidget.plot(t, y, pen=pen)
            elif self.function_box.currentText() == 'Differentiation':
                y = derivative(t, x)
                self.plotwidget.plot(t, y, pen=pen)
            elif self.function_box.currentText() == 'Integration':
                y = integration(t, x)
                self.plotwidget.plot(t, y, pen=pen)
            elif self.function_box.currentText() == 'Windowed Phasor':
                y, t_new = window_phasor(x, t, int(self.samplingrate.text()), float(self.forcycles.text()))
                self.plotwidget.plot(t_new, y, pen=pen)
            elif self.function_box.currentText() == 'User defined function':
                hyper = eval(self.hyperparams.text())
                y, tnew = user_func(t, x, hyper)
                self.plotwidget.plot(tnew, y, pen=pen)
# ----------------------------------------------------------------------------------------

    def selected(self):
        if self.function_box.currentText() in ['Low pass filter', 'High pass filter']:
            self.threshold.setEnabled(True)
            self.samplingrate.setEnabled(False)
            self.forcycles.setEnabled(False)
            self.hyperparams.setEnabled(False)
        elif self.function_box.currentText() == 'Windowed Phasor':
            self.threshold.setEnabled(False)
            self.samplingrate.setEnabled(True)
            self.hyperparams.setEnabled(False)
            self.forcycles.setEnabled(True)
        elif self.function_box.currentText() == 'User defined function':
            self.threshold.setEnabled(False)
            self.samplingrate.setEnabled(False)
            self.forcycles.setEnabled(False)
            self.hyperparams.setEnabled(True)
        else:
            self.threshold.setEnabled(False)
            self.samplingrate.setEnabled(False)
            self.forcycles.setEnabled(False)
            self.hyperparams.setEnabled(False)

# --------------------------------------------------------------------------------------------

    def clearPlot(self):
        self.plotwidget.clear()

    def changeGrid(self):
        if self.grid_check.isChecked():
            self.plotwidget.showGrid(x=True, y=True, alpha=1)
        else:
            self.plotwidget.showGrid(x=False, y=False)

    def setBackG(self):
        if self.background.isChecked():
            self.plotwidget.setBackground('black')
        else:
            self.plotwidget.setBackground('w')

    def useTest(self):
        if self.use_test.isChecked():
            self.time_signal.setEnabled(False)
            self.signal.setEnabled(False)
        else:
            self.time_signal.setEnabled(True)
            self.signal.setEnabled(True)

# --------------------------------------------------------------------------------------------------------

    def changeFormat(self):
        if self.fileinput.isChecked():
            self.time_signal.setEnabled(False)
            self.signal.setEnabled(False)
            self.function_box.setEnabled(True)
            self.plot_button.setEnabled(True)
            self.select_button.setEnabled(True)
            self.file_1.setEnabled(True)
            self.browse_button.setEnabled(True)
            self.file_signal_1.setEnabled(True)
            self.file_signal_2.setEnabled(True)
            self.use_test.setEnabled(False)
            self.clear_file.setEnabled(True)
        else:
            self.time_signal.setEnabled(True)
            self.signal.setEnabled(True)
            self.function_box.setEnabled(True)
            self.plot_button.setEnabled(True)
            self.select_button.setEnabled(True)
            self.file_1.setEnabled(False)
            self.browse_button.setEnabled(False)
            self.file_signal_1.setEnabled(False)
            self.file_signal_2.setEnabled(False)
            self.use_test.setEnabled(True)
            self.clear_file.setEnabled(False)

# ---------------------------------------------------------------------------------------------

    def getfile(self):
        self.file_signal_1.clear()
        self.file_signal_2.clear()
        dlg = QFileDialog(self)
        dlg.setFileMode
        filenames = QStringListModel()

        if dlg.exec():
            filenames = dlg.selectedFiles()
            f = open(filenames[0], 'r')
            with f:
                data = f.read()
                self.file_1.setText(filenames[0])

        df = pd.read_csv(self.file_1.text())

        self.file_signal_1.addItems([''])
        self.file_signal_1.addItems(df.columns)
        self.file_signal_2.addItems([''])
        self.file_signal_2.addItems(df.columns)

# -----------------------------------------------------------------------------------------------------

    def clearFile(self):
        self.file_1.clear()
        self.file_signal_1.clear()
        self.file_signal_2.clear()

    def About(self):
        QMessageBox.information(self, 'About', 'Created by Ajey Dikshit \n   June 2022')

# -----------------------------------------------------------------------------------------------------

#PDF = "C:/Users/dixit/OneDrive/Desktop/Ajey/Project/DRs/QT/Post_processing/hello.txt"

class guide1(QtWebEngineWidgets.QWebEngineView):
    def __init__(self):
        super(guide1, self).__init__()
        self.setWindowTitle('Tutorial')
        self.load(QUrl('https://drive.google.com/file/d/1zM1OHlAPwDCblt4KhIw_cARTkU13apvj/view?usp=sharing'))

if __name__ == "__main__":
    app = QApplication([])
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
