import numpy as np
import time
import centroid
from kungpao import config as kcfg
from kungpao.cameras import SimulatedCamera
from PyQt5.QtCore import (pyqtSignal, QMutex, QMutexLocker, QPoint, QSize, Qt,
                          QThread, QWaitCondition, QLine, QTimer)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb
from PyQt5.QtWidgets import QApplication, QWidget


class Sensor(QThread):

    def __init__(self,parent=None):
        super(Sensor, self).__init__(parent)
        self.counter = 0
        self.timer = QTimer()
        self.timer.moveToThread(self)
        self.timer.timeout.connect(self.update)
        self.started.connect(self.start_timer)
        self.start()
        
    def start_timer(self):
        self.timer.start(100)

    def update(self):
        print 'Sensor: %d'%self.counter
        self.counter+=1
        
class Mirror(QThread):

    def __init__(self,parent=None):
        super(Mirror, self).__init__(parent)
        self.counter = 0
        self.timer = QTimer()
        self.timer.moveToThread(self)
        self.timer.timeout.connect(self.update)
        self.started.connect(self.start_timer)
        self.start()
        
    def start_timer(self):
        self.timer.start(100)

    def update(self):
        print 'Mirror: %d'%self.counter
        self.counter+=1
        
class SensorOld(QThread):

    def __init__(self,parent=None):
        super(Sensor, self).__init__(parent)
        self.counter = 0
        self.timer = QTimer()
        self.timer.moveToThread(self)
        
    def run(self):
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

    def update(self):
        print 'Sensor: %d'%self.counter
        self.counter+=1

class Gui(QWidget):
    def __init__(self,loop,parent=None):
        super(Gui, self).__init__(parent)
        self.loop = loop
        

class Loop:
    def __init__(self,sensor,mirror,parent=None):
        self.sensor = sensor
        self.mirror = mirror
        self.sensor.start()
        self.mirror.start()
        
if __name__ == '__main__':

    import sys

    sensor = Sensor()
    mirror = Mirror()
    loop = Loop(sensor,mirror)
    app = QApplication(sys.argv)
    widget = Gui(loop)
    widget.show()
    sys.exit(app.exec_())
