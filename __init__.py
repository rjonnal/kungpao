import numpy as np
import time
import centroid
import config as kcfg
import cameras
from PyQt5.QtCore import (pyqtSignal, QMutex, QMutexLocker, QPoint, QSize, Qt,
                          QThread, QWaitCondition, QLine)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb
from PyQt5.QtWidgets import QApplication, QWidget


class ReferenceCoordinates:

    def __init__(self, x, y):
        self.x = x
        self.y = y

class SearchBoxes:

    def __init__(self, reference_coordinates, search_box_half_width):
        self.half_width = search_box_half_width
        self.reference_coordinates = reference_coordinates
        self.x1 = np.round(self.reference_coordinates.x - self.half_width).astype(np.int16)
        self.x2 = np.round(self.reference_coordinates.x + self.half_width).astype(np.int16)
        self.y1 = np.round(self.reference_coordinates.y - self.half_width).astype(np.int16)
        self.y2 = np.round(self.reference_coordinates.y + self.half_width).astype(np.int16)



class Sensor:

    def __init__(self, camera):

        self.cam = camera
        ref_temp = np.loadtxt(kcfg.reference_coordinates_filename)
        ref = ReferenceCoordinates(ref_temp[:,0],ref_temp[:,1])
        sb = SearchBoxes(ref,kcfg.search_box_half_width)

if __name__=='__main__':

    camera = cameras.SimulatedCamera()
    sensor = Sensor(camera)
    
