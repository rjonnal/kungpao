import numpy as np
import time
import centroid
import config as kcfg
import cameras
import sys
from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget
import time


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


class Sensor(QThread):

    def __init__(self,camera,parent=None):
        super(QThread,self).__init__(parent)
        self.update_rate = 30.0
        self.cam = camera
        self.spots = self.cam.get_image()
        self.downsample = 2
        try:
            self.cmax = np.load('.gui_settings/cmax.npy')
        except Exception as e:
            print e
            self.cmax = kcfg.contrast_maximum
        try:
            self.cmin = np.load('.gui_settings/cmin.npy')
        except Exception as e:
            print e
            self.cmin = kcfg.contrast_minimum
        
    def write_settings(self):
        np.save('.gui_settings/cmax.npy',self.cmax)
        np.save('.gui_settings/cmin.npy',self.cmin)
        
        
    def update(self):
        print "sensor update at %0.3f"%(time.time()%1000)
        self.spots = self.cam.get_image()
        self.bmp_spots = self.bmpscale(self.spots)

    def bmpscale(self,arr):
        return np.round(np.clip((arr[::self.downsample,::self.downsample]-self.cmin)/(self.cmax-self.cmin),0,1)*255).astype(np.uint8)
        
    def run(self):
        print "sensor thread started"
        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(1.0/self.update_rate*1000.0)
        self.exec_()


class Mirror(QThread):

    def __init__(self,parent=None):
        super(QThread,self).__init__(parent)
        self.update_rate = 100.0
        
    def update(self):
        print "mirror update at %0.3f"%(time.time()%1000)

    def run(self):
        print "mirror thread started"
        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(1.0/self.update_rate*1000.0)
        self.exec_()


class Loop:
    def __init__(self,sensor,mirror,parent=None):
        self.sensor = sensor
        self.mirror = mirror

    def start(self):
        self.sensor.start()
        self.mirror.start()

        

        
class Gui(QWidget):
    def __init__(self,loop):
        super(Gui,self).__init__()
        self.initUi()
        self.loop = loop
        self.loop.start()

    def initUi(self):
        self.setGeometry(500, 500, 300, 300)
        self.pb = QPushButton("Button", self)
        self.pb.move(50, 50)


if __name__ == '__main__':    
    app = QApplication(sys.argv)
    camera = cameras.SimulatedCamera()
    sensor = Sensor(camera)
    mirror = Mirror()
    loop = Loop(sensor,mirror)
    gui = Gui(loop)
    gui.show()
    sys.exit(app.exec_())
