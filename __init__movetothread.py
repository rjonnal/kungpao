import numpy as np
from matplotlib import pyplot as plt
import time, sys, os
import centroid
import config as kcfg
import cameras
import sys, os
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, Qt, QPoint, QLine, QObject
from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,
                             QHBoxLayout, QVBoxLayout, QGraphicsScene,
                             QLabel,QGridLayout, QCheckBox, QFrame, QGroupBox,
                             QSpinBox,QDoubleSpinBox,QSizePolicy,QFileDialog,
                             QErrorMessage,QTextEdit)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb, QPen, QBitmap, QPalette, QIcon
import psutil
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from tools import error_message, now_string, prepend, colortable, get_ram, get_process


control_type_keys = ['indicator','boolean','int','float','action']
control_types = dict(zip(control_type_keys,range(len(control_type_keys))))

class Control:
    '''The control class is an abstraction of the idea of an instrument control.
    It holds a type of control, a getter function and optional setter functions 
    and ranges of acceptable values. Implementation of the Control.build method
    may vary among application development approaches or widget toolkits.'''
    def __init__(self,label,getter,setter=None,min_value=None,max_value=None,step=None,dtype=None):
        self.label = label
        self.getter = getter
        self.setter = setter
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.dtype = dtype

    def get(self):
        return self.getter()

    def set(self,val):
        self.setter(val)
        
class SearchBoxes:

    def __init__(self, x, y, search_box_half_width):
        self.half_width = search_box_half_width
        self.x = x
        self.y = y
        self.xmax = kcfg.image_width_px
        self.ymax = kcfg.image_height_px
        self.x1 = np.round(self.x - self.half_width).astype(np.int16)
        self.x2 = np.round(self.x + self.half_width).astype(np.int16)
        self.y1 = np.round(self.y - self.half_width).astype(np.int16)
        self.y2 = np.round(self.y + self.half_width).astype(np.int16)
        self.n = len(self.x1)
        self.action_functions = [self.grow,self.shrink]
        self.numerical_functions = []
        self.numerical_functions.append((self.get_search_box_half_width,self.set_search_box_half_width,'Search box half width',(1,kcfg.search_box_half_width_max,1)))
        
    def set_search_box_half_width(self,val):
        print 'setting to %d'%val
        self.set_half_width(val)

    def get_search_box_half_width(self):
        return self.half_width

    def get_lenslet_index(self,x,y):
        d = np.sqrt((self.x-x)**2+(self.y-y)**2)
        return np.argmin(d)
        
    def grow(self):
        '''Grow searchboxes'''
        if (self.x1.min>0 and self.x2.max()<self.xmax-1 and
            self.y1.min>0 and self.y2.max()<self.ymax-1):
            self.x1 = self.x1 - 1
            self.x2 = self.x2 + 1
            self.y1 = self.y1 - 1
            self.y2 = self.y2 + 1
            self.half_width = self.half_width+2
        
    def shrink(self):
        '''Shrink searchboxes'''
        if (self.x2-self.x1).min()>3 and (self.y2-self.y1).min()>3:
            self.x1 = self.x1 + 1
            self.x2 = self.x2 - 1
            self.y1 = self.y1 + 1
            self.y2 = self.y2 - 1
            self.half_width = self.half_width-2

    def left(self):
        '''Bump to left.'''
        if self.x1.min()>1:
            self.x-=1.0
            self.x1-=1
            self.x2-=1
            
    def right(self):
        '''Bump to right.'''
        if self.x2.max()<kcfg.image_width_px-1:
            self.x+=1.0
            self.x1+=1
            self.x2+=1

    def up(self):
        '''Bump up.'''
        if self.y1.min()>1:
            self.y-=1.0
            self.y1-=1
            self.y2-=1
           
    def down(self):
        '''Bump down.'''
        if self.y2.max()<kcfg.image_height_px-1:
            self.y+=1.0
            self.y1+=1
            self.y2+=1

            
    def set_half_width(self,val):
        self.half_width = val
        self.x1 = np.round(self.x - self.half_width).astype(np.int16)
        self.x2 = np.round(self.x + self.half_width).astype(np.int16)
        self.y1 = np.round(self.y - self.half_width).astype(np.int16)
        self.y2 = np.round(self.y + self.half_width).astype(np.int16)



        
class Clock(QObject):

    def __init__(self,update_rate=1.0,fps_interval=1.0,parent=None):
        super(Clock,self).__init__(parent=parent)
        self.update_rate = update_rate
        self.count = 0.0
        self.t0 = time.time()
        self.t = 0.0
        self.fps_interval = fps_interval
        self.fps_window = int(round(self.update_rate*self.fps_interval))
        self.fps = 0.0
        
    def reset(self):
        self.count = 0
        self.t0 = time.time()
            
    def tick(self):
        self.count = self.count + 1
        self.t = time.time() - self.t0
        if self.count==self.fps_window:
            self.fps = float(self.count)/self.t



class SensorState:
    def __init__(self,a,b,c,d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
class Sensor(QObject):

    sensor_updated = pyqtSignal(object)
    def __init__(self,camera,parent=None):
        super(Sensor,self).__init__(parent=parent)
        self.clock = Clock(update_rate=kcfg.sensor_update_rate)
        self.state = SensorState(np.random.randn(10),'foo',lambda x: x+np.random.rand(),self.clock.fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1.0/self.clock.update_rate*1000.0)
        print 'timer started'
        self.name = 'frank'
        
    def update(self):
        print self.clock.t
        self.state = SensorState(np.random.randn(10),'foo',lambda x: x+np.random.rand(),self.clock.fps)
        self.sensor_updated.emit(self.state)
        self.clock.tick()

    def say_hello(self):
        print 'sensor: hello'
        

class Loop(QObject):
    hello = pyqtSignal()
    def __init__(self,sensor,parent=None):
        super(Loop,self).__init__(parent=parent)
        self.sensor = sensor
        self.sensor_thread = QThread()
        self.hello.connect(self.sensor.say_hello)
        self.sensor.sensor_updated.connect(self.on_sensor_updated)
        
        self.sensor.moveToThread(self.sensor_thread)
        
    def on_sensor_updated(self,state):
        return
        print state.a
        print state.b
        print state.c(7)
        print state.d
        
    def update(self):
        print 'loop updated'
        
    def say_hello(self):
        self.hello.emit()
        print 'loop: hello'
        
class Gui(QWidget):
    def __init__(self,loop):
        super(Gui,self).__init__()
        self.loop = loop
        self.layout = QHBoxLayout()
        self.button = QPushButton('Hey')
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.loop.say_hello)
    
        
if __name__ == '__main__':

    # construct the QApplication
    app = QApplication(sys.argv)
    # create a camera
    camera = cameras.SimulatedCamera()

    sensor = Sensor(camera)
    loop = Loop(sensor)
    #mirror = Mirror(update_rate=kcfg.mirror_update_rate)
    #loop = Loop(sensor,mirror)
    gui = Gui(loop)
    
    gui.show()
    sys.exit(app.exec_())
