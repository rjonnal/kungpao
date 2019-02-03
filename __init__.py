import numpy as np
import time
import centroid
import config as kcfg
import cameras
import sys
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, Qt, QPoint, QLine
from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,
                             QHBoxLayout, QVBoxLayout, QGraphicsScene,
                             QLabel,QGridLayout, QCheckBox, QFrame)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb
import time
import os
import psutil

process = psutil.Process(os.getpid())

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

    def copy(self):
        rx = self.x.copy()
        ry = self.y.copy()
        sb = SearchBoxes(rx,ry,self.half_width)
        return sb


class Sensor(QThread):

    # The sensor should emit a signal with 3 numpy arrays:
    # x-_centroids, y-_centroids, and the spots image
    sensor_state = pyqtSignal(np.ndarray,np.ndarray,np.ndarray)
    
    def __init__(self,camera,parent=None):
        super(QThread,self).__init__(parent)
        
        # load some configuration data
        ref_xy = np.loadtxt(kcfg.reference_coordinates_filename)
        self.x_ref = ref_xy[:,0]
        self.y_ref = ref_xy[:,1]
        self.search_boxes = SearchBoxes(self.x_ref,self.y_ref,kcfg.search_box_half_width)
        self.lenslet_focal_length_m = kcfg.lenslet_focal_length_m
        self.pixel_size_m = kcfg.pixel_size_m
        self.update_rate = 50.0
        self.cam = camera
        self.spots = self.cam.get_image()
        self.count = 0
        self.t0 = time.time()
        self.x_centroids = self.x_ref.copy()
        self.y_centroids = self.y_ref.copy()
        self.x_slopes = np.zeros(self.search_boxes.n)
        self.y_slopes = np.zeros(self.search_boxes.n)
        self.n_iterations = kcfg.centroiding_iterations

        # Boolean settings
        self.estimate_background = kcfg.estimate_background
        self.boolean_functions = [(self.get_estimate_background,self.set_estimate_background,'Estimate background')]
        
    def set_estimate_background(self,val):
        self.estimate_background = val
        
    def get_estimate_background(self):
        return self.estimate_background
        
    def update(self):
        self.spots = self.cam.get_image()

        if self.n_iterations > 1:
            sb = self.search_boxes
            xr = self.x_ref.copy()
            yr = self.y_ref.copy()
            half_width = sb.half_width
            for iteration in range(self.n_iterations):
                print sb.half_width
                centroid.compute_centroids(self.spots,
                                           sb.x1,sb.x2,
                                           sb.y1,sb.y2,
                                           xr,yr,self.estimate_background)
                if half_width<=kcfg.iterative_centroiding_step+1:
                    break
                half_width-=kcfg.iterative_centroiding_step
                sb = SearchBoxes(xr,yr,half_width)
            self.x_centroids = xr
            self.y_centroids = yr
        else:
            centroid.compute_centroids(self.spots,
                                       self.search_boxes.x1,self.search_boxes.x2,
                                       self.search_boxes.y1,self.search_boxes.y2,
                                       self.x_centroids,self.y_centroids,self.estimate_background)
        
        self.x_centroids-=(self.x_centroids.mean()-self.x_ref.mean())
        self.y_centroids-=(self.y_centroids.mean()-self.y_ref.mean())

        self.x_slopes = (self.x_centroids-self.x_ref)*self.pixel_size_m/self.lenslet_focal_length_m
        self.y_slopes = (self.y_centroids-self.y_ref)*self.pixel_size_m/self.lenslet_focal_length_m
        
        self.sensor_state.emit(self.x_slopes,self.y_slopes,self.spots)
        self.count = self.count + 1
        
    def get_fps(self):
        #print(process.memory_info().rss)//1024//1024
        return float(self.count)/(time.time()-self.t0)

    def run(self):
        print "sensor thread started"
        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(1.0/self.update_rate*1000.0)
        self.exec_()


class Mirror(QThread):

    def __init__(self,parent=None):
        super(QThread,self).__init__(parent)
        self.update_rate = 500.0
        self.count = 0.0
        self.t0 = time.time()

        
    def update(self):
        self.count = self.count + 1

    def run(self):
        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(1.0/self.update_rate*1000.0)
        self.exec_()

    def get_fps(self):
        return float(self.count)/(time.time()-self.t0)


class Loop:
    def __init__(self,sensor,mirror,parent=None):
        self.sensor = sensor
        self.mirror = mirror
        self.sensor.sensor_state.connect(self.update)
        
    def start(self):
        self.sensor.start()
        self.mirror.start()

    def update(self,xslopes,yslopes,spots):
        print 'Sensor fps: %0.1f, Mirror fps: %0.1f, Loop slope: %0.3f'%(self.sensor.get_fps(),self.mirror.get_fps(),xslopes[0])

        
class Gui(QWidget):
    def __init__(self,loop):
        super(Gui,self).__init__()
        self.pixmap = QPixmap()
        self.loop = loop
        self.loop.start()
        self.loop.sensor.sensor_state.connect(self.update_pixmap)
        self.bit_depth = kcfg.bit_depth
        self.cmax_abs = 2**self.bit_depth-1

        self.scale_factor = kcfg.interface_scale_factor
        self.downsample = int(round(1.0/self.scale_factor))
        try:
            self.cmax = np.loadtxt('.gui_settings/cmax.txt')
        except Exception as e:
            print e
            self.cmax = kcfg.contrast_maximum
        try:
            self.cmin = np.loadtxt('.gui_settings/cmin.txt')
        except Exception as e:
            print e
            self.cmin = kcfg.contrast_minimum

        self.bmp_spots = self.bmpscale(self.loop.sensor.spots)
        self.show_boxes = kcfg.show_boxes
        self.show_slopes = kcfg.show_slopes
        self.initUi()
        
    def initUi(self):
        #self.setGeometry(500, 500, 300, 300)
        self.layout = QGridLayout()
        self.spots_label = QLabel()
        self.spots_label.setPixmap(self.pixmap)

        self.layout.addWidget(self.spots_label, 0, 0)
        #self.flatten_button = QPushButton("Flatten", self)
        #self.layout.addWidget(self.flatten_button, 0, 1)

        sensor_layout = self.make_control_frame(self.loop.sensor,'Sensor')
        self.layout.addWidget(sensor_layout,0,1)

        self.setWindowTitle("kungpao")
        self.resize(800, self.loop.sensor.spots.shape[0]*self.scale_factor)

        self.setLayout(self.layout)


    def make_control_frame(self,obj,label=''):
        frame = QFrame()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label))
        try:
            for tup in obj.boolean_functions:
                cb = QCheckBox(tup[2])
                cb.setChecked(tup[0]())
                cb.stateChanged.connect(tup[1])
                layout.addWidget(cb)
        except:
            pass
        layout.setSpacing(0.0)
        frame.setLayout(layout)
        return frame
        
    def draw_clims(self):
        sy,sx = self.bmp.shape
        lheight = float(self.cmin)/float(self.cmax_abs)*sy
        rheight = float(self.cmax)/float(self.cmax_abs)*sy

        if self.cmax>=self.cmin:
            fill_value = (255,255,255,255)
        else:
            fill_value = (0,0,0,255)

        width = sx//50
        print lheight
        
        if lheight<0:
            self.draw_rect(0,0,width-5,lheight,fill_value)
        else:
            self.draw_rect(0,sy-lheight,width-5,lheight,fill_value)
            
        if rheight<0:
            self.draw_rect(sx-width,0,width-5,rheight,fill_value)
        else:
            self.draw_rect(sx-width,sy-rheight,width-5,rheight,fill_value)
    
    def bmpscale(self,arr):
        return np.round(np.clip((arr[::self.downsample,::self.downsample]-self.cmin)/(self.cmax-self.cmin),0,1)*255).astype(np.uint8)
        
    def update_pixmap(self, xslopes, yslopes, data):
        # receives slopes and image from sensor signal
        # first, convert the numpy image data into a QImage
        self.bmp = self.bmpscale(data)
        sy,sx = self.bmp.shape
        image = QImage(self.bmp,sx,sy,QImage.Format_Grayscale8)
        self.pixmap = QPixmap.fromImage(image)
        self.pixmap_scale = self.scale_factor
        self.update()
        
    def paintEvent(self, event):
        #painter = QPainter(self)
        #painter.fillRect(self.rect(), Qt.black)
        if self.show_boxes:
            self.draw_search_boxes()
        if self.show_slopes:
            self.draw_slopes()
        self.draw_clims()
        #painter.drawPixmap(QPoint(), self.pixmap)
        self.spots_label.setPixmap(self.pixmap)
        
    def draw_slopes(self,magnification=50):
        for xc,yc,xr,yr in zip(self.loop.sensor.x_centroids,
                               self.loop.sensor.y_centroids,
                               self.loop.sensor.x_ref,self.loop.sensor.y_ref):
            xrs = xr*self.scale_factor
            yrs = yr*self.scale_factor
            xcs = xc*self.scale_factor
            ycs = yc*self.scale_factor
            dx = xcs-xrs
            dy = ycs-yrs
            self.draw_line(xrs,yrs,xrs+dx*magnification,yrs+dy*magnification)
        
    def draw_search_boxes(self):
        #width = self.loop.sensor.search_boxes.half_width*2*self.scale_factor
        for x1a,y1a,x2a,y2a in zip(self.loop.sensor.search_boxes.x1,
                                   self.loop.sensor.search_boxes.y1,
                                   self.loop.sensor.search_boxes.x2,
                                   self.loop.sensor.search_boxes.y2):
            x1 = x1a*self.scale_factor
            y1 = y1a*self.scale_factor
            x2 = x2a*self.scale_factor
            y2 = y2a*self.scale_factor
            width = float(x2a-x1a)*self.scale_factor
            self.draw_square(x1,y1,width)

    def draw_square(self,x1,y1,wid,rgba=(0,127,255,127)):
        r,g,b,a = rgba
        painter = QPainter(self.pixmap)
        #painter.drawLine(QLine(x1,y1,x2,y2))
        painter.setPen(QColor(r,g,b,a))
        painter.drawRect(x1,y1,wid,wid)

    def draw_rect(self,x1,y1,wid,hei,rgba=(255,255,255,127)):
        r,g,b,a = rgba
        painter = QPainter(self.pixmap)
        painter.setPen(QColor(r,g,b,a))
        painter.drawRect(x1,y1,wid,hei)

    def draw_line(self,x1,y1,x2,y2,rgba=(127,255,127,127)):
        r,g,b,a = rgba
        painter = QPainter(self.pixmap)
        painter.setPen(QColor(r,g,b,a))
        painter.drawLine(QLine(x1,y1,x2,y2))
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.shutdown()
        elif event.key() == Qt.Key_Escape:
            self.shutdown()
        elif event.key() == Qt.Key_Minus:
            self.cmax = self.cmax - 50
        elif event.key() == Qt.Key_B:
            self.show_boxes = not self.show_boxes
        elif event.key() == Qt.Key_S:
            self.show_slopes = not self.show_slopes
        elif event.key() == Qt.Key_Equal:
            self.cmax = self.cmax + 50
        elif event.key() == Qt.Key_Underscore:
            self.cmin = self.cmin - 20
        elif event.key() == Qt.Key_Plus:
            self.cmin = self.cmin + 20
        elif event.key() == Qt.Key_PageUp:
            self.loop.sensor.search_boxes.grow()
        elif event.key() == Qt.Key_PageDown:
            self.loop.sensor.search_boxes.shrink()
        else:
            super(Gui, self).keyPressEvent(event)

    def shutdown(self):
        self.write_settings()
        self.close()
        
    def write_settings(self):
        print self.cmin,self.cmax
        np.savetxt('.gui_settings/cmax.txt',[self.cmax])
        np.savetxt('.gui_settings/cmin.txt',[self.cmin])
        

if __name__ == '__main__':

    # construct the QApplication
    app = QApplication(sys.argv)

    # create a camera
    camera = cameras.SimulatedCamera()

    sensor = Sensor(camera)
    mirror = Mirror()
    loop = Loop(sensor,mirror)
    gui = Gui(loop)
    
    gui.show()
    sys.exit(app.exec_())
