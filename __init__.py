import numpy as np
import time
import centroid
import config as kcfg
import cameras
import sys
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, Qt, QPoint, QLine
from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,
                             QHBoxLayout, QVBoxLayout, QGraphicsScene,
                             QLabel,QGridLayout, QCheckBox, QFrame, QGroupBox,
                             QSpinBox,QDoubleSpinBox)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb, QPen, QBitmap
import time
import os
import psutil
from matplotlib import pyplot as plt

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
        self.numerical_functions = []
        self.numerical_functions.append((self.get_search_box_half_width,self.set_search_box_half_width,'Search box half width',(1,kcfg.search_box_half_width_max,1)))


    def set_search_box_half_width(self,val):
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

    def set_half_width(self,val):
        self.half_width = val
        self.x1 = np.round(self.x - self.half_width).astype(np.int16)
        self.x2 = np.round(self.x + self.half_width).astype(np.int16)
        self.y1 = np.round(self.y - self.half_width).astype(np.int16)
        self.y2 = np.round(self.y + self.half_width).astype(np.int16)

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
        self.background_correction = kcfg.background_correction
        # Boolean settings
        self.estimate_background = kcfg.estimate_background
        self.show_search_boxes = kcfg.show_search_boxes
        self.show_slope_lines = kcfg.show_slope_lines
        self.boolean_functions = []
        self.boolean_functions.append((self.get_estimate_background,self.set_estimate_background,'Estimate &background'))
        self.numerical_functions = []
        self.numerical_functions.append((self.get_background_correction,self.set_background_correction,'Background correction',(-2**kcfg.bit_depth,2**kcfg.bit_depth,.5)))
        self.numerical_functions = self.numerical_functions + self.search_boxes.numerical_functions
        self.action_functions = []

    def set_background_correction(self,val):
        self.background_correction = val

    def get_background_correction(self):
        return self.background_correction
        
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
                                           xr,yr,self.estimate_background,
                                           self.background_correction)
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
                                       self.x_centroids,self.y_centroids,
                                       self.estimate_background,self.background_correction)
        
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
        self.update_rate = kcfg.mirror_update_rate
        self.count = 0.0
        self.t0 = time.time()
        self.action_functions = []
        self.numerical_functions = []
        self.boolean_functions = []
        self.action_functions.append((self.flatten,'&Flatten miror'))
        
    def flatten(self):
        pass
        
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
        self.spots_pixmap = QPixmap()
        self.single_spot_pixmap = QPixmap()
        
        self.loop = loop
        self.loop.start()
        self.loop.sensor.sensor_state.connect(self.update_pixmap)
        self.bit_depth = kcfg.bit_depth
        self.scale_factor = kcfg.interface_scale_factor
        self.downsample = int(round(1.0/self.scale_factor))
        if not self.scale_factor==1.0/float(self.downsample):
            sys.exit('kungpao.config.scale_factor must be 1.0, 0.5, 0.25, 0.2, or 0.1')
        
        try:
            self.cmax = np.loadtxt('.gui_settings/cmax.txt')
        except Exception as e:
            print e
            self.cmax = float(kcfg.contrast_maximum)
        try:
            self.cmin = np.loadtxt('.gui_settings/cmin.txt')
        except Exception as e:
            print e
            self.cmin = float(kcfg.contrast_minimum)

        self.slope_line_color = QColor(*kcfg.slope_line_color)
        self.slope_line_thickness = kcfg.slope_line_thickness
        self.slope_line_pen = QPen()
        self.slope_line_pen.setColor(self.slope_line_color)

        self.search_box_color = QColor(*kcfg.search_box_color)
        self.search_box_thickness = kcfg.search_box_thickness
        self.search_box_pen = QPen()
        self.search_box_pen.setColor(self.search_box_color)

        self.single_spot_color = QColor(*kcfg.single_spot_color)
        self.single_spot_thickness = kcfg.single_spot_thickness
        self.single_spot_pen = QPen()
        self.single_spot_pen.setColor(self.single_spot_color)
        
        self.spots_painter = QPainter(self.spots_pixmap)

        self.single_spot_index = 0
        
        #self.bmp_spots = self.bmpscale(self.loop.sensor.spots,self.downsample,self.cmin,self.cmax)
        
        self.show_search_boxes = kcfg.show_search_boxes
        self.show_slope_lines = kcfg.show_slope_lines
        self.boolean_functions = []
        self.boolean_functions.append((self.get_show_search_boxes,self.set_show_search_boxes,'Show &search boxes'))
        self.boolean_functions.append((self.get_show_slope_lines,self.set_show_slope_lines,'Show slope &lines'))
        self.numerical_functions = []

        self.numerical_functions.append((self.get_spots_cmin,self.set_spots_cmin,'Spots contrast min',(-2**kcfg.bit_depth,2**kcfg.bit_depth,1)))
        self.numerical_functions.append((self.get_spots_cmax,self.set_spots_cmax,'Spots contrast max',(-2**kcfg.bit_depth,2**kcfg.bit_depth,1)))
        self.action_functions = []
        self.action_functions.append((self.shutdown,'&Quit'))
        
        self.initUi()


    def quit(self):
        sys.exit()


    def set_spots_cmax(self,val):
        self.cmax = float(val)
    def get_spots_cmax(self):
        return self.cmax
        
    def set_spots_cmin(self,val):
        self.cmin = float(val)
    def get_spots_cmin(self):
        return self.cmin
        
    def set_show_search_boxes(self,val):
        self.show_search_boxes = val

    def get_show_search_boxes(self):
        return self.show_search_boxes
        
    def set_show_slope_lines(self,val):
        self.show_slope_lines = val

    def get_show_slope_lines(self):
        return self.show_slope_lines


    def select_single_spot(self,click):
        x = click.x()*self.downsample
        y = click.y()*self.downsample
        self.single_spot_index = self.loop.sensor.search_boxes.get_lenslet_index(x,y)
        
    def initUi(self):
        self.layout = QHBoxLayout()

        self.spots_label = QLabel()
        self.spots_label.mousePressEvent = self.select_single_spot
        self.spots_label.setPixmap(self.spots_pixmap)
        self.layout.addWidget(self.spots_label)

        images = QVBoxLayout()
        images.setAlignment(Qt.AlignTop)
        self.single_spot_label = QLabel()
        self.single_spot_label.setPixmap(self.single_spot_pixmap)
        images.addWidget(self.single_spot_label)
        self.layout.addLayout(images)
        
        controls = QVBoxLayout()
        sensor_controls = self.make_control_frame(self.loop.sensor,'Sensor')
        controls.addWidget(sensor_controls)

        mirror_controls = self.make_control_frame(self.loop.mirror,'Mirror')
        controls.addWidget(mirror_controls)
        
        ui_controls = self.make_control_frame(self,'User Interface')
        controls.addWidget(ui_controls)
        self.layout.addLayout(controls)

        self.setWindowTitle("kungpao")
        self.resize(800, self.loop.sensor.spots.shape[0]*self.scale_factor)
        self.setLayout(self.layout)

    def make_control_frame(self,obj,label=''):
        group_box = QGroupBox(label)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        idx = 0
        try:
            for tup in obj.action_functions:
                pb = QPushButton(tup[1])
                pb.clicked.connect(tup[0])
                layout.addWidget(pb)
                idx+=1
        except Exception as e:
            sys.exit(e)
        try:
            for tup in obj.boolean_functions:
                cb = QCheckBox(tup[2])
                cb.setChecked(tup[0]())
                cb.stateChanged.connect(tup[1])
                layout.addWidget(cb)
                idx+=1
        except Exception as e:
            pass
        try:
            for tup in obj.numerical_functions:
                step = tup[3][2]
                vmin = tup[3][0]
                vmax = tup[3][1]
                box = QHBoxLayout()
                box.addWidget(QLabel(tup[2]))
                print type(step)
                if type(step)==int:
                    sb = QSpinBox()
                    sb.valueChanged[int].connect(tup[1])
                elif type(step)==float:
                    sb = QDoubleSpinBox()
                    sb.valueChanged[float].connect(tup[1])
                sb.setValue(tup[0]())
                sb.setMinimum(tup[3][0])
                sb.setMaximum(tup[3][1])
                sb.setSingleStep(tup[3][2])
                box.addWidget(sb)
                layout.addLayout(box)
                idx+=1
        except Exception as e:
            pass
        group_box.setLayout(layout)
        return group_box

    
    def bmpscale(self,arr,downsample,cmin,cmax):
        return np.round(np.clip((arr[::downsample,::downsample]-cmin)/(cmax-cmin),0,1)*255).astype(np.uint8)
        
    def update_pixmap(self, xslopes, yslopes, data):
        # receives slopes and image from sensor signal
        # first, convert the numpy image data into a QImage
        k = self.single_spot_index
        self.bmp = self.bmpscale(data,self.downsample,self.cmin,self.cmax)
        single = data[self.loop.sensor.search_boxes.y1[k]:self.loop.sensor.search_boxes.y2[k]+1,
                      self.loop.sensor.search_boxes.x1[k]:self.loop.sensor.search_boxes.x2[k]+1]
        self.single_bmp = self.bmpscale(single,1,self.cmin,self.cmax)

        sy,sx = self.bmp.shape
        n_bytes = self.bmp.nbytes
        bytes_per_line = int(n_bytes/sy)
        image = QImage(self.bmp,sx,sy,bytes_per_line,QImage.Format_Grayscale8)
        self.spots_pixmap = QPixmap.fromImage(image)
        #self.spots_pixmap.convertFromImage(image)
        ssy,ssx = self.single_bmp.shape
        n_bytes = self.single_bmp.nbytes
        bytes_per_line = int(n_bytes/ssy)
        image = QImage(self.single_bmp,ssx,ssy,bytes_per_line,QImage.Format_Grayscale8)
        
        self.single_spot_pixmap = QPixmap.fromImage(image)
        #self.single_spot_pixmap.convertFromImage(image)
        self.update()

    def paintEvent(self, event):
        if self.show_search_boxes:
            self.draw_search_boxes()
        if self.show_slope_lines:
            self.draw_slopes()
        self.spots_label.setPixmap(self.spots_pixmap)
        self.single_spot_label.setPixmap(self.single_spot_pixmap.scaled(256,256))
        
    def draw_slopes(self,magnification=50):
        self.spots_painter.begin(self.spots_pixmap)
        self.spots_painter.setPen(self.slope_line_pen)
        for xc,yc,xr,yr in zip(self.loop.sensor.x_centroids,
                               self.loop.sensor.y_centroids,
                               self.loop.sensor.x_ref,self.loop.sensor.y_ref):
            xrs = xr*self.scale_factor
            yrs = yr*self.scale_factor
            xcs = xc*self.scale_factor
            ycs = yc*self.scale_factor
            dx = xcs-xrs
            dy = ycs-yrs
            self.spots_painter.drawLine(QLine(xrs,yrs,xrs+dx*magnification,yrs+dy*magnification))
        self.spots_painter.end()
        
    def draw_search_boxes(self):
        #width = self.loop.sensor.search_boxes.half_width*2*self.scale_factor
        self.spots_painter.begin(self.spots_pixmap)
        self.spots_painter.setPen(self.search_box_pen)
        for spot_index,(x1a,y1a,x2a,y2a) in enumerate(zip(self.loop.sensor.search_boxes.x1,
                                                          self.loop.sensor.search_boxes.y1,
                                                          self.loop.sensor.search_boxes.x2,
                                                          self.loop.sensor.search_boxes.y2)):
            x1 = x1a*self.scale_factor
            y1 = y1a*self.scale_factor
            x2 = x2a*self.scale_factor
            y2 = y2a*self.scale_factor
            width = float(x2a-x1a)*self.scale_factor
            if spot_index==self.single_spot_index:
                self.spots_painter.setPen(self.single_spot_pen)
            self.spots_painter.drawRect(x1,y1,width,width)
            if spot_index==self.single_spot_index:
                self.spots_painter.setPen(self.search_box_pen)
        self.spots_painter.end()

    def draw_square(self,x1,y1,wid,rgba=(0,127,255,127),thickness=1.0):
        painter = QPainter(self.spots_pixmap)
        #pen = QPen()
        #pen.setColor(self.
        painter.setPen(self.search_box_color)
        painter.drawRect(x1,y1,wid,wid)

    def draw_rect(self,x1,y1,wid,hei,rgba=(255,255,255,127)):
        r,g,b,a = rgba
        painter = QPainter(self.spots_pixmap)
        painter.setPen(QColor(r,g,b,a))
        painter.drawRect(x1,y1,wid,hei)

    def draw_line(self,x1,y1,x2,y2,rgba=(127,255,127,127)):
        r,g,b,a = rgba
        painter = QPainter(self.spots_pixmap)
        painter.setPen(QColor(r,g,b,a))
        painter.drawLine(QLine(x1,y1,x2,y2))
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.shutdown()
        elif event.key() == Qt.Key_Minus:
            self.cmax = self.cmax - 50
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
