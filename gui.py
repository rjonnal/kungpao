import numpy as np
import time
import centroid
from kungpao import config as kcfg
from kungpao.cameras import SimulatedCamera
from PyQt5.QtCore import (pyqtSignal, QMutex, QMutexLocker, QPoint, QSize, Qt,
                          QThread, QWaitCondition, QLine)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb
from PyQt5.QtWidgets import QApplication, QWidget


class Loop(QThread):
    spots_image = pyqtSignal(QImage, float)

    def __init__(self, parent=None):
        super(Loop, self).__init__(parent)

        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.colormap = []

        self.abort = False
        self.iteration = 0
        self.iteration_step = 1
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
            
        self.bit_depth = kcfg.bit_depth
        self.cmax_abs = 2**self.bit_depth-1

        reference_coordinates = np.loadtxt('config/kungpao_reference_coordinates.txt')
        self.ref_x = reference_coordinates[:,0]
        self.ref_y = reference_coordinates[:,1]
        self.search_box_radius = kcfg.search_box_radius
        self.downsample = 2
        self.cam = SimulatedCamera()
        self.arr = self.bmpscale(self.cam.get_image())

        
    def write_settings(self):
        np.save('.gui_settings/cmax.npy',self.cmax)
        np.save('.gui_settings/cmin.npy',self.cmin)
        
    def bmpscale(self,arr):
        temp = self.add_clims(arr)
        return np.round(np.clip((arr[::self.downsample,::self.downsample]-self.cmin)/(self.cmax-self.cmin),0,1)*255).astype(np.uint8)

    def add_clims(self,arr):
        sy,sx = arr.shape
        lheight = int(round(float(self.cmin)/float(self.cmax_abs)*sy))
        rheight = int(round(float(self.cmax)/float(self.cmax_abs)*sy))

        if self.cmax>=self.cmin:
            fill_value = self.cmax
        else:
            fill_value = self.cmin
        
        if lheight>0:
            arr[sy-lheight:sy,:sx//50] = fill_value
        else:
            arr[:-lheight,:sx//50] = fill_value
            
        if rheight>0:
            arr[sy-rheight:sy,-sx//50:] = fill_value
        else:
            arr[:-rheight,-sx//50:] = fill_value
        return arr
        
    def __del__(self):
        self.mutex.lock()
        self.abort = True
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()
        
    def render(self, window_size):
        locker = QMutexLocker(self.mutex)

        if not self.isRunning():
            self.start(QThread.LowPriority)
        else:
            self.condition.wakeOne()

    def run(self):
        self.t0 = time.time()
        fps = -1
        while True:
            print fps,self.iteration
            self.iteration = self.iteration + self.iteration_step
            #self.mutex.lock()
            #self.mutex.unlock()

            #image = QImage(QSize(512,512), QImage.Format_RGB32)
            #arr = np.round(np.random.rand(512,512)*255).astype(np.uint8)
            self.spots = self.cam.get_image()

            self.bmp_spots = self.bmpscale(self.spots)
            sy,sx = self.bmp_spots.shape
            image = QImage(self.bmp_spots,sx,sy,QImage.Format_Grayscale8)
            #self.mutex.lock()
            #self.mutex.unlock()

            self.spots_image.emit(image, 1.0)
            t = time.time()-self.t0
            fps = self.iteration/t
            
class Gui(QWidget):
    def __init__(self, parent=None):
        super(Gui, self).__init__(parent)

        self.loop = Loop()
        self.pixmap = QPixmap()
        self.loop.spots_image.connect(self.updatePixmap)

        self.setWindowTitle("kungpao")
        self.scale_factor = 1.0/self.loop.downsample
        self.loop.start()
        self.resize(1600, self.loop.arr.shape[0])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        #self.draw_search_boxes()
        painter.drawPixmap(QPoint(), self.pixmap)

    def draw_square(self,x1,y1,wid):
        painter = QPainter(self.pixmap)
        #painter.drawLine(QLine(x1,y1,x2,y2))
        painter.setPen(QColor(0,127,255))
        painter.drawRect(x1,y1,wid,wid)

    def draw_search_boxes(self):
        ref_x = self.loop.ref_x
        ref_y = self.loop.ref_y
        rad = self.loop.search_box_radius
        for x,y in zip(ref_x,ref_y):
            x1 = (x-rad)*self.scale_factor
            y1 = (y-rad)*self.scale_factor
            wid = rad*2*self.scale_factor
            self.draw_square(x1,y1,wid)
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.shutdown()
        elif event.key() == Qt.Key_Minus:
            self.loop.cmax = self.loop.cmax - 50
        elif event.key() == Qt.Key_Equal:
            self.loop.cmax = self.loop.cmax + 50
        elif event.key() == Qt.Key_Underscore:
            self.loop.cmin = self.loop.cmin - 20
        elif event.key() == Qt.Key_Plus:
            self.loop.cmin = self.loop.cmin + 20
        else:
            super(Gui, self).keyPressEvent(event)

    def updatePixmap(self, image, scaleFactor):
        self.pixmap = QPixmap.fromImage(image)
        self.pixmapOffset = QPoint()
        self.lastDragPosition = QPoint()
        self.pixmapScale = scaleFactor
        self.update()

    def shutdown(self):
        self.loop.write_settings()
        sys.exit()
        
if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    widget = Gui()
    widget.show()
    sys.exit(app.exec_())
