import numpy as np
import time
import centroid
import config as kcfg
from poke import Poke
import cameras
import sys
from PyQt5.QtCore import (QThread, QTimer, pyqtSignal, Qt, QPoint, QLine,
                          QMutex, QObject, pyqtSlot)

from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,
                             QHBoxLayout, QVBoxLayout, QGraphicsScene,
                             QLabel,QGridLayout, QCheckBox, QFrame, QGroupBox,
                             QSpinBox,QDoubleSpinBox,QSizePolicy,QFileDialog,
                             QErrorMessage, QSlider)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb, QPen, QBitmap, QPalette, QIcon
import time
import os
import psutil
from matplotlib import pyplot as plt
import datetime
from tools import error_message, now_string, prepend, colortable, get_ram, get_process
import copy
from zernike import Reconstructor
import cProfile
import scipy.io as sio
from poke_analysis import save_modes_chart
from ctypes import CDLL,c_void_p
try:
    from alpao.PyAcedev5 import *
except Exception as e:
    print e

try:
    assert os.path.exists(kcfg.kungpao_root)
except AssertionError as ae:
    sys.exit('Problem with config.kungpao_root')

try:
    assert os.path.exists(kcfg.logging_directory)
except AssertionError as ae:
    os.mkdir(kcfg.logging_directory)
    
class Mutex(QMutex):

    def __init__(self):
        super(Mutex,self).__init__()

    def lock(self):
        super(Mutex,self).lock()
        return

    def unlock(self):
        super(Mutex,self).unlock()
        return

sensor_mutex = QMutex()
mirror_mutex = QMutex()

class ReferenceGenerator:
    def __init__(self,camera,mask,x_offset=0.0,y_offset=0.0,spot_half_width=5,window_spots=False):
        self.cam = camera
        sensor_x = kcfg.image_width_px
        sensor_y = kcfg.image_height_px
        lenslet_pitch = kcfg.lenslet_pitch_m
        pixel_size = kcfg.pixel_size_m
        stride = lenslet_pitch/pixel_size
        my,mx = mask.shape
        xvec = np.arange(stride/2.0,mx*stride,stride)+x_offset
        yvec = np.arange(stride/2.0,my*stride,stride)+y_offset
        ref_xy = []
        for y in range(my):
            for x in range(mx):
                if mask[y,x]:
                    ref_xy.append((xvec[x],yvec[y]))
        self.xy = np.array(ref_xy)
        self.x_ref = self.xy[:,0]
        self.y_ref = self.xy[:,1]
        
        sim = np.zeros((kcfg.image_height_px,kcfg.image_width_px))
        for x,y in zip(self.x_ref,self.y_ref):
            ry,rx = int(round(y)),int(round(x))
            sim[ry-spot_half_width:ry+spot_half_width+1,
                rx-spot_half_width:rx+spot_half_width+1] = 1.0

        N = 10
        spots = self.cam.get_image().astype(np.float)
        for k in range(N-1):
            spots = spots + self.cam.get_image()
        spots = spots/np.float(N)
        
        if window_spots:
            sy,sx = sim.shape
            XX,YY = np.meshgrid(np.arange(sx),np.arange(sy))
            xcom = np.sum(spots*XX)/np.sum(spots)
            ycom = np.sum(spots*YY)/np.sum(spots)
            XX = XX-xcom
            YY = YY-ycom
            d = np.sqrt(XX**2+YY**2)
            sigma = kcfg.image_width_px//2
            g = np.exp((-d**2)/(2*sigma**2))
            spots = spots*g
        
        # cross-correlate it with a spots image to find the most likely offset
        nxc = np.abs(np.fft.ifft2(np.fft.fft2(spots)*np.conj(np.fft.fft2(sim))))

        sy,sx = nxc.shape
        ymax,xmax = np.unravel_index(np.argmax(nxc),nxc.shape)
        if ymax>sy//2:
            ymax = ymax-sy
        if xmax>sx//2:
            xmax = xmax-sx
        
        new_x_ref = self.x_ref+xmax
        new_y_ref = self.y_ref+ymax

        self.xy = np.vstack((new_x_ref,new_y_ref)).T
        
    def make_coords(self):
        outfn = os.path.join(kcfg.reference_directory,'%s_coords.txt'%now_string())
        print 'Reference coordinates saved in %s'%outfn
        print 'Please add the following line to config.py:'
        print "reference_coordinates_filename = '%s'"%outfn
        np.savetxt(outfn,self.xy)
        
class FrameTimer:
    def __init__(self,label,buffer_size=100,verbose=False):
        self.index = 0
        self.fps = 0.0
        self.frame_time = 0.0
        self.frame_rms = 0.0
        self.buff = np.zeros(buffer_size)
        self.buffer_size = buffer_size
        self.label = label
        self.verbose = verbose
        
    def tick(self):
        self.buff[self.index] = time.time()
        self.index = self.index + 1
        if self.index==self.buffer_size:
            # buffer full--compute
            dt = np.diff(self.buff)
            self.frame_time = dt.mean()
            self.frame_rms = dt.std()
            self.fps = 1.0/self.frame_time
            self.index=0
            if self.verbose:
                print '%s: %0.1f (ms) %0.1f (ms std) %0.1f (fps)'%(self.label,1000.*self.frame_time,1000.*self.frame_rms,self.fps)
                
class SearchBoxes(QObject):

    def __init__(self,x,y,half_width):
        super(SearchBoxes,self).__init__()
        self.x = x
        self.y = y
        self.half_width = half_width
        self.xmax = kcfg.image_width_px - 1
        self.ymax = kcfg.image_height_px - 1
        self.x1 = np.round(self.x - self.half_width).astype(np.int16)
        self.x2 = np.round(self.x + self.half_width).astype(np.int16)
        self.y1 = np.round(self.y - self.half_width).astype(np.int16)
        self.y2 = np.round(self.y + self.half_width).astype(np.int16)
        self.n = len(self.x1)
        if not self.in_bounds(self.x1,self.x2,self.y1,self.y2):
            sys.exit('Search boxes extend beyond image edges. x %d %d, y %d, %d.'%
                     (self.x1.min(),self.x2.max(),self.y1.min(),self.y2.max()))

    def resize(self,new_half_width):
        x1 = np.round(self.x - self.half_width).astype(np.int16)
        x2 = np.round(self.x + self.half_width).astype(np.int16)
        y1 = np.round(self.y - self.half_width).astype(np.int16)
        y2 = np.round(self.y + self.half_width).astype(np.int16)
        
        # Check to make sure none of the search boxes are out of bounds:
        if self.in_bounds(x1,x2,y1,y2):
            self.half_width = new_half_width
            self.x1 = x1
            self.x2 = x2
            self.y1 = y1
            self.y2 = y2

    def move(self,x,y):
        self.x = x
        self.y = y
        self.x1 = np.round(self.x - self.half_width).astype(np.int16)
        self.x2 = np.round(self.x + self.half_width).astype(np.int16)
        self.y1 = np.round(self.y - self.half_width).astype(np.int16)
        self.y2 = np.round(self.y + self.half_width).astype(np.int16)
        if not self.in_bounds(self.x1,self.x2,self.y1,self.y2):
            sys.exit('Search boxes extend beyond image edges. x %d %d, y %d, %d.'%
                     (self.x1.min(),self.x2.max(),self.y1.min(),self.y2.max()))

    def in_bounds(self,x1,x2,y1,y2):
        return (x1.min()>=0 and x2.max()<=self.xmax and
                y1.min()>=0 and y2.max()<=self.ymax)

    def get_index(self,x,y):
        d = np.sqrt((self.x-x)**2+(self.y-y)**2)
        return np.argmin(d)

    def copy(self):
        x = np.zeros(self.x.shape)
        y = np.zeros(self.y.shape)
        x[:] = self.x[:]
        y[:] = self.y[:]
        sb = SearchBoxes(x,y,self.half_width)
        return sb

        
class Sensor(QObject):

    finished = pyqtSignal()
    
    def __init__(self,camera):
        super(Sensor,self).__init__()
        self.image_width_px = kcfg.image_width_px
        self.image_height_px = kcfg.image_height_px
        self.lenslet_pitch_m = kcfg.lenslet_pitch_m
        self.lenslet_focal_length_m = kcfg.lenslet_focal_length_m
        self.pixel_size_m = kcfg.pixel_size_m
        self.beam_diameter_m = kcfg.beam_diameter_m
        self.wavelength_m = kcfg.wavelength_m
        self.background_correction = kcfg.background_correction
        self.centroiding_iterations = kcfg.centroiding_iterations
        self.iterative_centroiding_step = kcfg.iterative_centroiding_step
        self.filter_lenslets = kcfg.sensor_filter_lenslets
        self.estimate_background = kcfg.estimate_background
        self.mask = np.loadtxt(kcfg.reference_mask_filename)
        self.reconstruct_wavefront = kcfg.sensor_reconstruct_wavefront
        self.remove_tip_tilt = kcfg.sensor_remove_tip_tilt
        xy = np.loadtxt(kcfg.reference_coordinates_filename)
        self.search_boxes = SearchBoxes(xy[:,0],xy[:,1],kcfg.search_box_half_width)
        self.x0 = np.zeros(self.search_boxes.x.shape)
        self.y0 = np.zeros(self.search_boxes.y.shape)
        
        self.x0[:] = self.search_boxes.x[:]
        self.y0[:] = self.search_boxes.y[:]
        
        self.n = self.search_boxes.n
        n_lenslets = self.n
        self.image = np.zeros((kcfg.image_height_px,kcfg.image_width_px))
        self.x_slopes = np.zeros(n_lenslets)
        self.y_slopes = np.zeros(n_lenslets)
        self.x_centroids = np.zeros(n_lenslets)
        self.y_centroids = np.zeros(n_lenslets)
        self.box_maxes = np.zeros(n_lenslets)
        self.box_mins = np.zeros(n_lenslets)
        self.box_means = np.zeros(n_lenslets)
        self.box_backgrounds = np.zeros(n_lenslets)
        self.error = 0.0
        self.tip = 0.0
        self.tilt = 0.0
        self.zernikes = None
        self.wavefront = None

        self.cam = camera
        self.frame_timer = FrameTimer('Sensor',verbose=False)
        self.reconstructor = Reconstructor(self.search_boxes.x,
                                           self.search_boxes.y,self.mask)
        self.logging = False
        self.paused = False
        
    @pyqtSlot()
    def update(self):
        if not self.paused:
            try:
                self.sense()
            except Exception as e:
                print e
            if self.logging:
                self.log()
                
        self.finished.emit()
        self.frame_timer.tick()

    @pyqtSlot()
    def pause(self):
        print 'sensor paused'
        self.paused = True

    @pyqtSlot()
    def unpause(self):
        print 'sensor unpaused'
        self.paused = False
        #self.sense()

    def log(self):
        outfn = os.path.join(kcfg.logging_directory,'sensor_%s.mat'%(now_string(True)))
        d = {}
        d['x_slopes'] = self.x_slopes
        d['y_slopes'] = self.y_slopes
        d['x_centroids'] = self.x_centroids
        d['y_centroids'] = self.y_centroids
        d['search_box_x1'] = self.search_boxes.x1
        d['search_box_x2'] = self.search_boxes.x2
        d['search_box_y1'] = self.search_boxes.y1
        d['search_box_y2'] = self.search_boxes.y2
        d['ref_x'] = self.search_boxes.x
        d['ref_y'] = self.search_boxes.y
        d['error'] = self.error
        d['tip'] = self.tip
        d['tilt'] = self.tilt
        d['wavefront'] = self.wavefront
        d['zernikes'] = self.zernikes
        
        sio.savemat(outfn,d)

    def set_background_correction(self,val):
        sensor_mutex.lock()
        self.background_correction = val
        sensor_mutex.unlock()


    def set_logging(self,val):
        self.logging = val


    def set_defocus(self,val):
        self.pause()
        
        newx = self.x0 + self.reconstructor.defocus_dx*val*kcfg.zernike_dioptric_equivalent
        
        newy = self.y0 + self.reconstructor.defocus_dy*val*kcfg.zernike_dioptric_equivalent
        self.search_boxes.move(newx,newy)
        
        self.unpause()
        
    def sense(self):
        image = cam.get_image()
        sensor_mutex.lock()
        sb = self.search_boxes
        xr = np.zeros(self.search_boxes.x.shape)
        xr[:] = self.search_boxes.x[:]
        yr = np.zeros(self.search_boxes.y.shape)
        yr[:] = self.search_boxes.y[:]
        half_width = sb.half_width
        for iteration in range(self.centroiding_iterations):
            #QApplication.processEvents()
            msi = iteration==self.centroiding_iterations-1
            centroid.compute_centroids(spots_image=image,
                                       sb_x1_vec=sb.x1,
                                       sb_x2_vec=sb.x2,
                                       sb_y1_vec=sb.y1,
                                       sb_y2_vec=sb.y2,
                                       x_out=xr,
                                       y_out=yr,
                                       mean_intensity = self.box_means,
                                       maximum_intensity = self.box_maxes,
                                       minimum_intensity = self.box_mins,
                                       background_intensity = self.box_backgrounds,
                                       estimate_background = self.estimate_background,
                                       background_correction = self.background_correction,
                                       num_threads = 1,
                                       modify_spots_image = msi)
            half_width-=self.iterative_centroiding_step
            sb = SearchBoxes(xr,yr,half_width)
            
        self.x_centroids[:] = xr[:]
        self.y_centroids[:] = yr[:]
        self.x_slopes = (self.x_centroids-self.search_boxes.x)*self.pixel_size_m/self.lenslet_focal_length_m
        self.y_slopes = (self.y_centroids-self.search_boxes.y)*self.pixel_size_m/self.lenslet_focal_length_m
        self.tilt = np.mean(self.x_slopes)
        self.tip = np.mean(self.y_slopes)
        if self.remove_tip_tilt:
            self.x_slopes-=self.tilt
            self.y_slopes-=self.tip
        self.image = image
        if self.reconstruct_wavefront:
            self.zernikes,self.wavefront,self.error = self.reconstructor.get_wavefront(self.x_slopes,self.y_slopes)
        sensor_mutex.unlock()

    
    def record_reference(self):
        print 'recording reference'
        self.pause()
        xcent = []
        ycent = []
        for k in range(kcfg.reference_n_measurements):
            print 'measurement %d of %d'%(k+1,kcfg.reference_n_measurements),
            self.sense()
            print '...done'
            xcent.append(self.x_centroids)
            ycent.append(self.y_centroids)
            
        x_ref = np.array(xcent).mean(0)
        y_ref = np.array(ycent).mean(0)
        self.search_boxes = SearchBoxes(x_ref,y_ref,self.search_boxes.half_width)
        outfn = os.path.join(kcfg.reference_directory,prepend('coords.txt',now_string()))
        refxy = np.array((x_ref,y_ref)).T
        np.savetxt(outfn,refxy,fmt='%0.2f')
        self.unpause()
        time.sleep(1)


class MirrorController(object):
    def __init__(self):
        self.cmax = kcfg.mirror_command_max
        self.cmin = kcfg.mirror_command_min
        self.command = np.zeros(kcfg.mirror_n_actuators,dtype=np.double)
        self.clipped = False
        
    def clip(self):
        self.clipped = (self.command.max()>=self.cmax or self.command.min()<=self.cmin)
        if self.clipped:
            self.command = np.clip(self.command,self.cmin,self.cmax)
                
    def set(self,vec):
        self.command[:] = vec[:]
        
    def send(self):
        return 1
        
class MirrorControllerCtypes(MirrorController):
    def __init__(self):
        super(MirrorControllerCtypes,self).__init__()
        self.acedev5 = CDLL("acedev5")
        self.mirror_id = self.acedev5.acedev5Init(0)
        self.command = np.zeros(kcfg.mirror_n_actuators,dtype=np.double)
        self.command_ptr = self.command.ctypes.data_as(c_void_p)
        self.send()
        
    def set(self,vec):
        assert len(vec)==len(self.command)
        self.command[:] = vec[:]

    def send(self):
        self.clip()
        return self.acedev5.acedev5Send(self.mirror_id,self.command_ptr)
        
        
class MirrorControllerPython(MirrorController):
    def __init__(self):
    
        super(MirrorControllerPython,self).__init__()        
        self.mirror_id = kcfg.mirror_id
        self.dm = PyAcedev5(self.mirror_id)
        self.command = self.dm.values
        self.send()
        
    def set(self,vec):
        self.dm.values[:] = vec[:]
        
    def send(self):
        self.clip()
        return self.dm.Send()
        
class Mirror(QObject):
    finished = pyqtSignal(QObject)
    
    def __init__(self):
        super(Mirror,self).__init__()
        
        try:
            self.controller = MirrorControllerPython()
        except Exception as e:
            print e
            try:
                self.controller = MirrorControllerCtypes()
            except Exception as e:
                print e
                print 'No mirror driver found. Using virtual mirror.'
                self.controller = MirrorController()
            
        
        self.mask = np.loadtxt(kcfg.mirror_mask_filename)
        self.n = kcfg.mirror_n_actuators
        self.flat = np.loadtxt(kcfg.mirror_flat_filename)
        self.command_max = kcfg.mirror_command_max
        self.command_min = kcfg.mirror_command_min
        self.settling_time = kcfg.mirror_settling_time_s
        self.update_rate = kcfg.mirror_update_rate
        
        self.flatten()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1.0/self.update_rate*1000.0)
        self.frame_timer = FrameTimer('Mirror',verbose=False)
        self.logging = False
        self.paused = False
        
    @pyqtSlot()
    def update(self):
        if not self.paused:
            self.send()
        if self.logging:
            self.log()
        self.frame_timer.tick()

    @pyqtSlot()
    def pause(self):
        print 'mirror paused'
        self.paused = True

    @pyqtSlot()
    def unpause(self):
        print 'mirror unpaused'
        self.paused = False

    def send(self):
        self.controller.send()

    def flatten(self):
        mirror_mutex.lock()
        self.controller.set(self.flat)
        mirror_mutex.unlock()
        self.send()

    def set_actuator(self,index,value):
        mirror_mutex.lock()
        self.controller.command[index] = value
        mirror_mutex.unlock()
        self.send()
        
    def set_command(self,vec):
        self.controller.set(vec)
        
    def get_command(self):
        return self.controller.command
        
    def log(self):
        outfn = os.path.join(kcfg.logging_directory,'mirror_%s.mat'%(now_string(True)))
        d = {}
        d['command'] = self.controller.command
        sio.savemat(outfn,d)

    def set_logging(self,val):
        self.logging = val
        
        
class Loop(QObject):

    finished = pyqtSignal()
    pause_signal = pyqtSignal()
    unpause_signal = pyqtSignal()
    
    def __init__(self,sensor,mirror):
        super(Loop,self).__init__()
        self.mirror_thread = QThread()
        self.sensor_thread = QThread()

        self.sensor = sensor
        self.active_lenslets = np.ones(self.sensor.n).astype(int)
        self.mirror = mirror

        n_lenslets = self.sensor.n
        n_actuators = self.mirror.n
        dummy = np.ones((2*n_lenslets,n_actuators))
        outfn = os.path.join(kcfg.poke_directory,'dummy_poke.txt')
        np.savetxt(outfn,dummy)



        #DEBUG
        
        self.sensor.moveToThread(self.sensor_thread)
        #self.sensor.update()

        self.mirror.moveToThread(self.mirror_thread)

        self.sensor_thread.started.connect(self.sensor.update)
        self.finished.connect(self.sensor.update)
        self.sensor.finished.connect(self.update)
        
        self.pause_signal.connect(self.sensor.pause)
        self.pause_signal.connect(self.mirror.pause)
        self.unpause_signal.connect(self.sensor.unpause)
        self.unpause_signal.connect(self.mirror.unpause)
        
        self.poke = None
        self.closed = False
        
        try:
            self.load_poke(kcfg.poke_filename)
        except Exception as e:
            self.load_poke()
        self.gain = kcfg.loop_gain
        self.loss = kcfg.loop_loss
        self.paused = False

    def has_poke(self):
        return self.poke is not None

    def start(self):
        self.sensor_thread.start()
        self.mirror_thread.start()

    def pause(self):
        self.pause_signal.emit()
        self.paused = True

    def unpause(self):
        self.unpause_signal.emit()
        self.paused = False
        self.finished.emit()
        print 'loop unpaused'
        
    @pyqtSlot()
    def update(self):
        if not self.paused:
            sensor_mutex.lock()
            mirror_mutex.lock()
            
            # compute the mirror command here
            if self.closed and self.has_poke():
                current_active_lenslets = np.zeros(self.active_lenslets.shape)
                current_active_lenslets[np.where(self.sensor.box_maxes>kcfg.spots_threshold)] = 1
                if not all(self.active_lenslets==current_active_lenslets):
                    self.active_lenslets[:] = current_active_lenslets[:]
                    self.poke.invert(mask=self.active_lenslets)

                xs = self.sensor.x_slopes[np.where(self.active_lenslets)[0]]
                ys = self.sensor.y_slopes[np.where(self.active_lenslets)[0]]
                slope_vec = np.hstack((xs,ys))
                command = self.gain * np.dot(self.poke.ctrl,slope_vec)
                command = self.mirror.get_command()*(1-self.loss) - command
                self.mirror.set_command(command)
                
            self.finished.emit()
            sensor_mutex.unlock()
            mirror_mutex.unlock()
                
    def load_poke(self,poke_filename=None):
        sensor_mutex.lock()
        mirror_mutex.lock()
        try:
            poke = np.loadtxt(poke_filename)
        except Exception as e:
            error_message('Could not find %s.'%poke_filename)
            options = QFileDialog.Options()
            #options |= QFileDialog.DontUseNativeDialog
            poke_filename, _ = QFileDialog.getOpenFileName(
                            None,
                            "Please select a poke file.",
                            kcfg.poke_directory,
                            "Text Files (*.txt)",
                            options=options)
            poke = np.loadtxt(poke_filename)

        py,px = poke.shape
        expected_py = self.sensor.n*2
        expected_px = self.mirror.n
        
        try:
            assert py==expected_py
        except AssertionError as ae:
            error_message('Poke matrix has %d rows, but %d expected.'%(py,expected_py))
            return
        try:
            assert px==expected_px
        except AssertionError as ae:
            error_message('Poke matrix has %d columns, but %d expected.'%(px,expected_px))
            return
        
        self.poke = Poke(poke)
        sensor_mutex.unlock()
        mirror_mutex.unlock()

    def invert(self):
        if self.poke is not None:
            self.pause()
            time.sleep(1)
            self.poke.invert()
            time.sleep(1)
            QApplication.processEvents()
            self.unpause()
            time.sleep(1)

    def set_n_modes(self,n):
        try:
            self.poke.n_modes = n
        except Exception as e:
            print e

    def get_n_modes(self):
        out = -1
        try:
            out = self.poke.n_modes
        except Exception as e:
            print e
        return out

    def get_condition_number(self):
        out = -1
        try:
            out = self.poke.cutoff_cond
        except Exception as e:
            print e
        return out
            
    def run_poke(self):
        cmin = kcfg.poke_command_min
        cmax = kcfg.poke_command_max
        n_commands = kcfg.poke_n_command_steps
        commands = np.linspace(cmin,cmax,n_commands)

        self.pause()
        time.sleep(1)
        
        n_lenslets = self.sensor.n
        n_actuators = self.mirror.n
        
        x_mat = np.zeros((n_lenslets,n_actuators,n_commands))
        y_mat = np.zeros((n_lenslets,n_actuators,n_commands))
        
        for k_actuator in range(n_actuators):
            self.mirror.flatten()
            for k_command in range(n_commands):
                cur = commands[k_command]
                print k_actuator,cur
                self.mirror.set_actuator(k_actuator,cur)
                QApplication.processEvents()
                time.sleep(.01)
                self.sensor.sense()
                sensor_mutex.lock()
                x_mat[:,k_actuator,k_command] = self.sensor.x_slopes
                y_mat[:,k_actuator,k_command] = self.sensor.y_slopes
                sensor_mutex.unlock()
                self.finished.emit()
        # print 'done'
        self.mirror.flatten()
        
        d_commands = np.mean(np.diff(commands))
        d_x_mat = np.diff(x_mat,axis=2)
        d_y_mat = np.diff(y_mat,axis=2)

        x_response = np.mean(d_x_mat/d_commands,axis=2)
        y_response = np.mean(d_y_mat/d_commands,axis=2)
        poke = np.vstack((x_response,y_response))
        ns = now_string()
        poke_fn = os.path.join(kcfg.poke_directory,'%s_poke.txt'%ns)
        command_fn = os.path.join(kcfg.poke_directory,'%s_currents.txt'%ns)
        chart_fn = os.path.join(kcfg.poke_directory,'%s_modes.pdf'%ns)
        np.savetxt(poke_fn,poke)
        np.savetxt(command_fn,commands)
        save_modes_chart(chart_fn,poke,commands,self.mirror.mask)

        self.poke = Poke(poke)
        
        time.sleep(1)
        self.unpause()

    def set_closed(self,val):
        self.closed = val

class ImageDisplay(QWidget):
    def __init__(self,name,downsample=1,clim=None,colormap=None,mouse_event_handler=None,image_min=None,image_max=None,width=512,height=512,zoom_height=kcfg.zoom_height,zoom_width=kcfg.zoom_width,zoomable=False,draw_boxes=False,draw_lines=False):
        super(ImageDisplay,self).__init__()
        self.name = name
        self.autoscale = False
        self.sx = width
        self.sy = height
        self.draw_boxes = draw_boxes
        self.draw_lines = draw_lines
        self.zoomable = zoomable
        
        if clim is None:
            try:
                clim = np.loadtxt('.gui_settings/clim_%s.txt'%name)
            except Exception as e:
                self.autoscale = True
        
        self.clim = clim
        self.pixmap = QPixmap()
        self.label = QLabel()
        self.image_max = image_max
        self.image_min = image_min
        self.zoom_width = zoom_width
        self.zoom_height = zoom_height
        
        layout = QHBoxLayout()
        layout.addWidget(self.label)

        if image_min is not None and image_max is not None and not self.autoscale:
            self.n_steps = 100
        
            self.cmin_slider = QSlider(Qt.Vertical)
            self.cmax_slider = QSlider(Qt.Vertical)

            self.cmin_slider.setMinimum(0)
            self.cmax_slider.setMinimum(0)

            self.cmin_slider.setSingleStep(1)
            self.cmax_slider.setSingleStep(1)

            self.cmin_slider.setPageStep(10)
            self.cmax_slider.setPageStep(10)

            self.cmin_slider.setMaximum(self.n_steps)
            self.cmax_slider.setMaximum(self.n_steps)

            self.cmin_slider.setValue(self.real2slider(self.clim[0]))
            self.cmax_slider.setValue(self.real2slider(self.clim[1]))

            self.cmin_slider.valueChanged.connect(self.set_cmin)
            self.cmax_slider.valueChanged.connect(self.set_cmax)
            
            layout.addWidget(self.cmin_slider)
            layout.addWidget(self.cmax_slider)

        
        self.setLayout(layout)
        
        self.zoomed = False
        self.colormap = colormap
        if self.colormap is not None:
            self.colortable = colortable(self.colormap)
        if mouse_event_handler is not None:
            self.mousePressEvent = mouse_event_handler
        else:
            self.mousePressEvent = self.zoom
            
        self.downsample = downsample
        
        data = np.random.rand(100,100)
        self.show(data)
        
        self.zoom_x1 = 0
        self.zoom_x2 = self.sx-1
        self.zoom_y1 = 0
        self.zoom_y2 = self.sy-1



        
    def real2slider(self,val):
        # convert a real value into a slider value
        return round(int((val-float(self.image_min))/float(self.image_max-self.image_min)*self.n_steps))

    def slider2real(self,val):
        # convert a slider integer into a real value
        return float(val)/float(self.n_steps)*(self.image_max-self.image_min)+self.image_min
    
    def set_cmax(self,slider_value):
        self.clim = (self.clim[0],self.slider2real(slider_value))
        np.savetxt('.gui_settings/clim_%s.txt'%self.name,self.clim)

    def set_cmin(self,slider_value):
        self.clim = (self.slider2real(slider_value),self.clim[1])
        np.savetxt('.gui_settings/clim_%s.txt'%self.name,self.clim)
        
    def show(self,data,boxes=None,lines=None,mask=None):

        if mask is None:
            if boxes is not None:
                mask = np.ones(boxes[0].shape)
            elif lines is not None:
                mask = np.ones(lines[0].shape)
            else:
                assert (boxes is None) and (mask is None)
        
#        if self.name=='mirror':
#            print data[6,6]
            
        if self.autoscale:
            clim = (data.min(),data.max())
        else:
            clim = self.clim

        cmin,cmax = clim
        downsample = self.downsample
        data = data[::downsample,::downsample]
        
        if self.zoomed:
            x_scale = float(data.shape[1])/float(self.sx)
            y_scale = float(data.shape[0])/float(self.sy)

            zy1 = int(round(self.zoom_y1*y_scale))
            zy2 = int(round(self.zoom_y2*y_scale))
            zx1 = int(round(self.zoom_x1*x_scale))
            zx2 = int(round(self.zoom_x2*x_scale))
            
            #data = data[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
            data = data[zy1:zy2,zx1:zx2]
            
        bmp = np.round(np.clip((data.astype(np.float)-cmin)/(cmax-cmin),0,1)*255).astype(np.uint8)
        sy,sx = bmp.shape
        n_bytes = bmp.nbytes
        bytes_per_line = int(n_bytes/sy)
        image = QImage(bmp,sy,sx,bytes_per_line,QImage.Format_Indexed8)
        if self.colormap is not None:
            image.setColorTable(self.colortable)
        self.pixmap.convertFromImage(image)

        
        
        if boxes is not None and self.draw_boxes:
            x1vec,x2vec,y1vec,y2vec = boxes
            pen = QPen()
            pen.setColor(QColor(*kcfg.active_search_box_color))
            pen.setWidth(kcfg.search_box_thickness)
            painter = QPainter()
            painter.begin(self.pixmap)
            painter.setPen(pen)
            for index,(x1,y1,x2,y2) in enumerate(zip(x1vec,y1vec,x2vec,y2vec)):
                if mask[index]:
                    width = float(x2 - x1 + 1)/float(self.downsample)
                    painter.drawRect(x1/float(self.downsample)-self.zoom_x1,y1/float(self.downsample)-self.zoom_y1,width,width)
            painter.end()
            
        if lines is not None and self.draw_lines:
            x1vec,x2vec,y1vec,y2vec = lines
            pen = QPen()
            pen.setColor(QColor(*kcfg.slope_line_color))
            pen.setWidth(kcfg.slope_line_thickness)
            painter = QPainter()
            painter.begin(self.pixmap)
            painter.setPen(pen)
            for index,(x1,y1,x2,y2) in enumerate(zip(x1vec,y1vec,x2vec,y2vec)):
                if mask[index]:
                    painter.drawLine(QLine(x1/float(self.downsample)- self.zoom_x1,y1/float(self.downsample)-self.zoom_y1,x2/float(self.downsample)- self.zoom_x1,y2/float(self.downsample)- self.zoom_y1))
            painter.end()

        if sy==self.sy and sx==self.sx:
            self.label.setPixmap(self.pixmap)
        else:
            self.label.setPixmap(self.pixmap.scaled(self.sy,self.sx))
        
    def set_clim(self,clim):
        self.clim = clim

    def zoom(self,event):
        if not self.zoomable:
            return
        
        if self.zoom_width>=self.sx or self.zoom_height>=self.sy:
            return
        
        x,y = event.x(),event.y()
        
        if self.zoomed:
            self.zoomed = False
            self.zoom_x1 = 0
            self.zoom_x2 = self.sx-1
            self.zoom_y1 = 0
            self.zoom_y2 = self.sy-1
        else:
            self.zoomed = True
            self.zoom_x1 = x-self.zoom_width//2
            self.zoom_x2 = x+self.zoom_width//2
            self.zoom_y1 = y-self.zoom_height//2
            self.zoom_y2 = y+self.zoom_height//2
            if self.zoom_x1<0:
                dx = -self.zoom_x1
                self.zoom_x1+=dx
                self.zoom_x2+=dx
            if self.zoom_x2>self.sx-1:
                dx = self.zoom_x2-(self.sx-1)
                self.zoom_x1-=dx
                self.zoom_x2-=dx
            if self.zoom_y1<0:
                dy = -self.zoom_y1
                self.zoom_y1+=dy
                self.zoom_y2+=dy
            if self.zoom_y2>self.sy-1:
                dy = self.zoom_y2-(self.sy-1)
                self.zoom_y1-=dy
                self.zoom_y2-=dy
            #print 'zooming to %d,%d,%d,%d'%(self.zoom_x1,self.zoom_x2,self.zoom_y1,self.zoom_y2)

    def set_draw_lines(self,val):
        self.draw_lines = val

    def set_draw_boxes(self,val):
        self.draw_boxes = val
        
class UI(QWidget):

    def __init__(self,loop):
        super(UI,self).__init__()
        self.loop = loop
        self.loop.finished.connect(self.update)
        self.init_UI()
        self.frame_timer = FrameTimer('UI',verbose=False)
        self.show()

    def init_UI(self):
        layout = QHBoxLayout()
        imax = 2**kcfg.bit_depth-1
        imin = 0
        self.id_spots = ImageDisplay('spots',downsample=2,clim=None,colormap=kcfg.spots_colormap,image_min=imin,image_max=imax,draw_boxes=kcfg.show_search_boxes,draw_lines=kcfg.show_slope_lines,zoomable=True)
        #self.id_spots = ImageDisplay('spots',downsample=2,clim=(50,1000),colormap=kcfg.spots_colormap,image_min=imin,image_max=imax,draw_boxes=kcfg.show_search_boxes,draw_lines=kcfg.show_slope_lines,zoomable=True)
        layout.addWidget(self.id_spots)

        self.id_mirror = ImageDisplay('mirror',downsample=1,clim=(-0.5,0.5),colormap=kcfg.mirror_colormap,image_min=kcfg.mirror_command_min,image_max=kcfg.mirror_command_max,width=256,height=256)
        self.id_wavefront = ImageDisplay('wavefront',downsample=1,clim=(-1.0e-8,1.0e-8),colormap=kcfg.wavefront_colormap,image_min=-1.0e-5,image_max=1.0e-5,width=256,height=256)
        column_1 = QVBoxLayout()
        column_1.setAlignment(Qt.AlignTop)
        column_1.addWidget(self.id_mirror)
        column_1.addWidget(self.id_wavefront)
        layout.addLayout(column_1)

        column_2 = QVBoxLayout()
        column_2.setAlignment(Qt.AlignTop)
        self.cb_closed = QCheckBox('Loop &closed')
        self.cb_closed.setChecked(self.loop.closed)
        self.cb_closed.stateChanged.connect(self.loop.set_closed)

        self.cb_draw_boxes = QCheckBox('Draw boxes')
        self.cb_draw_boxes.setChecked(self.id_spots.draw_boxes)
        self.cb_draw_boxes.stateChanged.connect(self.id_spots.set_draw_boxes)

        self.cb_draw_lines = QCheckBox('Draw lines')
        self.cb_draw_lines.setChecked(self.id_spots.draw_lines)
        self.cb_draw_lines.stateChanged.connect(self.id_spots.set_draw_lines)

        self.cb_logging = QCheckBox('Logging')
        self.cb_logging.setChecked(False)
        self.cb_logging.stateChanged.connect(self.loop.sensor.set_logging)
        self.cb_logging.stateChanged.connect(self.loop.mirror.set_logging)
        
        self.pb_poke = QPushButton('Poke')
        self.pb_poke.clicked.connect(self.loop.run_poke)
        self.pb_record_reference = QPushButton('Record reference')
        self.pb_record_reference.clicked.connect(self.loop.sensor.record_reference)
        self.pb_flatten = QPushButton('&Flatten')
        self.pb_flatten.clicked.connect(self.loop.mirror.flatten)
        self.pb_quit = QPushButton('&Quit')
        self.pb_quit.clicked.connect(sys.exit)

        poke_layout = QHBoxLayout()
        poke_layout.addWidget(QLabel('Modes:'))
        self.modes_spinbox = QSpinBox()
        self.modes_spinbox.setMaximum(kcfg.mirror_n_actuators)
        self.modes_spinbox.setMinimum(0)
        self.modes_spinbox.valueChanged.connect(self.loop.set_n_modes)
        self.modes_spinbox.setValue(self.loop.get_n_modes())
        poke_layout.addWidget(self.modes_spinbox)
        self.pb_invert = QPushButton('Invert')
        self.pb_invert.clicked.connect(self.loop.invert)
        poke_layout.addWidget(self.pb_invert)

        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel('Background correction:'))
        self.bg_spinbox = QSpinBox()
        self.bg_spinbox.setValue(self.loop.sensor.background_correction)
        self.bg_spinbox.valueChanged.connect(self.loop.sensor.set_background_correction)
        bg_layout.addWidget(self.bg_spinbox)


        f_layout = QHBoxLayout()
        f_layout.addWidget(QLabel('Defocus:'))
        self.f_spinbox = QDoubleSpinBox()
        self.f_spinbox.setValue(0.0)
        self.f_spinbox.setSingleStep(0.01)
        self.f_spinbox.setMaximum(1.0)
        self.f_spinbox.setMinimum(-1.0)
        self.f_spinbox.valueChanged.connect(self.loop.sensor.set_defocus)
        f_layout.addWidget(self.f_spinbox)
        
        self.lbl_error = QLabel()
        self.lbl_error.setAlignment(Qt.AlignRight)
        self.lbl_tip = QLabel()
        self.lbl_tip.setAlignment(Qt.AlignRight)
        self.lbl_tilt = QLabel()
        self.lbl_tilt.setAlignment(Qt.AlignRight)
        self.lbl_cond = QLabel()
        self.lbl_cond.setAlignment(Qt.AlignRight)
        self.lbl_sensor_fps = QLabel()
        self.lbl_sensor_fps.setAlignment(Qt.AlignRight)
        self.lbl_mirror_fps = QLabel()
        self.lbl_mirror_fps.setAlignment(Qt.AlignRight)
        self.lbl_ui_fps = QLabel()
        self.lbl_ui_fps.setAlignment(Qt.AlignRight)
        
        column_2.addWidget(self.pb_flatten)
        column_2.addWidget(self.cb_closed)
        column_2.addLayout(f_layout)
        column_2.addLayout(bg_layout)
        column_2.addLayout(poke_layout)
        column_2.addWidget(self.cb_draw_boxes)
        column_2.addWidget(self.cb_draw_lines)
        column_2.addWidget(self.pb_quit)
        
        column_2.addWidget(self.lbl_error)
        column_2.addWidget(self.lbl_tip)
        column_2.addWidget(self.lbl_tilt)
        column_2.addWidget(self.lbl_cond)
        column_2.addWidget(self.lbl_sensor_fps)
        column_2.addWidget(self.lbl_mirror_fps)
        column_2.addWidget(self.lbl_ui_fps)
        
        column_2.addWidget(self.pb_poke)
        column_2.addWidget(self.pb_record_reference)
        column_2.addWidget(self.cb_logging)
        
        layout.addLayout(column_2)
        
        self.setLayout(layout)
        

    @pyqtSlot()
    def update(self):
        
        try:
            sensor = self.loop.sensor
            mirror = self.loop.mirror

            sb = sensor.search_boxes

            if self.id_spots.draw_boxes:
                boxes = [sb.x1,sb.x2,sb.y1,sb.y2]
            else:
                boxes = None

            if self.id_spots.draw_lines:
                lines = [sb.x,sb.x+sensor.x_slopes*kcfg.slope_line_magnification,
                         sb.y,sb.y+sensor.y_slopes*kcfg.slope_line_magnification]
            else:
                lines = None
                
            self.id_spots.show(sensor.image,boxes=boxes,lines=lines,mask=self.loop.active_lenslets)

            mirror_map = np.zeros(mirror.mask.shape)
            mirror_map[np.where(mirror.mask)] = mirror.get_command()[:]
            
            self.id_mirror.show(mirror_map)

            self.id_wavefront.show(sensor.wavefront)
            
            self.lbl_error.setText(kcfg.wavefront_error_fmt%(sensor.error*1e9))
            self.lbl_tip.setText(kcfg.tip_fmt%(sensor.tip*1000000))
            self.lbl_tilt.setText(kcfg.tilt_fmt%(sensor.tilt*1000000))
            self.lbl_cond.setText(kcfg.cond_fmt%(self.loop.get_condition_number()))
            self.lbl_sensor_fps.setText(kcfg.sensor_fps_fmt%sensor.frame_timer.fps)
            self.lbl_mirror_fps.setText(kcfg.mirror_fps_fmt%mirror.frame_timer.fps)
            self.lbl_ui_fps.setText(kcfg.ui_fps_fmt%self.frame_timer.fps)
        except Exception as e:
            print e
            
    def select_single_spot(self,click):
        print 'foo'
        x = click.x()*self.downsample
        y = click.y()*self.downsample
        self.single_spot_index = self.loop.sensor.search_boxes.get_lenslet_index(x,y)

    def paintEvent(self,event):
        self.frame_timer.tick()
        
if __name__=='__main__':
    app = QApplication(sys.argv)

    ####
    #mirror = Mirror()
    #t = QThread()
    #mirror.moveToThread(t)
    #t.start()
    #sys.exit(app.exec_())
    ####

    try:
        cam = cameras.PylonCamera()
    except Exception as e:
        print e
        print 'Using simulated camera'
        cam = cameras.SimulatedCamera()

    # look at a test image and make sure it agrees with kcfg settings
    im = cam.get_image()
    height,width = im.shape
    try:
        assert height==kcfg.image_height_px
        assert width==kcfg.image_width_px
    except Exception as e:
        print e
    
    try:
        xy = np.loadtxt(kcfg.reference_coordinates_filename)
    except Exception as e:
        mask = np.loadtxt(kcfg.reference_mask_filename)
        rg = ReferenceGenerator(cam,mask)
        rg.make_coords()
        sys.exit()

    sensor = Sensor(cam)
    mirror = Mirror()
    loop = Loop(sensor,mirror)
    ui = UI(loop)
    loop.start()

    sys.exit(app.exec_())
