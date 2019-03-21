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

sensor_state_mutex = QMutex()
sensor_configuration_mutex = QMutex()
mirror_state_mutex = QMutex()
mirror_configuration_mutex = QMutex()

class FrameTimer:
    def __init__(self,label,buffer_size=100,verbose=True):
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
                     (x1.min(),x2.max(),y1.min(),y2.max()))

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

class SensorConfiguration(QObject):

    def __init__(self):
        super(SensorConfiguration,self).__init__()
        self.image_width_px = kcfg.image_width_px
        self.image_height_px = kcfg.image_height_px
        self.lenslet_pitch_m = kcfg.lenslet_pitch_m
        self.lenslet_focal_length_m = kcfg.lenslet_focal_length_m
        self.pixel_size_m = kcfg.pixel_size_m
        self.beam_diameter_m = kcfg.beam_diameter_m
        self.wavelength_m = kcfg.wavelength_m
        self.background_correction = kcfg.background_correction
        xy = np.loadtxt(kcfg.reference_coordinates_filename)
        self.search_boxes = SearchBoxes(xy[:,0],xy[:,1],kcfg.search_box_half_width)
        self.n = self.search_boxes.n
        self.centroiding_iterations = kcfg.centroiding_iterations
        self.iterative_centroiding_step = kcfg.iterative_centroiding_step
        self.filter_lenslets = kcfg.sensor_filter_lenslets
        self.estimate_background = kcfg.estimate_background
        self.mask = np.loadtxt(kcfg.reference_mask_filename)
        self.reconstruct_wavefront = kcfg.sensor_reconstruct_wavefront
        self.remove_tip_tilt = kcfg.sensor_remove_tip_tilt
        
    def copy(self):
        sc = SensorConfiguration()
        sc.image_width_px = self.image_width_px
        sc.image_height_px = self.image_height_px
        sc.lenslet_pitch_m = self.lenslet_pitch_m
        sc.lenslet_focal_length_m = self.lenslet_focal_length_m
        sc.pixel_size_m = self.pixel_size_m
        sc.beam_diameter_m = self.beam_diameter_m
        sc.wavelength_m = self.wavelength_m
        sc.background_correction = self.background_correction
        sc.search_boxes = self.search_boxes.copy()
        sc.n = self.n
        sc.centroiding_iterations = self.centroiding_iterations
        sc.iterative_centroiding_step = self.iterative_centroiding_step
        sc.filter_lenslets = self.filter_lenslets
        sc.estimate_background = self.estimate_background
        sc.mask = self.mask
        sc.reconstruct_wavefront = self.reconstruct_wavefront
        sc.remove_tip_tilt = self.remove_tip_tilt
        return sc
        #return copy.deepcopy(self)

        
class SensorState(QObject):

    def __init__(self,n_lenslets):
        super(SensorState,self).__init__()
        self.n = n_lenslets
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
        
    def copy(self):
        new_state = SensorState(self.n)
        new_state.image = self.image.copy()
        new_state.x_slopes = self.x_slopes.copy()
        new_state.y_slopes = self.y_slopes.copy()
        new_state.x_centroids = self.x_centroids.copy()
        new_state.y_centroids = self.y_centroids.copy()
        new_state.box_maxes = self.box_maxes.copy()
        new_state.box_mins = self.box_mins.copy()
        new_state.box_means = self.box_means.copy()
        new_state.box_backgrounds = self.box_backgrounds.copy()
        new_state.error = self.error
        new_state.tip = self.tip
        new_state.tilt = self.tilt
        new_state.zernikes = self.zernikes
        new_state.wavefront = self.wavefront
        return new_state
        
    def __str__(self):
        sensor_state_mutex.lock()
        out = 'Sensor state: '
        out = out + 'x-centroids [%0.1f, %0.1f,...], '%(self.x_centroids[0],self.x_centroids[1])
        out = out + 'y-centroids [%0.1f, %0.1f,...], '%(self.y_centroids[0],self.y_centroids[1])
        out = out + 'x-slopes [%0.2e, %0.2e,...], '%(self.x_slopes[0],self.x_slopes[1])
        out = out + 'y-slopes [%0.2e, %0.2e,...]'%(self.y_slopes[0],self.y_slopes[1])
        sensor_state_mutex.unlock()
        return out
        
class Sensor(QObject):

    finished = pyqtSignal(QObject)
    
    def __init__(self,camera):
        super(Sensor,self).__init__()
        self.cfg = SensorConfiguration()
        self.cam = camera
        self.state = SensorState(self.cfg.n)
        self.frame_timer = FrameTimer('Sensor',verbose=True)
        self.reconstructor = Reconstructor(self.cfg.search_boxes.x,
                                           self.cfg.search_boxes.y,self.cfg.mask)

        self.paused = False
        
    @pyqtSlot()
    def update(self):
        if not self.paused:
            self.sense()
        self.finished.emit(self.state)
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
        
    def sense(self):
        image = cam.get_image()
        sensor_configuration_mutex.lock()
        local_cfg = self.cfg.copy()
        sensor_configuration_mutex.unlock()

        sensor_state_mutex.lock()
        local_state = self.state.copy()
        sensor_state_mutex.unlock()
        
        sb = local_cfg.search_boxes
        xr = np.zeros(local_cfg.search_boxes.x.shape)
        xr[:] = local_cfg.search_boxes.x[:]
        yr = np.zeros(local_cfg.search_boxes.y.shape)
        yr[:] = local_cfg.search_boxes.y[:]

        half_width = sb.half_width
        for iteration in range(local_cfg.centroiding_iterations):
            centroid.compute_centroids(image,
                                       sb.x1,sb.x2,
                                       sb.y1,sb.y2,
                                       xr,yr,
                                       local_state.box_means,
                                       local_state.box_maxes,
                                       local_state.box_mins,
                                       local_state.box_backgrounds,
                                       local_cfg.estimate_background,
                                       local_cfg.background_correction,
                                       1)
            half_width-=local_cfg.iterative_centroiding_step
            sb = SearchBoxes(xr,yr,half_width)
            
        local_state.x_centroids[:] = xr[:]
        local_state.y_centroids[:] = yr[:]
        local_state.x_slopes = (local_state.x_centroids-local_cfg.search_boxes.x)*local_cfg.pixel_size_m/local_cfg.lenslet_focal_length_m
        local_state.y_slopes = (local_state.y_centroids-local_cfg.search_boxes.y)*local_cfg.pixel_size_m/local_cfg.lenslet_focal_length_m
        local_state.tilt = np.mean(local_state.x_slopes)
        local_state.tip = np.mean(local_state.y_slopes)
        if local_cfg.remove_tip_tilt:
            local_state.x_slopes-=local_state.tilt
            local_state.y_slopes-=local_state.tip
        local_state.image = image
        if local_cfg.reconstruct_wavefront:
            local_state.zernikes,local_state.wavefront,local_state.error = self.reconstructor.get_wavefront(local_state.x_slopes,local_state.y_slopes)
        sensor_state_mutex.lock()
        self.state = local_state
        sensor_state_mutex.unlock()

    def get_configuration(self):
        sensor_configuration_mutex.lock()
        cfg = self.cfg.copy()
        sensor_configuration_mutex.unlock()
        return cfg
    
    def set_configuration(self,new_cfg):
        sensor_configuration_mutex.lock()
        self.cfg = new_cfg
        sensor_configuration_mutex.unlock()
        
    def get_state(self):
        sensor_state_mutex.lock()
        ms = self.state.copy()
        sensor_state_mutex.unlock()
        return ms
        
    def set_state(self,new_state):
        sensor_state_mutex.lock()
        self.state = new_state
        sensor_state_mutex.unlock()
    
    def auto_center(self,window_spots=False,spot_half_width=5):
        self.pause()
        # make a simulated spots image
        sim = np.zeros((kcfg.image_height_px,kcfg.image_width_px))
        for x,y in zip(self.x_ref,self.y_ref):
            ry,rx = int(round(y)),int(round(x))
            sim[ry-spot_half_width:ry+spot_half_width+1,
                rx-spot_half_width:rx+spot_half_width+1] = 1.0

        spots = self.cam.get_image()

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
        config = self.get_configuration()
        search_boxes = SearchBoxes(new_x_ref,new_y_ref,config.search_boxes.half_width)
        config.search_boxes = search_boxes
        self.set_configuration(config)
        self.unpause()

    def record_reference(self,verbose=True):
        if verbose:
            print 'recording reference'
        self.pause()
        xcent = []
        ycent = []
        for k in range(kcfg.reference_n_measurements):
            if verbose:
                print 'measurement %d of %d'%(k+1,kcfg.reference_n_measurements)
            self.sense()
            state = self.get_state()
            xcent.append(state.x_centroids)
            ycent.append(state.y_centroids)
            
        x_ref = np.array(xcent).mean(0)
        y_ref = np.array(ycent).mean(0)
        config = self.get_configuration()
        search_boxes = SearchBoxes(x_ref,y_ref,config.search_boxes.half_width)
        config.search_boxes = search_boxes
        self.set_configuration(config)
        outfn = os.path.join(kcfg.reference_directory,prepend('coords.txt',now_string()))
        refxy = np.array((x_ref,y_ref)).T
        np.savetxt(outfn,refxy,fmt='%0.2f')
        self.unpause()
        
    def make_reference_coordinates(self,x_offset=0.0,y_offset=0.0):
        self.pause()
        sensor_x = kcfg.image_width_px
        sensor_y = kcfg.image_height_px
        config = self.get_configuration()
        mask = config.mask
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
        ref_xy = np.array(ref_xy)
        self.unpause()
        return ref_xy
    
class MirrorConfiguration(QObject):
    def __init__(self):
        super(MirrorConfiguration,self).__init__()
        self.mask = np.loadtxt(kcfg.mirror_mask_filename)
        self.n = kcfg.mirror_n_actuators
        self.flat = np.loadtxt(kcfg.mirror_flat_filename)
        self.command_max = kcfg.mirror_command_max
        self.command_min = kcfg.mirror_command_min
        self.settling_time = kcfg.mirror_settling_time_s
        self.update_rate = kcfg.mirror_update_rate
        
    def copy(self):
        return copy.deepcopy(self)

class MirrorState(QObject):
    def __init__(self,n_actuators):
        super(MirrorState,self).__init__()
        self.command = np.zeros(n_actuators)
        self.n = n_actuators

    def copy(self):
        ms = MirrorState(self.n)
        ms.command[:] = self.command[:]
        return ms

class Mirror(QObject):
    finished = pyqtSignal(QObject)
    
    def __init__(self):
        super(Mirror,self).__init__()
        self.cfg = MirrorConfiguration()
        self.state = MirrorState(self.cfg.n)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1.0/self.cfg.update_rate*1000.0)
        self.frame_timer = FrameTimer('Mirror',verbose=True)
        self.paused = False
        
    @pyqtSlot()
    def update(self):
        if not self.paused:
            self.send()
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
        mirror_state_mutex.lock()
        mirror_configuration_mutex.lock()
        cmax = self.cfg.command_max
        cmin = self.cfg.command_min
        command = self.state.command
        command = np.clip(command,cmin,cmax)
        self.state.command = command
        # SEND TO MIRROR HERE
        mirror_state_mutex.unlock()
        mirror_configuration_mutex.unlock()

    def flatten(self):
        #self.pause()
        mirror_configuration_mutex.lock()
        flat = self.cfg.flat.copy()
        mirror_configuration_mutex.unlock()
        mirror_state_mutex.lock()
        self.state.command = flat
        mirror_state_mutex.unlock()
        self.send()
        #self.unpause()

    def set_actuator(self,index,value):
        mirror_state_mutex.lock()
        self.state.command[index] = value
        mirror_state_mutex.unlock()
        
    def get_state(self):
        mirror_state_mutex.lock()
        ms = self.state.copy()
        mirror_state_mutex.unlock()
        return ms
        
    def set_state(self,new_state):
        mirror_state_mutex.lock()
        self.state = new_state
        mirror_state_mutex.unlock()
    
    def get_configuration(self):
        mirror_configuration_mutex.lock()
        cfg = self.cfg.copy()
        mirror_configuration_mutex.unlock()
        return cfg
    
    def set_configuration(self,new_cfg):
        mirror_configuration_mutex.lock()
        self.cfg = new_cfg
        mirror_configuration_mutex.unlock()
        

class Loop(QObject):

    finished = pyqtSignal()
    pause_signal = pyqtSignal()
    unpause_signal = pyqtSignal()
    final_states = pyqtSignal(QObject,QObject)
    
    def __init__(self,sensor,mirror):
        super(Loop,self).__init__()
        self.mirror_thread = QThread()
        self.sensor_thread = QThread()

        self.sensor = sensor
        self.active_lenslets = np.zeros(self.sensor.cfg.n).astype(int)
        self.mirror = mirror
        self.sensor.moveToThread(self.sensor_thread)
        #self.mirror.moveToThread(self.mirror_thread)

        self.sensor_thread.started.connect(self.sensor.update)
        self.finished.connect(self.sensor.update)
        self.sensor.finished.connect(self.update)
        
        self.pause_signal.connect(self.sensor.pause)
        self.pause_signal.connect(self.mirror.pause)
        self.unpause_signal.connect(self.sensor.unpause)
        self.unpause_signal.connect(self.mirror.unpause)
        
        self.poke = None
        self.closed = True
        self.load_poke(kcfg.poke_filename)
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
        
    @pyqtSlot(QObject)
    def update(self,sensor_state):
        if not self.paused:
            print 'loop updating'
            sensor_state_mutex.lock()
            local_sensor_state = sensor_state.copy()
            sensor_state_mutex.unlock()

            mirror_state_mutex.lock()
            local_mirror_state = self.mirror.state.copy()
            mirror_state_mutex.unlock()

            current_active_lenslets = self.decide_lenslets(local_sensor_state)
            # compute the mirror command here
            if self.closed and self.has_poke():
                if not all(self.active_lenslets==current_active_lenslets):
                    self.active_lenslets[:] = current_active_lenslets[:]
                    self.poke.invert(mask=self.active_lenslets)

                xs = local_sensor_state.x_slopes[np.where(self.active_lenslets)[0]]
                ys = local_sensor_state.y_slopes[np.where(self.active_lenslets)[0]]
                slope_vec = np.hstack((xs,ys))
                command = self.gain * np.dot(self.poke.ctrl,slope_vec)
                command = local_mirror_state.command*(1-self.loss) - command
                local_mirror_state.command = command
                self.mirror.set_state(local_mirror_state)
                self.finished.emit()
                self.final_states.emit(local_sensor_state,local_mirror_state)
                
    def decide_lenslets(self,sensor_state):
        active_lenslets = np.zeros(sensor_state.x_slopes.shape).astype(int)
        active_lenslets[np.where(sensor_state.box_maxes>kcfg.spots_threshold)] = 1
        return active_lenslets

    def load_poke(self,poke_filename=None):
        try:
            poke = np.loadtxt(poke_filename)
        except IOError as ioe:
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


        sensor_cfg = self.sensor.get_configuration()
        mirror_cfg = self.mirror.get_configuration()
        py,px = poke.shape
        expected_py = sensor_cfg.n*2
        expected_px = mirror_cfg.n
        
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

    def run_poke(self):
        cmin = kcfg.poke_command_min
        cmax = kcfg.poke_command_max
        n_commands = kcfg.poke_n_command_steps
        commands = np.linspace(cmin,cmax,n_commands)
        self.pause()
        time.sleep(1)
        local_sensor_cfg = self.sensor.get_configuration()
        local_mirror_cfg = self.mirror.get_configuration()
        n_lenslets = local_sensor_cfg.n
        n_actuators = local_mirror_cfg.n
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
                local_sensor_state = self.sensor.get_state()
                local_mirror_state = self.mirror.get_state()
                x_mat[:,k_actuator,k_command] = local_sensor_state.x_slopes
                y_mat[:,k_actuator,k_command] = local_sensor_state.y_slopes
                self.final_states.emit(local_sensor_state,local_mirror_state)
        # print 'done'
        self.mirror.flatten()
        
        d_commands = np.mean(np.diff(commands))
        d_x_mat = np.diff(x_mat,axis=2)
        d_y_mat = np.diff(y_mat,axis=2)

        x_response = np.mean(d_x_mat/d_commands,axis=2)
        y_response = np.mean(d_y_mat/d_commands,axis=2)
        poke = np.vstack((x_response,y_response))
        poke_fn = os.path.join(kcfg.poke_directory,'%s_poke.txt'%now_string())
        np.savetxt(poke_fn,poke)
        self.poke = Poke(poke)
        time.sleep(1)
        self.unpause()


class ImageDisplay0(QLabel):
    def __init__(self,name,downsample=1,clim=None,colormap=None,mouse_event_handler=None):
        super(ImageDisplay,self).__init__()
        self.name = name
        if clim is None:
            try:
                clim = np.loadtxt('.gui_settings/clim_%s.txt'%name)
            except Exception as e:
                pass
        self.clim = clim
        self.pixmap = QPixmap()
        self.downsample = downsample
        self.colormap = colormap
        self.zoomed = False
        if self.colormap is not None:
            self.colortable = colortable(self.colormap)
        if mouse_event_handler is not None:
            self.mousePressEvent = mouse_event_handler
        else:
            self.mousePressEvent = self.zoom
            
        data = np.random.rand(100,100)
        self.show(data)
        self.zoom_x1 = 0
        self.zoom_x2 = self.sx-1
        self.zoom_y1 = 0
        self.zoom_y2 = self.sy-1
        
    def show(self,data,boxes=None,lines=None):
        if self.clim is None:
            clim = (data.min(),data.max())
        else:
            clim = self.clim

        cmin,cmax = clim
        downsample = self.downsample
        data = data[::downsample,::downsample]
        self.sy,self.sx = data.shape
        
        if self.zoomed:
            data = data[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
        
        bmp = np.round(np.clip((data.astype(np.float)-cmin)/(cmax-cmin),0,1)*255).astype(np.uint8)
        #np.save('bmp.npy',bmp)
        sy,sx = bmp.shape
        n_bytes = bmp.nbytes
        bytes_per_line = int(n_bytes/sy)
        #QApplication.processEvents()
        image = QImage(bmp,sy,sx,bytes_per_line,QImage.Format_Indexed8)
        #image = QImage(bmp,sy,sx,bytes_per_line,QImage.Format_Grayscale8)
        if self.colormap is not None:
            image.setColorTable(self.colortable)
        #QApplication.processEvents()
        self.pixmap.convertFromImage(image)
        #QApplication.processEvents()

        #self.overlay.draw(self.pixmap)

        #painter = QPainter(self.pixmap)
        if boxes is not None:
            x1vec,x2vec,y1vec,y2vec = boxes
            pen = QPen()
            pen.setColor(QColor(*kcfg.active_search_box_color))
            pen.setWidth(kcfg.search_box_thickness)
            painter = QPainter()
            painter.begin(self.pixmap)
            painter.setPen(pen)
            for index,(x1,y1,x2,y2) in enumerate(zip(x1vec,y1vec,x2vec,y2vec)):
                width = float(x2 - x1 + 1)/float(self.downsample)
                painter.drawRect(x1/float(self.downsample)-self.zoom_x1,y1/float(self.downsample)-self.zoom_y1,width,width)
            painter.end()
            
        if lines is not None:
            x1vec,x2vec,y1vec,y2vec = lines
            pen = QPen()
            pen.setColor(QColor(*kcfg.slope_line_color))
            pen.setWidth(kcfg.slope_line_thickness)
            painter = QPainter()
            painter.begin(self.pixmap)
            painter.setPen(pen)
            for index,(x1,y1,x2,y2) in enumerate(zip(x1vec,y1vec,x2vec,y2vec)):
                painter.drawLine(QLine(x1/float(self.downsample)- self.zoom_x1,y1/float(self.downsample)-self.zoom_y1,x2/float(self.downsample)- self.zoom_x1,y2/float(self.downsample)- self.zoom_y1))
            painter.end()
            
        self.setPixmap(self.pixmap.scaled(self.sy,self.sx))
        
        
    def set_clim(self,clim):
        self.clim = clim

    def zoom(self,event):
        x,y = event.x(),event.y()
        if self.zoomed:
            self.zoomed = False
            self.zoom_x1 = 0
            self.zoom_x2 = self.sx-1
            self.zoom_y1 = 0
            self.zoom_y2 = self.sy-1
        else:
            self.zoomed = True
            self.zoom_x1 = x-kcfg.zoom_width//2
            self.zoom_x2 = x+kcfg.zoom_width//2
            self.zoom_y1 = y-kcfg.zoom_height//2
            self.zoom_y2 = y+kcfg.zoom_height//2
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
        
class ImageDisplay(QWidget):
    def __init__(self,name,downsample=1,clim=None,colormap=None,mouse_event_handler=None,image_min=None,image_max=None,width=512,height=512):
        super(ImageDisplay,self).__init__()
        self.name = name
        self.autoscale = False
        self.sx = width
        self.sy = height
        if clim is None:
            try:
                clim = np.loadtxt('.gui_settings/clim_%s.txt'%name)
            except Exception as e:
                self.autoscale = True
        
        self.clim = clim
        self.pixmap = QPixmap()
        self.label = QLabel()
        layout = QHBoxLayout()
        layout.addWidget(self.label)

        if image_min is not None and image_max is not None and not self.autoscale:
            n_steps = 1000.
            step_size = (image_max-image_min)/n_steps
        
            self.cmin_slider = QSlider(Qt.Vertical)
            self.cmax_slider = QSlider(Qt.Vertical)

            self.cmin_slider.setMinimum(image_min)
            self.cmax_slider.setMinimum(image_min)

            self.cmin_slider.setSingleStep(step_size)
            self.cmax_slider.setSingleStep(step_size)

            self.cmin_slider.setPageStep(step_size*10)
            self.cmax_slider.setPageStep(step_size*10)

            self.cmin_slider.setMaximum(image_max)
            self.cmax_slider.setMaximum(image_max)

            self.cmin_slider.setValue(self.clim[0])
            self.cmax_slider.setValue(self.clim[1])
            
            layout.addWidget(self.cmin_slider)
            layout.addWidget(self.cmax_slider)

            self.cmin_slider.valueChanged.connect(self.set_cmin)
            self.cmax_slider.valueChanged.connect(self.set_cmax)
        
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
        

    def set_cmax(self,val):
        self.clim = (self.clim[0],val)

    def set_cmin(self,val):
        self.clim = (val,self.clim[1])
        
    def show(self,data,boxes=None,lines=None):
        if self.autoscale:
            clim = (data.min(),data.max())
        else:
            clim = self.clim

        cmin,cmax = clim
        downsample = self.downsample
        data = data[::downsample,::downsample]
        
        if self.zoomed:
            data = data[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
        
        bmp = np.round(np.clip((data.astype(np.float)-cmin)/(cmax-cmin),0,1)*255).astype(np.uint8)
        sy,sx = bmp.shape
        n_bytes = bmp.nbytes
        bytes_per_line = int(n_bytes/sy)
        image = QImage(bmp,sy,sx,bytes_per_line,QImage.Format_Indexed8)
        if self.colormap is not None:
            image.setColorTable(self.colortable)
        self.pixmap.convertFromImage(image)
        
        if boxes is not None:
            x1vec,x2vec,y1vec,y2vec = boxes
            pen = QPen()
            pen.setColor(QColor(*kcfg.active_search_box_color))
            pen.setWidth(kcfg.search_box_thickness)
            painter = QPainter()
            painter.begin(self.pixmap)
            painter.setPen(pen)
            for index,(x1,y1,x2,y2) in enumerate(zip(x1vec,y1vec,x2vec,y2vec)):
                width = float(x2 - x1 + 1)/float(self.downsample)
                painter.drawRect(x1/float(self.downsample)-self.zoom_x1,y1/float(self.downsample)-self.zoom_y1,width,width)
            painter.end()
            
        if lines is not None:
            x1vec,x2vec,y1vec,y2vec = lines
            pen = QPen()
            pen.setColor(QColor(*kcfg.slope_line_color))
            pen.setWidth(kcfg.slope_line_thickness)
            painter = QPainter()
            painter.begin(self.pixmap)
            painter.setPen(pen)
            for index,(x1,y1,x2,y2) in enumerate(zip(x1vec,y1vec,x2vec,y2vec)):
                painter.drawLine(QLine(x1/float(self.downsample)- self.zoom_x1,y1/float(self.downsample)-self.zoom_y1,x2/float(self.downsample)- self.zoom_x1,y2/float(self.downsample)- self.zoom_y1))
            painter.end()

        if sy==self.sy and sx==self.sx:
            self.label.setPixmap(self.pixmap)
        else:
            self.label.setPixmap(self.pixmap.scaled(self.sy,self.sx))
        
    def set_clim(self,clim):
        self.clim = clim

    def zoom(self,event):
        x,y = event.x(),event.y()
        if self.zoomed:
            self.zoomed = False
            self.zoom_x1 = 0
            self.zoom_x2 = self.sx-1
            self.zoom_y1 = 0
            self.zoom_y2 = self.sy-1
        else:
            self.zoomed = True
            self.zoom_x1 = x-kcfg.zoom_width//2
            self.zoom_x2 = x+kcfg.zoom_width//2
            self.zoom_y1 = y-kcfg.zoom_height//2
            self.zoom_y2 = y+kcfg.zoom_height//2
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
class UI(QWidget):

    def __init__(self,loop):
        super(UI,self).__init__()
        self.loop = loop
        self.show_boxes = kcfg.show_search_boxes
        self.show_lines = kcfg.show_slope_lines
        
        self.loop.final_states.connect(self.update_states)
        self.init_UI()
        
        self.show()


    def init_UI(self):
        layout = QHBoxLayout()
        imax = 2**kcfg.bit_depth-1
        imin = 0
        self.id_spots = ImageDisplay('spots',downsample=2,clim=(50,1000),colormap=None,image_min=imin,image_max=imax)
        layout.addWidget(self.id_spots)
        self.pb_poke = QPushButton('Poke')
        self.pb_poke.clicked.connect(self.loop.run_poke)
        self.pb_quit = QPushButton('Quit')
        self.pb_quit.clicked.connect(sys.exit)
        layout.addWidget(self.pb_poke)
        layout.addWidget(self.pb_quit)
        self.setLayout(layout)
        

    @pyqtSlot(QObject,QObject)
    def update_states(self,sensor_state,mirror_state):
        try:
            self.sensor_state = sensor_state.copy()
            #print self.sensor_state.image.max()
            #print self.sensor_state.image.min()
            sensor_cfg = self.loop.sensor.get_configuration()
            #self.id_spots.draw_boxes(sensor_cfg.search_boxes)


            sb = sensor_cfg.search_boxes

            if self.show_boxes:
                boxes = [sb.x1,sb.x2,sb.y1,sb.y2]
            else:
                boxes = None

            if self.show_lines:
                lines = [sb.x,sb.x+self.sensor_state.x_slopes*kcfg.slope_line_magnification,
                         sb.y,sb.y+self.sensor_state.y_slopes*kcfg.slope_line_magnification]
            else:
                lines = None
                
            self.id_spots.show(self.sensor_state.image,boxes=boxes,lines=lines)
            
        except Exception as e:
            print e
            
    def select_single_spot(self,click):
        print 'foo'
        x = click.x()*self.downsample
        y = click.y()*self.downsample
        self.single_spot_index = self.loop.sensor.search_boxes.get_lenslet_index(x,y)
        
        
app = QApplication(sys.argv)
cam = cameras.SimulatedCamera()
sensor = Sensor(cam)
mirror = Mirror()
loop = Loop(sensor,mirror)
ui = UI(loop)
loop.start()

sys.exit(app.exec_())

class Foo:

    def auto_center(self,window_spots=False,spot_half_width=5):
        self.pause()
        # make a simulated spots image
        sim = np.zeros((kcfg.image_height_px,kcfg.image_width_px))
        for x,y in zip(self.x_ref,self.y_ref):
            ry,rx = int(round(y)),int(round(x))
            sim[ry-spot_half_width:ry+spot_half_width+1,
                rx-spot_half_width:rx+spot_half_width+1] = 1.0

        spots = self.cam.get_image()

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

        #plt.imshow(nxc,interpolation='none')
        #plt.savefig('nxc.png',dpi=300)
        #sys.exit()

        
        sy,sx = nxc.shape
        ymax,xmax = np.unravel_index(np.argmax(nxc),nxc.shape)
        if ymax>sy//2:
            ymax = ymax-sy
        if xmax>sx//2:
            xmax = xmax-sx
        
        new_x_ref = self.x_ref+xmax
        new_y_ref = self.y_ref+ymax
        self.search_boxes = SearchBoxes(new_x_ref,new_y_ref,kcfg.search_box_half_width)
        self.x_ref = new_x_ref
        self.y_ref = new_y_ref

        self.unpause()

    def record_reference(self):
        self.pause()
        xcent = []
        ycent = []
        for k in range(kcfg.reference_n_measurements):
            self.step()
            xcent.append(self.x_centroids)
            ycent.append(self.y_centroids)
        self.x_ref = np.array(xcent).mean(0)
        self.y_ref = np.array(ycent).mean(0)
        self.search_boxes = SearchBoxes(self.x_ref,self.y_ref,kcfg.search_box_half_width)
        outfn = os.path.join(kcfg.reference_directory,prepend('coords.txt',now_string()))
        refxy = np.array((self.x_ref,self.y_ref)).T
        np.savetxt(outfn,refxy,fmt='%0.2f')
        self.unpause()
        
    def make_reference_coordinates(self,x_offset=0.0,y_offset=0.0):
        sensor_x = kcfg.image_width_px
        sensor_y = kcfg.image_height_px
        mask = np.loadtxt(kcfg.reference_mask_filename)
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
        ref_xy = np.array(ref_xy)
        return ref_xy
    
    def set_background_correction(self,val):
        self.background_correction = val

    def get_background_correction(self):
        return self.background_correction
        
    def set_estimate_background(self,val):
        self.estimate_background = val
        
    def get_estimate_background(self):
        return self.estimate_background
        
    def step(self):
        self.spots = self.cam.get_image()
        self.mutex.lock()
        sb = self.search_boxes
        xr = self.x_ref.copy()
        yr = self.y_ref.copy()
        half_width = sb.half_width
        for iteration in range(self.n_iterations):
            if iteration==0:
                estimate_background = self.estimate_background
                background_correction = self.background_correction
            else:
                estimate_background = False
                background_correction = 0.0
            centroid.compute_centroids(self.spots,
                                       sb.x1,sb.x2,
                                       sb.y1,sb.y2,
                                       xr,yr,
                                       self.total_intensity,
                                       self.maximum_intensity,
                                       self.minimum_intensity,
                                       self.background_intensity,
                                       estimate_background,
                                       background_correction,
                                       kcfg.centroiding_num_threads)

            if half_width<=kcfg.iterative_centroiding_step+1:
                break
            half_width-=kcfg.iterative_centroiding_step
            sb = SearchBoxes(xr,yr,half_width)
        self.x_centroids = xr
        self.y_centroids = yr

        self.x_slopes = (self.x_centroids-self.x_ref)*self.pixel_size_m/self.lenslet_focal_length_m
        self.y_slopes = (self.y_centroids-self.y_ref)*self.pixel_size_m/self.lenslet_focal_length_m
        self.error = np.sqrt(np.var(self.x_slopes[np.where(self.active_lenslets)[0]])+np.var(self.y_slopes[np.where(self.active_lenslets)[0]]))
        self.tilt = np.mean(self.x_slopes)
        self.tip = np.mean(self.y_slopes)
        self.x_slopes-=self.tilt
        self.y_slopes-=self.tip
        self.set_active_lenslets()
        self.mutex.unlock()
        self.ping.emit()
        
        
    def get_fps(self):
        return self.fps

    # def run(self):
    #     print "sensor thread started"
    #     timer = QTimer()
    #     timer.timeout.connect(self.update)
    #     timer.start(1.0/self.update_rate*1000.0)
    #     self.exec_()


class Mirror(Component):

    def __init__(self,**kwargs):
        super(Mirror,self).__init__(**kwargs)
        self.flat = np.loadtxt(kcfg.mirror_flat_filename)
        self.mask = np.loadtxt(kcfg.mirror_mask_filename)
        self.map = np.zeros(self.mask.shape)
        self.command = np.zeros(self.flat.shape)
        if not len(self.command)==np.sum(self.mask):
            sys.exit('Disagreement between mirror flat and mirror mask.')
        self.n_actuators = len(self.command)
        self.command[:] = self.flat[:]
        self.map[np.where(self.mask)] = self.command
        self.action_functions = []
        self.numerical_functions = []
        self.boolean_functions = []
        self.boolean_functions.append((self.get_paused,self.set_paused,'Paused'))
        self.action_functions.append((self.flatten,'&Flatten miror'))
        self.action_functions.append((self.replace_flat,'Replace flat'))
        print 'Computing mirror maximum rate...'
        self.max_rate = self.get_max_rate(100)
        print 'Maximum rate is %0.2f Hz.'%self.max_rate

    def flatten(self):
        self.pause()
        self.command[:] = self.flat[:]
        self.update()
        self.unpause()

    def replace_flat(self):
        out_fn = prepend(kcfg.mirror_flat_filename,now_string())
        self.flat[:] = self.command[:]
        np.savetxt(out_fn,self.flat)

    def step(self):
        # ACTUALLY UPDATE MIRROR HERE
        self.map[np.where(self.mask)] = self.command
        self.ping.emit()
        time.sleep(kcfg.mirror_settling_time_s)
        
    def get_fps(self):
        return self.fps

    def set_actuator(self,actuator_index,value):
        self.command[:] = self.flat[:]
        self.command[actuator_index]=value
        self.step()
        self.ping.emit()


class Loop:
    def __init__(self,sensor,mirror,parent=None):
        self.n_modes = kcfg.loop_n_control_modes
        self.gain = kcfg.loop_gain
        self.loss = kcfg.loop_loss
        self.count = 0
        self.t0 = time.time()
        self.paused = False
        self.closed = False
        self.sensor = sensor
        self.mutex = sensor.mutex
        self.mirror = mirror
        self.active_lenslets = np.ones(sensor.active_lenslets.shape,dtype=np.int)
        self.sensor.ping.connect(self.update)
        self.action_functions = []
        self.numerical_functions = []
        
        self.numerical_functions.append((self.get_gain,self.set_gain,'Loop gain',(0.0,1.0,.01),self.has_poke))
        self.numerical_functions.append((self.get_loss,self.set_loss,'Loop loss',(0.0,1.0,.01),self.has_poke))
        
        self.numerical_functions.append((self.get_n_modes,self.set_n_modes,'Control modes',(1,kcfg.mirror_n_actuators,1),self.has_poke))
        self.boolean_functions = []
        self.boolean_functions.append((self.get_closed,self.set_closed,'&Closed',self.has_poke))
        self.boolean_functions.append((self.get_paused,self.set_paused,'Paused'))
        self.action_functions.append((self.load_poke,'Load poke'))
        self.action_functions.append((self.run_poke,'Run poke'))
        self.action_functions.append((self.step,'Step'))
        self.poke = None

    def get_gain(self):
        return self.gain

    def set_gain(self,val):
        self.gain = val
        
    def get_loss(self):
        return self.loss

    def set_loss(self,val):
        self.loss = val
        
    def get_n_modes(self):
        if self.poke is None:
            return kcfg.loop_n_control_modes
        else:
            return self.poke.n_modes

    def set_n_modes(self,val):
        self.poke.n_modes = val
        self.poke.invert(mask=self.active_lenslets)

    def get_cutoff_cond(self):
        try:
            return self.poke.cutoff_cond
        except Exception as e:
            return np.nan
        
    def get_full_cond(self):
        try:
            return self.poke.full_cond
        except Exception as e:
            return np.nan
        
    def has_poke(self):
        return self.poke is not None

    def load_poke(self):
        options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        poke_filename, _ = QFileDialog.getOpenFileName(
                        None,
                        "Please select a poke file.",
                        kcfg.poke_directory,
                        "Text Files (*.txt)",
                        options=options)

        try:
            poke = np.loadtxt(poke_filename)
        except IOError as ioe:
            error_message('Could not find %s.'%poke_filename)
            return
        
        py,px = poke.shape

        try:
            assert py==2*self.sensor.n_lenslets
        except AssertionError as ae:
            error_message('Poke matrix has %d rows, but %d expected.'%(py,2*self.sensor.n_lenslets))
            return
        try:
            assert px==self.mirror.n_actuators
        except AssertionError as ae:
            error_message('Poke matrix has %d columns, but %d expected.'%(px,2*self.mirror.n_actuators))
            return

        self.poke = Poke(poke)

            
    def set_closed(self,val):
        self.closed = val
            
    def get_closed(self):
        return self.closed
    
    def set_paused(self,val):
        self.paused = val
        if not val:
            self.count = 0
            self.t0 = time.time()
        else:
            self.count = 0
            
    def get_paused(self):
        return self.paused
    
    def pause(self):
        self.set_paused(True)

    def unpause(self):
        self.set_paused(False)
        
    def start(self):
        self.sensor.start()
        self.mirror.start()

    def update(self):
        QApplication.processEvents()
        if self.paused:
            return
        self.step()
        self.count = self.count + 1

    def step(self):
        if self.closed and self.has_poke():
            if not all(self.active_lenslets==sensor.active_lenslets):
                self.active_lenslets[:] = sensor.active_lenslets[:]
                self.poke.invert(mask=self.active_lenslets)
            self.mutex.lock()
            xs = self.sensor.x_slopes[np.where(self.active_lenslets)[0]]
            ys = self.sensor.y_slopes[np.where(self.active_lenslets)[0]]
            self.mutex.unlock()
            slope_vec = np.hstack((xs,ys))

            command = self.gain * np.dot(self.poke.ctrl,slope_vec)
            command = self.mirror.command*(1-self.loss) - command
            self.mirror.command = command
            
    def get_n_ctrl_stored(self):
        try:
            return self.poke.n_ctrl_stored
        except Exception as e:
            return 0

        
        
        
class Indicator(QLabel):
    def __init__(self,fmt,func,signed=False):
        super(Indicator,self).__init__()
        self.fmt = fmt
        self.func = func
        self.signed = signed
        self.update()
        
    def update(self):
        if not self.signed:
            self.setText(self.fmt%self.func())
        else:
            val = self.func()
            if val<0:
                sign = '-'
            else:
                sign = '+'
            self.setText(sign+self.fmt%val)
        
        
class Gui(QWidget):
    def __init__(self,loop):
        super(Gui,self).__init__()
        self.fps = 0.0
        self.fps_interval = 2.0
        self.counter = 0
        self.timer = QTimer()
        self.timer.setInterval(1000.0*self.fps_interval)
        self.timer.timeout.connect(self.compute_fps)
        self.timer.start()
        
        self.spots_pixmap = QPixmap()
        self.single_spot_pixmap = QPixmap()
        self.mirror_pixmap = QPixmap()

        self.loop = loop
        self.mutex = self.loop.sensor.mutex
        self.bit_depth = kcfg.bit_depth
        self.scale_factor = kcfg.interface_scale_factor
        self.downsample = int(round(1.0/self.scale_factor))
        self.setWindowIcon(QIcon('./icons/kungpao.png'))
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


        self.mirror_color_table = colortable('jet')
        
        self.single_spot_index = 0

        #self.bmp_spots = self.bmpscale(self.loop.sensor.spots,self.downsample,self.cmin,self.cmax)
        
        self.show_search_boxes = kcfg.show_search_boxes
        self.show_slope_lines = kcfg.show_slope_lines

        self.boolean_functions = []
        self.boolean_functions.append((self.get_show_search_boxes,self.set_show_search_boxes,'Show &search boxes'))
        self.boolean_functions.append((self.get_show_slope_lines,self.set_show_slope_lines,'Show slope &lines'))
        self.numerical_functions = []
        self.numerical_functions.append((self.get_spots_cmin,self.set_spots_cmin,'Spots contrast min',(-2**kcfg.bit_depth,2**kcfg.bit_depth,1.0)))
        self.numerical_functions.append((self.get_spots_cmax,self.set_spots_cmax,'Spots contrast max',(-2**kcfg.bit_depth,2**kcfg.bit_depth,1.0)))
        self.action_functions = []
        self.action_functions.append((self.shutdown,'&Quit'))
        
        self.active_check_list = []
        
        self.init_pens()
        self.initUi()

        self.loop.sensor.ping.connect(self.update_spots_pixmap)
        self.loop.mirror.ping.connect(self.update_mirror_pixmap)
        self.loop.start()
        

    def init_pens(self):
        self.slope_line_color = QColor(*kcfg.slope_line_color)
        self.slope_line_thickness = kcfg.slope_line_thickness
        self.slope_line_pen = QPen()
        self.slope_line_pen.setColor(self.slope_line_color)
        self.slope_line_pen.setWidth(self.slope_line_thickness)
        
        self.search_box_thickness = kcfg.search_box_thickness

        self.active_search_box_color = QColor(*kcfg.active_search_box_color)
        self.active_search_box_pen = QPen()
        self.active_search_box_pen.setColor(self.active_search_box_color)
        self.active_search_box_pen.setWidth(self.search_box_thickness)
        
        self.inactive_search_box_color = QColor(*kcfg.inactive_search_box_color)
        self.inactive_search_box_pen = QPen()
        self.inactive_search_box_pen.setColor(self.inactive_search_box_color)
        self.inactive_search_box_pen.setWidth(self.search_box_thickness)
        
        self.single_spot_color = QColor(*kcfg.single_spot_color)
        self.single_spot_thickness = kcfg.single_spot_thickness
        self.single_spot_pen = QPen()
        self.single_spot_pen.setColor(self.single_spot_color)
        self.single_spot_pen.setWidth(self.single_spot_thickness)
        
        self.spots_painter = QPainter(self.spots_pixmap)
        
        
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
        self.mirror_label = QLabel()
        self.mirror_label.setPixmap(self.mirror_pixmap)
        images.addWidget(self.mirror_label)
        self.layout.addLayout(images)
        
        controls = QGridLayout()
        sensor_controls = self.make_control_frame(self.loop.sensor,'Sensor')
        controls.addWidget(sensor_controls,0,0)

        mirror_controls = self.make_control_frame(self.loop.mirror,'Mirror')
        controls.addWidget(mirror_controls,0,1)
        
        loop_controls = self.make_control_frame(self.loop,'Loop')
        controls.addWidget(loop_controls,1,0)
        
        ui_controls = self.make_control_frame(self,'User Interface')
        controls.addWidget(ui_controls,1,1)

        self.indicators = []
        self.indicators.append(Indicator(kcfg.ui_fps_fmt,self.get_fps))
        self.indicators.append(Indicator(kcfg.sensor_fps_fmt,self.loop.sensor.get_fps))
        self.indicators.append(Indicator(kcfg.mirror_fps_fmt,self.loop.mirror.get_fps))
        self.indicators.append(Indicator(kcfg.wavefront_error_fmt,self.loop.sensor.get_error))
        self.indicators.append(Indicator(kcfg.tip_fmt,self.loop.sensor.get_tip,True))
        self.indicators.append(Indicator(kcfg.tilt_fmt,self.loop.sensor.get_tilt,True))
        self.indicators.append(Indicator('%0.2f ADU',lambda: self.loop.sensor.total_intensity.mean()))
        self.indicators.append(Indicator('%0.2e (full condition)',self.loop.get_full_cond))
        self.indicators.append(Indicator('%0.2e (cutoff condition)',self.loop.get_cutoff_cond))
        self.indicators.append(Indicator('%d (control matrices stored)',self.loop.get_n_ctrl_stored))
        
        indicator_box = QGroupBox('Indicators')
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        for ind in self.indicators:
            layout.addWidget(ind)
        indicator_box.setLayout(layout)
        controls.addWidget(indicator_box,0,2)
        
        self.layout.addLayout(controls)

        self.setWindowTitle("kungpao")
        self.resize(1200, self.loop.sensor.spots.shape[0]*self.scale_factor)
        self.setLayout(self.layout)
        
    def make_control_frame(self,obj,label=''):
        group_box = QGroupBox(label)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        idx = 0
        try:
            for tup in obj.action_functions:
                pb = QPushButton(tup[1])
                try:
                    self.active_check_list.append((pb,tup[2]))
                except Exception as e:
                    pass
                pb.clicked.connect(tup[0])
                layout.addWidget(pb)
                idx+=1
        except Exception as e:
            sys.exit(e)
        try:
            for tup in obj.boolean_functions:
                cb = QCheckBox(tup[2])
                try:
                    self.active_check_list.append((cb,tup[3]))
                except Exception as e:
                    pass
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
                if type(step)==int:
                    sb = QSpinBox()
                elif type(step)==float:
                    sb = QDoubleSpinBox()
                sb.setMinimum(tup[3][0])
                sb.setMaximum(tup[3][1])
                sb.setValue(tup[0]())
                sb.setSingleStep(tup[3][2])
                if type(step)==int:
                    sb.valueChanged[int].connect(tup[1])
                elif type(step)==float:
                    sb.valueChanged[float].connect(tup[1])
                box.addWidget(sb)
                layout.addLayout(box)
                try:
                    self.active_check_list.append((sb,tup[4]))
                except Exception as e:
                    pass
                idx+=1
        except Exception as e:
            pass
        group_box.setLayout(layout)
        return group_box

    
    def bmpscale(self,arr,downsample,cmin,cmax):
        return np.round(np.clip((arr[::downsample,::downsample]-cmin)/(cmax-cmin),0,1)*255).astype(np.uint8)
        
    def update_spots_pixmap(self):
        # receives slopes and image from sensor signal
        # first, convert the numpy image data into a QImage
        k = self.single_spot_index

        self.mutex.lock()
        data = self.loop.sensor.spots
        self.bmp = self.bmpscale(data,self.downsample,self.cmin,self.cmax)
        single = data[self.loop.sensor.search_boxes.y1[k]:self.loop.sensor.search_boxes.y2[k]+1,
                      self.loop.sensor.search_boxes.x1[k]:self.loop.sensor.search_boxes.x2[k]+1]
        self.single_bmp = self.bmpscale(single,1,self.cmin,self.cmax)
        self.mutex.unlock()
        
        sy,sx = self.bmp.shape
        n_bytes = self.bmp.nbytes
        bytes_per_line = int(n_bytes/sy)
        image = QImage(self.bmp,sx,sy,bytes_per_line,QImage.Format_Grayscale8)
        #self.spots_pixmap = QPixmap.fromImage(image)
        self.spots_pixmap.convertFromImage(image)

        ssy,ssx = self.single_bmp.shape
        n_bytes = self.single_bmp.nbytes
        bytes_per_line = int(n_bytes/ssy)
        image = QImage(self.single_bmp,ssx,ssy,bytes_per_line,QImage.Format_Grayscale8)
        
        #self.single_spot_pixmap = QPixmap.fromImage(image)
        self.single_spot_pixmap.convertFromImage(image)
        #self.update()

    def update_mirror_pixmap(self):
        data = self.loop.mirror.map
        self.mirror_bmp = self.bmpscale(data,1,kcfg.mirror_command_min,kcfg.mirror_command_max)
        msy,msx = self.mirror_bmp.shape
        n_bytes = self.mirror_bmp.nbytes
        bytes_per_line = int(n_bytes/msy)
        image = QImage(self.mirror_bmp,msx,msy,bytes_per_line,QImage.Format_Indexed8)
        image.setColorTable(self.mirror_color_table)
        self.mirror_pixmap = QPixmap.fromImage(image)
        #self.update()

    def compute_fps(self):
        self.fps = float(self.counter)/float(self.fps_interval)
        self.counter = 0

    def get_fps(self):
        return self.fps

    def activate_widgets(self):
        for wid,func in self.active_check_list:
            wid.setEnabled(func())
        
    def paintEvent(self, event):
        self.counter = self.counter + 1
        if self.show_search_boxes:
            self.draw_search_boxes()
        if self.show_slope_lines:
            self.draw_slopes()
        self.spots_label.setPixmap(self.spots_pixmap)
        self.single_spot_label.setPixmap(self.single_spot_pixmap.scaled(256,256))
        self.mirror_label.setPixmap(self.mirror_pixmap.scaled(256,256))
        for ind in self.indicators:
            ind.update()
        self.activate_widgets()
        
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
        for spot_index,(x1a,y1a,x2a,y2a) in enumerate(zip(self.loop.sensor.search_boxes.x1,
                                                          self.loop.sensor.search_boxes.y1,
                                                          self.loop.sensor.search_boxes.x2,
                                                          self.loop.sensor.search_boxes.y2)):
            if spot_index==self.single_spot_index:
                self.spots_painter.setPen(self.single_spot_pen)
            elif self.loop.sensor.active_lenslets[spot_index]:
                self.spots_painter.setPen(self.active_search_box_pen)
            else:
                self.spots_painter.setPen(self.inactive_search_box_pen)
            x1 = x1a*self.scale_factor
            y1 = y1a*self.scale_factor
            x2 = x2a*self.scale_factor
            y2 = y2a*self.scale_factor
            width = float(x2a-x1a)*self.scale_factor
            self.spots_painter.drawRect(x1,y1,width,width)
        self.spots_painter.end()
        
    def keyPressEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
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
        elif event.key() == Qt.Key_I:
            if modifiers==Qt.ControlModifier:
                self.loop.sensor.search_boxes.up()
        elif event.key() == Qt.Key_M:
            if modifiers==Qt.ControlModifier:
                self.loop.sensor.search_boxes.down()
        elif event.key() == Qt.Key_K:
            if modifiers==Qt.ControlModifier:
                self.loop.sensor.search_boxes.right()
        elif event.key() == Qt.Key_J:
            if modifiers==Qt.ControlModifier:
                self.loop.sensor.search_boxes.left()
        else:
            super(Gui, self).keyPressEvent(event)

    def shutdown(self):
        self.write_settings()
        self.close()
        
    def write_settings(self):
        print 'writing %0.2f, %0.2f to cmin, cmax'%(self.cmin,self.cmax)
        np.savetxt('.gui_settings/cmax.txt',[self.cmax])
        np.savetxt('.gui_settings/cmin.txt',[self.cmin])
        

if __name__ == '__main__':

    # construct the QApplication
    app = QApplication(sys.argv)
    QThread.currentThread().setPriority(QThread.LowPriority)
    # create a camera

    mutex = QMutex()
    camera = cameras.SimulatedCamera(mutex)

    sensor = Sensor(camera,mutex,update_rate=kcfg.sensor_update_rate)
    mirror = Mirror(update_rate=kcfg.mirror_update_rate)
    loop = Loop(sensor,mirror)
    gui = Gui(loop)
    
    gui.show()
    sys.exit(app.exec_())
