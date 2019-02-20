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
                             QSpinBox,QDoubleSpinBox,QSizePolicy,QFileDialog,
                             QErrorMessage)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb, QPen, QBitmap, QPalette, QIcon
import time
import os
import psutil
from matplotlib import pyplot as plt
import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

process = psutil.Process(os.getpid())

def error_message(message):
    error_dialog = QErrorMessage()
    error_dialog.setWindowModality(Qt.WindowModal)
    error_dialog.showMessage(message)
    error_dialog.exec_()


def now_string():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def prepend(full_path_fn,prefix):
    p,f = os.path.split(full_path_fn)
    return os.path.join(p,'%s_%s'%(prefix,f))

def colortable(colormap_name):
    try:
        cmapobj = plt.get_cmap(colormap_name)
    except AttributeError as ae:
        print '\'%s\' is not a valid colormap name'%colormap_name
        print 'using \'bone\' instead'
        cmapobj = plt.get_cmap('bone')
    ncolors = cmapobj.N


    cmap = np.uint8(cmapobj(range(ncolors))[:,:3]*255.0)
    table = []
    for row in xrange(cmap.shape[0]):
        table.append(qRgb(cmap[row,0],cmap[row,1],cmap[row,2]))
    return table


class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()
 
 
    def plot(self):
        data = np.random.randn(25)
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        ax.set_title('PyQt Matplotlib Example')
        self.draw()


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

    def copy(self):
        rx = self.x.copy()
        ry = self.y.copy()
        sb = SearchBoxes(rx,ry,self.half_width)
        return sb


class Component(QThread):

    ping = pyqtSignal()
    
    def __init__(self,update_rate=30.0,fps_interval=1.0,initial_paused=False,parent=None):
        super(QThread,self).__init__(parent)
        self.update_rate = update_rate
        self.count = 0.0
        self.t0 = time.time()
        self.paused = initial_paused
        self.fps_interval = fps_interval
        self.fps_window = int(round(self.update_rate*self.fps_interval))
        self.fps = 0.0
        
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
        
    def update(self):
        if self.paused:
            return
        self.step()
        self.count = self.count + 1
        if self.count==self.fps_window:
            self.fps = float(self.count)/(time.time()-self.t0)
            self.count = 0
            self.t0 = time.time()
        
    def step(self):
        pass

    def run(self):
        timer = QTimer()
        timer.timeout.connect(self.update)
        timer.start(1.0/self.update_rate*1000.0)
        self.exec_()

    def get_max_rate(self,n_iterations=100):
        self.pause()
        t0 = time.time()
        for k in range(n_iterations):
            self.step()
        t = time.time()-t0
        self.unpause()
        return float(n_iterations)/t
            

class Sensor(Component):
    # Currently this subclasses Component, which means that it runs on its
    # own clock/thread. The disadvantage of this approach is that its
    # update rate has to be set such that each camera frame is new (i.e.
    # it shouldn't update faster than the exposure/transfer cycle. We
    # should think about re-implementing this so that it initializes its
    # own camera object and passes its update function to the camera thread
    # to be called on exposure, as described in this example:
    # https://pgi-jcns.fz-juelich.de/portal/pages/using-c-from-python.html
    
    def __init__(self,camera,**kwargs):#update_rate=30.0,fps_window=100,initial_paused=False,parent=None):
        super(Sensor,self).__init__(**kwargs)#update_rate,fps_window,initial_paused,parent)
        
        # load some configuration data
        try:
            ref_xy = np.loadtxt(kcfg.reference_coordinates_filename)
        except Exception as e:
            # Create search boxes using basic parameters of the lenset
            # array and sensor
            # The only reason to do this is to start up the software
            # when there's never been a reference coordinate set. The next
            # step should be to identify a reasonable guess for their position
            # and record the centroids as reference coordinates
            ref_xy = self.make_reference_coordinates()
        self.x_ref = ref_xy[:,0]
        self.y_ref = ref_xy[:,1]
        self.n_lenslets = len(self.x_ref)
        self.search_boxes = SearchBoxes(self.x_ref,self.y_ref,kcfg.search_box_half_width)
        self.lenslet_focal_length_m = kcfg.lenslet_focal_length_m
        self.pixel_size_m = kcfg.pixel_size_m
        self.cam = camera
        self.spots = self.cam.get_image()
        self.x_centroids = self.x_ref.copy()
        self.y_centroids = self.y_ref.copy()
        self.x_slopes = np.zeros(self.search_boxes.n)
        self.y_slopes = np.zeros(self.search_boxes.n)
        self.total_intensity = np.zeros(self.search_boxes.n)
        self.maximum_intensity = np.zeros(self.search_boxes.n)
        self.minimum_intensity = np.zeros(self.search_boxes.n)
        self.background_intensity = np.zeros(self.search_boxes.n)
        self.active_lenslets = np.ones(self.n_lenslets)
        self.n_iterations = kcfg.centroiding_iterations
        self.filter_lenslets = kcfg.sensor_filter_lenslets
        self.error = 0.0
        self.tilt = 0.0
        self.tip = 0.0
        self.background_correction = kcfg.background_correction
        # Boolean settings
        self.estimate_background = kcfg.estimate_background
        self.show_search_boxes = kcfg.show_search_boxes
        self.show_slope_lines = kcfg.show_slope_lines
        self.boolean_functions = []
        self.boolean_functions.append((self.get_estimate_background,self.set_estimate_background,'Estimate background'))
        self.boolean_functions.append((self.get_filter_lenslets,self.set_filter_lenslets,'Filter lenslets'))
        self.boolean_functions.append((self.get_paused,self.set_paused,'Paused'))
        self.boolean_functions.append((self.cam.get_opacity,self.cam.set_opacity,'Opacity'))
        self.numerical_functions = []
        self.numerical_functions.append((self.get_background_correction,self.set_background_correction,'Background correction',(-2**kcfg.bit_depth,2**kcfg.bit_depth,.5)))
        self.numerical_functions = self.numerical_functions + self.search_boxes.numerical_functions
        self.action_functions = []
        self.action_functions.append((self.step,'Step'))
        self.action_functions.append((self.auto_center,'Auto center'))
        self.action_functions.append((self.record_reference,'Record reference'))
        print 'Computing sensor maximum rate...'
        self.max_rate = self.get_max_rate(50)
        print 'Maximum rate is %0.2f Hz.'%self.max_rate

    def set_active_lenslets(self):
        #self.active_lenslets[:] = 0
        #self.active_lenslets[np.where(self.maximum_intensity-self.minimum_intensity>kcfg.spots_threshold)[0]] = 1
        if self.filter_lenslets:
            self.active_lenslets[:] = 1
            thresh = self.total_intensity.mean()-self.total_intensity.std()*3
            self.active_lenslets[np.where(self.total_intensity<thresh)]=0

    def get_filter_lenslets(self):
        return self.filter_lenslets

    def set_filter_lenslets(self,val):
        self.filter_lenslets = val
            
    def get_error(self):
        return self.error*1e6

    def get_tilt(self):
        return self.tilt*1000

    def get_tip(self):
        return self.tip*1000

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

        sb = self.search_boxes
        xr = self.x_ref.copy()
        yr = self.y_ref.copy()
        half_width = sb.half_width
        for iteration in range(self.n_iterations):
            centroid.compute_centroids(self.spots,
                                       sb.x1,sb.x2,
                                       sb.y1,sb.y2,
                                       xr,yr,
                                       self.total_intensity,
                                       self.maximum_intensity,
                                       self.minimum_intensity,
                                       self.background_intensity,
                                       self.estimate_background,
                                       self.background_correction,
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
        self.ping.emit()
        
        
    def get_fps(self):
        #print(process.memory_info().rss)//1024//1024
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
        self.max_rate = self.get_max_rate(500)
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
            xs = self.sensor.x_slopes[np.where(self.active_lenslets)[0]]
            ys = self.sensor.y_slopes[np.where(self.active_lenslets)[0]]
            slope_vec = np.hstack((xs,ys))

            command = self.gain * np.dot(self.poke.ctrl,slope_vec)
            command = self.mirror.command*(1-self.loss) - command
            self.mirror.command = command
            
    def run_poke(self):
        cmin = kcfg.poke_current_min
        cmax = kcfg.poke_current_max
        n_currents = kcfg.poke_n_current_steps
        currents = np.linspace(cmin,cmax,n_currents)
        self.sensor.pause()
        self.mirror.pause()
        n_lenslets = self.sensor.n_lenslets
        n_actuators = self.mirror.n_actuators
        x_mat = np.zeros((n_lenslets,n_actuators,n_currents))
        y_mat = np.zeros((n_lenslets,n_actuators,n_currents))
        for k_actuator in range(n_actuators):
            for k_current in range(n_currents):
                cur = currents[k_current]
                print k_actuator,cur
                self.mirror.set_actuator(k_actuator,cur)
                self.sensor.step()
                QApplication.processEvents()
                x_mat[:,k_actuator,k_current] = self.sensor.x_slopes
                y_mat[:,k_actuator,k_current] = self.sensor.y_slopes
        self.mirror.flatten()
        self.sensor.unpause()
        self.mirror.unpause()
        d_currents = np.mean(np.diff(currents))
        d_x_mat = np.diff(x_mat,axis=2)
        d_y_mat = np.diff(y_mat,axis=2)

        x_response = np.mean(d_x_mat/d_currents,axis=2)
        y_response = np.mean(d_y_mat/d_currents,axis=2)
        poke = np.vstack((x_response,y_response))
        poke_fn = prepend(kcfg.poke_filename,now_string())
        np.savetxt(poke_fn,poke)
        self.poke = Poke(poke)

    def get_n_ctrl_stored(self):
        try:
            return self.poke.n_ctrl_stored
        except Exception as e:
            return 0

        
        
        
class Poke:
    def __init__(self,poke_matrix):
        self.poke = poke_matrix
        self.n_modes = kcfg.loop_n_control_modes
        self.n_ctrl_stored = 0
        self.ctrl_dict = {}
        self.ctrl_key_list = []
        self.invert()

    def mask_to_key1(self,mask):
        if mask is None:
            return ''
        else:
            return ''.join(['%d'%k for k in list(mask)])

    def mask_to_key(self,mask):
        if mask is None:
            return 'None'
        else:
            return '_'.join(['%d'%idx for idx in np.where(mask==0)[0]])+'_%d'%self.n_modes

    def get_stored_ctrl(self,mask):
        key = self.mask_to_key(mask)
        try:
            out = self.ctrl_dict[key]
        except KeyError as ke:
            out = None

    def store_ctrl(self,mask,ctrl):
        key = self.mask_to_key(mask)
        if key in self.ctrl_dict.keys():
            return
        self.ctrl_dict[key] = ctrl
        self.ctrl_key_list.append(key)
        self.n_ctrl_stored+=1
        self.print_dict_info()
        assert self.n_ctrl_stored==len(self.ctrl_dict.keys())
        assert self.n_ctrl_stored==len(self.ctrl_key_list)

    def print_dict_info(self):
        print 'N stored:',self.n_ctrl_stored
        print 'Current dictionary:'
        print self.ctrl_dict.keys()
        print 'Current key list:'
        print self.ctrl_key_list
        print

        
    def trim_ctrl_dict(self):
        if self.n_ctrl_stored<=kcfg.ctrl_dictionary_max_size:
            return
        else:
            n_to_remove = self.n_ctrl_stored-kcfg.ctrl_dictionary_max_size
            print 'Removing %d'%n_to_remove
            for k in range(n_to_remove):
                key = self.ctrl_key_list[k]
                self.print_dict_info()
                print 'Key to remove: %s'%key
                del self.ctrl_dict[key]
                self.n_ctrl_stored-=1
                self.ctrl_key_list.remove(key)
        
    def invert(self,subtract_mean=False,mask=None):
        self.ctrl = self.get_stored_ctrl(mask)
        if self.ctrl is not None:
            return
        
        t0 = time.time()

        poke = self.poke.copy()

        #mask = np.round(np.random.rand(poke.shape[0]//2)).astype(np.int)
        if mask is not None:
            double_mask = np.hstack((mask,mask))
            poke = poke[np.where(double_mask)[0],:]

        double_n_lenslets,n_actuators = poke.shape

        if subtract_mean:
            # subtract mean influence across actuators from
            # each actuator's influence
            # transpose, broadcast, transpose back:
            m_poke = np.mean(poke,axis=1)
            poke = (poke.T - m_poke).T

        U,s,V = np.linalg.svd(poke)


        self.full_cond = (s[0]/s).max()
        self.cutoff_cond = s[0]/s[self.n_modes-1]
        # zero upper modes
        if self.n_modes<n_actuators:
            s[self.n_modes:] = 0

        term1 = V.T
        term2 = np.zeros([n_actuators,double_n_lenslets])
        term2[:n_actuators,:n_actuators] = np.linalg.pinv(np.diag(s))
        term3 = U.T
        ctrlmat = np.dot(np.dot(term1,term2),term3)
        dt = time.time()-t0

        sanity_check = False
        if sanity_check:
            # double check the explicit Moore-Penrose pseudoinverse
            # above with LAPACK implementation (pinv)
            cutoff_cond = s[n_modes]/s[0]
            test = np.linalg.pinv(poke,cutoff_cond)
            if np.allclose(test,ctrlmat):
                print 'Pseudoinverse is correct.'
                sys.exit()
            else:
                print 'Pseudoinverse is incorrect.'
                sys.exit()
            
        self.ctrl = ctrlmat
        print 'SVD %d modes %0.4e'%(self.n_modes,self.cutoff_cond)
        self.store_ctrl(mask,self.ctrl)
        self.trim_ctrl_dict()

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
        data = self.loop.sensor.spots
        self.bmp = self.bmpscale(data,self.downsample,self.cmin,self.cmax)
        single = data[self.loop.sensor.search_boxes.y1[k]:self.loop.sensor.search_boxes.y2[k]+1,
                      self.loop.sensor.search_boxes.x1[k]:self.loop.sensor.search_boxes.x2[k]+1]
        self.single_bmp = self.bmpscale(single,1,self.cmin,self.cmax)

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
        self.mirror_bmp = self.bmpscale(data,1,kcfg.mirror_current_min,kcfg.mirror_current_max)
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
    camera = cameras.SimulatedCamera()

    sensor = Sensor(camera,update_rate=kcfg.sensor_update_rate)
    mirror = Mirror(update_rate=kcfg.mirror_update_rate)
    loop = Loop(sensor,mirror)
    gui = Gui(loop)
    
    gui.show()
    sys.exit(app.exec_())
