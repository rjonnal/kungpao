import numpy as np
import time
import config as kcfg
import sys
from PyQt5.QtCore import (QThread, QTimer, pyqtSignal, Qt, QPoint, QLine,
                          QMutex, QObject, pyqtSlot)

from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,
                             QHBoxLayout, QVBoxLayout, QGraphicsScene,
                             QLabel,QGridLayout, QCheckBox, QFrame, QGroupBox,
                             QSpinBox,QDoubleSpinBox,QSizePolicy,QFileDialog,
                             QErrorMessage, QSlider)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb, QPen, QBitmap, QPalette, QIcon
import os
from matplotlib import pyplot as plt
import datetime
from tools import error_message, now_string, prepend, colortable, get_ram, get_process
from zernike import Zernike
from search_boxes import SearchBoxes
from frame_timer import FrameTimer
from reference_generator import ReferenceGenerator

class Simulator(QObject):

    def __init__(self):

        super(Simulator,self).__init__()

        self.frame_timer = FrameTimer('simulator')
        self.sy = kcfg.image_height_px
        self.sx = kcfg.image_width_px
        self.wavefront = np.zeros((self.sy,self.sx))
        self.dc = 100
        self.spots_range = 2000
        self.spots = np.ones((self.sy,self.sx))*self.dc
        self.spots = self.noise(self.spots)
        self.pixel_size_m = kcfg.pixel_size_m

        # compute single spot
        pitch = kcfg.lenslet_pitch_m
        self.f = kcfg.lenslet_focal_length_m
        L = kcfg.wavelength_m
        fwhm_px = (1.22*L*self.f/pitch)/self.pixel_size_m
        
        self.disc = np.zeros((self.sy,self.sx))
        self.disc_diameter = 110
        
        xvec = np.arange(self.sx)
        yvec = np.arange(self.sy)
        xvec = xvec-xvec.mean()
        yvec = yvec-yvec.mean()
        XX,YY = np.meshgrid(xvec,yvec)
        d = np.sqrt(XX**2+YY**2)
        
        self.disc[np.where(d<=self.disc_diameter)] = 1.0
        
        if False:
            # verify that the spot size is about right
            spot = np.abs(np.fft.fftshift(np.fft.fft2(self.disc)))
            py,px = np.unravel_index(np.argmax(spot),spot.shape)
            prof = spot[py,:]
            print len(np.where(prof>prof.max()/2.0)[0])
            plt.plot(prof)
            plt.show()
            sys.exit()
            
        
        d = kcfg.beam_diameter_m
        self.beam_radius = d/2.0
        self.X = np.arange(self.sx,dtype=np.float)*self.pixel_size_m
        self.Y = np.arange(self.sy,dtype=np.float)*self.pixel_size_m
        self.X = self.X-self.X.mean()
        self.Y = self.Y-self.Y.mean()
        
        self.XX,self.YY = np.meshgrid(self.X,self.Y)

        self.RR = np.sqrt(self.XX**2+self.YY**2)
        self.mask = np.zeros(self.RR.shape)
        self.mask[np.where(self.RR<=self.beam_radius)] = 1.0
        
        self.mirror_mask = np.loadtxt(kcfg.mirror_mask_filename)
        self.n_actuators = int(np.sum(self.mirror_mask))

        self.command = np.zeros(self.n_actuators)
        
        # virtual actuator spacing in magnified or demagnified
        # plane of camera
        actuator_spacing = 9e-4
        ay,ax = np.where(self.mirror_mask)
        ay = ay*actuator_spacing
        ax = ax*actuator_spacing
        ay = ay-ay.mean()
        ax = ax-ax.mean()

        self.n_zernike_terms = kcfg.n_zernike_terms
        actuator_sigma = actuator_spacing
        key = '%d'%hash((tuple(ax),tuple(ay),actuator_sigma,tuple(self.X),tuple(self.Y),self.n_zernike_terms))
        key = key.replace('-','m')

        cfn = os.path.join(kcfg.simulator_cache_directory,'%s_actuator_basis.npy'%key)
        try:
            self.actuator_basis = np.load(cfn)
            print 'Loading cached actuator basis set...'
        except Exception as e:
            actuator_basis = []
            print 'Building actuator basis set...'
            for x,y in zip(ax,ay):
                xx = self.XX - x
                yy = self.YY - y
                surf = np.exp((-(xx**2+yy**2)/(2*actuator_sigma**2)))
                actuator_basis.append(surf.ravel())

            self.actuator_basis = np.array(actuator_basis)
            np.save(cfn,self.actuator_basis)


        zfn = os.path.join(kcfg.simulator_cache_directory,'%s_zernike_basis.npy'%key)
        try:
            self.zernike_basis = np.load(zfn)
            print 'Loading cached zernike basis set...'
        except Exception as e:
            zernike_basis = []
            print 'Building zernike basis set...'
            zernike = Zernike()
            for z in range(self.n_zernike_terms):
                surf = zernike.get_j_surface(z,self.XX,self.YY)
                zernike_basis.append(surf.ravel())

            self.zernike_basis = np.array(zernike_basis)
            np.save(zfn,self.zernike_basis)

        self.new_error_sigma = np.ones(self.n_zernike_terms)*10
        self.new_error_sigma[:3] = 0.0
        
        # load reference information
        try:
            xy = np.loadtxt(kcfg.reference_coordinates_filename)
        except Exception as e:
            print e
            sys.exit('Simulation mode needs a reference coordinate filename.')

        self.search_boxes = SearchBoxes(xy[:,0],xy[:,1],kcfg.search_box_half_width)
        self.update()
        self.paused = False

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False

    def set_logging(self,val):
        self.logging = val

    def flatten(self):
        self.command[:] = 0.0

    def get_command(self):
        return self.command

    def set_command(self,vec):
        self.command[:] = vec[:]
        self.update()
        
    def set_actuator(self,index,value):
        self.command[index]=value
        self.update()
        
    def noise(self,im):
        noiserms = np.random.randn(im.shape[0],im.shape[1])*np.sqrt(im)
        return im+noiserms

    def get_new_error(self):
        #self.new_error_sigma = np.ones(self.n_zernike_terms)
        coefs = np.random.randn(self.n_zernike_terms)*self.new_error_sigma
        return np.reshape(np.dot(coefs,self.zernike_basis),(self.sy,self.sx))

    def defocus_animation(self):
        err = np.zeros(self.n_zernike_terms)
        for k in np.arange(0.0,100.0):
            err[4] = np.random.randn()
            im = np.reshape(np.dot(err,self.zernike_basis),(self.sy,self.sx))
            plt.clf()
            plt.imshow(im-im.min())
            plt.colorbar()
            plt.pause(.1)
    
    def plot_actuators(self):
        edge = self.XX.min()
        wid = self.XX.max()-edge
        plt.imshow(self.mask,extent=[edge,edge+wid,edge,edge+wid])
        plt.autoscale(False)
        plt.plot(ax,ay,'ks')
        plt.show()

    def update(self):
        
        mirror = np.reshape(np.dot(self.command,self.actuator_basis),(self.sy,self.sx))
        err = self.get_new_error()*0.0
        dx = np.diff(err,axis=1)
        dy = np.diff(err,axis=0)
        sy,sx = err.shape
        col = np.zeros((sy,1))
        row = np.zeros((1,sx))
        dx = np.hstack((col,dx))
        dy = np.vstack((row,dy))
        #err = err - dx - dy
        
        self.wavefront = mirror+err
        y_slope_vec = []
        x_slope_vec = []
        self.spots[:] = 0.0
        for idx,(x,y,x1,x2,y1,y2) in enumerate(zip(self.search_boxes.x,
                                                   self.search_boxes.y,
                                                   self.search_boxes.x1,
                                                   self.search_boxes.x2,
                                                   self.search_boxes.y1,
                                                   self.search_boxes.y2)):
            subwf = self.wavefront[y1:y2+1,x1:x2+1]
            yslope = np.mean(np.diff(subwf.mean(1)))
            dy = yslope*self.f/self.pixel_size_m
            ycentroid = y+dy
            ypx = int(round(y+dy))
            xslope = np.mean(np.diff(subwf.mean(0)))
            dx = xslope*self.f/self.pixel_size_m
            xcentroid = x+dx
            self.spots = self.interpolate_dirac(xcentroid,ycentroid,self.spots)
            x_slope_vec.append(xslope)
            y_slope_vec.append(yslope)
            QApplication.processEvents()
        self.spots = np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(self.spots))*self.disc))
        self.x_slopes = np.array(x_slope_vec)
        self.y_slopes = np.array(y_slope_vec)
        self.frame_timer.tick()
        
    def get_image(self):
        spots = (self.spots-self.spots.min())/(self.spots.max()-self.spots.min())*self.spots_range+self.dc
        nspots = self.noise(spots)
        nspots = np.clip(nspots,0,4095)
        nspots = np.round(nspots).astype(np.int16)
        return nspots
        
    def interpolate_dirac(self,x,y,frame):
        # take subpixel precision locations x and y and insert an interpolated
        # delta w/ amplitude 1 there
        x1 = int(np.floor(x))
        x2 = x1+1
        y1 = int(np.floor(y))
        y2 = y1+1
        
        for yi in [y1,y2]:
            for xi in [x1,x2]:
                yweight = 1.0-(abs(yi-y))
                xweight = 1.0-(abs(xi-x))
                frame[yi,xi] = yweight*xweight
        return frame
            
    def wavefront_to_spots(self):
        
        pass

    def show_zernikes(self):
        for k in range(self.n_zernike_terms):
            b = np.reshape(self.zernike_basis[k,:],(self.sy,self.sx))
            plt.clf()
            plt.imshow(b)
            plt.colorbar()
            plt.pause(.5)
        
if __name__=='__main__':

    sim = Simulator()
    for k in np.linspace(0.0,200000.0,10.0):
        err = np.zeros(kcfg.n_zernike_terms)
        err[3] = k
        sim.update(err)
        im = sim.get_image()
        plt.cla()
        plt.imshow(im)
        plt.pause(.1)
