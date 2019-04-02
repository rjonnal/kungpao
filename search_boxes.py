from PyQt5.QtCore import QObject
import numpy as np
import sys
import config as kcfg

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

        
