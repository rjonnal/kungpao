import numpy as np
import glob

class SimulatedCamera:

    def __init__(self):
        self.image_list = sorted(glob.glob('./data/spots/*.npy'))
        self.n_images = len(self.image_list)
        self.index = 0
        self.images = [np.load(fn) for fn in self.image_list]

    def get_image(self):
        im = np.load(self.image_list[self.index])
        #im = self.images[self.index]
        self.index = (self.index + 1)%self.n_images
        return im
        
    
