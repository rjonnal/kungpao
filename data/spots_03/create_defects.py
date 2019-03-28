import numpy as np
from matplotlib import pyplot as plt
import glob,os,sys

image_list = sorted(glob.glob('*.npy'))
n_images = len(image_list)
index = 0

sy,sx = np.load(image_list[0]).shape
oy = int(round(np.random.rand()*sy//2+sy//4))
ox = int(round(np.random.rand()*sx//2+sx//4))
XX,YY = np.meshgrid(np.arange(sx),np.arange(sy))


def get_image(self):
    im = np.load(image_list[index])

def opacify(im,sigma=30):
    xx,yy = XX-ox,YY-oy
    d = np.sqrt(xx**2+yy**2)
    mask = np.ones(d.shape)
    mask[np.where(d<=sigma)] = 0.2
    out = np.round(im*mask).astype(np.int16)
    return out
    
for index,in_fn in enumerate(image_list):
    print index,in_fn

    im = np.load(in_fn)
    
    plt.subplot(1,2,1)
    plt.cla()
    plt.imshow(im,cmap='gray')
    
    im = opacify(im)
    oy = oy+np.random.randn()*.5
    ox = ox+np.random.randn()*.5
    
    plt.subplot(1,2,2)
    plt.cla()
    plt.imshow(im,cmap='gray')
    plt.pause(.1)
    np.save(in_fn,im)
