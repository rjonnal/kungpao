import numpy as np
from matplotlib import pyplot as plt
import glob,os,sys

files = glob.glob('*.npy')
files.sort()

test = np.load(files[0])

def centroid(test):
    sy,sx = test.shape
    XX,YY = np.meshgrid(np.arange(sx),np.arange(sy))
    im = np.zeros(test.shape)
    im[...] = test[...]
    im = im - im.mean()
    im = np.clip(im,0,im.max())
    xc = np.sum(im*XX)/np.sum(im)
    yc = np.sum(im*YY)/np.sum(im)
    return xc,yc

def hroll(im,n=1):
    right = im[:,n:]
    left = im[:,:n]
    im = np.hstack((right,left))
    return im

def vroll(im,n=1):
    top = im[n:,:]
    bottom = im[:n,:]
    im = np.vstack((top,bottom))
    return im


xc,yc = centroid(test)
nh = int(round((xc-512)//1))
nv = int(round((yc-512)//1))

for fn in files:
    im = np.load(fn)
    newim = hroll(im,nh)
    newim = vroll(newim,nv)
    outfn = os.path.join('../spots_02',fn)
    np.save(outfn,newim)

    plt.imshow(newim,cmap='gray')
    plt.show()
