from ctypes import *
import numpy as np
from time import time

def info(numpy_array):
    print numpy_array

ctex = cdll.LoadLibrary('./ctypes_examples.so')
ctex.fill_c_buffer(123)

im = np.ascontiguousarray(np.zeros((5,1)).astype(np.uint16),dtype=np.uint16)
c_uint16_p = POINTER(c_ushort)
im_ptr = im.ctypes.data_as(c_uint16_p)
#im_ptr = im.ctypes.data


def timeit(func,arg,N=100):
    t0 = time()
    for k in range(N):
        func(arg)
    t = time()
    print 'elapsed time: %f ms'%(t-t0)

#ctex.get_buffer(im_ptr)
#ctex.copy_buffer(im_ptr)

N = 1000000
timeit(ctex.get_buffer,im_ptr,N)
timeit(ctex.copy_buffer,im_ptr,N)




