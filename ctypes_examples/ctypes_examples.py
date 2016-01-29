from ctypes import *
import numpy as np

def info(numpy_array):
    print numpy_array

ctex = cdll.LoadLibrary('./ctypes_examples.so')
ctex.fill_c_buffer(123)

im = np.ascontiguousarray(np.zeros((5,1)).astype(np.uint16),dtype=np.uint16)
c_uint16_p = POINTER(c_ushort)
im_ptr = im.ctypes.data_as(c_uint16_p)
#im_ptr = im.ctypes.data

print im
ctex.get_buffer(im_ptr)
print im
ctex.copy_buffer(im_ptr)
print im





