import numpy as np
cimport numpy as np
from matplotlib import pyplot as plt
import cython
from cython.parallel import prange
ctypedef np.uint16_t uint16_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_centroids(np.ndarray[np.int16_t,ndim=2] spots_image,
                        np.ndarray[np.int16_t,ndim=1] sb_x1_vec,
                        np.ndarray[np.int16_t,ndim=1] sb_x2_vec,
                        np.ndarray[np.int16_t,ndim=1] sb_y1_vec,
                        np.ndarray[np.int16_t,ndim=1] sb_y2_vec,
                        np.ndarray[np.float_t,ndim=1] x_out,
                        np.ndarray[np.float_t,ndim=1] y_out,
                        np.ndarray[np.float_t,ndim=1] total_intensity,
                        np.ndarray[np.float_t,ndim=1] maximum_intensity,
                        np.ndarray[np.float_t,ndim=1] minimum_intensity,
                        np.ndarray[np.float_t,ndim=1] background_intensity,
                        estimate_background = True,
                        background_correction = 0.0,
                        num_threads = 4):

    cdef np.int_t n_spots = len(sb_x1_vec)
    cdef np.int_t k
    cdef np.float_t intensity
    cdef np.float_t background
    cdef np.float_t xprod
    cdef np.float_t yprod
    cdef np.int_t x
    cdef np.int_t y
    cdef np.float_t imax
    cdef np.float_t imin
    cdef np.float_t pixel
    cdef np.float_t edge_counter
    cdef np.int_t estimate_background_t
    cdef np.float_t background_correction_t
    if estimate_background:
        estimate_background_t = 1
    else:
        estimate_background_t = 0
    background_correction_t = float(background_correction)
    #for k in prange(n_spots,nogil=True,num_threads=num_threads):
    for k in range(n_spots):
        if estimate_background_t:
            edge_counter = 0.0
            background = 0.0
            for x in range(sb_x1_vec[k],sb_x2_vec[k]+1):
                pixel = float(spots_image[sb_y1_vec[k],x])
                background = background + pixel
                edge_counter = edge_counter + 1
                pixel = float(spots_image[sb_y2_vec[k],x])
                background = background + pixel
                edge_counter = edge_counter + 1
            for y in range(sb_y1_vec[k]+1,sb_y2_vec[k]):
                pixel = float(spots_image[y,sb_x1_vec[k]])
                background = background + pixel
                edge_counter = edge_counter + 1
                pixel = float(spots_image[y,sb_x1_vec[k]])
                background = background + pixel
                edge_counter = edge_counter + 1
            background = background/edge_counter
        else:
            background = 0.0
        intensity = 0.0
        xprod = 0.0
        yprod = 0.0
        imin = 2**15
        imax = -2**15
        for x in range(sb_x1_vec[k],sb_x2_vec[k]+1):
            for y in range(sb_y1_vec[k],sb_y2_vec[k]+1):
                pixel = float(spots_image[y,x])-(background+background_correction_t)
                if pixel<0.0:
                    pixel = 0.0
                spots_image[y,x] = <np.int_t>pixel
                xprod = xprod + pixel*x
                yprod = yprod + pixel*y
                intensity = intensity + pixel
                if pixel<imin:
                    imin=pixel
                if pixel>imax:
                    imax=pixel
        if intensity==0.0:
            intensity = 1.0
        total_intensity[k] = intensity
        background_intensity[k] = background
        maximum_intensity[k] = imax
        minimum_intensity[k] = imin
        x_out[k] = xprod/intensity
        y_out[k] = yprod/intensity
    return x_out,y_out
