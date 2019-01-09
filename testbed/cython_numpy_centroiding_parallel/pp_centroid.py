import numpy as np
from matplotlib import pyplot as plt

def compute_centroids_loop(spots_image,sb_x1_vec,sb_x2_vec,sb_y1_vec,sb_y2_vec):
    n_spots = len(sb_x1_vec)
    x_out = np.zeros(n_spots)
    y_out = np.zeros(n_spots)
    for k in range(n_spots):
        xprod = 0.0
        yprod = 0.0
        intensity = 0.0
        for x in range(sb_x1_vec[k],sb_x2_vec[k]+1):
            for y in range(sb_y1_vec[k],sb_y2_vec[k]+1):
                pixel = float(spots_image[y,x])
                xprod = xprod + pixel*x
                yprod = yprod + pixel*y
                intensity = intensity + pixel
        x_out[k] = xprod/intensity
        y_out[k] = yprod/intensity
    return x_out,y_out
                
def compute_centroids_vector(spots_image,sb_x1_vec,sb_x2_vec,sb_y1_vec,sb_y2_vec):
    n_spots = len(sb_x1_vec)
    x_out = np.zeros(n_spots)
    y_out = np.zeros(n_spots)
    for k in range(n_spots):
        xprod = 0.0
        yprod = 0.0
        intensity = 0.0
        XX,YY = np.meshgrid(np.arange(sb_x1_vec[k],sb_x2_vec[k]+1),np.arange(sb_y1_vec[k],sb_y2_vec[k]+1))
        subim = spots_image[sb_y1_vec[k]:sb_y2_vec[k]+1,sb_x1_vec[k]:sb_x2_vec[k]+1].astype(np.float)
        x_out[k] = np.sum(XX*subim)/np.sum(subim)
        y_out[k] = np.sum(YY*subim)/np.sum(subim)
    return x_out,y_out
                
