import numpy as np
from matplotlib import pyplot as plt

ref = np.loadtxt('referenceCoordinates.txt')

x = ref[:ref.shape[0]//2,:]
y = ref[ref.shape[0]//2:,:]
x = x[np.where(x>-1)]
y = y[np.where(y>-1)]

ref = np.vstack((x,y)).T

np.savetxt('kungpao_reference_coordinates.txt',ref,fmt='%0.3f')


