import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import sys
from numpy.fft import fft2, fftshift
import scipy
from scipy.integrate import simps
import seaborn as sns # Used only to set up plots

sns.set_context(context = 'talk')
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#272b30'
plt.rcParams['image.cmap'] = 'viridis' 

print('Python version:\n{}\n'.format(sys.version))
print('Numpy version:\t\t{}'.format(np.__version__))
print('matplotlib version:\t{}'.format(matplotlib.__version__))
print('Scipy version:\t\t{}'.format(scipy.__version__))
print('Seaborn version:\t{}'.format(sns.__version__))

# Setup the simulation parameters
wavelength = 0.68   # microns
NA         = 0.7    # Numerical aperture of the objective
pixelSize  = 0.1    # microns
numPixels  = 2048   # Number of pixels in the camera; keep this even
power      = 0.1    # Watts
Z0         = 376.73 # Ohms; impedance of free space


# Create the image plane coordinates
x = np.linspace(-pixelSize * numPixels / 2, pixelSize * numPixels / 2, num = numPixels, endpoint = True)

# Create the Fourier plane
dx = x[1] - x[0]    # Sampling period, microns
fS = 1 / dx         # Spatial sampling frequency, inverse microns
df = fS / numPixels # Spacing between discrete frequency coordinates, inverse microns
fx = np.arange(-fS / 2, fS / 2, step = df) # Spatial frequency, inverse microns


# Create the pupil, which is defined by the numerical aperture
fNA             = NA / wavelength # radius of the pupil, inverse microns
pupilRadius     = fNA / df        # pixels
pupilCenter     = numPixels / 2   # assumes numPixels is even
W, H            = np.meshgrid(np.arange(0, numPixels), np.arange(0, numPixels)) # coordinates of the array indexes
pupilMask       = np.sqrt((W - pupilCenter)**2 + (H - pupilCenter)**2) <= pupilRadius



# Compute normalizing constant
norm_factor = simps(simps(np.abs(pupilMask)**2, dx = df), dx = df) / Z0 # A^2
print('Normalization constant:\t\t{:.4f} W'.format(np.sqrt(norm_factor)))

# Renormalize the pupil values
pupil = pupilMask * (1 + 0j)
normedPupil = pupil * np.sqrt(power / norm_factor)

new_power = simps(simps(np.abs(normedPupil)**2, dx = df), dx = df) / Z0
print('User-defined power:\t\t{:.4f} W'.format(power))
print('Power now carried by field:\t{:.4f} W'.format(new_power))


# Show the pupil
ax = plt.imshow(np.abs(pupil), extent = [fx[0], fx[-1], fx[0], fx[-1]])
plt.grid(False)
cb = plt.colorbar(ax)
cb.set_label('$P \, ( f_x, f_y ) $')
plt.xlabel('$f_x$, $\mu m^{-1}$')
plt.ylabel('$f_y$, $\mu m^{-1}$')
plt.show()

# Compute the power
power_pupil = simps(simps(np.abs(normedPupil)**2, dx = df), dx = df) / Z0
print('Power in pupil plane: {:.4f} W'.format(power_pupil))



