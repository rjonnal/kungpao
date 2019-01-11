BEAM_DIAMETER_M = 10e-3
LENSLET_PITCH_M = 500e-6
LENSLET_F_M = 30e-3

MASK_PADDING = 0.4
SENSOR_XY = 1024 # Assume square sensor
SENSOR_X = SENSOR_XY
SENSOR_Y = SENSOR_XY

# Granularity of wavefront simulation
N_WAVEFRONT = 1024
GAUSSIAN_BEAM_SIGMA_M = 3e-3

# Number of Zernike orders to simulate
N_ORDERS = 4
N_ZERNIKE = sum(range(N_ORDERS+1))

WAVELENGTH_M = 830e-9
PSF_NA = 6.75/16.67
PIXEL_SIZE_M = BEAM_DIAMETER_M/SENSOR_X

POWER_W = 1e-6
Z0_OHMS = 376.73 # impedence of free space


