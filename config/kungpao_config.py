# A set of name-value pairs specifying local configuration of
# kungpao installation. Where appropriate, each parameter's final
# characters represent units. 
#
# When the parameter is a filename e.g. the name of the file
# containing reference coordinates, only the filename should be given,
# not the full path. The path is determined by rules established in
# pyao.settings, which are not really intended to be user
# modifiable. See the Installation page in documentation for details.
import numpy as np
import os,sys

LENSLET_PITCH_M = 500e-6
LENSLET_FOCAL_LENGTH_M = 30e-3
LENSLETS_ACROSS_ARRAY = 20
CONFIG_PATH = '/home/rjonnal/code/kungpao/config'
REFERENCE_COORDINATE_FILENAME = os.path.join(CONFIG_PATH,'kungpao_reference_coordinates.txt')

REFERENCE_COORDINATES = np.loadtxt(REFERENCE_COORDINATE_FILENAME)
SEARCH_BOX_WIDTH_PX = 39
SENSOR_WIDTH_PX = 1024
SENSOR_HEIGHT_PX = 1024
BEAM_DIAMETER_M = 20e-3
PIXEL_SIZE_M = 11e-6



# #################################################################################
# # A unique, permanent identifier for the optical system
# # associated with this installation of kungpao:
# systemID = 'simulator'

# #################################################################################
# # mirror configuration (look in directory specified by environment variable 
# # %ACECFG% for same-named configuration files)
# mirrorID = 'alpaoDM97-15-012'

# #################################################################################
# # camera configuration:
# cameraFilename = 'acA2040-180km-4tap-12bit_reloaded.dcf'
# cameraBinningPx = 2
# cameraPixelSizeM = 5.5e-6
# cameraIntegrationTimeS = 50.0e-3
# # On some versions of the camera, binning must be called in both dimensions, and
# # on others, just in the X dimension. Set the following accordingly.
# cameraBinY = False 
# cameraMaxIntensity = 4095
# cameraMinIntensity = 0
# nHistogramBins = 100

# #################################################################################
# # wavefront sensor configuration:
# lensletFocalLengthM = 30.0e-3
# lensletPitchM = 500.0e-6
# lensletsAcrossArray = 20
# beaconWavelengthM = 800.0e-9
# referenceCoordinateFilename = 'referenceCoordinates.txt'
# pupilDiameterM = .01

# audioFeedback = True
# audioWavefrontThreshold = 80e-9

# referenceShiftStepPx = 0.001

# # spotsImageThreshold is subtracted from the spots image and the spots
# # image is then clipped at 0
# DCMethod = 'filter'
# #DCMethod = 'auto'
# DCFilterSigma = 10.0
# useAdaptiveDC = True

# # peakThreshold is the value, above spotsImageThreshold, that the peak
# # within any search box must exceed, in order for that subaperture to
# # be considered valid, for purposes of wavefront reconstruction and
# # compensation
# peakThreshold = 20.0
# searchBoxSizePx = 39

# #################################################################################
# # display configuration:
# spotsDisplayScaleFactor = 1.0
# guiScale = 0.9
# spotsContrastLimitStep = 100

# #################################################################################
# # AO loop configuration:
# defaultGain = 0.3
# defaultLoss = 0.01
# loopGainStep = 0.01
# loopLossStep = 0.002
# loopSafeMode = True

# #pokeMatrixFilename = '20140515151448_poke.txt'
# # pokeMatrixFilename = '20140616100944_poke.txt' # low stroke poke matrix (-.1,.1)
# #pokeMatrixFilename = '20140616101701_poke.txt' # high stroke poke matrix (-.25,.25)
# # pokeMatrixFilename = '20150305171400_poke.txt'
# pokeMatrixFilename = '20150515171245_poke.txt'
# ctrlMatrixSVDModes = 30



# #################################################################################
# #################################################################################
# #################################################################################
# # END OF USER CONFIGURATION
# # Please do not edit anything below this block. What happens below
# # is, essentially, prepending paths from kungpao.settings (e.g. the paths
# # to camera files, reference files, etc) to the filenames, and printing
# # them out to the console, or errors.
# from kungpao import settings
# try:
#     cameraFilename = settings.dcfPath + cameraFilename
#     print 'Camera file: ',cameraFilename
# except Exception as e:
#     print 'Error:',e

# try:
#     referenceCoordinateFilename = settings.refPath + referenceCoordinateFilename
#     print 'Reference file: ',referenceCoordinateFilename
# except Exception as e:
#     print 'Error:',e

# try:
#     pokeMatrixFilename = settings.ctrlPath + pokeMatrixFilename
#     print 'Poke file: ',pokeMatrixFilename
# except Exception as e:
#     print 'Error:',e
