# A set of name-value pairs specifying local configuration of
# kungpao installation. Where appropriate, each parameter's final
# characters represent units. 
#
# When the parameter is a filename e.g. the name of the file
# containing reference coordinates, only the filename should be given,
# not the full path. The path is determined by rules established in
# pyao.settings, which are not really intended to be user
# modifiable. See the Installation page in documentation for details.


#################################################################################
# A unique, permanent identifier for the optical system
# associated with this installation of kungpao:
#systemID = 'AO-OCT-2G'
system_id = 'simulator'

#################################################################################
# mirror configuration (look in directory specified by environment variable 
# %ACECFG% for same-named configuration files)
mirror_id = 'alpaoDM97-15-012'

#################################################################################
# camera configuration:
camera_filename = 'acA2040-180km-4tap-12bit_reloaded.dcf'
camera_physical_pixel_size = 5.5e-6
camera_physical_width = 2048
camera_physical_height = 2048
camera_binning = 1

#################################################################################
# wavefront sensor configuration:
lenslet_focal_length_m = 30.0e-3
lenslet_pitch_m = 500.0e-6
beacon_wavelength_m = 850e-9

reference_coordinate_filename = 'reference_coordinates.txt'

pupilDiameterM = .01

audioFeedback = True
audioWavefrontThreshold = 80e-9

referenceShiftStepPx = 0.001

# spotsImageThreshold is subtracted from the spots image and the spots
# image is then clipped at 0
DCMethod = 'filter'
#DCMethod = 'auto'
DCFilterSigma = 10.0

# peakThreshold is the value, above spotsImageThreshold, that the peak
# within any search box must exceed, in order for that subaperture to
# be considered valid, for purposes of wavefront reconstruction and
# compensation
peakThreshold = 20.0
searchBoxSizePx = 39

#################################################################################
# display configuration:
spotsDisplayScaleFactor = 1.0
guiScale = 0.9
spotsContrastLimitStep = 100

#################################################################################
# AO loop configuration:
defaultGain = 0.3
defaultLoss = 0.01
loopGainStep = 0.01
loopLossStep = 0.002
loopSafeMode = True

#pokeMatrixFilename = '20140515151448_poke.txt'
# pokeMatrixFilename = '20140616100944_poke.txt' # low stroke poke matrix (-.1,.1)
#pokeMatrixFilename = '20140616101701_poke.txt' # high stroke poke matrix (-.25,.25)
# pokeMatrixFilename = '20150305171400_poke.txt'
pokeMatrixFilename = '20150515171245_poke.txt'
ctrlMatrixSVDModes = 30



#################################################################################
#################################################################################
#################################################################################
# END OF USER CONFIGURATION
# Please do not edit anything below this block. What happens below
# is, essentially, prepending paths from pyao.settings (e.g. the paths
# to camera files, reference files, etc) to the filenames, and printing
# them out to the console, or errors.
from pyao import settings
try:
    cameraFilename = settings.dcfPath + cameraFilename
    print 'Camera file: ',cameraFilename
except Exception as e:
    print 'Error:',e

try:
    referenceCoordinateFilename = settings.refPath + referenceCoordinateFilename
    print 'Reference file: ',referenceCoordinateFilename
except Exception as e:
    print 'Error:',e

try:
    pokeMatrixFilename = settings.ctrlPath + pokeMatrixFilename
    print 'Poke file: ',pokeMatrixFilename
except Exception as e:
    print 'Error:',e
