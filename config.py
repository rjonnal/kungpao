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
image_width_px = 1024
image_height_px = 1024
bit_depth = 12
contrast_maximum = 2000.0
contrast_minimum = 200.0
reference_coordinates_filename = '/home/rjonnal/code/kungpao/etc/ref/coords.txt'
lenslet_pitch_m = 500e-6
lenslet_focal_length_m = 30.0e-3
pixel_size_m = 11e-6
beam_diameter_m = 10e-3
interface_scale_factor = 0.5
wavelength_m = 840e-9
estimate_background = True
background_correction = +3.5
search_box_half_width = 18
search_box_color = (127,127,255,255)
search_box_thickness = 10
show_search_boxes = True
show_slope_lines = True
slope_line_thickness = 1.0
slope_line_color = (0,255,0,255)
single_spot_color = (255,63,63,255)
single_spot_thickness = 1.0
mirror_update_rate = 100.0
mirror_n_actuators = 97

#search_box_half_width_max = 30
search_box_half_width_max = int(lenslet_pitch_m/pixel_size_m)//2

rigorous_iteration = False
if rigorous_iteration:
    # First, calculate the PSF FWHM for the lenslets:
    import math
    lenslet_psf_fwhm_m = 1.22*wavelength_m*lenslet_focal_length_m/lenslet_pitch_m
    # Now see how many pixels this is:
    lenslet_psf_fwhm_px = lenslet_psf_fwhm_m/pixel_size_m 

    diffraction_limited_width_px = round(math.ceil(lenslet_psf_fwhm_px))
    if diffraction_limited_width_px%2==0:
        diffraction_limited_width_px+=1
    diffraction_limited_half_width_px = (diffraction_limited_width_px-1)//2

    iterative_centroiding_step = 1
    centroiding_iterations = int(round((search_box_half_width-diffraction_limited_half_width_px)//iterative_centroiding_step))
else:
    iterative_centroiding_step = 2
    centroiding_iterations = 4




# #################################################################################
# # Paths
# # Specify a single path with 'XXX_path', or a list of paths with 'XXX_paths'. The
# # latter overrides the former, if one of the paths in XXX_paths exists. (See end
# # of this file for logic).
# #################################################################################
# # Path to shared libraries
# lib_paths = ['C:/code/kungpao/lib','/home/rjonnal/code/kungpao/lib']
# #################################################################################
# # Path to camera configuration files
# dcf_paths = ['C:/code/kungpao_etc/config/dcf/','/home/rjonnal/data/Dropbox/kungpao_etc/config/dcf']
# #################################################################################
# #################################################################################
# #################################################################################
# #################################################################################



# #################################################################################
# # mirror configuration (look in directory specified by environment variable 
# # %ACECFG% for same-named configuration files)
# mirror_id = 'alpaoDM97-15-012'

# #################################################################################
# # camera configuration:
# camera_filename = 'acA2040-180km-4tap-12bit_reloaded.dcf'
# camera_physical_pixel_size = 5.5e-6
# camera_physical_width = 2048
# camera_physical_height = 2048
# camera_binning = 1

# #################################################################################
# # wavefront sensor configuration:
# lenslet_focal_length_m = 30.0e-3
# lenslet_pitch_m = 500.0e-6
# beacon_wavelength_m = 850e-9

# reference_coordinate_filename = 'reference_coordinates.txt'

# pupilDiameterM = .01

# audioFeedback = True
# audioWavefrontThreshold = 80e-9

# referenceShiftStepPx = 0.001

# # spotsImageThreshold is subtracted from the spots image and the spots
# # image is then clipped at 0
# DCMethod = 'filter'
# #DCMethod = 'auto'
# DCFilterSigma = 10.0

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
# # If lib_paths exists and is non-empty, set lib_path to the first existing path
# # in the list.
# import os
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def find_path(path_list):
#     ''' Return the first valid (existing) path in path_list.'''
#     for test in path_list:
#         print test
#         if os.path.exists(test):
#             print 'found'
#             return test
    
# lib_path = find_path(lib_paths)
# print lib_path
# dcf_path = find_path(dcf_paths)
# print dcf_path
