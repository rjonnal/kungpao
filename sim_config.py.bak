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
#system_id = 'simulator'
system_id = 'aossffoct'
mirror_id = 'alpaoDM97-15-010'

image_width_px = 512
image_height_px = 512
bit_depth = 12

contrast_maximum = 2000.0
contrast_minimum = 200.0

kungpao_root = 'c:/code/kungpao'
#kungpao_root = '/home/rjonnal/code/kungpao'

reference_directory = kungpao_root + '/etc/ref/'

reference_n_measurements = 10

logging_directory = kungpao_root + '/log'

poke_directory = kungpao_root + '/etc/ctrl/'
poke_filename = poke_directory + '20190514104827_poke.txt'

simulator_cache_directory = '.simulator_cache'

n_zernike_terms = 66
zernike_dioptric_equivalent = 1.5

lenslet_pitch_m = 500e-6
lenslet_focal_length_m = 20.0e-3
pixel_size_m = 22e-6
beam_diameter_m = 6.5e-3

interface_scale_factor = 0.5
wavelength_m = 840e-9
estimate_background = True
background_correction = -100
search_box_half_width = 12

active_search_box_color = (127,16,16,255)
inactive_search_box_color = (0,63,127,255)

search_box_thickness = 1.0
show_search_boxes = True
show_slope_lines = True

slope_line_thickness = 5.0
slope_line_color = (200,100,100,155)
slope_line_magnification = 5e4

spots_colormap = 'gray'
wavefront_colormap = 'jet'
mirror_colormap = 'mirror'

zoom_width = 50
zoom_height = 50

single_spot_color = (255,63,63,255)
single_spot_thickness = 2.0

sensor_update_rate = 100.0
sensor_filter_lenslets = False
sensor_reconstruct_wavefront = True
sensor_remove_tip_tilt = True

mirror_update_rate = 20.0

#mirror_n_actuators = 97
#mirror_flat_filename = kungpao_root + '/etc/dm/flat.txt'
#mirror_mask_filename = kungpao_root + '/etc/dm/mirror_mask.txt'

mirror_n_actuators = 37
mirror_flat_filename = kungpao_root + '/etc/dm/small_flat.txt'
mirror_mask_filename = kungpao_root + '/etc/dm/small_mirror_mask.txt'

mirror_command_max = 0.3
mirror_command_min = -0.3
mirror_settling_time_s = 0.001

poke_command_max = 0.05
poke_command_min = -0.05
poke_n_command_steps = 5

ctrl_dictionary_max_size = 10

loop_n_control_modes = 10
loop_gain = 0.3
loop_loss = 0.01

#spots_threshold = 200.0
spots_threshold = 100.0

ui_fps_fmt = '%0.2f Hz (UI)'
sensor_fps_fmt = '%0.2f Hz (Sensor)'
mirror_fps_fmt = '%0.2f Hz (Mirror)'
wavefront_error_fmt = '%0.1f nm RMS (Error)'
tip_fmt = '%0.4f mrad (Tip)'
tilt_fmt = '%0.4f mrad (Tilt)'
cond_fmt = '%0.2f (Condition)'
centroiding_num_threads = 1

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
    centroiding_iterations = 1

