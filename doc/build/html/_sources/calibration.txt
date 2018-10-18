=====================================
Calibration of AO systems, using PyAO
=====================================

These instructions apply both to the first generation (1G) system and second generation (2G) system. Where differences exist, they will be explicitly addressed.

Setting up reference coordinates
--------------------------------

#. Flip up the mirror ("reference mirror", hereafter) just downstream of the OCT source entry to the sample channel. In both systems, this mirror back-reflects the beam toward the collimating lens, before which the beam is partially reflected by a pellicle toward the SHWS.

#. Using an IR card with a pinhole punched in it, verify that the OCT beam returns roughly to the center of the source collimator.

#. Using the real-time view of the OCT spectrum (or, alternatively, a power meter attached to the unused fiber of the OCT fiber beam splitter), adjust the tip and tilt of the reference mirror to maximize the intensity of detected light. Since the pupil of the system is finite, the spot formed at the fiber tip has a sinc profile with side lobes. As such, it is possible to mistake local maxima for the spot's true peak.

#. Run the script ``pyao/calibration/plot_spots_profiles.py``, which shows real-time marginal plots (vertical and horizontal averages) of the spots image. 

#. The number of peaks in both dimensions should be greater than or equal to the number of lenslets across the pupil. If the number of peaks is smaller than the number of lenslets across the pupil, increase the diameter of the aperture downstream of the collimator (2nd generation system only).

#. While viewing the peaks in the plots, translate the SHWS in the x- and y-directions, in order to center the spots image on the detector. In addition to centering the spots, it is good to ensure that the profiles are symmetric, mainly by looking at the peaks away from the center of the beam. If the number of lenslets across the pupil is even, the peaks should be centered about two rows/columns of spots (or an imaginary line between the two brightest peaks). If the number of lenslets is odd, the peaks should be centered about a single row/column of spots. The y- and x-dimension centerlines are shown on the plots as well.

#. While watching the marginal profile plots, rotate the tube containing the lenslet array until the peaks in both profiles are brightest. This is an indication that the spots are square with the axes of the image (i.e. the orientation of the SHWS sensor). It may be necessary to iterate the previous step and this one.

#. Stop ``plot_spots_profiles`` by typing Ctrl-C with the Python console active.

#. Next run the script ``pyao/calibration/make_reference_spots.py``. When it is finished running, launch a gui (e.g. ``pyao/pyao_gui/closed_loop_gui.py``) and check to see if the reference coordinates (and corresponding search boxes) overlap the beam well. If not, edit ``make_reference_spots.py``, and adjust the values of ``xoff`` and/or ``yoff`` accordingly. For example, if the search boxes are too low (by a row), decrease yoff by 1; if the search boxes are too far to the left, increase xoff by 1; and so forth. You should only need to adjust these values once. Run ``make_reference_spots.py`` again, and check the reference coordinates using ``closed_loop_gui.py`` again. 

#. Since the spots images should be nearly identical to the images used to generate the reference, the measured wavefront error should be very low--significantly less than the diffraction limit. Typical values here are 20 - 30 nm RMS. If the error is greater than that, try manually adjusting the threshold/DC level to minimize the error. If it is still too high, the previous steps should be repeated, with an emphasis on squaring the lenslet array to the detector and making the spots symmetrical about the appropriate horizontal and vertical midlines. A certain amount of error (20 - 30 nm) is unavoidable--due to the finite number of Zernike terms used to model the wavefront (fitting error), as well as photon noise.

#. Flip down the reference mirror. We have observed that flipping this mirror down and back up again, even very gently, changes the angle of reflection in both axes. In other words, you cannot move this mirror up and down during the course of reference coordinate measurement. If you flip it down and back up, you have to start again from step 1 above.

Measuring the poke matrix (aka system matrix aka influence function)
--------------------------------------------------------------------

#. Make sure the OCT scanner driver is on, and set the OCT software to scan angles of 0.0 and offsets of 0.0. Also enable a live view of the OCT spectrum (or, as above, use the unused fiber of the OCT beam splitter plugged into a power meter).

#. Continue running ``closed_loop_gui.py``, with the mirror flat and the loop open. Ideally a system flat should be used, i.e. a set of mirror currents (or voltages) that correct the bulk of the system aberrations. This will give the most sensitive measure of recoupling light into the OCT fiber, since the PSF will be steepest. If a system flat is unavailable, use another flat file, collect a system flat later, and repeat this section.

#. Disconnect the OCT source fiber, *downstream of its optical isolator*, and connect it with the calibration fiber (labeled 'alignment' in the 1G system).

#. Flip up the calibration flip mirror. Ideally, light from the calibration source will be visible in the OCT spectrum and in the spots image. The spots image may be highly saturated--this is expected.

#. In both the 1G and 2G systems, there are two mirrors in the calibration channels but not in the normal sample channel. These mirrors are to be used to align the calibration source. In both systems, the non-flipping mirror tends to be more useful 1) because it is closer to the system's pupil plane, thereby having a greater effect on spot position in the image plane, and 2) because it is fixed and the folding hinges of the other mirrors can be moved easily while turning their tip/tilt knobs.

#. Using the appropriate mirror in the calibration channel, adjust the tip and tilt of the mirror to maximize signal on the OCT spectrum (or power meter). As of this writing, in both systems it is possible to saturate the spectrometer camera, across almost the entire bandwidth of the spectrum. If this is possible, reduce the power of the source (by disconnecting, slightly, the fiber downstream of the optical isolator from the calibration/alignment fiber). Once the entire spectrum is visible again (unsaturated), further tip/tilt adjustments can be made using the mirror. If the spectrum saturates again, the fibers may need to be disconnected further, etc.

#. Due to system aberrations and/or an imperfect flat file for the deformable mirror, the previous step is susceptible to finding a local PSF peak rather than the global peak. The global peak will show a precipitous loss of intensity whichever way the mirror is tipped or tilted. Local maxima may not exhibit this behavior.

#. If a reliable system flat is in use, now is a good time to adjust the collimation of the calibration source (the distance between the fiber tip and the collimating lens), while monitoring two things: 1) the OCT spectrum (or power meter), and/or 2) the AO wavefront error. Ideally, wavefront error should be minimized while maximizing the OCT spectrum. If these two things do not occur at the same time, it may be an indication that the OCT source delivery collimation is poor. If the dominant source of wavefront error (when the OCT signal is maximized) is defocus, OCT source collimation is the likley suspect. If the dominant source is astigmatism, the OCT/SHWS pellicle may be the culprit.

#. Once the OCT spectrum has been maximized, further disconnect the fibers until the AO spots image is no longer saturated. If using ``closed_loop_gui.py``, keep an eye on the histogram at the lower right corner of the UI. We typically collect influence functions with the brightest pixels about half-way up the dynamic range of the camera. (~2048 for 12-bit cameras). If the intensity profile of the beam is not reasonably flat (i.e. if the peripheral spots are much dimmer than the central spots), brighter spots images may have to be generated by reconnecting the fibers slightly.

#. Quit ``closed_loop_gui.py`` and run ``influence_function_gui.py``. If parameters need to be adjusted, such as DC/threshold level, they may be adjusted now. When you are ready to collect the poke matrix, press *z* and wait.

#. When the poke matrix is collected, it is saved to the directory ``pyao_etc/ctrl/`` and named using the date and time of acquisition. The most recent one is the the one you want to test. Locate it and copy it's filename, then paste the filename in the correct place in ``pyao_etc/pyao_config.py``.

#. Continuing to use the calibration source, run ``closed_loop_gui.py`` again and try closing the loop. The residual wavefront error should converge rapidly on a value below the diffraction limit.

#. Finally, swap the fiber back to the OCT beamsplitter, put a model eye in the sample channel, and test the system. If the loop converges well, congratulations--you're done! After correcting, you can write a new flat file (Shift-F).




Notes
-----

#. This procedure assumes that the beam is collimated very well by the first collimator in the source launch. If the wavefront generated by the reference mirrors is not planar, errors in the reference coordinates will exist, and these will lead to convergence of the closed-loop wavefront on a non-planar shape. Most likely this will result in a beam that is focused away from the natural plane of focus of the AO system, and can be compensated by using the Zernike reference offsets (see ``pyao.sensors.applyReferenceOffsets``).

#. The effectiveness of the calibration procedure may be sensitive to the DC subtraction method employed (specified in ``pyao_config.py``); 

