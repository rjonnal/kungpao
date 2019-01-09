import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
import config
from zernike import Zernike
import scipy.interpolate as spi

class Beam:
    def __init__(self):
        self.diameter = config.BEAM_DIAMETER_M
        NZ = config.N_ZERNIKE
        self.zc = np.zeros(NZ)
        N = config.N_WAVEFRONT
        self.gaussian_sigma = config.GAUSSIAN_BEAM_SIGMA_M
        self.XX,self.YY = np.meshgrid(np.linspace(-self.diameter/2.0,+self.diameter/2.0,N),np.linspace(-self.diameter/2.0,+self.diameter/2.0,N))
        self.beam_profile = np.exp(-(self.XX**2+self.YY**2)/(2*self.gaussian_sigma**2))
        self.beam_profile = self.beam_profile/np.sum(self.beam_profile)

        ### Prepopulate Zernike surfaces
        self.zernike = Zernike()
        self.zernike_surfaces = []
        # Set up some aligned vectors of Zernike index (j), order (n),
        # frequency (m), and tuples of order and frequency (nm).
        j_vec = range(len(self.zc))
        nm_vec = np.array([self.zernike.j2nm(x) for x in j_vec])
        n_vec = np.array([self.zernike.j2nm(x)[0] for x in j_vec])
        m_vec = np.array([self.zernike.j2nm(x)[1] for x in j_vec])
        maxOrder = n_vec.max()
        
        unity = 1.0
        xx,yy = np.meshgrid(np.linspace(-unity,unity,N),np.linspace(-unity,unity,N))
        self.mask = np.zeros((N,N)) # circular mask to define unit pupil
        d = np.sqrt(xx**2 + yy**2)
        self.mask[np.where(d<unity)] = 1
        # We build up the simulated wavefront by summing the modes of
        # interest. The simulated wavefront has units of pupil radius; to
        # convert to meters, we have to multiply by the pupil radius in
        # meters.
        for (n,m) in nm_vec:
            self.zernike_surfaces.append(self.zernike.getSurface(n,m,xx,yy,'h'))
            print 'Computing zernike mode %d,%d'%(n,m)

    def set_zernike_coefficients(self,coefs):
        self.zc = coefs

    def get_phase(self):
        phase = np.zeros(self.XX.shape) # matrix to accumulate wavefront error

        # We build up the simulated wavefront by summing the modes of
        # interest. The simulated wavefront has units of pupil radius; to
        # convert to meters, we have to multiply by the pupil radius in
        # meters.
        phase = self.zernike_surfaces[0]*self.zc[0]
        for surf,coef in zip(self.zernike_surfaces[1:],self.zc[1:]):
            phase = phase + surf*coef
        return phase*self.mask

    def get_pupil_function(self):
        return self.beam_profile*np.exp(self.get_phase()*1j)

    def get_psf(self):
        return np.abs(np.fft.fftshift(np.fft.fft2(self.get_pupil_function())))

    

class SHWS:
    """Represents a Shack-Hartmann wavefront sensor.
    Should take a Wavefront object and return a Spots
    image."""

    def __init__(self):
        # Don't configure the sensor here; use more readable configuration
        # parameters in config.py

        # d and r of beam measured in lenslets
        beam_diameter = config.BEAM_DIAMETER_M
        lenslet_pitch_m = config.LENSLET_PITCH_M
        pixel_size = config.PIXEL_SIZE_M
        lenslet_pitch_px = lenslet_pitch_m/pixel_size
        diameter_lenslets = int(np.ceil(beam_diameter/lenslet_pitch_m))
        radius_lenslets = diameter_lenslets/2.0
        sensor_x = config.SENSOR_X
        sensor_y = config.SENSOR_Y

        
        
        # Make a mask of the subapertures, using -1 where there's no beam and a
        # non-negative integer where there is one. The latter number will be used
        # to index the lenslets.
        self.subaperture_mask = np.ones((diameter_lenslets,diameter_lenslets))*(-1)
        XX,YY = np.meshgrid(np.arange(diameter_lenslets)-(diameter_lenslets-1)/2.0,np.arange(diameter_lenslets)-(diameter_lenslets-1)/2.0)
        d = np.sqrt(XX**2+YY**2)
        yvec,xvec = np.where(d<=radius_lenslets-config.MASK_PADDING)
        self.n_lenslets = len(yvec)

        self.lenslet_array = np.ones((sensor_y,sensor_x))
        self.ref_x = np.zeros(self.n_lenslets)
        self.ref_y = np.zeros(self.n_lenslets)

        XX,YY = np.meshgrid(np.arange(sensor_x),np.arange(sensor_y))
        self.sensor_XX = XX
        self.sensor_YY = YY

        # Make a centered mask
        centered_xx = self.sensor_XX - self.sensor_XX.mean()
        centered_yy = self.sensor_YY - self.sensor_YY.mean()
        centered_d = np.sqrt(centered_xx**2+centered_yy**2)
        self.centered_mask = np.zeros(XX.shape)
        self.centered_mask[np.where(centered_d<=lenslet_pitch_px/2.0)] = 1.0
        self.centered_y_coords, self.centered_x_coords = np.where(self.centered_mask)
        
        try:
            self.subaperture_mask = np.load(os.path.join('cache','subaperture_mask.npy'))
            self.ref_x = np.load(os.path.join('cache','ref_x.npy'))
            self.ref_y = np.load(os.path.join('cache','ref_y.npy'))
            self.lenslet_array = np.load(os.path.join('cache','lenslet_array.npy'))
        except Exception as e:
            for index,(y,x) in enumerate(zip(yvec,xvec)):
                self.subaperture_mask[y,x] = index
                ref_x = x*lenslet_pitch_px+lenslet_pitch_px/2.0
                ref_y = y*lenslet_pitch_px+lenslet_pitch_px/2.0
                print 'Configuring lenslet %d at %0.1f,%0.1f'%(index,ref_y,ref_x)
                self.ref_x[index] = ref_x
                self.ref_y[index] = ref_y
                xx = XX - ref_x
                yy = YY - ref_y
                d = np.sqrt(xx**2+yy**2)
                self.lenslet_array[np.where(d<=lenslet_pitch_px/2.0)]=index
                np.save(os.path.join('cache','subaperture_mask.npy'),self.subaperture_mask)
                np.save(os.path.join('cache','ref_x.npy'),self.ref_x)
                np.save(os.path.join('cache','ref_y.npy'),self.ref_y)
                np.save(os.path.join('cache','lenslet_array.npy'),self.lenslet_array)
            
    def get_spots_image(self,beam):
        pf = beam.get_pupil_function()
        spots_image = np.zeros(pf.shape)

        get_x_coords = self.centered_x_coords
        get_y_coords = self.centered_y_coords
        
        for k in range(self.n_lenslets):
            print k
            ref_x = self.ref_x[k]
            ref_y = self.ref_y[k]
            put_x_coords = np.round(get_x_coords-get_x_coords.mean()+ref_x).astype(np.integer)
            put_y_coords = np.round(get_y_coords-get_y_coords.mean()+ref_y).astype(np.integer)

            subpupil = np.zeros(spots_image.shape,dtype=np.complex)
            subpupil[np.where(self.lenslet_array==k)] = pf[np.where(self.lenslet_array==k)]
            sub_spots = np.abs(np.fft.fftshift(np.fft.fft2(subpupil)))

            for xg,yg,xp,yp in zip(get_x_coords,get_y_coords,put_x_coords,put_y_coords):
                try:
                    spots_image[yp,xp] = sub_spots[yg,xg]
                except:
                    pass
            
        plt.imshow(spots_image)
        plt.colorbar()
        plt.show()
        
if __name__=='__main__':
    shws = SHWS()
    beam = Beam()
    coefs = np.random.randn(config.N_ZERNIKE)
    #beam.set_zernike_coefficients(coefs)
    shws.get_spots_image(beam)
    sys.exit()
    #coefs = np.zeros(5)
    #coefs[4] = 1.0
    #coefs[:4] = 0.0
    coefs[:3] = 0.0
    wf = beam.get_phase()
    psf = beam.get_psf()
    psf_profile = np.max(psf,axis=0)
    plt.plot(psf_profile)
    plt.show()

