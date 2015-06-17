"""
.. module:: cameras
   :platform: Windows, Linux, Mac
   :synopsis: Provides an interface to the wavefront sensor's camera.

.. moduleauthor:: Ravi S. Jonnal <rjonnal@gmail.com>

"""
from ctypes import *
from ctypes.util import find_library
import numpy as np
from numpy.fft import fft,ifft,fft2,ifft2,fftshift
from time import time,sleep
import sys
from types import *
from glob import glob
import scipy

# import milc
# from utils import *
# from settings import dcfPath,dataPath
# import pyao_config


class Camera:
    """A template for subclassing AO camera classes.
    A the functions `grab` and

    """
    
    def grab(self):
        """Updates this object's image. Can be implemented in a variety
        of ways, depending on hardware (using the camera's API, in combination
        with ctypes (see :mod:`pyao.cameras.AOCameraMatrox`) or by reading 
        images from the disk (see :mod:`pyao.cameras.AOCameraSim`), for 
        example.)

        Args:
            none
        Kwargs:
            none
        Returns:
            
        """
        pass

    def getImage(self):
        """Calls this object's updateImage() and gets its _im array.

        Args:
            none
        Kwargs:
            none
        Returns:
            numpy.ndarray of type numpy.uint16, the image

        """
        self.updateImage()
        return self._im

    def getImagePtr(self):
        """Gets a pointer to this object's _im array.

        Args:
            none
        Kwargs:
            none
        Returns:
            ctypes pointer, reference to the image

        """
        return self._im_ptr

    def getShape(self):
        """Gets this object's shape. For example:

            >>> imHeight,imWidth = cam.getShape()
            >>> print imHeight
            512

        Args:
            none
        Kwargs:
            none
        Returns:
            tuple containing height,width of image

        """
        return self._im.shape

    def getIntegrationTime(self):
        """Gets this object's integration time, in seconds.

        Args:
            none
        Kwargs:
            none
        Returns:
            float, integration time in seconds

        """
        return self._integrationTimeS

    def getPixelSize(self):
        """Gets this object's physical pixel size, in meters.

        Args:
            none
        Kwargs:
            none
        Returns:
            float, size of binned pixel in meters

        """
        return self._pixelSizeM

    def close(self):
        """Close this camera interface.

        Args:
            none
        Kwargs:
            none
        Returns:
            None

        """
        pass

    def save(self,filename):
        np.save(filename,self._im)


# class AOCameraSim(AOCamera):
#     """A class representing a simulated camera"""
    
#     _physicalPixelSizeM = 12.0e-6
#     _binningPx = 2.0
#     _pixelSizeM = _physicalPixelSizeM * _binningPx
#     _imageindex = -1


#     def __init__(self,noisy=False,binning=1,dataPathIn=None,nImages=None):
#         if dataPathIn is None:
#             self.dataPath = dataPath
#         else:
#             self.dataPath = dataPathIn

#         self._noisy = noisy
#         self.binning = binning
#         self.nImages = nImages
#         self.loadImages()
        
#     def loadImages(self):
#         imfnlist = glob(os.path.join(self.dataPath,'*.npy'))
#         if self.nImages is None:
#             imfnlist = sorted(imfnlist)
#         else:
#             imfnlist = sorted(imfnlist)[:self.nImages]

#         self.ims = []
#         for imfn in imfnlist:
#             print 'AOCameraSim: loading %s'%imfn
#             im = np.load(imfn).astype(np.int16)
#             self.ims.append(im)

#     def updateImage1(self):
#         self._imageindex = (self._imageindex + 1)%20
#         fn = dataPath+'im_%02d.npy'%self._imageindex
#         self._im = np.load(fn)

#     def updateImage(self):
#         self._imageindex = (self._imageindex + 1)%len(self.ims)
#         # Important: when assigning one of the images from the image list,
#         # make sure to copy it, otherwise the 'original', in the list,
#         # will be modified after any in-place operations.
#         self._im = self.ims[self._imageindex].copy()
#         if self._noisy:
#             # coarse shot noise simulation:
#             self._im = self._im + (np.random.randn(*(self._im.shape))*np.sqrt(self._im)).astype(np.int16)


# class AOCameraAce(AOCamera):


#     try:
#         val = pyao_config.cameraBinningPx
#         if type(val)!=IntType:
#             raise Exception('cameraBinningPx should be an integer')
#         _binningPx = float(val)
#     except Exception as e:
#         print 'Configuration file error:',e
#         sys.exit()

#     try:
#         val = pyao_config.cameraPixelSizeM
#         if type(val)!=FloatType:
#             raise Exception('cameraPixelSizeM should be a float')
#         _physicalPixelSizeM = val
#     except Exception as e:
#         print 'Configuration file error:',e
#         sys.exit()

#     _physicalPixelSizeM = pyao_config.cameraPixelSizeM

#     _pixelSizeM = _physicalPixelSizeM * _binningPx

#     try:
#         val = pyao_config.cameraFilename
#         if type(val)!=StringType:
#             raise Exception('cameraFilename should be a string')
#         _cameraFilename = val
#     except Exception as e:
#         print 'Configuration file error:',e
#         sys.exit()


#     _MilImage0 = c_longlong(0)
#     _MilImage1 = c_longlong(0)
#     _InitFlag = c_longlong(milc.M_PARTIAL)
#     #_InitFlagD = c_longlong(milc.M_DEFAULT)


#     _MilApplication = c_longlong()
#     _MilSystem = c_longlong()
#     _MilDigitizer = c_longlong()
#     _MilImage0 = c_longlong()
#     _MilImage1 = c_longlong()

#     def __init__(self,mode=0,binning=pyao_config.cameraBinningPx):


#         if sys.platform=='win32':
#             self._mil = windll.LoadLibrary("mil")
#         else:
#             sys.exit('pyao.cameras assumes Windows DLL shared libraries')



#         self._mode = mode

#         self._mil.MappAllocW.argtypes = [c_longlong, POINTER(c_longlong)] 
#         self._mil.MsysAllocW.argtypes = [c_wchar_p, c_longlong, c_longlong, POINTER(c_longlong)]
#         self._mil.MdigAllocW.argtypes = [c_longlong, c_longlong, c_wchar_p, c_longlong, POINTER(c_longlong)]
#         self._mil.MbufAllocColor.argtypes = [c_longlong, c_longlong, c_longlong, c_longlong, c_longlong, c_longlong, POINTER(c_longlong)]

#         self._mil.MappAllocW(self._InitFlag,byref(self._MilApplication))
#         self.printMilError()
#         print 'MIL App Identifier: %d'%self._MilApplication.value

#         cSystemName = c_wchar_p('M_SYSTEM_SOLIOS')
#         self._mil.MsysAllocW(cSystemName, milc.M_DEFAULT, 
#                            self._InitFlag, byref(self._MilSystem))
#         self.printMilError()
#         print 'MIL Sys Identifier: %d'%self._MilSystem.value

        
#         cDcfFn = c_wchar_p(self._cameraFilename)
#         print self._cameraFilename

#         self._mil.MdigAllocW(self._MilSystem, milc.M_DEFAULT, cDcfFn, milc.M_DEFAULT, byref(self._MilDigitizer))
#         self.printMilError()
#         print 'MIL Dig Identifier: %d'%self._MilDigitizer.value

#         nBands = c_longlong(1)
#         bufferType = c_longlong(milc.M_SIGNED + 16)
#         bufferAttribute = c_longlong(milc.M_GRAB + milc.M_IMAGE)
        
#         self._xSizePx = 2048/binning
#         self._ySizePx = 2048/binning
#         #print binning
#         #sys.exit()

#         if self._mode==0: # double buffer / continuous grab
#             self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage0))
#             self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage1))
#             self.printMilError()
#             print 'MIL Img Identifiers: %d,%d'%(self._MilImage0.value, self._MilImage1.value)

#             self._mil.MdigGrabContinuous(self._MilDigitizer,self._MilImage0)
#             self.printMilError()
#         elif self._mode==1: # double buffer / single grabs
#             self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage0))
#             self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage1))
#             self.printMilError()
#             print 'MIL Img Identifiers: %d,%d'%(self._MilImage0.value, self._MilImage1.value)
#             self._mil.MdigControlInt64(self._MilDigitizer,milc.M_GRAB_MODE,milc.M_ASYNCHRONOUS)
#             self._mil.MdigGrab(self._MilDigitizer,self._MilImage0)
#         elif self._mode==2: # single buffer / single grabs
#             self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage0))
#             self.printMilError()
#             print 'MIL Img Identifiers: %d,%d'%(self._MilImage0.value, self._MilImage1.value)
#             self._mil.MdigControlInt64(self._MilDigitizer,milc.M_GRAB_MODE,milc.M_ASYNCHRONOUS)
#             self._mil.MdigGrab(self._MilDigitizer,self._MilImage0)
            
            


#         self._im = np.zeros([np.int16(self._ySizePx),np.int16(self._xSizePx)]).astype(np.int16)
#         self._im_ptr = self._im.ctypes.data


        

#         # there must be a bug in camera initialization code above because if the camera has been
#         # sitting, on, for a while, the first few frames (up to 3, anecdotally) may have serious
#         # geometry problems (via, it seems, tap reordering). A quick and dirty fix: grab a few images
#         # upon initialization:
#         nBad = 5
#         for iBad in range(nBad):
#             bad = self.getImage()


#     def close(self):
#         print 'Closing camera...'
#         self._mil.MdigHalt(self._MilDigitizer)
#         self.printMilError()
#         self._mil.MbufFree(self._MilImage0)
#         self.printMilError()
#         self._mil.MbufFree(self._MilImage1)
#         self.printMilError()
#         self._mil.MdigFree(self._MilDigitizer)
#         self.printMilError()
#         self._mil.MsysFree(self._MilSystem)
#         self.printMilError()
#         self._mil.MappFree(self._MilApplication)

#     def updateImage(self):
#         t0 = time()
#         if self._mode==0:
#             self._mil.MdigGrabWait(self._MilDigitizer, milc.M_GRAB_FRAME_END );
#             self._mil.MbufCopy(self._MilImage0,self._MilImage1)
#             self._mil.MbufGet(self._MilImage1,self._im_ptr)
#         elif self._mode==1:
#             self._mil.MdigGrabWait(self._MilDigitizer, milc.M_GRAB_FRAME_END )
#             self._mil.MbufCopy(self._MilImage0,self._MilImage1)
#             self._mil.MdigGrab(self._MilDigitizer,self._MilImage0)
#             self._mil.MbufGet(self._MilImage1,self._im_ptr)
#         elif self._mode==2:
#             #self._mil.MdigGrabWait(self._MilDigitizer, milc.M_GRAB_FRAME_END )
#             self._mil.MdigGrab(self._MilDigitizer,self._MilImage0)
#             self._mil.MbufGet(self._MilImage0,self._im_ptr)
#         t1 = time()
            
#     def printMilError(self):
#         err = c_longlong(0)
#         self._mil.MappGetError(2L,byref(err))
#         print 'MIL Error Code: %d'%err.value
    


# class AOCameraMatrox(AOCamera):
#     """A class representing a Matrox camera"""
#     if sys.platform=='win32':
#         _mil = windll.LoadLibrary("mil")

#     try:
#         val = pyao_config.cameraBinningPx
#         if type(val)!=IntType:
#             raise Exception('cameraBinningPx should be an integer')
#         _binningPx = float(val)
#     except Exception as e:
#         print 'Configuration file error:',e
#         sys.exit()

#     try:
#         val = pyao_config.cameraPixelSizeM
#         if type(val)!=FloatType:
#             raise Exception('cameraPixelSizeM should be a float')
#         _physicalPixelSizeM = val
#     except Exception as e:
#         print 'Configuration file error:',e
#         sys.exit()



#     _physicalPixelSizeM = pyao_config.cameraPixelSizeM

#     _pixelSizeM = _physicalPixelSizeM * _binningPx

#     try:
#         val = pyao_config.cameraFilename
#         if type(val)!=StringType:
#             raise Exception('cameraFilename should be a string')
#         _cameraFilename = val
#     except Exception as e:
#         print 'Configuration file error:',e
#         sys.exit()

#     _MilApplication = c_long(0)
#     _MilSystem = c_long(0)
#     _MilDisplay = c_long(0)
#     _MilDigitizer = c_long(0)
#     _MilImage0 = c_long(0)
#     _MilImage1 = c_long(0)
#     _InitFlag = milc.M_PARTIAL

#     try:
#         val = pyao_config.cameraIntegrationTimeS
#         if type(val)!=FloatType:
#             raise Exception('integration time should be a float')
#         _integrationTimeS = val
#     except Exception as e:
#         print 'Configuration file error:',e
#         sys.exit()

#     def __init__(self):
#         if sys.platform=='win32':
# #            mil = windll.LoadLibrary("mil")
# #            mil = CDLL("mil")
#             pass
#         else:
#             err = 'Attempting to load AOCameraMatrox on non-Windows32 platform. Please use AOCameraSim instead.'
#             sys.exit(err)

#         self._mil.MappAlloc(self._InitFlag,byref(self._MilApplication))

#         self._mil.MsysAlloc(milc.M_DEF_SYSTEM_TYPE, milc.M_DEF_SYSTEM_NUM, 
#                            self._InitFlag, byref(self._MilSystem))

#         self._mil.MdigAlloc(self._MilSystem, milc.M_DEFAULT, 
#                            self._cameraFilename, milc.M_DEFAULT, 
#                            byref(self._MilDigitizer))

#         self._mil.MdigControl(self._MilDigitizer, milc.M_GRAB_SCALE_X, 
#                              c_double( 1.0 / self._binningPx ) );

#         try:
#             binY = pyao_config.cameraBinY
#         except Exception:
#             binY = False

#         if binY:
#             self._mil.MdigControl(self._MilDigitizer, milc.M_GRAB_SCALE_Y, 
#                                   c_double( 1.0 / self._binningPx ) );

#         self._xSizePx = c_long(long(self._mil.MdigInquire(
#                     self._MilDigitizer,milc.M_SIZE_X,None)/self._binningPx))

#         self._ySizePx = c_long(long(self._mil.MdigInquire(
#                     self._MilDigitizer,milc.M_SIZE_Y,None)/self._binningPx))

#         attribute0 = c_longlong(milc.M_IMAGE + milc.M_GRAB + milc.M_PROC 
#                                 + milc.M_ON_BOARD)
#         attribute1 = c_longlong(milc.M_IMAGE + milc.M_GRAB + milc.M_PROC)
#         type = c_long(16 + milc.M_UNSIGNED)
#         self._mil.MbufAlloc2d(self._MilSystem, self._xSizePx, self._ySizePx, type, 
#                              attribute0, byref(self._MilImage0) )
#         self._mil.MbufAlloc2d(self._MilSystem, self._xSizePx, self._ySizePx, type, 
#                              attribute1, byref(self._MilImage1) )

#         self._mil.MdigGrabContinuous(self._MilDigitizer,self._MilImage0)

#         #self._im = np.zeros([np.int16(self._ySizePx),np.int16(self._xSizePx)]).astype(np.uint16)
#         self._im = np.zeros([np.int16(self._ySizePx),np.int16(self._xSizePx)]).astype(np.int16)
#         self._im_ptr = self._im.ctypes.data


#     def close(self):
#         self._mil.MdigHalt(self._MilDigitizer)
#         self._mil.MbufFree(self._MilImage0)
#         self._mil.MbufFree(self._MilImage1)
#         self._mil.MdigFree(self._MilDigitizer)
#         self._mil.MsysFree(self._MilSystem)
#         self._mil.MappFree(self._MilApplication)

#     def updateImage(self):
#         # to profile the loop, comment out the wait command below; this
#         # may result in the same image being grabbed more than once, but
#         # gives an estimate of the loop rate based only on wavefront
#         # computations
#         self._mil.MdigGrabWait(self._MilDigitizer, milc.M_GRAB_NEXT_FRAME );
#         self._mil.MbufCopy(self._MilImage0,self._MilImage1)
#         self._mil.MbufGet(self._MilImage1,self._im_ptr)
#         sy,sx = self.getShape()
#         self._im[sy-2:sy,:] = self._im[sy-4:sy-2,:]

#     def setExposureTime(self,expTime=.017):
#         #sprintf_s( cmdStr, sizeof( cmdStr ), "set %f\r", integrationTime );
#         #MdigControl( MilDigitizer, M_UART_WRITE_STRING, M_PTR_TO_DOUBLE( cmdStr ) );
#         #cmdStr = 'set %f\r'%expTime
#         #self._mil.MdigControl(self._MilDigitizer,milc.M_UART_WRITE_STRING,cmdStr)
#         x = c_double(10.0)
#         self._mil.MdigInquire(self._MilDigitizer,milc.M_GRAB_EXPOSURE_TIME,byref(x))
#         self._mil.MdigControl(self._MilDigitizer,milc.M_GRAB_EXPOSURE_TIME,c_double(expTime))

    
    
