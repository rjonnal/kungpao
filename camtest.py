"""
.. module:: cameras
   :platform: Windows, Linux, Mac
   :synopsis: Provides an interface to the wavefront sensor's camera.

.. moduleauthor:: Ravi S. Jonnal <rjonnal@gmail.com>

"""
from ctypes import *
from ctypes.util import find_library
import numpy as np
import scipy as sp
from time import time,sleep
import sys
import kungpao_config

BUFFERING_SIZE_MAX = 7

from pyao import milc
# from utils import *
# from settings import dcfPath,dataPath
# import pyao_config

MilApplication = c_longlong()
MilSystem = c_longlong()
MilDigitizer = c_longlong()
MilDisplay = c_longlong()
MilImageDisp = c_longlong()
mil = windll.LoadLibrary("mil")

# attempt 1:
MilGrabBufferListType = c_longlong * BUFFERING_SIZE_MAX
MilGrabBufferList = MilGrabBufferListType()

# attempt 2:
#MilGrabBufferList = [c_longlong()]*BUFFERING_SIZE_MAX

ProcessFrameCount = c_long(0)
NbFrames = c_long(0)
ProcessFrameRate = c_double(0)

def printMilError(header='generic'):
    err = c_longlong(0)
    mil.MappGetError(milc.M_CURRENT,byref(err))
    errmsg = create_string_buffer(' ' * 64)
    mil.MappGetError(milc.M_CURRENT+milc.M_MESSAGE,byref(errmsg))
    print '%s: err %d, msg %s'%(header,err.value,errmsg.raw)

    
class HookDataStruct(Structure):

    _fields_ = [("MilImageDisp",c_longlong),
                ("ProcessedImageCount",c_long)]
                
    @classmethod
    def from_param(cls, obj):
        if obj is None:
            return c_void_p()
        else:
            return obj.c_ptr                

            

UserHookData = HookDataStruct()
def quit():
    print 'Closing camera...'
    mil.MdigHalt(MilDigitizer)
    printMilError()
    mil.MdigFree(MilDigitizer)
    printMilError()
    mil.MsysFree(MilSystem)
    printMilError()
    mil.MappFree(MilApplication)
    sys.exit()

mil.MappAllocW.argtypes = [c_longlong, POINTER(c_longlong)] 
mil.MsysAllocW.argtypes = [c_wchar_p, c_longlong, c_longlong, POINTER(c_longlong)]
mil.MdigAllocW.argtypes = [c_longlong, c_longlong, c_wchar_p, c_longlong, POINTER(c_longlong)]

InitFlag = c_longlong(milc.M_PARTIAL)
mil.MappAllocW(InitFlag,byref(MilApplication))
printMilError('MappAllocW')

cSystemName = c_wchar_p('M_SYSTEM_SOLIOS')
mil.MsysAllocW(cSystemName, milc.M_DEFAULT, InitFlag, byref(MilSystem))
printMilError('MsysAllocW')
print 'MIL App Identifier: %d'%(MilApplication.value)
print 'MIL Sys Identifier: %d'%(MilSystem.value)


cameraFilename = 'C:/pyao_etc/config/dcf/acA2040-180km-4tap-12bit_reloaded.dcf'
cDcfFn = c_wchar_p(cameraFilename)
mil.MdigAllocW(MilSystem, milc.M_DEFAULT, cDcfFn, milc.M_DEFAULT, byref(MilDigitizer))
printMilError('MdigAllocW')

xsize = c_long(kungpao_config.camera_physical_width)
ysize = c_long(kungpao_config.camera_physical_height)

for MilGrabBufferListSize in range(BUFFERING_SIZE_MAX):
    mil.MbufAlloc2d(MilSystem,
                    xsize,
                    ysize,
                    milc.M_DEF_IMAGE_TYPE,
                    milc.M_IMAGE+milc.M_GRAB+milc.M_PROC,
                    cast(MilGrabBufferList[MilGrabBufferListSize], POINTER(c_longlong)))
    printMilError('MbufAlloc2d')

# free one buffer for possible temporary buffering:
mil.MbufFree(cast(MilGrabBufferList[BUFFERING_SIZE_MAX-1], POINTER(c_longlong)))
printMilError('MbufFree')

UserHookData.MilImageDisp = MilImageDisp
UserHookData.ProcessedImageCount = c_long(0)

def processing_function(hook_type_long, hook_id_longlong, hook_data_pointer):
    print 'hi'
    return c_long(333)

#print pointer(UserHookData)
#print POINTER(HookDataStruct)
processing_function_ptr = WINFUNCTYPE(c_long,c_long,c_longlong,POINTER(HookDataStruct))

mil.MdigProcess.argtypes = [c_longlong,MilGrabBufferListType,c_long,c_longlong,c_longlong,processing_function_ptr,POINTER(HookDataStruct)]

mil.MdigProcess(MilDigitizer, MilGrabBufferList, BUFFERING_SIZE_MAX, milc.M_START, milc.M_DEFAULT,
                processing_function_ptr(processing_function), byref(UserHookData))
printMilError('MdigProcess')

# free all the buffers now that we're done:
for MilGrabBufferListSize in range(BUFFERING_SIZE_MAX):
    mil.MbufFree(cast(MilGrabBufferList[MilGrabBufferListSize], POINTER(c_longlong)))
    printMilError('MbufFree')

quit()



print self._cameraFilename

self._mil.MdigAllocW(self._MilSystem, milc.M_DEFAULT, cDcfFn, milc.M_DEFAULT, byref(self._MilDigitizer))
self.printMilError()
print 'MIL Dig Identifier: %d'%self._MilDigitizer.value




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

    
    
