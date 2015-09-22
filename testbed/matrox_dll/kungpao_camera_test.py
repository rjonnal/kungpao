from ctypes import *
from ctypes.util import find_library
import numpy as np
import scipy as sp
from time import time,sleep
import sys
from matplotlib import pyplot as plt

def processing_function(hook_type_long, hook_id_longlong, hook_data_pointer):
    print hook_type_long
    #print hook_id_longlong
    #print hook_data_pointer
    #print 'hi'
    return 0

class HookDataStruct(Structure):
    _fields_ = [("MilImageDisp",c_longlong),
                ("ProcessedImageCount",c_long)]

UserHookData = HookDataStruct(0,0)

processing_function_type = WINFUNCTYPE(c_long,c_long,c_longlong,c_long)
processing_function_ptr = processing_function_type(processing_function)


kcam = windll.LoadLibrary('./kungpao_camera')

#kcam.setup_default()
system_name = c_char_p('M_SYSTEM_SOLIOS')
camera_filename = c_char_p('C:\\pyao_etc\\config\\dcf\\acA2040-180km-4tap-12bit_reloaded.dcf')
kcam.setup(system_name,camera_filename)
# sys.exit()

kcam.go.argtypes = [processing_function_type,HookDataStruct]
kcam.go(processing_function_ptr, UserHookData)
x = raw_input()
kcam.stop(processing_function_ptr, UserHookData)
