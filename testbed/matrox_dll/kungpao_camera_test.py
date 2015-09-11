from ctypes import *
from ctypes.util import find_library
import numpy as np
import scipy as sp
from time import time,sleep
import sys

def processing_function(hook_type_long, hook_id_longlong, hook_data_pointer):
    print hook_data_pointer
    return 0

#print pointer(UserHookData)
#print POINTER(HookDataStruct)
processing_function_type = WINFUNCTYPE(c_long,c_long,c_longlong,c_long)
processing_function_ptr = processing_function_type(processing_function)

kcam = windll.LoadLibrary('./kungpao_camera')
kcam.go(processing_function_ptr)
x = raw_input()
kcam.stop(processing_function_ptr)
