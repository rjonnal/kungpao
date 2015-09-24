from ctypes import c_char_p, windll
import numpy as np
import scipy as sp
from time import time,sleep
import sys
from matplotlib import pyplot as plt



kcam = windll.LoadLibrary('./kungpao_camera')
system_name = c_char_p('M_SYSTEM_SOLIOS')
camera_filename = c_char_p('C:\\pyao_etc\\config\\dcf\\acA2040-180km-4tap-12bit_reloaded.dcf')
kcam.setup(system_name,camera_filename)



size_x = kcam.get_size_x()
size_y = kcam.get_size_y()
print size_x,size_y
im = np.zeros((size_y,size_x)).astype(np.uint16)
im_ptr = im.ctypes.data




kcam.start()

plt.figure()
ph = plt.imshow(im)
for k in range(10c):
    kcam.get_current_image(im_ptr)
    plt.cla()
    plt.imshow(im)
    #ph.set_array(im)
    plt.pause(.001)

kcam.stop()
plt.show()


