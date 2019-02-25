import sys, os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QErrorMessage
from matplotlib import pyplot as plt
import datetime
import psutil

def get_process():
    return psutil.Process(os.getpid())

def get_ram():
    return (get_process().memory_info().rss)//1024//1024

def error_message(message):
    error_dialog = QErrorMessage()
    error_dialog.setWindowModality(Qt.WindowModal)
    error_dialog.showMessage(message)
    error_dialog.exec_()

def now_string():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def prepend(full_path_fn,prefix):
    p,f = os.path.split(full_path_fn)
    return os.path.join(p,'%s_%s'%(prefix,f))

def colortable(colormap_name):
    try:
        cmapobj = plt.get_cmap(colormap_name)
    except AttributeError as ae:
        print '\'%s\' is not a valid colormap name'%colormap_name
        print 'using \'bone\' instead'
        cmapobj = plt.get_cmap('bone')
    ncolors = cmapobj.N


    cmap = np.uint8(cmapobj(range(ncolors))[:,:3]*255.0)
    table = []
    for row in xrange(cmap.shape[0]):
        table.append(qRgb(cmap[row,0],cmap[row,1],cmap[row,2]))
    return table
