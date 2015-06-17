from kungpao.cameras import Camera
#from pyao.sensors import WavefrontSensor
#from pyao.mirrors import AOMirrorAlpao
#from pyao.loops import ClosedLoop

'''A minimal AO loop, written using PyAO.'''

cam = Camera()
wfs = WavefrontSensor(cam)
dm = AOMirrorAlpao()
loop = ClosedLoop(wfs,dm)

while True:
    loop.step()
    

