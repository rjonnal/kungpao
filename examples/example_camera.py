from kungpao.cameras import Camera

'''Interacting with a camera using kungpao.

'''

cam = Camera()
wfs = WavefrontSensor(cam)
dm = AOMirrorAlpao()
loop = ClosedLoop(wfs,dm)

while True:
    loop.step()
    

