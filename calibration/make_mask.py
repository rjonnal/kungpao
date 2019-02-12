import numpy as np
import sys

if len(sys.argv)<4:
    print 'To run, call as follows:'
    print 'python make_mask.py N rad filename.txt'
    print 'where N is the width/height of the mask, rad is the radius in which to write ones'
    print 'e.g., for the SHWS use: python make_mask.py 20 9.6 reference_mask.txt'
    print 'and for the mirror mask use: python make_mask.py 11 5.5 mirror_mask.txt'
    sys.exit()
    
N = int(sys.argv[1])
rad = float(sys.argv[2])
outfn = sys.argv[3]


xx,yy = np.meshgrid(np.arange(N),np.arange(N))

xx = xx - float(N-1)/2.0
yy = yy - float(N-1)/2.0

d = np.sqrt(xx**2+yy**2)

mask = np.zeros(xx.shape,dtype=np.uint8)
mask[np.where(d<=rad)] = 1

np.savetxt(outfn,mask,fmt='%d')
print mask
print '%d active elements'%np.sum(mask)

