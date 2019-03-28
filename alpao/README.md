# Methods for controlling the Alpao mirror

1. The old way, by exposing the functions in acedev5.dll and acecs.dll, using ctypes.

2. The new way, using the Python interface supplied by Alpao (SWIG-wrapper of the DLLs).

The goal is to offer both of these options, in case one of them doesn't work properly on a given system. In the previous version of the software, both existed but had very different APIs because they were written ad hoc to serve the purposes of the two AO systems. Now, I want to provide an interface based on ctypes but which looks and acts like the Python interface.

First, here's how the Alpao Python interface works:

```
from PyAcedev5 import *
import numpy as np
mirror_id = kungpao.config.mirror_id # this would be something like alpaoDM97-15-010
dm = PyAcedev5(mirror_id)
command = np.zeros(97,dtype=np.double) # dtype doesn't matter since we copy into dm.values
self.dm.values[:] = command[:]
self.dm.Send()
```

Now here's how the current version of the ctypes interface works:

```
from ctypes import *
from ctypes.util import find_library
import numpy as np

acecs = CDLL("acecs")
acedev5 = CDLL("acedev5")
mirror_id = acedev5.acedev5Init(0)
command = np.zeros(97,dtype=np.double) # here the dtype matters because we pass a pointer
ptr = command.ctypes.data_as(c_void_p)
result = acedev5.acedev5Send(mirror_id,ptr)
```