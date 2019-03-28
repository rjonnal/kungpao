#!/usr/bin/python
# -*- coding: utf8 -*-

import sys
import os
import time
import struct

''' Import DEv5 class '''
from alpao.PyAcedev5 import *

''' Start example '''
def main(args):
    serial = raw_input("Please enter the S/N within the following format BXXYYY (see DM backside): ")
    
    print("Connect the mirror")
    dm = PyAcedev5( serial )
    
    print("Retreive number of actuators")
    nbAct = dm.GetNbActuator()
    
    print("Send 0 on each actuators")
    dm.values = [0] * nbAct
    dm.Send()
    
    print("We will send on all actuator 12% for 1/2 second")
    for i in range( nbAct ):
        dm.values[i] = 0.12
        # Send values vector
        dm.Send()
        print("Send 0.12 on actuator "+str(i))
        time.sleep(0.5) # Wait for 0.5 second
        dm.values[i] = 0
    
    print("Send 0 on each actuators")
    dm.values = [0] * nbAct
    dm.Send()
    
    print("Get offset")
    offset = dm.GetOffset()
    print offset
    
    print("Reset")
    dm.DACReset()
    
    print("Exit")

if __name__=="__main__":
    main(sys.argv)